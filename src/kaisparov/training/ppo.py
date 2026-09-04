"""Minimal, self-contained PPO implementation for graph-based chess agents.

Design notes
------------
* Model-agnostic: the rollout stores everything the update needs (the graph
  ``Data``, the chosen action index, the legal-action mask, old log-prob/value,
  reward and done). ``train_one_epoch`` never touches engine internals.
* Self-play credit assignment: advantages use negamax GAE (``self_play=True``),
  flipping the opponent's value/advantage so a move that lets the opponent win
  gets a negative advantage. Set ``self_play=False`` for a plain single-agent
  trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data


@dataclass
class PPOBuffer:
    """Fixed-horizon rollout buffer with GAE advantage estimation."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    self_play: bool = True

    states: list[Data] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    legal_masks: list[Tensor] = field(default_factory=list)

    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.legal_masks.clear()
        self.advantages.clear()
        self.returns.clear()

    def __len__(self) -> int:
        return len(self.actions)

    def is_empty(self) -> bool:
        return len(self) == 0

    def add(
        self,
        *,
        state: Data,
        action: int,
        log_prob: Tensor | float,
        value: Tensor | float,
        reward: float,
        done: bool,
        legal_mask: Tensor,
    ) -> None:
        self.states.append(state.cpu())
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.legal_masks.append(legal_mask.detach().cpu())

    def compute_returns_and_advantages(self) -> None:
        """Fill ``advantages`` and ``returns`` with GAE, resetting at episode ends.

        In self-play mode (the default) consecutive steps belong to opposing
        players, so bootstrapping and advantage accumulation use the opponent's
        value/advantage negated (negamax): from the mover's viewpoint the next
        state is worth ``-V(next)`` and the trailing advantage flips sign. This
        gives correct zero-sum credit — a move that lets the opponent win gets a
        negative advantage.
        """
        n = len(self)
        self.advantages = [0.0] * n
        self.returns = [0.0] * n
        flip = -1.0 if self.self_play else 1.0

        gae = 0.0
        for t in reversed(range(n)):
            done = self.dones[t]
            next_value = 0.0 if (done or t + 1 >= n) else flip * self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            gae = delta if done else delta + self.gamma * self.gae_lambda * flip * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]


def _masked_categorical(logits: Tensor, legal_mask: Tensor) -> torch.distributions.Categorical:
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    return torch.distributions.Categorical(logits=masked)


def train_one_epoch(
    agent: torch.nn.Module,
    buffer: PPOBuffer,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
    update_epochs: int = 4,
    clip_eps: float = 0.2,
    value_coef: float = 0.25,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    normalize_advantages: bool = True,
) -> dict[str, float]:
    """Run several PPO update passes over the collected buffer.

    Returns a dict of scalar metrics (loss components + diagnostics).
    """
    if buffer.is_empty():
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "loss": 0.0,
            "steps": 0.0,
            "nonfinite_skips": 0.0,
        }

    buffer.compute_returns_and_advantages()
    device = torch.device(device)
    agent.train()

    states = buffer.states
    edge_counts = [int(s.edge_index.shape[1]) for s in states]
    batch = Batch.from_data_list(states).to(device)

    actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    returns = torch.tensor(buffer.returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(buffer.advantages, dtype=torch.float32, device=device)
    legal_masks = [m.to(device) for m in buffer.legal_masks]

    if normalize_advantages and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "loss": 0.0}
    nonfinite_skips = 0
    for _ in range(update_epochs):
        action_scores, values = agent(batch)
        values = values.reshape(-1)
        per_graph_scores = torch.split(action_scores, edge_counts, dim=0)

        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        for i, scores in enumerate(per_graph_scores):
            dist = _masked_categorical(scores, legal_masks[i])
            log_probs.append(dist.log_prob(actions[i]))
            entropies.append(dist.entropy())
        new_log_probs = torch.stack(log_probs)
        entropy = torch.stack(entropies).mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        # Huber (smooth L1) instead of MSE: robust to the occasional huge return
        # (a king capture is worth many material swings), so a single outlier can't
        # dominate the critic gradient and inflate the shared trunk.
        value_loss = F.smooth_l1_loss(values, returns)
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        # A non-finite loss (e.g. exp(ratio) overflow) would turn every gradient into
        # NaN; clip_grad_norm_ then propagates it (NaN norm -> NaN clip coef) and the
        # optimizer writes NaN into every weight. Skip the step instead of corrupting
        # the model — the next batch usually recovers.
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            nonfinite_skips += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        metrics["policy_loss"] = float(policy_loss.item())
        metrics["value_loss"] = float(value_loss.item())
        metrics["entropy"] = float(entropy.item())
        metrics["loss"] = float(loss.item())

    metrics["steps"] = float(len(buffer))
    metrics["nonfinite_skips"] = float(nonfinite_skips)
    return metrics


__all__ = ["PPOBuffer", "train_one_epoch"]
