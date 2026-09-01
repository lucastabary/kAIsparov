"""Config-driven PPO self-play trainer with tracking and periodic evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.core.board import ChessGame
from kaisparov.eval.arena import evaluate
from kaisparov.models.factory import load_backend, load_backend_spec
from kaisparov.tracking.run import RunManager
from kaisparov.training.config import TrainConfig
from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum
from kaisparov.training.reward import make_reward_fn

BEST_METRIC = "elo_vs_random"


def _resolve_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        _seed_everything(config.seed)
        self.device = _resolve_device(config.device)

        self.module = load_backend(config.model)
        self.spec = load_backend_spec(config.model)

        self.agent = self.spec.model_class.create_agent(
            device=self.device, hidden_dim=config.hidden_dim
        )
        self.optimizer = self.spec.model_class.create_optimizer(
            self.agent, learning_rate=config.ppo.learning_rate
        )
        self.buffer = self.spec.buffer_class(
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            self_play=config.ppo.self_play,
        )
        self.curriculum = PieceCountCurriculum(
            PhaseConfig(
                name=config.curriculum.name,
                max_pieces_per_side=config.curriculum.max_pieces_per_side,
                allow_major=config.curriculum.allow_major,
                allow_minor=config.curriculum.allow_minor,
                allow_pawns=config.curriculum.allow_pawns,
            ),
            seed=config.seed,
        )
        self._processor = self.spec.processor_class()
        self._num_params = sum(p.numel() for p in self.agent.parameters())
        self.reward_fn = make_reward_fn(config.reward)

        # Resume: load weights, optimizer + RNG state, and continue epoch numbering.
        self.start_epoch = 0
        if config.resume_from is not None:
            self.agent.load_state_dict(torch.load(config.resume_from, map_location=self.device))
            print(f"Resumed weights from {config.resume_from}")
            self._restore_trainer_state(config.resume_from)
            if config.parent_run_id is not None:
                from kaisparov.tracking.registry import Registry

                parent = Registry(config.runs_dir).get(config.parent_run_id)
                self.start_epoch = int(parent.get("epochs_completed", 0))

        self.run = RunManager(
            root=config.runs_dir,
            model=config.model,
            config=config.to_dict(),
            seed=config.seed,
            device=str(self.device),
            num_params=self._num_params,
            title=config.title,
            description=config.description,
            notes=config.notes,
            parent_run_id=config.parent_run_id,
            resumed_from=config.resume_from,
        )

    # -------------------------------------------------------------- resume state
    def _trainer_state(self, epoch: int) -> dict:
        """Optimizer + RNG snapshot, saved beside a checkpoint for exact resume."""
        return {
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

    def _restore_trainer_state(self, model_path: str) -> None:
        state_path = Path(model_path).with_name(Path(model_path).stem + ".state.pth")
        if not state_path.exists():
            print("(no optimizer/RNG state beside the checkpoint — optimizer starts fresh)")
            return
        state = torch.load(state_path, map_location="cpu")
        self.optimizer.load_state_dict(state["optimizer"])
        for opt_state in self.optimizer.state.values():
            for key, value in opt_state.items():
                if isinstance(value, torch.Tensor):
                    opt_state[key] = value.to(self.device)
        rng = state.get("rng", {})
        if rng:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        print(f"Restored optimizer + RNG state from {state_path.name}")

    # ------------------------------------------------------------------ training
    def train(self) -> str:
        cfg = self.config
        print(f"Run {self.run.run_id} | device={self.device} | params={self._num_params}")
        if cfg.title:
            print(f"  {cfg.title}")
        game = ChessGame()
        last_eval: dict[str, float] = {}
        metrics: dict[str, float] = {}

        with self.run:
            for epoch in range(self.start_epoch + 1, self.start_epoch + cfg.epochs + 1):
                self.buffer.clear()
                self.spec.collect_data(
                    self.agent,
                    game,
                    self.buffer,
                    num_episodes=cfg.rollout.episodes_per_epoch,
                    max_steps_per_episode=cfg.rollout.max_steps_per_episode,
                    model_module=self.module,
                    curriculum=self.curriculum,
                    reward_fn=self.reward_fn,
                )
                metrics = self.spec.train_one_epoch(
                    self.agent,
                    self.buffer,
                    self.optimizer,
                    device=self.device,
                    update_epochs=cfg.ppo.update_epochs,
                    clip_eps=cfg.ppo.clip_eps,
                    value_coef=cfg.ppo.value_coef,
                    entropy_coef=cfg.ppo.entropy_coef,
                    max_grad_norm=cfg.ppo.max_grad_norm,
                )
                self.run.log_metrics(epoch, metrics, section="train")

                if cfg.eval.enabled and epoch % cfg.eval.every == 0:
                    last_eval = self.evaluate()
                    self.run.log_eval(epoch, last_eval)

                self._print_progress(epoch, metrics, last_eval)

                if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0:
                    self.run.save_checkpoint(
                        self.agent,
                        epoch,
                        {**metrics, **last_eval},
                        best_metric=BEST_METRIC,
                        best_mode="max",
                        trainer_state=self._trainer_state(epoch),
                    )

            self.run.finish(final_metrics={**metrics, **last_eval})

        print(f"Done. Artifacts in {self.run.dir}")
        return self.run.run_id

    # ---------------------------------------------------------------- evaluation
    def evaluate(self) -> dict[str, float]:
        from kaisparov.agents.neural_agent import NeuralAgent

        neural = NeuralAgent(self.agent, self._processor, deterministic=True)
        cfg = self.config.eval
        vs_random = evaluate(
            neural, RandomAgent(seed=self.config.seed), cfg.games, max_plies=cfg.max_plies
        )
        vs_material = evaluate(
            neural, MaterialAgent(seed=self.config.seed), cfg.games, max_plies=cfg.max_plies
        )
        return {
            "winrate_vs_random": vs_random.score_a,
            "elo_vs_random": vs_random.elo_diff,
            "winrate_vs_material": vs_material.score_a,
            "elo_vs_material": vs_material.elo_diff,
        }

    def _print_progress(self, epoch: int, metrics: dict, last_eval: dict) -> None:
        line = (
            f"[epoch {epoch:>3}] loss={metrics.get('loss', 0):+.4f} "
            f"policy={metrics.get('policy_loss', 0):+.4f} "
            f"value={metrics.get('value_loss', 0):.4f} "
            f"entropy={metrics.get('entropy', 0):.3f}"
        )
        if last_eval and self.config.eval.every and epoch % self.config.eval.every == 0:
            line += (
                f" | vs_random={last_eval['winrate_vs_random']:.0%} "
                f"(elo {last_eval['elo_vs_random']:+.0f})"
            )
        print(line)
