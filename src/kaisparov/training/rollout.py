"""Batched self-play rollout: fill a PPOBuffer efficiently.

All episodes of an epoch are played *in parallel*: at each ply every still-running
game is graphified and the model runs a single batched forward pass, instead of one
tiny forward per game per move. This keeps the CPU (or GPU) busy and is the main
training-throughput lever.

Per-episode transitions are buffered separately and flushed contiguously when the
game ends, so the negamax GAE in :mod:`kaisparov.training.ppo` sees clean episode
boundaries. The agent plays both sides.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch

from kaisparov.core.board import ChessGame
from kaisparov.core.pieces import PieceType
from kaisparov.training.curriculum import BaseCurriculum
from kaisparov.training.ppo import PPOBuffer


def _model_device(agent: torch.nn.Module) -> torch.device:
    try:
        return next(agent.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _resolve_module(model_module, model_name: str | None):
    if model_module is not None:
        return model_module
    from kaisparov.models.factory import load_backend  # lazy: avoid import cycles

    return load_backend(model_name)


def _new_game(curriculum: BaseCurriculum | None) -> ChessGame:
    if curriculum is not None:
        return ChessGame(initial_board=curriculum.get_initial_board())
    return ChessGame()


def collect_data(
    agent: torch.nn.Module,
    game: ChessGame,
    buffer: PPOBuffer,
    num_episodes: int = 5,
    max_steps_per_episode: int = 100,
    *,
    model_name: str | None = None,
    model_module=None,
    deterministic: bool = False,
    curriculum: BaseCurriculum | None = None,
    reward_fn=None,
) -> dict[str, float]:
    module = _resolve_module(model_module, model_name)
    processor = module.PROCESSOR_CLASS()
    device = _model_device(agent)
    edge_index = processor.static_graph_edges[0].to(device)
    agent.eval()

    games = [_new_game(curriculum) for _ in range(num_episodes)]
    pending: list[list[dict]] = [[] for _ in range(num_episodes)]
    steps = [0] * num_episodes
    active = [True] * num_episodes

    # How episodes ended, for live monitoring.
    n_king = n_truncated = n_stalemate = total_plies = 0

    def flush(i: int) -> None:
        transitions = pending[i]
        if transitions:
            transitions[-1]["done"] = True  # episode boundary for GAE
            for t in transitions:
                buffer.add(**t)
        pending[i] = []
        active[i] = False

    with torch.no_grad():
        while any(active):
            idxs = [i for i in range(num_episodes) if active[i]]
            states = [processor.graphify(games[i]) for i in idxs]
            batch = Batch.from_data_list(states).to(device)
            action_scores, values = agent(batch)

            edge_counts = [int(s.edge_index.shape[1]) for s in states]
            per_game_scores = torch.split(action_scores, edge_counts, dim=0)
            values = values.reshape(-1)

            for k, i in enumerate(idxs):
                g = games[i]
                legal_mask = module.get_legal_mask(g, edge_index)
                if not legal_mask.any():
                    n_stalemate += 1
                    total_plies += steps[i]
                    flush(i)
                    continue

                action = processor.process_output(
                    (per_game_scores[k], values[k]),
                    g,
                    deterministic=deterministic,
                    legal_mask=legal_mask,
                )
                captured = g.play(*action.move_coords)
                if reward_fn is not None:
                    reward = reward_fn(g, captured)
                else:
                    reward = module.compute_reward(g, captured)
                king_captured = captured is not None and captured.type == PieceType.KING

                steps[i] += 1
                done = king_captured or steps[i] >= max_steps_per_episode
                pending[i].append(
                    {
                        "state": states[k],
                        "action": action.action_index,
                        "log_prob": action.log_prob,
                        "value": action.value,
                        "reward": reward,
                        "done": done,
                        "legal_mask": legal_mask,
                    }
                )
                if done:
                    if king_captured:
                        n_king += 1
                    else:
                        n_truncated += 1
                    total_plies += steps[i]
                    flush(i)

    n = max(num_episodes, 1)
    return {
        "king_capture_rate": n_king / n,
        "truncated_rate": n_truncated / n,
        "stalemate_rate": n_stalemate / n,
        "avg_plies": total_plies / n,
        "transitions": float(len(buffer)),
    }


__all__ = ["collect_data"]
