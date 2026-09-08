"""Single-agent rollout: the learner plays against a *fixed* opponent.

Unlike self-play (both sides are the learner, negamax GAE), here the opponent is part
of the environment. So only the learner's transitions are stored, with a per-step
reward = (what the learner captured) - (what the opponent captured in reply) - step
cost, and terminal win/loss on king capture. The PPO buffer must use ``self_play=False``
(standard GAE) for this data.

Episodes are played *batched*: at each ply every still-running game is graphified and
the learner's model runs a single batched forward pass (instead of one tiny forward per
game per move), which keeps the CPU/GPU busy. Opponent replies stay per game (they are
plain Python / small minimax). Per-episode transitions are flushed contiguously with a
terminal ``done`` so the GAE sees clean episode boundaries.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import torch
from torch_geometric.data import Batch

from kaisparov.core.board import ChessGame
from kaisparov.core.movegen import all_moves
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.core.utils import get_piece_value
from kaisparov.training.config import RewardSettings
from kaisparov.training.curriculum import BaseCurriculum


def _gain(settings: RewardSettings, captured: Piece | None) -> float:
    if captured is None:
        return 0.0
    # A king capture is the flat win bonus, decoupled from the king's sentinel
    # material value (mirrors kaisparov.training.reward.make_reward_fn).
    if captured.type == PieceType.KING:
        return settings.king_capture
    return settings.material * get_piece_value(captured.type)


def _is_king(piece: Piece | None) -> bool:
    return piece is not None and piece.type == PieceType.KING


def _new_game(curriculum: BaseCurriculum | None) -> ChessGame:
    return ChessGame(initial_board=curriculum.get_initial_board() if curriculum else None)


def _opponent_reply(game: ChessGame, opponent) -> tuple[bool, Piece | None]:
    """Play the opponent's move. Returns (moved, captured_piece)."""
    if not all_moves(game.grid, game.turn, game.en_passant_target):
        return False, None
    move = opponent.select_move(game)
    if move is None:
        return False, None
    return True, game.play(*move)


def collect_vs_opponent(
    agent: torch.nn.Module,
    buffer,
    num_episodes: int,
    max_steps_per_episode: int,
    *,
    model_module,
    curriculum: BaseCurriculum | None,
    reward_settings: RewardSettings,
    opponent: Any | None = None,
    sample_opponent: Callable[[], Any] | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Collect learner transitions against a fixed or per-episode-sampled opponent.

    Pass ``opponent`` for a single fixed opponent, or ``sample_opponent`` (a zero-arg
    callable, e.g. ``pool.sample``) to draw a fresh opponent for *each* episode — the
    latter keeps a single epoch's batch from being homogeneous (all-vs-Material games
    are short and swing the critic; all-vs-Random games drift long), which otherwise
    makes the PPO gradient high-variance from one epoch to the next.
    """
    if (opponent is None) == (sample_opponent is None):
        raise ValueError("Pass exactly one of `opponent` or `sample_opponent`.")

    processor = model_module.PROCESSOR_CLASS()
    device = next(agent.parameters()).device
    edge_index = processor.static_graph_edges[0].to(device)
    agent.eval()
    rng = random.Random(seed)

    def pick_opponent():
        return opponent if sample_opponent is None else sample_opponent()

    # Per-game state, indexed by episode.
    games: list[ChessGame] = []
    opponents: list[Any] = []
    learners: list[Player] = []
    pending: list[list[dict]] = [[] for _ in range(num_episodes)]
    steps = [0] * num_episodes
    active = [True] * num_episodes

    wins = losses = draws = total_plies = 0

    def finish(i: int, result: str) -> None:
        nonlocal wins, losses, draws, total_plies
        transitions = pending[i]
        if transitions:
            transitions[-1]["done"] = True  # episode boundary for GAE
            for t in transitions:
                buffer.add(**t)
        pending[i] = []
        active[i] = False
        total_plies += steps[i]
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            draws += 1

    # Initialise each game; if the opponent is on move first, let it play (not stored).
    for i in range(num_episodes):
        opp = pick_opponent()
        game = _new_game(curriculum)
        learner = rng.choice([Player.WHITE, Player.BLACK])
        games.append(game)
        opponents.append(opp)
        learners.append(learner)
        if game.turn != learner:
            moved, captured = _opponent_reply(game, opp)
            if not moved:
                finish(i, "draw")
            elif _is_king(captured):
                finish(i, "loss")

    # Batched play: at loop top, every active game has the learner to move.
    with torch.no_grad():
        while any(active):
            idxs = [i for i in range(num_episodes) if active[i]]
            states = processor.graphify_batch([games[i] for i in idxs])
            batch = Batch.from_data_list(states).to(device)
            action_scores, values = agent(batch)
            edge_counts = [int(s.edge_index.shape[1]) for s in states]
            per_game_scores = torch.split(action_scores, edge_counts, dim=0)
            values = values.reshape(-1)

            for k, i in enumerate(idxs):
                game = games[i]
                learner = learners[i]
                opp = opponents[i]

                legal_mask = model_module.get_legal_mask(game, edge_index)
                if not legal_mask.any():
                    finish(i, "draw")  # learner has no move
                    continue

                action = processor.process_output(
                    (per_game_scores[k], values[k]),
                    game,
                    deterministic=False,
                    legal_mask=legal_mask,
                )
                captured_l = game.play(*action.move_coords)
                steps[i] += 1
                reward = _gain(reward_settings, captured_l) - reward_settings.step_penalty
                done = False
                result = "draw"

                if _is_king(captured_l):
                    result, done = "win", True
                else:
                    # King safety: did the learner leave its own king capturable?
                    if reward_settings.king_safety and game.is_in_check(learner):
                        reward -= reward_settings.king_safety
                    moved, captured_o = _opponent_reply(game, opp)
                    if not moved:
                        done = True  # opponent stuck -> draw
                    else:
                        steps[i] += 1
                        reward -= _gain(reward_settings, captured_o)
                        if _is_king(captured_o):
                            result, done = "loss", True
                    if not done and steps[i] >= max_steps_per_episode:
                        done = True  # truncated (stays a draw)

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
                    finish(i, result)

    n = max(num_episodes, 1)
    return {
        # A decisive game here ends with a king capture by either side (win + loss);
        # named to match the self-play rollout's key so the trainer can log/print it
        # uniformly. ``winrate`` is the learner's own king-capture rate.
        "king_capture_rate": (wins + losses) / n,
        "winrate": wins / n,
        "lossrate": losses / n,
        "drawrate": draws / n,
        "avg_plies": total_plies / n,
        "transitions": float(len(buffer)),
    }


__all__ = ["collect_vs_opponent"]
