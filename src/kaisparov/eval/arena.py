"""Play policies against each other and summarise the results.

Used to benchmark a trained agent against the baselines and to produce
showcase-friendly win-rates and a rough Elo gap.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from kaisparov.agents.base import Policy
from kaisparov.core.pieces import Piece, Player
from kaisparov.envs.chess_env import ChessEnv

Grid = list[list["Piece | None"]]


@dataclass
class GameResult:
    winner: Player | None
    plies: int
    reason: str


@dataclass
class MatchStats:
    agent_a: str
    agent_b: str
    games: int
    wins_a: int
    wins_b: int
    draws: int

    @property
    def score_a(self) -> float:
        """Points for agent A per game (win = 1, draw = 0.5)."""
        return (self.wins_a + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def elo_diff(self) -> float:
        """Estimated Elo of A minus B from the score (clamped for 0%/100%)."""
        score = min(max(self.score_a, 1e-4), 1 - 1e-4)
        return -400.0 * math.log10(1.0 / score - 1.0)


def _other(player: Player) -> Player:
    return Player.BLACK if player == Player.WHITE else Player.WHITE


def play_game(
    white: Policy,
    black: Policy,
    *,
    board: Grid | None = None,
    max_plies: int = 200,
) -> GameResult:
    env = ChessEnv(max_plies=max_plies)
    game = env.reset(board=board)
    policies = {Player.WHITE: white, Player.BLACK: black}

    reason = "max_plies"
    while not env.done:
        mover = game.turn
        move = policies[mover].select_move(game)
        if move is None:
            # The mover produced no move. If legal moves existed, it forfeits.
            if env.legal_moves():
                env.done, env.winner, reason = True, _other(mover), "forfeit"
            else:
                env.done, env.winner, reason = True, None, "stalemate"
            break

        result = env.step(move)
        if result.done:
            reason = "king_captured" if env.winner is not None else "stalemate_or_limit"

    return GameResult(winner=env.winner, plies=env.plies, reason=reason)


def evaluate(
    agent_a: Policy,
    agent_b: Policy,
    games: int = 20,
    *,
    max_plies: int = 200,
) -> MatchStats:
    """Play ``games`` games, alternating colors so first-move bias cancels out."""
    wins_a = wins_b = draws = 0
    for i in range(games):
        a_is_white = i % 2 == 0
        white, black = (agent_a, agent_b) if a_is_white else (agent_b, agent_a)
        result = play_game(white, black, max_plies=max_plies)

        if result.winner is None:
            draws += 1
        else:
            winner_is_a = (result.winner == Player.WHITE) == a_is_white
            if winner_is_a:
                wins_a += 1
            else:
                wins_b += 1

    return MatchStats(
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        games=games,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
    )


def _format(stats: MatchStats) -> str:
    return (
        f"{stats.agent_a:>10} vs {stats.agent_b:<10} | "
        f"{stats.wins_a:>3}W {stats.wins_b:>3}L {stats.draws:>3}D | "
        f"score={stats.score_a:5.1%} | elo_diff={stats.elo_diff:+.0f}"
    )


def _load_neural(model_name, checkpoint, device, hidden_dim):
    from kaisparov.agents.neural_agent import NeuralAgent
    from kaisparov.models.factory import load_backend_spec

    spec = load_backend_spec(model_name)
    model, path = spec.model_class.load_agent_for_inference(
        device=device, model_path=checkpoint, hidden_dim=hidden_dim
    )
    print(f"Loaded neural agent from {path}")
    return NeuralAgent(model, spec.processor_class(), deterministic=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kaisparov eval", description="Evaluate agents.")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None, help="Also evaluate this neural backend.")
    parser.add_argument("--checkpoint", default=None, help="Explicit weights path for --model.")
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    from kaisparov.agents.material_agent import MaterialAgent
    from kaisparov.agents.random_agent import RandomAgent

    random_agent = RandomAgent(seed=args.seed)
    material_agent = MaterialAgent(seed=args.seed)

    print(f"\n=== Arena ({args.games} games each, alternating colors) ===")
    print(_format(evaluate(material_agent, random_agent, args.games, max_plies=args.max_plies)))

    if args.model is not None:
        import torch

        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        neural = _load_neural(args.model, args.checkpoint, device, args.hidden_dim)
        print(_format(evaluate(neural, random_agent, args.games, max_plies=args.max_plies)))
        print(_format(evaluate(neural, material_agent, args.games, max_plies=args.max_plies)))
    print()


if __name__ == "__main__":
    main()
