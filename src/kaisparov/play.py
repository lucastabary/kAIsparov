"""Interactive play in the pygame window: human vs human, or human vs AI.

python -m kaisparov.play                         # human vs human
python -m kaisparov.play --vs-ai --checkpoint runs/<id>/checkpoints/best.pth
python -m kaisparov.play --vs-ai                 # uses the best tracked checkpoint
"""

from __future__ import annotations

import argparse

import pygame
import torch

from kaisparov.core.board import ChessGame
from kaisparov.core.game_interface import GameInterface
from kaisparov.core.pieces import PieceType, Player
from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum


def _initial_board(from_curriculum: bool, seed: int | None):
    if not from_curriculum:
        return None
    phase = PhaseConfig(name="Phase 1: 6 pieces", max_pieces_per_side=6, allow_major=False)
    return PieceCountCurriculum(phase, seed=seed).get_initial_board()


def _resolve_checkpoint(checkpoint: str | None, runs_dir: str, use_best: bool = False) -> str:
    if checkpoint is not None:
        return checkpoint
    from kaisparov.tracking.registry import Registry

    reg = Registry(runs_dir)
    if use_best:
        best = reg.best_run("elo_vs_random")
        if best is None:
            raise SystemExit("No tracked run has an eval metric. Pass --checkpoint <path>.")
        return str(reg.resolve_checkpoint(best[0]["run_id"], "best"))

    runs = reg.list_runs()
    if not runs:
        raise SystemExit("No tracked runs found. Train first, or pass --checkpoint <path>.")
    return str(reg.resolve_checkpoint(runs[0]["run_id"], "latest"))  # newest run, latest ckpt


def _build_ai(checkpoint: str, hidden_dim: int, device: torch.device):
    from kaisparov.agents.neural_agent import NeuralAgent
    from kaisparov.models.factory import load_backend_spec

    spec = load_backend_spec()
    model, path = spec.model_class.load_agent_for_inference(
        device=device, model_path=checkpoint, hidden_dim=hidden_dim
    )
    print(f"AI loaded from {path}")
    return NeuralAgent(model, spec.processor_class(), deterministic=True)


def play_vs_ai(ai, game: ChessGame, human_color: Player) -> None:
    ui = GameInterface(game)
    ui._ensure_initialized()
    print(f"You are {human_color.name}. Close the window to quit.")

    while True:
        if game.turn == human_color:
            move = ui._get_single_move(use_pov=True)
            if move is None:
                break
        else:
            move = ai.select_move(game)
            if move is None:
                print("AI has no legal move.")
                break

        captured = game.play(*move)
        ui._draw_frame(use_pov=True)
        pygame.display.flip()

        if captured is not None and captured.type == PieceType.KING:
            print(f"Game over — {game.turn.name} wins by capturing the king!")
            break

    pygame.quit()
    ui._initialized = False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kaisparov play", description="Play in a pygame window.")
    parser.add_argument("--vs-ai", action="store_true", help="Play against the neural agent.")
    parser.add_argument(
        "--checkpoint", default=None, help="AI weights (default: newest run's latest checkpoint)."
    )
    parser.add_argument("--best", action="store_true", help="Use the best-Elo checkpoint instead.")
    parser.add_argument(
        "--color", choices=["white", "black"], default="white", help="Your color vs the AI."
    )
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument(
        "--curriculum", action="store_true", help="Start from a curriculum position."
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    board = _initial_board(args.curriculum, args.seed)
    game = ChessGame(initial_board=board)

    if not args.vs_ai:
        GameInterface(game).play_game(use_pov=True)
        return

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = _resolve_checkpoint(args.checkpoint, args.runs_dir, use_best=args.best)
    ai = _build_ai(checkpoint, args.hidden_dim, device)
    human_color = Player.WHITE if args.color == "white" else Player.BLACK
    play_vs_ai(ai, game, human_color)


if __name__ == "__main__":
    main()
