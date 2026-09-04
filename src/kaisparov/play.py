"""Interactive play in the pygame window.

Run with no mode flag to open the in-window menu (solo / vs AI / AI vs AI):

    python -m kaisparov.play

Or pick a mode straight from the command line (skips the menu):

    python -m kaisparov.play --vs-ai --checkpoint runs/<id>/checkpoints/best.pth
    python -m kaisparov.play --vs-ai --best        # best tracked checkpoint
    python -m kaisparov.play --ai-vs-ai --dev       # watch two models, with analysis
    python -m kaisparov.play --solo                 # two humans, one keyboard

``--dev`` turns on developer mode: while a side backed by a trained model is to
move, the board shows that model's top candidate moves as arrows and its value
estimate in the side panel (see :mod:`kaisparov.insights`).
"""

from __future__ import annotations

import argparse

import pygame

from kaisparov.core.board import ChessGame
from kaisparov.core.coords import Coord
from kaisparov.core.game_interface import GameInterface, MatchSetup, MoveArrow
from kaisparov.core.pieces import PieceType, Player
from kaisparov.insights import PositionAnalysis
from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum


def _other(player: Player) -> Player:
    return Player.BLACK if player == Player.WHITE else Player.WHITE


def _algebraic(coord: Coord) -> str:
    return f"{chr(ord('a') + coord[0])}{coord[1] + 1}"


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
        best = reg.best_run("elo_vs_material")  # the metric that actually discriminates skill
        if best is None:
            raise SystemExit("No tracked run has an eval metric. Pass --checkpoint <path>.")
        return str(reg.resolve_checkpoint(best[0]["run_id"], "best"))

    runs = reg.list_runs()
    if not runs:
        raise SystemExit("No tracked runs found. Train first, or pass --checkpoint <path>.")
    return str(reg.resolve_checkpoint(runs[0]["run_id"], "latest"))  # newest run, latest ckpt


# ---------------------------------------------------------------------- model

# state_dict keys whose first dimension equals hidden_dim, tried in order.
_HIDDEN_DIM_KEYS = ("chess_rgcn.conv1.bias", "critic_head.0.bias", "actor_head.0.bias")


def _infer_hidden_dim(state_dict) -> int | None:
    """Read hidden_dim off a checkpoint so it need not be passed on the CLI.

    Checkpoints are plain ``state_dict``s with no metadata, but every candidate key
    is a 1-D tensor of length ``hidden_dim`` — so its shape tells us the width the
    model was trained at, tracked run or raw path alike.
    """
    for key in _HIDDEN_DIM_KEYS:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.dim() >= 1:
            return int(tensor.shape[0])
    return None


def _load_model(checkpoint: str, hidden_dim: int | None, device):
    """Load the backend once; agents and the analyzer are cheap wrappers over it.

    ``hidden_dim=None`` means "infer it from the checkpoint" (the default), which
    avoids the size-mismatch crash when a model was trained at a non-default width.
    """
    import torch

    from kaisparov.models.factory import load_backend_spec

    spec = load_backend_spec()
    state_dict = torch.load(checkpoint, map_location=device)

    if hidden_dim is None:
        hidden_dim = _infer_hidden_dim(state_dict)
        if hidden_dim is None:
            hidden_dim = 8
            print("Could not infer hidden_dim from the checkpoint; falling back to 8.")
        else:
            print(f"Using hidden_dim={hidden_dim} (inferred from the checkpoint).")

    model = spec.model_class.create_agent(device=device, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model, spec.processor_class(), str(checkpoint)


def _build_ai(args, device, *, deterministic: bool, allow_fallback: bool):
    """Return ``(policy, analyzer)`` for an AI seat.

    When no checkpoint is available and ``allow_fallback`` is set (menu-driven
    play on a fresh clone), degrade to the material baseline with no analyzer so
    the window still works instead of raising.
    """
    try:
        checkpoint = _resolve_checkpoint(args.checkpoint, args.runs_dir, use_best=args.best)
        model, processor, path = _load_model(checkpoint, args.hidden_dim, device)
    except SystemExit:
        if not allow_fallback:
            raise
        from kaisparov.agents.material_agent import MaterialAgent

        print("No trained checkpoint found — falling back to the material baseline.")
        return MaterialAgent(), None

    from kaisparov.agents.neural_analyzer import NeuralAnalyzer

    analyzer = NeuralAnalyzer(model, processor)
    if args.minimax_depth > 0:
        from kaisparov.agents.minimax_agent import MinimaxAgent

        print(f"AI loaded from {path} (minimax depth {args.minimax_depth})")
        return MinimaxAgent(model, processor, depth=args.minimax_depth), analyzer

    from kaisparov.agents.neural_agent import NeuralAgent

    print(f"AI loaded from {path}")
    return NeuralAgent(model, processor, deterministic=deterministic), analyzer


# --------------------------------------------------------------- match wiring


def _prepare_match(setup: MatchSetup, args, device):
    """Turn a :class:`MatchSetup` into per-side controllers and analyzers.

    ``controllers[color]`` is ``None`` for a human seat or a policy for an AI one.
    ``analyzers[color]`` is the analyzer to consult when that side is to move
    (developer mode only); it may be ``None``.
    """
    controllers: dict[Player, object | None] = {Player.WHITE: None, Player.BLACK: None}
    analyzers: dict[Player, object | None] = {Player.WHITE: None, Player.BLACK: None}
    shared_analyzer: object | None = None

    if setup.mode == "vs_ai":
        ai_color = _other(setup.human_color)
        agent, analyzer = _build_ai(args, device, deterministic=True, allow_fallback=True)
        controllers[ai_color] = agent
        analyzers[ai_color] = analyzer
        shared_analyzer = analyzer
    elif setup.mode == "ai_vs_ai":
        # Share one loaded model across both seats; sample moves so games vary.
        try:
            checkpoint = _resolve_checkpoint(args.checkpoint, args.runs_dir, use_best=args.best)
            model, processor, path = _load_model(checkpoint, args.hidden_dim, device)
            from kaisparov.agents.neural_agent import NeuralAgent
            from kaisparov.agents.neural_analyzer import NeuralAnalyzer

            print(f"AI loaded from {path}")
            shared_analyzer = NeuralAnalyzer(model, processor)
            for color in (Player.WHITE, Player.BLACK):
                controllers[color] = NeuralAgent(model, processor, deterministic=False)
                analyzers[color] = shared_analyzer
        except SystemExit:
            from kaisparov.agents.material_agent import MaterialAgent

            print("No trained checkpoint found — pitting two material baselines instead.")
            controllers[Player.WHITE] = MaterialAgent()
            controllers[Player.BLACK] = MaterialAgent()
    # "solo": both seats stay human.

    # In developer mode, analyze human seats too (comment on the running game) by
    # reusing whichever model we already loaded, if any.
    if setup.dev_mode:
        if shared_analyzer is None and setup.mode == "solo":
            shared_analyzer = _try_build_analyzer(args, device)
        for color in (Player.WHITE, Player.BLACK):
            if analyzers[color] is None:
                analyzers[color] = shared_analyzer

    return controllers, analyzers


def _try_build_analyzer(args, device):
    """Best-effort analyzer for developer mode in solo play; ``None`` if no model."""
    try:
        checkpoint = _resolve_checkpoint(args.checkpoint, args.runs_dir, use_best=args.best)
        model, processor, path = _load_model(checkpoint, args.hidden_dim, device)
    except SystemExit:
        print("Developer mode: no checkpoint found, analysis overlay disabled.")
        return None
    from kaisparov.agents.neural_analyzer import NeuralAnalyzer

    print(f"Developer analyzer loaded from {path}")
    return NeuralAnalyzer(model, processor)


def _overlay_from_analysis(analysis: PositionAnalysis | None):
    """Map a model analysis onto UI arrows + panel text (both empty if ``None``)."""
    if analysis is None:
        return [], []

    arrows = [
        MoveArrow(source=c.move[0], dest=c.move[1], intensity=c.score, label=c.label)
        for c in analysis.candidates
    ]

    status = ["-- Mode developpeur --"]
    if analysis.value is not None:
        status.append(f"Eval (trait): {analysis.value:+.2f}")
    if analysis.best is not None:
        src, dst = analysis.best.move
        status.append(f"Idee: {_algebraic(src)}->{_algebraic(dst)}  {analysis.best.label}")
    return arrows, status


def _pump_events(ui: GameInterface, delay_ms: int) -> bool:
    """Keep the window responsive for ``delay_ms`` between AI moves.

    Returns ``False`` if the user closed the window.
    """
    assert ui._clock is not None
    end = pygame.time.get_ticks() + delay_ms
    while pygame.time.get_ticks() < end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        ui._clock.tick(60)
    return True


def run_match(
    ui: GameInterface,
    game: ChessGame,
    controllers: dict[Player, object | None],
    analyzers: dict[Player, object | None],
    dev_mode: bool,
    ai_delay_ms: int = 500,
) -> None:
    """Drive a game where each side is a human (mouse) or a policy.

    Humans see the developer overlay while they think; AI seats show what they are
    about to consider, pause briefly (so AI-vs-AI is watchable), then move.
    """
    ui._ensure_initialized()

    while True:
        side = game.turn
        agent = controllers[side]
        analyzer = analyzers.get(side)

        arrows, status = [], []
        if dev_mode and analyzer is not None:
            arrows, status = _overlay_from_analysis(analyzer.analyze(game))

        if agent is None:  # human seat
            move = ui._get_single_move(use_pov=True, analysis_arrows=arrows, status_lines=status)
            if move is None:
                break
        else:  # AI seat
            ui._draw_frame(use_pov=True, analysis_arrows=arrows, status_lines=status)
            pygame.display.flip()
            if not _pump_events(ui, ai_delay_ms):
                break
            move = agent.select_move(game)
            if move is None:
                print(f"{side.name} (AI) has no legal move.")
                break

        captured = game.play(*move)
        ui._draw_frame(use_pov=True)
        pygame.display.flip()

        if captured is not None and captured.type == PieceType.KING:
            print(f"Game over — {game.turn.name} wins by capturing the king!")
            break

    pygame.quit()
    ui._initialized = False


# ----------------------------------------------------------------------- CLI


def _setup_from_args(args) -> MatchSetup | None:
    """A mode chosen on the command line bypasses the menu; otherwise ``None``."""
    human_color = Player.WHITE if args.color == "white" else Player.BLACK
    if args.vs_ai:
        return MatchSetup("vs_ai", human_color, args.dev)
    if args.ai_vs_ai:
        return MatchSetup("ai_vs_ai", human_color, args.dev)
    if args.solo:
        return MatchSetup("solo", human_color, args.dev)
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kaisparov play", description="Play in a pygame window.")
    parser.add_argument("--vs-ai", action="store_true", help="Play against the neural agent.")
    parser.add_argument("--ai-vs-ai", action="store_true", help="Watch two AIs play each other.")
    parser.add_argument("--solo", action="store_true", help="Two humans on one keyboard.")
    parser.add_argument("--dev", action="store_true", help="Developer mode: show model analysis.")
    parser.add_argument(
        "--checkpoint", default=None, help="AI weights (default: newest run's latest checkpoint)."
    )
    parser.add_argument("--best", action="store_true", help="Use the best-Elo checkpoint instead.")
    parser.add_argument(
        "--color", choices=["white", "black"], default="white", help="Your color vs the AI."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None, help="AI width (default: inferred from checkpoint)."
    )
    parser.add_argument(
        "--minimax-depth", type=int, default=0, help="Let the AI search (negamax on the critic)."
    )
    parser.add_argument(
        "--ai-delay", type=int, default=500, help="Pause in ms between AI moves (AI vs AI)."
    )
    parser.add_argument(
        "--curriculum", action="store_true", help="Start from a curriculum position."
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    board = _initial_board(args.curriculum, args.seed)
    game = ChessGame(initial_board=board)
    ui = GameInterface(game)

    setup = _setup_from_args(args)
    if setup is None:
        setup = ui.select_setup()  # in-window menu
        if setup is None:
            return  # window closed on the menu

    # Only touch torch when a seat (or the developer overlay) actually needs a model,
    # so plain solo play stays torch-free.
    device = None
    if setup.mode != "solo" or setup.dev_mode:
        import torch

        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    controllers, analyzers = _prepare_match(setup, args, device)
    run_match(ui, game, controllers, analyzers, setup.dev_mode, ai_delay_ms=args.ai_delay)


if __name__ == "__main__":
    main()
