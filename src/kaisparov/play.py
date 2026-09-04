"""Interactive play in the pygame window.

Run with no mode flag to open the in-window menu (solo / vs AI / AI vs AI):

    python -m kaisparov.play

Or pick a mode straight from the command line (skips the menu for the first game):

    python -m kaisparov.play --vs-ai --checkpoint runs/<id>/checkpoints/best.pth
    python -m kaisparov.play --vs-ai --best        # best tracked checkpoint
    python -m kaisparov.play --ai-vs-ai --dev       # watch two models, with analysis
    python -m kaisparov.play --solo                 # two humans, one keyboard

In the menu, the AI modes open a model picker: choose which tracked run (or the
Material / Random baseline) plays the opponent in vs-AI, and each of the two seats
independently in AI vs AI. AI vs AI keeps White at the bottom and advances one move
at a time when you click "Coup suivant" (or press Space). When a game ends the window
returns to the menu instead of closing.

``--dev`` turns on developer mode: while a side backed by a trained model is to
move, the board shows that model's top candidate moves as arrows and its value
estimate in the side panel (see :mod:`kaisparov.insights`).
"""

from __future__ import annotations

import argparse

import pygame

from kaisparov.agents.base import Policy
from kaisparov.core.board import ChessGame
from kaisparov.core.coords import Coord
from kaisparov.core.game_interface import GameInterface, MatchSetup, ModelOption, MoveArrow
from kaisparov.core.pieces import PieceType, Player
from kaisparov.insights import Analyzer, PositionAnalysis
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


# --------------------------------------------------------------- model catalogue

_BASELINE_KEYS = ("material", "random")


def _available_models(runs_dir: str) -> list[ModelOption]:
    """The models offered in the menu: every tracked run (newest first) that has a
    checkpoint, plus the two torch-free baselines. Never raises — a fresh clone with
    no runs still gets the baselines so AI modes remain playable.
    """
    options: list[ModelOption] = []
    try:
        from kaisparov.tracking.registry import Registry

        for run in Registry(runs_dir).list_runs():
            run_id = run["run_id"]
            title = (run.get("title") or "").strip()
            label = f"{title[:26]}  [{run_id[:13]}]" if title else run_id
            options.append(ModelOption(key=run_id, label=label))
    except Exception as exc:  # registry is best-effort; baselines always work
        print(f"Could not list tracked runs ({exc}); offering baselines only.")

    options.append(ModelOption(key="material", label="Material (baseline gloutonne)"))
    options.append(ModelOption(key="random", label="Random (baseline aleatoire)"))
    return options


def _make_baseline(key: str):
    if key == "material":
        from kaisparov.agents.material_agent import MaterialAgent

        return MaterialAgent()
    from kaisparov.agents.random_agent import RandomAgent

    return RandomAgent()


def _controller_from_key(key, args, device, *, deterministic: bool, model_cache: dict):
    """Build ``(policy, analyzer)`` for a model chosen in the menu.

    ``key`` is a baseline name or a tracked ``run_id`` (its best checkpoint is used).
    Loaded backends are memoised in ``model_cache`` so the two AI-vs-AI seats sharing
    a run only pay for one load. Baselines have no analyzer.
    """
    if key in _BASELINE_KEYS:
        return _make_baseline(key), None

    if key not in model_cache:
        from kaisparov.tracking.registry import Registry

        try:
            checkpoint = str(Registry(args.runs_dir).resolve_checkpoint(key, "best"))
            model_cache[key] = _load_model(checkpoint, args.hidden_dim, device)
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            print(f"Could not load run '{key}' ({exc}); using the material baseline.")
            model_cache[key] = None

    loaded = model_cache[key]
    if loaded is None:
        return _make_baseline("material"), None

    model, processor, path = loaded
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
    (developer mode only); it may be ``None``. When the menu supplied explicit model
    keys they win; otherwise we fall back to the default checkpoint resolution (the
    path taken by the ``--vs-ai`` / ``--ai-vs-ai`` command-line shortcuts).
    """
    controllers: dict[Player, Policy | None] = {Player.WHITE: None, Player.BLACK: None}
    analyzers: dict[Player, Analyzer | None] = {Player.WHITE: None, Player.BLACK: None}
    model_cache: dict = {}
    shared_analyzer: Analyzer | None = None

    if setup.mode == "vs_ai":
        ai_color = _other(setup.human_color)
        if setup.ai_model is not None:
            agent, analyzer = _controller_from_key(
                setup.ai_model, args, device, deterministic=True, model_cache=model_cache
            )
        else:
            agent, analyzer = _build_ai(args, device, deterministic=True, allow_fallback=True)
        controllers[ai_color] = agent
        analyzers[ai_color] = analyzer
        shared_analyzer = analyzer
    elif setup.mode == "ai_vs_ai":
        if setup.white_model is not None or setup.black_model is not None:
            # A model per seat (they may differ). Sample moves so games vary.
            for color, key in (
                (Player.WHITE, setup.white_model),
                (Player.BLACK, setup.black_model),
            ):
                agent, analyzer = _controller_from_key(
                    key, args, device, deterministic=False, model_cache=model_cache
                )
                controllers[color] = agent
                analyzers[color] = analyzer
            shared_analyzer = analyzers[Player.WHITE] or analyzers[Player.BLACK]
        else:
            # CLI shortcut: share one loaded model across both seats, sampling moves.
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
                print("No trained checkpoint found — pitting two material baselines instead.")
                controllers[Player.WHITE] = _make_baseline("material")
                controllers[Player.BLACK] = _make_baseline("material")
    # "solo": both seats stay human.

    # In developer mode, analyze human/baseline seats too (comment on the running
    # game) by reusing whichever model we already loaded, if any.
    if setup.dev_mode:
        if shared_analyzer is None:
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
    controllers: dict[Player, Policy | None],
    analyzers: dict[Player, Analyzer | None],
    dev_mode: bool,
    *,
    view_pov: bool = True,
    step_mode: bool = False,
    ai_delay_ms: int = 500,
) -> str:
    """Drive a game where each side is a human (mouse) or a policy.

    ``view_pov`` fixes the board orientation: ``True`` follows the side to move (POV),
    ``False`` keeps White at the bottom throughout (used for AI vs AI so the view does
    not flip every move). ``step_mode`` makes AI seats wait for the user to request
    each move (the "Coup suivant" button) instead of auto-advancing on a timer.

    Returns ``"quit"`` if the window was closed, or ``"menu"`` when the game ends (so
    the caller can return to the start menu without tearing down the window).
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
            move = ui._get_single_move(
                use_pov=view_pov, analysis_arrows=arrows, status_lines=status
            )
            if move is None:
                return "quit"
        else:  # AI seat
            if step_mode:
                turn_status = status + [f"Trait aux {_fr_color(side)}"]
                if not ui.wait_for_step(
                    use_pov=view_pov, analysis_arrows=arrows, status_lines=turn_status
                ):
                    return "quit"
            else:
                ui._draw_frame(use_pov=view_pov, analysis_arrows=arrows, status_lines=status)
                pygame.display.flip()
                if not _pump_events(ui, ai_delay_ms):
                    return "quit"
            move = agent.select_move(game)
            if move is None:
                winner = _other(side)
                print(f"{side.name} (AI) has no legal move — {winner.name} wins.")
                msg = f"Les {_fr_color(side)} n'ont aucun coup.\nLes {_fr_color(winner)} gagnent !"
                return "quit" if not ui.show_game_over(msg, use_pov=view_pov) else "menu"

        captured = game.play(*move)
        ui._draw_frame(use_pov=view_pov)
        pygame.display.flip()

        if captured is not None and captured.type == PieceType.KING:
            # On a king capture the turn does not advance, so game.turn is the winner.
            winner = game.turn
            print(f"Game over — {winner.name} wins by capturing the king!")
            msg = f"Roi capture !\nLes {_fr_color(winner)} gagnent."
            return "quit" if not ui.show_game_over(msg, use_pov=view_pov) else "menu"


def _fr_color(player: Player) -> str:
    return "Blancs" if player == Player.WHITE else "Noirs"


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

    ui = GameInterface(ChessGame(initial_board=_initial_board(args.curriculum, args.seed)))
    models = _available_models(args.runs_dir)
    device = None  # created lazily, the first time a seat or the overlay needs torch

    # A command-line mode plays the first match without the menu; afterwards (and on
    # "back to menu" from any game) control returns to the in-window menu.
    cli_setup = _setup_from_args(args)
    setup: MatchSetup | None
    try:
        while True:
            if cli_setup is not None:
                setup, cli_setup = cli_setup, None
            else:
                setup = ui.select_setup(models)
                if setup is None:
                    break  # window closed on the menu

            if device is None and (setup.mode != "solo" or setup.dev_mode):
                import torch

                device = torch.device(
                    "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
                )

            game = ChessGame(initial_board=_initial_board(args.curriculum, args.seed))
            ui.set_game(game)
            controllers, analyzers = _prepare_match(setup, args, device)

            # AI vs AI: keep White at the bottom (no per-move flip) and advance one
            # move at a time on the user's request rather than on a timer.
            result = run_match(
                ui,
                game,
                controllers,
                analyzers,
                setup.dev_mode,
                view_pov=setup.mode != "ai_vs_ai",
                step_mode=setup.mode == "ai_vs_ai",
                ai_delay_ms=args.ai_delay,
            )
            if result == "quit":
                break
    finally:
        pygame.quit()
        ui._initialized = False


if __name__ == "__main__":
    main()
