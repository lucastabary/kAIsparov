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

``--review`` turns on the move review: every move played gets a chess.com-style
grade badge on its destination square (``!!`` brilliant ... ``??`` blunder), and the
end-of-game banner reports each side's accuracy. Both switches are independent and
also live on the start menu, so either can be toggled without restarting. See
:mod:`kaisparov.analysis` for how a move is graded.

What each badge means is one click away — from the menu, or from the side panel
during the game (or the ``L`` key). The wording lives in ``_QUALITY_HELP`` here; the
interface only lays the rows out.
"""

from __future__ import annotations

import argparse

import pygame

from kaisparov.agents.base import Policy
from kaisparov.analysis import GameReview, HeuristicEvaluator, MaterialEvaluator, MoveJudge
from kaisparov.core.board import ChessGame
from kaisparov.core.coords import Coord
from kaisparov.core.game_interface import (
    GameInterface,
    LegendEntry,
    MatchSetup,
    ModelOption,
    MoveArrow,
    MoveBadge,
)
from kaisparov.core.pieces import PieceType, Player
from kaisparov.insights import Analyzer, MoveQuality, MoveVerdict, PositionAnalysis
from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum

# Grades worth naming in the end-of-game recap; the rest is ordinary play.
_NOTABLE = (
    MoveQuality.BRILLIANT,
    MoveQuality.GREAT,
    MoveQuality.MISTAKE,
    MoveQuality.MISS,
    MoveQuality.BLUNDER,
)

# What each grade means, in one line, for the legend panel. The badge itself is two
# characters wide, so this table is the key to reading it — every quality appears,
# in the enum's own best-to-worst order.
_QUALITY_HELP: dict[MoveQuality, str] = {
    MoveQuality.BRILLIANT: "Sacrifice sain: vous donnez du materiel, et ca marche.",
    MoveQuality.GREAT: "Le seul coup qui tenait la position.",
    MoveQuality.BEST: "Le premier choix du moteur.",
    MoveQuality.EXCELLENT: "A un cheveu du meilleur (moins de 2 pts).",
    MoveQuality.GOOD: "Correct, sans plus (moins de 5 pts).",
    MoveQuality.FORCED: "Aucun autre coup n'etait possible.",
    MoveQuality.INACCURACY: "Vous lachez 5 a 10 pts.",
    MoveQuality.MISTAKE: "Vous lachez 10 a 20 pts.",
    MoveQuality.MISS: "Un gain immediat etait la, vous l'avez laisse passer.",
    MoveQuality.BLUNDER: "Plus de 20 pts jetes par la fenetre.",
}


def _legend_entries() -> list[LegendEntry]:
    """The badge legend, best grade first."""
    return [
        LegendEntry(
            symbol=quality.symbol,
            title=quality.caption,
            detail=_QUALITY_HELP[quality],
            tone=quality.name.lower(),
        )
        for quality in MoveQuality
    ]


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
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)

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
    # game) by reusing whichever model we already loaded, if any. The critic-backed
    # review needs the same thing, so it shares the lookup.
    wants_model = setup.dev_mode or (setup.review_mode and args.judge_eval == "critic")
    if wants_model and shared_analyzer is None:
        shared_analyzer = _try_build_analyzer(args, device)
    if setup.dev_mode:
        for color in (Player.WHITE, Player.BLACK):
            if analyzers[color] is None:
                analyzers[color] = shared_analyzer

    return controllers, analyzers, _build_judge(args, shared_analyzer, setup.review_mode)


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


def _build_judge(args, analyzer: Analyzer | None, enabled: bool) -> MoveJudge | None:
    """The move grader for this match, or ``None`` when the review is off.

    ``--judge-eval critic`` reuses the model already loaded for the AI seat (or the
    developer overlay) rather than loading a second one; with no model around it
    falls back to the handcrafted evaluator instead of disabling the review, since
    the whole point is that the feature works on a fresh clone.
    """
    if not enabled:
        return None

    if args.judge_eval == "critic":
        model = getattr(analyzer, "model", None)
        processor = getattr(analyzer, "processor", None)
        if model is not None and processor is not None:
            from kaisparov.analysis.critic import CriticEvaluator

            print("Move review: grading with the model's critic.")
            return MoveJudge(CriticEvaluator(model, processor), lookahead=args.judge_depth)
        print("Move review: no model available, grading with the handcrafted evaluator.")

    evaluator = MaterialEvaluator() if args.judge_eval == "material" else HeuristicEvaluator()
    return MoveJudge(evaluator, lookahead=args.judge_depth)


def _board_orientation(setup: MatchSetup) -> Player | None:
    """Which side sits at the bottom of the board for this match.

    Only two humans on one keyboard want the board to turn over between moves. Against
    an AI it stays on the human's side: letting it follow the side to move flipped the
    whole board for the fraction of a second the AI was thinking. AI vs AI keeps White
    at the bottom so the spectator's view never moves either.
    """
    if setup.mode == "vs_ai":
        return setup.human_color
    if setup.mode == "ai_vs_ai":
        return Player.WHITE
    return None  # solo: follow the side to move


def _badge_from_verdict(verdict: MoveVerdict) -> MoveBadge:
    """Map a verdict onto the mark the interface draws on the destination square."""
    return MoveBadge(
        square=verdict.move[1],
        symbol=verdict.quality.symbol,
        tone=verdict.quality.name.lower(),
        caption=verdict.quality.caption,
    )


def _review_lines(side: Player, verdict: MoveVerdict) -> list[str]:
    """Side-panel text for the move just played."""
    src, dst = verdict.move
    lines = [
        f"{_fr_color(side)}: {_algebraic(src)}->{_algebraic(dst)}",
        f"{verdict.quality.caption}  (-{verdict.loss:.1f} pts)",
    ]
    if verdict.best_move is not None and verdict.best_move != verdict.move:
        best_src, best_dst = verdict.best_move
        lines.append(f"Mieux: {_algebraic(best_src)}->{_algebraic(best_dst)}")
    return lines


def _review_summary(review: GameReview) -> list[str]:
    """End-of-game recap: accuracy per side, then the moves worth talking about."""
    lines: list[str] = []
    accuracies = {player: review.accuracy(player) for player in (Player.WHITE, Player.BLACK)}
    if any(value is not None for value in accuracies.values()):
        parts = [
            f"{_fr_color(player)} {value:.0f}%"
            for player, value in accuracies.items()
            if value is not None
        ]
        lines.append("Precision: " + "  ".join(parts))

    for player in (Player.WHITE, Player.BLACK):
        counts = review.counts(player)
        notable = [
            f"{count} {quality.caption.lower()}"
            for quality, count in counts.items()
            if count and quality in _NOTABLE
        ]
        if notable:
            lines.append(f"{_fr_color(player)}: " + ", ".join(notable))
    return lines


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
    judge: MoveJudge | None = None,
    view_as: Player | None = None,
    step_mode: bool = False,
    ai_delay_ms: int = 500,
) -> str:
    """Drive a game where each side is a human (mouse) or a policy.

    ``view_as`` fixes the board orientation to one side. ``None`` follows the side to
    move, which flips the board every ply -- only ever right when two humans share a
    keyboard. ``step_mode`` makes AI seats wait for the user to request each move (the
    "Coup suivant" button) instead of auto-advancing on a timer.
    ``judge`` (optional) grades each move as it is played and drives the badge on the
    board, the panel text, and the end-of-game accuracy recap.

    Returns ``"quit"`` if the window was closed, or ``"menu"`` when the game ends (so
    the caller can return to the start menu without tearing down the window).
    """
    ui._ensure_initialized()
    review = GameReview()
    badge: MoveBadge | None = None
    review_status: list[str] = []

    def finish(message: str) -> str:
        summary = _review_summary(review) if judge is not None else []
        full = "\n".join([message, *summary])
        return "quit" if not ui.show_game_over(full, view_as=view_as) else "menu"

    while True:
        side = game.turn
        agent = controllers[side]
        analyzer = analyzers.get(side)

        arrows, status = [], []
        if dev_mode and analyzer is not None:
            arrows, status = _overlay_from_analysis(analyzer.analyze(game))
        status = status + review_status

        if agent is None:  # human seat
            move = ui._get_single_move(
                view_as=view_as,
                analysis_arrows=arrows,
                status_lines=status,
                badge=badge,
                legend_button=judge is not None,
            )
            if move is None:
                return "quit"
        else:  # AI seat
            if step_mode:
                turn_status = status + [f"Trait aux {_fr_color(side)}"]
                if not ui.wait_for_step(
                    view_as=view_as,
                    analysis_arrows=arrows,
                    status_lines=turn_status,
                    badge=badge,
                    legend_button=judge is not None,
                ):
                    return "quit"
            else:
                ui._draw_frame(
                    view_as=view_as,
                    analysis_arrows=arrows,
                    status_lines=status,
                    badge=badge,
                    legend_button=judge is not None,
                )
                pygame.display.flip()
                if not _pump_events(ui, ai_delay_ms):
                    return "quit"
            move = agent.select_move(game)
            if move is None:
                winner = _other(side)
                print(f"{side.name} (AI) has no legal move — {winner.name} wins.")
                return finish(
                    f"Les {_fr_color(side)} n'ont aucun coup.\nLes {_fr_color(winner)} gagnent !"
                )

        # Grading must happen before the move is played: the verdict compares it
        # against every alternative in the position it was played from.
        if judge is not None:
            verdict = judge.judge(game, move)
            if verdict is not None:
                review.add(side, verdict)
                badge = _badge_from_verdict(verdict)
                review_status = _review_lines(side, verdict)

        captured = game.play(*move)
        ui._draw_frame(
            view_as=view_as,
            badge=badge,
            status_lines=review_status,
            legend_button=judge is not None,
        )
        pygame.display.flip()

        if captured is not None and captured.type == PieceType.KING:
            # On a king capture the turn does not advance, so game.turn is the winner.
            winner = game.turn
            print(f"Game over — {winner.name} wins by capturing the king!")
            return finish(f"Roi capture !  Les {_fr_color(winner)} gagnent.")


def _fr_color(player: Player) -> str:
    return "Blancs" if player == Player.WHITE else "Noirs"


# ----------------------------------------------------------------------- CLI


def _setup_from_args(args) -> MatchSetup | None:
    """A mode chosen on the command line bypasses the menu; otherwise ``None``."""
    human_color = Player.WHITE if args.color == "white" else Player.BLACK
    if args.vs_ai:
        return MatchSetup("vs_ai", human_color, args.dev, args.review)
    if args.ai_vs_ai:
        return MatchSetup("ai_vs_ai", human_color, args.dev, args.review)
    if args.solo:
        return MatchSetup("solo", human_color, args.dev, args.review)
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kaisparov play", description="Play in a pygame window.")
    parser.add_argument("--vs-ai", action="store_true", help="Play against the neural agent.")
    parser.add_argument("--ai-vs-ai", action="store_true", help="Watch two AIs play each other.")
    parser.add_argument("--solo", action="store_true", help="Two humans on one keyboard.")
    parser.add_argument("--dev", action="store_true", help="Developer mode: show model analysis.")
    parser.add_argument(
        "--review", action="store_true", help="Grade every move played (chess.com-style badges)."
    )
    parser.add_argument(
        "--judge-eval",
        choices=["heuristic", "material", "critic"],
        default="heuristic",
        help="What the review grades against (default: handcrafted, no model needed).",
    )
    parser.add_argument(
        "--judge-depth",
        type=int,
        default=None,
        help="Plies searched per candidate move when grading (default: per evaluator).",
    )
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
    ui.legend = _legend_entries()  # menu + in-game key to the grade badges
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

            needs_torch = (
                setup.mode != "solo"
                or setup.dev_mode
                or (setup.review_mode and args.judge_eval == "critic")
            )
            if device is None and needs_torch:
                import torch

                device = torch.device(
                    "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
                )

            game = ChessGame(initial_board=_initial_board(args.curriculum, args.seed))
            ui.set_game(game)
            controllers, analyzers, judge = _prepare_match(setup, args, device)

            # AI vs AI: keep White at the bottom (no per-move flip) and advance one
            # move at a time on the user's request rather than on a timer.
            result = run_match(
                ui,
                game,
                controllers,
                analyzers,
                setup.dev_mode,
                judge=judge,
                view_as=_board_orientation(setup),
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
