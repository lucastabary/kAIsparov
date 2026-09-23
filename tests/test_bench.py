"""The benchmark framework: positions, oracle, tasks, generators, suites, runner, report.

Torch-free — only the baseline contestants are built here.
"""

from __future__ import annotations

import random

import pytest

from kaisparov.bench import (
    AvoidMoves,
    BaselineContestant,
    BenchmarkReport,
    BenchmarkRunner,
    Contestant,
    FindMove,
    Oracle,
    Outcome,
    PlayOut,
    Position,
    Problem,
    ProblemGenerator,
    SamplingGenerator,
    Suite,
    SuiteSpec,
    Task,
    TaskContext,
    move_to_uci,
    uci_to_move,
)
from kaisparov.bench.generators import GenerationError
from kaisparov.bench.position import START_FEN, mirror_move
from kaisparov.bench.report import wilson_interval
from kaisparov.core.game import ChessGame
from kaisparov.core.move import Move
from kaisparov.core.pieces import PieceType, Player

# ------------------------------------------------------------------ positions


def test_start_fen_round_trips_to_the_engine_start():
    game = Position(START_FEN).to_game()
    fresh = ChessGame()
    assert Position.from_game(game).fen == START_FEN
    assert game.zobrist == fresh.zobrist  # castling rights and pawn flags agree too


def test_fen_castling_rights_decide_has_moved():
    game = Position("r3k2r/8/8/8/8/8/8/R3K2R w K - 0 1").to_game()
    assert not game.grid[4][0].has_moved and not game.grid[7][0].has_moved
    assert game.grid[0][0].has_moved  # no Q right
    assert (6, 0) in game.possible_moves((4, 0))  # O-O offered
    assert (2, 0) not in game.possible_moves((4, 0))  # O-O-O not


def test_pawns_off_their_start_rank_cannot_double_push():
    game = Position("4k3/8/8/8/8/P7/8/4K3 w - - 0 1").to_game()
    assert game.possible_moves((0, 2)) == [(0, 3)]


def test_setup_moves_arm_en_passant():
    position = Position("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1", moves=("d7d5",))
    game = position.to_game()
    assert game.en_passant_target == (3, 5)
    assert (3, 5) in game.possible_moves((4, 4))
    assert position.turn is Player.WHITE


def test_bad_fen_and_illegal_setup_moves_are_rejected():
    with pytest.raises(ValueError):
        Position("not a fen")
    with pytest.raises(ValueError):
        Position(START_FEN, moves=("e2e5",)).to_game()


def test_uci_round_trip():
    assert uci_to_move("e2e4") == Move((4, 1), (4, 3))
    assert move_to_uci(Move((4, 1), (4, 3))) == "e2e4"
    # Promotion rides in the fifth character, both ways.
    assert uci_to_move("e7e8n") == Move((4, 6), (4, 7), PieceType.KNIGHT)
    assert move_to_uci(Move((4, 6), (4, 7), PieceType.QUEEN)) == "e7e8q"


def test_mirror_swaps_colours_and_is_an_involution():
    position = Position("r3k3/8/8/8/8/8/8/4K2R w Kq - 0 1")
    mirrored = position.mirrored()
    assert mirrored.fen == "4k2r/8/8/8/8/8/8/R3K3 b Qk - 0 1"
    assert mirrored.mirrored() == position
    assert mirror_move(Move((4, 1), (4, 3))) == Move((4, 6), (4, 4))


# --------------------------------------------------------------------- oracle


def test_oracle_sees_a_mate_in_one():
    game = Position(MATE_IN_ONE).to_game()
    oracle = Oracle()
    assert oracle.mates_in_one(game) == [Move((0, 0), (0, 7))]  # Ra1-a8#
    assert oracle.win_depth(game, 3) == 1


def test_oracle_finds_a_mate_in_two():
    """King and rook against a bare king: Kf6 takes the opposition, then Rd8 mates."""
    game = Position("4k3/6K1/8/8/8/8/3R4/8 w - - 0 1").to_game()
    oracle = Oracle()
    assert not oracle.wins_within(game, 1)  # nothing mates on the spot
    assert oracle.winning_moves(game, 2) == [Move((6, 6), (5, 5))]  # Kg7-f6, and only that
    assert oracle.win_depth(game, 2) == 2


def test_oracle_counts_a_stalemate_as_no_win():
    """Qc7 would box the lone king on a8 in: every reply hangs it, but that is a draw."""
    game = Position("k7/8/8/8/8/8/8/2Q4K w - - 0 1").to_game()
    assert ((2, 0), (2, 6)) not in Oracle().winning_moves(game, 2)


def test_oracle_leaves_the_game_untouched():
    game = Position("7k/6pp/8/8/8/8/8/R5K1 w - - 0 1").to_game()
    before = (game.zobrist, list(game.position_history), game.turn)
    Oracle().win_depth(game, 3)
    assert (game.zobrist, list(game.position_history), game.turn) == before


# ---------------------------------------------------------------------- tasks


class _Fixed:
    """A policy that always answers the same thing."""

    name = "fixed"

    def __init__(self, move):
        self.move = move

    def select_move(self, game):
        if isinstance(self.move, Exception):
            raise self.move
        return self.move


def _attempt(task: Task, fen: str, policy) -> Outcome:
    return task.attempt(Position(fen).to_game(), policy, TaskContext())


# Back-rank mate in one: Ra1-a8# is the answer, Rh1-h5 a plausible wrong rook move.
# The old fixture put a king en prise, which standard chess simply cannot reach.
MATE_IN_ONE = "7k/5ppp/8/8/8/8/5PPP/R6K w - - 0 1"


def test_find_move_grades_accepted_partial_and_wrong_moves():
    task = FindMove(accepted=("a1a8",), partial=(("a1a7", 0.25),))
    assert _attempt(task, MATE_IN_ONE, _Fixed(uci_to_move("a1a8"))).solved
    partial = _attempt(task, MATE_IN_ONE, _Fixed(uci_to_move("a1a7")))
    assert (partial.score, partial.solved) == (0.25, False)
    assert _attempt(task, MATE_IN_ONE, _Fixed(uci_to_move("h2h3"))).score == 0.0


def test_a_crashing_passing_or_illegal_policy_fails_instead_of_raising():
    task = FindMove(accepted=("a1a8",))
    for answer, error in [
        (RuntimeError("boom"), "RuntimeError: boom"),
        (None, "no move"),
        (uci_to_move("a1b8"), "illegal move a1b8"),
    ]:
        outcome = _attempt(task, MATE_IN_ONE, _Fixed(answer))
        assert (outcome.solved, outcome.error) == (False, error)


def test_avoid_moves_passes_anything_else():
    fen = "3r1r1k/8/8/8/8/8/P7/4K3 w - - 0 1"
    task = AvoidMoves(forbidden=("e1d1", "e1f1"))
    assert _attempt(task, fen, _Fixed(uci_to_move("e1e2"))).solved
    assert not _attempt(task, fen, _Fixed(uci_to_move("e1d1"))).solved


def test_task_validation_catches_moves_that_do_not_exist():
    game = Position(MATE_IN_ONE).to_game()
    with pytest.raises(ValueError):
        FindMove(accepted=("a1b8",)).validate(game)


def test_play_out_converts_king_and_queen_against_random():
    task = PlayOut(opponent="random", max_plies=80, goal="win")
    outcome = _attempt(task, "8/8/8/4k3/8/8/2Q5/4K3 w - - 0 1", _material())
    assert outcome.details["result"] in ("win", "draw", "loss")
    assert outcome.plies > 0 and outcome.move is not None


def test_play_out_scores_the_goal():
    """Mate in one, played by an agent that sees it: a win, and not a loss."""
    for goal in PlayOut.GOALS:
        outcome = _attempt(
            PlayOut(opponent="random", goal=goal), MATE_IN_ONE, _Fixed(uci_to_move("a1a8"))
        )
        assert outcome.solved and outcome.plies == 1
        assert outcome.details == {"result": "win", "end": "checkmate"}


@pytest.mark.parametrize(
    "task",
    [
        FindMove(accepted=("e2e4", "d2d4"), partial=(("c2c4", 0.5),)),
        AvoidMoves(forbidden=("f2f3",)),
        PlayOut(opponent="material+safe", max_plies=30, goal="not_lose"),
    ],
)
def test_tasks_round_trip_through_dicts_and_mirror_twice_to_themselves(task):
    assert Task.from_dict(task.to_dict()) == task
    assert task.mirrored().mirrored() == task


def test_mirrored_problem_keeps_its_answer():
    problem = Problem("p", "t", Position(MATE_IN_ONE), FindMove(accepted=("a1a8",)))
    mirrored = problem.mirrored()
    mirrored.validate()
    assert mirrored.task == FindMove(accepted=("a8a1",))
    assert mirrored.tags == ("mirrored",) and mirrored.mirrored().tags == ()


# ----------------------------------------------------------------- generators


def test_registry_lists_the_builtin_generators():
    assert {"fixed", "mate_in_one"} <= set(ProblemGenerator.available())
    with pytest.raises(ValueError):
        ProblemGenerator.create("no_such_generator")
    with pytest.raises(TypeError):
        ProblemGenerator.create("mate_in_one", {"no_such_param": 1})


def test_mate_in_one_problems_are_valid_and_reproducible():
    make = lambda: ProblemGenerator.create("mate_in_one").generate(15, random.Random(7))  # noqa: E731
    problems = make()
    assert problems == make()
    assert len({p.position for p in problems}) == 15
    assert {p.position.turn for p in problems} == {Player.WHITE, Player.BLACK}  # mirroring
    oracle = Oracle()
    for problem in problems:
        problem.validate()
        game = problem.position.to_game()
        assert set(problem.task.accepted) == {move_to_uci(m) for m in oracle.mates_in_one(game)}


def test_a_generator_that_never_accepts_gives_up_loudly():
    class Never(SamplingGenerator):
        def propose(self, rng):
            return None

    with pytest.raises(GenerationError):
        Never(attempts_per_problem=3).generate(2, random.Random(0))


# --------------------------------------------------------------------- suites

# Same shape as SPEC, but on a theme the greedy baseline actually solves.
CAPTURE_SPEC = {
    "name": "c",
    "seed": 3,
    "problems": [
        {"generator": "free_capture", "count": 4},
        {
            "generator": "fixed",
            "params": {
                "theme": "hand",
                "problems": [
                    {
                        # A hanging queen on a6, straight down the open a-file.
                        "fen": "7k/8/q7/8/8/8/8/R3K3 w - - 0 1",
                        "task": {"kind": "find_move", "accepted": ["a1a6"]},
                    }
                ],
            },
        },
    ],
}

SPEC = {
    "name": "t",
    "seed": 3,
    "problems": [
        {"generator": "mate_in_one", "count": 4},
        {
            "generator": "fixed",
            "params": {
                "theme": "hand",
                "problems": [
                    {"fen": MATE_IN_ONE, "task": {"kind": "find_move", "accepted": ["a1a8"]}}
                ],
            },
        },
    ],
}


def test_suite_spec_builds_the_same_suite_every_time():
    first, second = SuiteSpec.from_dict(SPEC).build(), SuiteSpec.from_dict(SPEC).build()
    assert first.problems == second.problems
    assert [p.id for p in first][-1] == "fixed-0000"
    assert first.themes() == ["hand", "mate_in_one"]


def test_adding_an_entry_does_not_reshuffle_the_others():
    base = SuiteSpec.from_dict(SPEC).build()
    grown = dict(
        SPEC,
        problems=[{"generator": "mate_in_one", "count": 2, "name": "extra"}, *SPEC["problems"]],
    )
    rebuilt = SuiteSpec.from_dict(grown).build()
    assert [p for p in rebuilt if not p.id.startswith("extra")] == base.problems


def test_entries_sharing_a_generator_need_names():
    spec = dict(SPEC, problems=[{"generator": "mate_in_one"}, {"generator": "mate_in_one"}])
    with pytest.raises(ValueError, match="name"):
        SuiteSpec.from_dict(spec)


def test_frozen_suite_round_trips(tmp_path):
    suite = SuiteSpec.from_dict(SPEC).build()
    loaded = Suite.load(suite.save(tmp_path / "suite.jsonl"))
    assert loaded.problems == suite.problems
    assert (loaded.name, loaded.seed) == ("t", 3)


def test_select_filters_by_theme_and_caps_per_theme():
    suite = SuiteSpec.from_dict(SPEC).build()
    assert {p.theme for p in suite.select(themes=["hand"])} == {"hand"}
    assert len(suite.select(limit=2)) == 3


# ---------------------------------------------------------------- contestants


def _material():
    return BaselineContestant(kind="material").build(seed=0)


def test_contestant_specs_parse():
    assert Contestant.parse("random").name == "random"
    safe = Contestant.parse("material+safe")
    assert isinstance(safe, BaselineContestant) and safe.modifiers.safe
    labelled = Contestant.parse("greedy=material")
    assert (labelled.name, labelled.spec) == ("greedy", "greedy=material")
    for bad in ("stockfish", "random+minimax2", "material+sample", "random+wings", "run:"):
        with pytest.raises(ValueError):
            Contestant.parse(bad)


def test_material_minimax_contestant_builds_the_search():
    from kaisparov.agents.minimax_agent import MinimaxAgent
    from kaisparov.analysis.evaluators import HeuristicEvaluator, MaterialEvaluator

    contestant = Contestant.parse("material+minimax2")
    assert contestant.name == "material+minimax2"
    agent = contestant.build(seed=0)
    assert isinstance(agent, MinimaxAgent) and agent.depth == 2
    assert isinstance(agent.evaluator, MaterialEvaluator)
    # heuristic always searches: one ply alone, N with +minimax<N>.
    alone, deeper = (
        Contestant.parse("heuristic").build(0),
        Contestant.parse("heuristic+minimax3").build(0),
    )
    assert isinstance(alone.evaluator, HeuristicEvaluator) and alone.depth == 1
    assert deeper.depth == 3


def test_run_spec_resolves_through_the_registry(tmp_path):
    run_dir = tmp_path / "20260101-000000_rgcn"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "run.json").write_text(
        '{"run_id": "20260101-000000_rgcn", "model": "rgcn", "config": {"hidden_dim": 16},'
        ' "checkpoints": [{"epoch": 10, "file": "epoch10.pth"}]}',
        encoding="utf-8",
    )
    contestant = Contestant.parse("run:20260101-000000_rgcn+minimax2", runs_dir=tmp_path)
    assert contestant.checkpoint.name == "epoch10.pth"  # latest by default
    assert (contestant.hidden_dim, contestant.modifiers.search_depth) == (16, 2)
    assert contestant.name == "20260101-000000_rgcn@latest+minimax2"
    assert (
        Contestant.parse("run:20260101-000000_rgcn@10", runs_dir=tmp_path).checkpoint.name
        == "epoch10.pth"
    )


# --------------------------------------------------------------- runner/report


def _decision(result):
    """What a result says about the play, leaving out the (noisy) timing."""
    return result.problem_id, result.outcome.move, result.outcome.score, result.outcome.error


def test_runner_ranks_material_above_random_on_free_captures(tmp_path):
    """The greedy baseline must beat the random one where greed is the right answer.

    On free_capture, not on mate_in_one: taking a hanging piece is exactly what the
    material agent does, while a mate in one is invisible to it.
    """
    suite = SuiteSpec.from_dict(CAPTURE_SPEC).build()
    runner = BenchmarkRunner(suite, seed=1)
    report = runner.run([Contestant.parse("random"), Contestant.parse("material")])
    random_result, material_result = report.contestants
    assert material_result.overall().solve_rate == 1.0
    assert material_result.overall().solve_rate > random_result.overall().solve_rate
    assert "free_capture" in report.format_table()

    loaded = BenchmarkReport.load(report.save(tmp_path / "report.json"))
    assert [_decision(r) for r in loaded.contestants[1].results] == [
        _decision(r) for r in material_result.results
    ]
    assert material_result.metrics()["bench_hand_solve_rate"] == 1.0


def test_outcomes_do_not_depend_on_the_rest_of_the_suite():
    suite = SuiteSpec.from_dict(SPEC).build()
    full = BenchmarkRunner(suite).evaluate(Contestant.parse("random"))
    part = BenchmarkRunner(suite.select(themes=["mate_in_one"])).evaluate(
        Contestant.parse("random")
    )
    assert [_decision(r) for r in part.results] == [
        _decision(r) for r in full.results if r.theme == "mate_in_one"
    ]


def test_runner_refuses_duplicate_names():
    suite = SuiteSpec.from_dict(SPEC).build()
    with pytest.raises(ValueError):
        BenchmarkRunner(suite).run([Contestant.parse("random"), Contestant.parse("random")])


def test_wilson_interval_is_sane():
    low, high = wilson_interval(10, 20)
    assert low < 0.5 < high
    assert wilson_interval(0, 20)[0] == 0.0 and wilson_interval(20, 20)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_mates_in_one_lists_every_move_that_mates():
    # The queen on c6 mates the king in the h-file corner two ways: Qh6 and Qg2.
    game = Position("8/6R1/2Q5/5K2/8/8/7k/8 w - - 0 1").to_game()
    assert set(Oracle().mates_in_one(game)) == {Move((2, 5), (7, 5)), Move((2, 5), (6, 1))}
    assert game.grid[2][5].type is PieceType.QUEEN
