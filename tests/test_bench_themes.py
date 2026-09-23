"""The benchmark themes: oracle extensions, the new task kinds, and every generator.

Torch-free. Each generator is exercised on a couple of problems, and the themes whose
answer key is enumerated are re-checked against the oracle from scratch.
"""

from __future__ import annotations

import random

import pytest

from kaisparov.bench import BenchmarkReport, ContestantResult, Oracle, Position, ProblemResult
from kaisparov.bench.analyzers import MaterialAnalyzer
from kaisparov.bench.generators import ProblemGenerator
from kaisparov.bench.oracle import WIN
from kaisparov.bench.position import mirror_move, move_to_uci, uci_to_move
from kaisparov.bench.suite import SuiteSpec
from kaisparov.bench.tasks import (
    Outcome,
    PolicyRank,
    SameMove,
    Task,
    TaskContext,
    ValueSign,
    WinMaterial,
)
from kaisparov.core.draw import REPETITION, STALEMATE
from kaisparov.core.pieces import PieceType, Player
from kaisparov.insights import MoveInsight, PositionAnalysis

# ---------------------------------------------------------------------- oracle


def test_material_gain_sees_a_free_piece_and_a_defended_one():
    # White queen d1 can take a loose knight on d5, or a pawn-defended knight on h5.
    game = Position("4k3/8/6p1/3n3n/8/8/8/3QK3 w - - 0 1").to_game()
    oracle = Oracle()
    assert oracle.material_gain(game, uci_to_move("d1d5"), 3) == 3.0
    assert oracle.material_gain(game, uci_to_move("d1h5"), 3) < 0  # Qxh5 gxh5


def test_material_search_counts_a_mate_as_a_win():
    """Mate is worth WIN, not the material on the board.

    Ra8# leaves White a rook up and nothing else, so a search that only counted
    material would score it 0 and rank it alongside any other quiet rook move.
    """
    game = Position("7k/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1").to_game()
    assert Oracle().material_gain(game, uci_to_move("a1a8"), 1) == WIN


def test_safe_moves_threats_and_forced_losses():
    oracle = Oracle()
    # Back rank: if White passes, the rook mates on a1. That is what threatens() asks.
    game = Position("r5k1/8/8/8/8/8/5PPP/6K1 w - - 0 1").to_game()
    assert oracle.threatens(game, 2)

    # Black to move with a lone king on g8: the queen on c7 covers the 7th rank and
    # the rook swings to a8, so both squares it can run to are mated next ply.
    lost = Position("6k1/2Q5/8/8/2K5/R7/8/8 b - - 0 1").to_game()
    assert lost.legal_moves()  # it is not already over...
    assert oracle.safe_moves(lost) == []  # ...but nothing survives
    assert oracle.loses_within(lost, 1)


def test_draws_after_names_stalemate_and_repetition():
    oracle = Oracle()
    stalemate = Position("k7/8/8/8/8/8/8/2Q4K w - - 0 1").to_game()
    assert oracle.draws_after(stalemate, uci_to_move("c1c7")) == STALEMATE

    # Two rooks shuffling on and off their corner. Neither side has castling rights
    # to lose, so every cycle lands on exactly the position it started from: after
    # one and three quarter cycles, going home is the third occurrence.
    shuffle = ("a1b1", "a8b8", "b1a1", "b8a8") * 2
    repeated = Position("r5k1/8/8/8/8/8/8/R5K1 w - - 0 1", shuffle[:-1]).to_game()
    assert oracle.draws_after(repeated, uci_to_move("b8a8")) == REPETITION
    assert oracle.draws_after(repeated, uci_to_move("b8c8")) is None


def test_mirroring_keeps_setup_moves_and_their_en_passant_right():
    position = Position("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1", moves=("d7d5",))
    mirrored = position.mirrored()
    assert mirrored.moves == ("d2d4",)
    assert mirrored.to_game().en_passant_target == (3, 2)
    assert mirrored.mirrored() == position


# ----------------------------------------------------------------------- tasks


class _Fixed:
    name = "fixed"

    def __init__(self, *moves):
        self.moves = list(moves)

    def select_move(self, game):
        return uci_to_move(self.moves.pop(0))


class _Analyzer:
    name = "stub"

    def __init__(self, value, ranking):
        self.analysis = PositionAnalysis(
            candidates=tuple(MoveInsight(uci_to_move(m), w) for m, w in ranking), value=value
        )

    def analyze(self, game):
        return self.analysis


LOOSE_KNIGHT = "4k3/8/6p1/3n3n/8/8/8/3QK3 w - - 0 1"


def test_win_material_accepts_any_move_that_wins_enough():
    task = WinMaterial(min_gain=3, best_gain=3, plies=3, reference=("d1d5",))
    game = Position(LOOSE_KNIGHT).to_game()
    assert task.attempt(game, _Fixed("d1d5"), TaskContext()).solved
    poisoned = task.attempt(Position(LOOSE_KNIGHT).to_game(), _Fixed("d1h5"), TaskContext())
    assert not poisoned.solved and poisoned.score == 0.0


def test_same_move_compares_the_contestant_with_itself():
    position = Position(LOOSE_KNIGHT)
    mirror = SameMove(variant=position.mirrored(), mirror=True)
    flipped = move_to_uci(mirror_move(uci_to_move("d1d5")))
    assert mirror.attempt(position.to_game(), _Fixed("d1d5", flipped), TaskContext()).solved
    assert not mirror.attempt(position.to_game(), _Fixed("d1d5", "d8d7"), TaskContext()).solved


def test_probes_read_the_analyzer_and_skip_without_one():
    game = Position(LOOSE_KNIGHT).to_game()
    rank = PolicyRank(accepted=("d1d5",), k=2)
    analyzer = _Analyzer(2.5, [("d1h5", 3.0), ("d1d5", 1.0), ("e1e2", 0.0)])
    outcome = rank.attempt(game, _Fixed(), TaskContext(analyzer=analyzer))
    assert outcome.solved and outcome.details["rank"] == 2
    assert outcome.score == pytest.approx(0.25)
    assert ValueSign(expected=1).attempt(game, _Fixed(), TaskContext(analyzer=analyzer)).solved
    assert not ValueSign(expected=-1).attempt(game, _Fixed(), TaskContext(analyzer=analyzer)).solved
    assert ValueSign(expected=1).attempt(game, _Fixed(), TaskContext()).skipped


@pytest.mark.parametrize(
    "task",
    [
        WinMaterial(2.0, 5.0, 3, ("e2e4",)),
        SameMove(Position(LOOSE_KNIGHT).mirrored(), mirror=True),
        ValueSign(-1),
        PolicyRank(("d1d5", "e1e2"), k=5),
    ],
)
def test_new_tasks_round_trip_and_mirror_twice_to_themselves(task):
    assert Task.from_dict(task.to_dict()) == task
    assert task.mirrored().mirrored() == task


def test_material_analyzer_ranks_the_free_piece_first():
    analysis = MaterialAnalyzer().analyze(Position(LOOSE_KNIGHT).to_game())
    assert move_to_uci(analysis.candidates[0].move) == "d1d5"
    assert analysis.value > 0  # a queen against two knights and a pawn


# ---------------------------------------------------------------------- report


def test_skipped_outcomes_are_counted_apart_and_shown_as_na():
    skipped = ContestantResult(
        "random", "random", [ProblemResult("p", "value_sign", 1, Outcome.skip("x"))]
    )
    played = ContestantResult(
        "material", "material", [ProblemResult("p", "value_sign", 1, Outcome(1.0, True))]
    )
    report = BenchmarkReport.merge(
        [BenchmarkReport("s", [skipped]), BenchmarkReport("s", [played])]
    )
    assert skipped.overall().count == 0 and skipped.overall().skipped == 1
    assert "n/a" in report.format_table()
    with pytest.raises(ValueError):
        BenchmarkReport.merge([BenchmarkReport("s", [played]), BenchmarkReport("other", [played])])


# ------------------------------------------------------------------ generators

CHEAP = {
    "avoid_mate": {},
    "conversion": {},
    "distractor_invariance": {},
    "escape_check": {},
    "fork": {},
    "free_capture": {},
    "hold": {},
    "mate_in_one": {},
    "material_edge": {},
    "mirror_consistency": {},
    "repetition_trap": {},
    "special_rules": {},
    "stalemate_trap": {},
    "value_sign": {},
    "win_in_n": {"depth": 2},
    "policy_rank": {"source": "escape_check"},
    "avoid_piece_hang": {},
    "parry_threat": {},
}


def test_every_registered_generator_is_covered_here():
    assert set(ProblemGenerator.available()) - {"fixed"} == set(CHEAP)


class _Generated(dict):
    """Each generator's two problems, drawn on first use: under pytest-xdist a worker
    only pays for the generators its own tests need (avoid_mate alone takes ~20 s)."""

    def __missing__(self, name):
        problems = ProblemGenerator.create(name, CHEAP[name]).generate(
            2, random.Random(f"test:{name}")
        )
        self[name] = problems
        return problems


@pytest.fixture(scope="module")
def generated():
    return _Generated()


@pytest.mark.parametrize("name", sorted(CHEAP))
def test_generators_are_valid_reproducible_and_mirrorable(name, generated):
    problems = generated[name]
    assert len(problems) == 2
    again = ProblemGenerator.create(name, CHEAP[name]).generate(2, random.Random(f"test:{name}"))
    assert again == problems
    for problem in problems:
        problem.validate()
        problem.mirrored().validate()
        assert Task.from_dict(problem.task.to_dict()) == problem.task


def _games(generated, name):
    return [(p, p.position.to_game()) for p in generated[name]]


def test_escape_check_answers_are_exactly_the_safe_moves(generated):
    for problem, game in _games(generated, "escape_check"):
        assert game.is_in_check(game.turn)
        assert set(problem.task.accepted) == {move_to_uci(m) for m in Oracle().safe_moves(game)}


def test_draw_traps_forbid_exactly_the_drawing_moves(generated):
    oracle = Oracle()
    for name, reason in (("stalemate_trap", STALEMATE), ("repetition_trap", REPETITION)):
        for problem, game in _games(generated, name):
            moves = game.legal_moves()
            drawing = {move_to_uci(m) for m in moves if oracle.draws_after(game, m) == reason}
            assert set(problem.task.forbidden) == drawing


def test_special_rules_leave_one_safe_move_and_it_is_special(generated):
    for _problem, game in _games(generated, "special_rules"):
        (move,) = Oracle().safe_moves(game)
        piece = game.grid[move[0][0]][move[0][1]]
        castles = piece.type == PieceType.KING and abs(move[1][0] - move[0][0]) == 2
        assert castles or move[1] == game.en_passant_target


def test_win_in_2_is_exact(generated):
    oracle = Oracle()
    for problem, game in _games(generated, "win_in_n"):
        assert oracle.win_depth(game, 2) == 2
        assert set(problem.task.accepted) == {move_to_uci(m) for m in oracle.winning_moves(game, 2)}


def test_value_sign_labels_agree_with_the_oracle(generated):
    oracle = Oracle()
    for problem, game in _games(generated, "value_sign"):
        if problem.task.expected > 0:
            assert oracle.wins_within(game, 2)
        else:
            assert oracle.loses_within(game, 1)


def test_policy_rank_on_a_shared_stream_asks_about_the_same_positions():
    spec = SuiteSpec.from_dict(
        {
            "problems": [
                {"generator": "escape_check", "count": 3},
                {
                    "generator": "policy_rank",
                    "name": "policy",
                    "stream": "escape_check",
                    "count": 3,
                    "params": {"source": "escape_check"},
                },
            ]
        }
    )
    suite = spec.build()
    moves = [p for p in suite if p.theme == "escape_check"]
    ranks = [p for p in suite if p.theme == "policy_escape_check"]
    assert [p.position for p in moves] == [p.position for p in ranks]
    assert all(r.task.accepted == m.task.accepted for m, r in zip(moves, ranks, strict=True))
    assert {p.position.turn for p in moves} <= {Player.WHITE, Player.BLACK}
