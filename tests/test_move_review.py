"""Move review: win-probability maths, the labels, and the per-game recap.

Torch-free — the judge only ever sees an evaluator, and the handcrafted ones are
pure Python.
"""

from __future__ import annotations

import os

import pytest

# The legend rows live in the play module, which pulls in pygame. Importing it is
# harmless headless as long as SDL has a driver it can open without a display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from kaisparov.analysis import (
    GameReview,
    HeuristicEvaluator,
    MaterialEvaluator,
    MoveJudge,
    accuracy_from_loss,
    win_probability,
)
from kaisparov.analysis.judge import WIN
from kaisparov.core.board import ChessGame
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.insights import MoveQuality, MoveVerdict

W, B = Player.WHITE, Player.BLACK


def position(pieces: dict, turn: Player = W) -> ChessGame:
    """A board holding only ``{(col, row): (player, piece_type)}``."""
    game = ChessGame(initial_board=[[None] * 8 for _ in range(8)], turn=turn)
    for (col, row), (player, piece_type) in pieces.items():
        game.grid[col][row] = Piece(player, piece_type)
    return game


# ------------------------------------------------------------------ the curve


def test_win_probability_is_centred_and_monotonic():
    slope = MaterialEvaluator.slope
    assert win_probability(0.0, slope) == pytest.approx(0.5)
    assert win_probability(1.0, slope) > win_probability(0.0, slope)
    assert win_probability(-1.0, slope) < win_probability(0.0, slope)
    assert win_probability(WIN, slope) == 1.0
    assert win_probability(-WIN, slope) == 0.0


def test_win_probability_saturates_so_a_pawn_matters_less_when_winning():
    """The whole reason the review works in win probability rather than in pawns."""
    slope = MaterialEvaluator.slope
    near_equal = win_probability(1.0, slope) - win_probability(0.0, slope)
    already_won = win_probability(11.0, slope) - win_probability(10.0, slope)
    assert near_equal > 5 * already_won


def test_accuracy_from_loss_is_bounded_and_decreasing():
    assert accuracy_from_loss(0.0) == pytest.approx(100.0, abs=0.1)
    assert accuracy_from_loss(5.0) > accuracy_from_loss(25.0)
    assert 0.0 <= accuracy_from_loss(100.0) <= 100.0


# ----------------------------------------------------------------- the labels


def test_ranking_prefers_a_developing_move_to_a_rook_pawn():
    judge = MoveJudge(HeuristicEvaluator())
    game = ChessGame()
    assert judge.judge(game, ((4, 1), (4, 3))).quality is MoveQuality.BEST  # e4
    assert judge.judge(game, ((0, 1), (0, 2))).quality is not MoveQuality.BEST  # a3


def test_hanging_a_queen_is_punished():
    """The queen walks onto a square a pawn covers; the alternative keeps it safe."""
    game = position(
        {
            (4, 0): (W, PieceType.KING),
            (3, 0): (W, PieceType.QUEEN),
            (4, 7): (B, PieceType.KING),
            (2, 5): (B, PieceType.PAWN),
            (0, 7): (B, PieceType.ROOK),
        }
    )
    judge = MoveJudge(MaterialEvaluator())
    hangs = judge.judge(game, ((3, 0), (3, 4)))  # Qd5, met by cxd5
    safe = judge.judge(game, ((3, 0), (3, 3)))  # Qd4
    assert hangs.loss > safe.loss
    assert hangs.quality in {MoveQuality.MISTAKE, MoveQuality.BLUNDER, MoveQuality.MISS}


def test_missing_a_king_capture_is_a_miss():
    game = position(
        {
            (4, 0): (W, PieceType.KING),
            (0, 4): (W, PieceType.ROOK),  # Ra5 takes the king on e5
            (7, 0): (W, PieceType.ROOK),
            (4, 4): (B, PieceType.KING),
        }
    )
    judge = MoveJudge(MaterialEvaluator())
    assert judge.judge(game, ((0, 4), (4, 4))).quality is MoveQuality.BEST
    missed = judge.judge(game, ((7, 0), (7, 1)))
    assert missed.quality is MoveQuality.MISS
    assert missed.win_prob_before == 1.0


def test_a_sound_queen_sacrifice_is_brilliant():
    """Qxg7 gives up the queen; Kxg7 loses the king to Rxg7, so the sac is sound."""
    game = position(
        {
            (0, 0): (W, PieceType.KING),
            (3, 3): (W, PieceType.QUEEN),
            (6, 0): (W, PieceType.ROOK),
            (7, 7): (B, PieceType.KING),
            (6, 6): (B, PieceType.PAWN),
            (7, 5): (B, PieceType.ROOK),
            (2, 4): (B, PieceType.QUEEN),
        }
    )
    verdict = MoveJudge(HeuristicEvaluator()).judge(game, ((3, 3), (6, 6)))
    assert verdict.quality is MoveQuality.BRILLIANT
    assert verdict.sacrificed >= 2.0


def test_a_quiet_best_move_is_not_a_sacrifice():
    verdict = MoveJudge(HeuristicEvaluator()).judge(ChessGame(), ((4, 1), (4, 3)))
    assert verdict.sacrificed == 0.0
    assert verdict.quality is MoveQuality.BEST


def test_the_only_legal_move_is_forced():
    """A single pawn one square from the last rank: it can push, and that is all.

    Contrived on purpose: the variant filters nothing for legality, so a king always
    has its steps available and can never be the piece that runs out of moves.
    """
    game = position({(0, 6): (W, PieceType.PAWN), (7, 7): (B, PieceType.KING)})
    judge = MoveJudge(MaterialEvaluator())
    assert [move for move, _ in judge.rank(game)] == [((0, 6), (0, 7))]
    assert judge.judge(game, ((0, 6), (0, 7))).quality is MoveQuality.FORCED


def test_an_illegal_move_gets_no_verdict():
    judge = MoveJudge(MaterialEvaluator())
    assert judge.judge(ChessGame(), ((4, 1), (4, 5))) is None


def test_judge_leaves_the_board_untouched():
    """Grading make/unmakes thousands of times; the caller's game must survive it."""
    game = ChessGame()
    before = [[piece and (piece.player, piece.type) for piece in col] for col in game.grid]
    MoveJudge(HeuristicEvaluator()).judge(game, ((4, 1), (4, 3)))
    after = [[piece and (piece.player, piece.type) for piece in col] for col in game.grid]
    assert before == after
    assert game.turn is W
    assert game.count == 0


# ------------------------------------------------------------------ the recap


def _verdict(quality: MoveQuality, loss_points: float) -> MoveVerdict:
    return MoveVerdict(
        move=((0, 0), (0, 1)),
        quality=quality,
        win_prob_before=0.5,
        win_prob_after=0.5 - loss_points / 100.0,
    )


def test_game_review_scores_the_cleaner_side_higher():
    review = GameReview()
    review.add(W, _verdict(MoveQuality.BEST, 0.0))
    review.add(W, _verdict(MoveQuality.EXCELLENT, 1.0))
    review.add(B, _verdict(MoveQuality.BLUNDER, 30.0))
    review.add(B, _verdict(MoveQuality.MISTAKE, 15.0))

    assert review.accuracy(W) > review.accuracy(B)
    assert review.counts(B)[MoveQuality.BLUNDER] == 1
    assert review.counts(W)[MoveQuality.BLUNDER] == 0


def test_game_review_has_no_accuracy_for_a_side_that_never_moved():
    assert GameReview().accuracy(W) is None


# ----------------------------------------------------------------- the legend


def test_the_legend_explains_every_grade_the_judge_can_hand_out():
    """A badge nobody can decode is a bug — the key must cover the whole enum."""
    from kaisparov.play import _legend_entries

    entries = _legend_entries()
    assert [entry.tone for entry in entries] == [q.name.lower() for q in MoveQuality]
    assert [entry.symbol for entry in entries] == [q.symbol for q in MoveQuality]
    assert all(entry.title and entry.detail for entry in entries)


def test_legend_tones_all_have_a_colour():
    from kaisparov.core.game_interface import GameInterface
    from kaisparov.play import _legend_entries

    palette = GameInterface()._quality_colors
    assert all(entry.tone in palette for entry in _legend_entries())
