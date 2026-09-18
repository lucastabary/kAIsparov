"""Move review: win-probability maths, the labels, and the per-game recap.

Torch-free — the judge only ever sees an evaluator, and the handcrafted ones are
pure Python.
"""

from __future__ import annotations

import pytest

from kaisparov.analysis import (
    GameReview,
    HeuristicEvaluator,
    MaterialEvaluator,
    MoveJudge,
    accuracy_from_loss,
    win_probability,
)
from kaisparov.analysis.judge import WIN
from kaisparov.core.game import ChessGame
from kaisparov.core.move import Move
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.insights import MoveQuality, MoveVerdict

W, B = Player.WHITE, Player.BLACK


def game_from(fen: str) -> ChessGame:
    """A position straight from a FEN — clearer than listing squares, for real chess."""
    import chess

    return ChessGame(board=chess.Board(fen))


def position(pieces: dict, turn: Player = W) -> ChessGame:
    """A board holding only ``{(col, row): (player, piece_type)}``."""
    game = ChessGame(initial_board=[[None] * 8 for _ in range(8)], turn=turn)
    for (col, row), (player, piece_type) in pieces.items():
        game.place((col, row), Piece(player, piece_type))
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


def test_missing_a_mate_is_a_miss():
    """Ra8 is mate on the back rank; shuffling the h-pawn instead throws the win away."""
    game = game_from("7k/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    judge = MoveJudge(MaterialEvaluator())
    mate = judge.judge(game, ((0, 0), (0, 7)))  # Ra8#
    # It is the only move that wins, so it earns GREAT rather than plain BEST.
    assert mate.quality in {MoveQuality.BEST, MoveQuality.GREAT}
    assert mate.best_move == Move((0, 0), (0, 7))
    assert mate.win_prob_after == 1.0

    missed = judge.judge(game, ((7, 1), (7, 2)))  # h2h3
    assert missed.quality is MoveQuality.MISS
    assert missed.win_prob_before == 1.0


def test_a_sound_queen_sacrifice_is_brilliant():
    """Philidor's legacy: Qg8+ hands over the queen, and Rxg8 is forced into Nf7#."""
    game = game_from("5r1k/6pp/7N/8/8/1Q6/8/6K1 w - - 0 1")
    verdict = MoveJudge(HeuristicEvaluator(), lookahead=2).judge(game, ((1, 2), (6, 7)))
    assert verdict.quality is MoveQuality.BRILLIANT
    assert verdict.sacrificed >= 2.0


def test_a_quiet_best_move_is_not_a_sacrifice():
    verdict = MoveJudge(HeuristicEvaluator()).judge(ChessGame(), ((4, 1), (4, 3)))
    assert verdict.sacrificed == 0.0
    assert verdict.quality is MoveQuality.BEST


def test_the_only_legal_move_is_forced():
    """The rook on g7 takes away g8 and h7, leaving the black king a single square."""
    game = game_from("7k/6R1/8/8/8/8/8/6K1 b - - 0 1")
    judge = MoveJudge(MaterialEvaluator())
    ranked = [move for move, _ in judge.rank(game)]
    assert len(ranked) == 1
    assert judge.judge(game, ranked[0]).quality is MoveQuality.FORCED


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


def test_stalemating_the_opponent_scores_as_a_draw_not_a_win():
    """Qc7 boxes in the lone king on a8 with no legal reply: a draw, not a win.

    A search that only asks "does the opponent have a move?" would read the empty
    reply list as a forced loss for Black and hand White the game.
    """
    game = game_from("k7/8/8/8/8/8/8/2Q4K w - - 0 1")
    judge = MoveJudge(MaterialEvaluator(), lookahead=1)
    scores = dict(judge.rank(game))

    assert scores[Move((2, 0), (2, 6))] == 0.0  # Qc1-c7, stalemate
    assert max(scores.values()) > 0.0  # keeping the king boxed without stalemating is better
