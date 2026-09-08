"""Grade a played move — the algorithm behind chess.com's move classification.

The whole thing rests on one quantity: how much **winning probability** a move gave
away compared with the best move available.

1. Score every legal move in the position *before* the move, from the mover's point
   of view (:meth:`MoveJudge.rank`). That is a negamax of ``lookahead`` plies whose
   leaves come from an :class:`~kaisparov.analysis.evaluators.Evaluator`.
2. Convert both the best score and the played move's score to a winning probability
   with a logistic curve (:func:`win_probability`). This is the step that matters:
   giving away one pawn from a dead-equal position is a disaster, giving away one
   pawn while eight pieces up is nothing — centipawns say those are identical,
   winning chances do not.
3. The drop in winning chances, in percentage points, picks the label
   (:class:`ReviewThresholds`). A handful of special cases override it: a move with
   only one legal alternative is ``Force``, the only move that holds the position is
   ``Tres bon``, a sound material sacrifice is ``Brillant``, and throwing away a won
   position is ``Occasion manquee`` rather than a plain ``Erreur``.

Everything is calibrated in win-probability points, so the same thresholds apply
whether the evaluator counts material or asks a trained critic.

Torch-free on purpose: the neural evaluator is injected, never imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from kaisparov.analysis.evaluators import Evaluator
from kaisparov.core.board import ChessGame
from kaisparov.core.movegen import all_moves
from kaisparov.core.pieces import BOARD_SIZE, PieceType, Player
from kaisparov.core.rules import attacked_squares
from kaisparov.core.utils import get_piece_value
from kaisparov.insights import Move, MoveQuality, MoveVerdict

WIN = 1e6  # sentinel score for "this move captures the king", i.e. wins outright
EPS = 1e-6  # scores closer than this count as tied


def win_probability(value: float, slope: float) -> float:
    """Map an evaluation onto the mover's winning chances in ``0..1``.

    A logistic curve, which is what makes an evaluation *loss* comparable across
    positions: it is steep around equality (where a pawn decides the game) and flat
    at the extremes (where a pawn changes nothing).
    """
    if value >= WIN:
        return 1.0
    if value <= -WIN:
        return 0.0
    return 1.0 / (1.0 + math.exp(-slope * value))


def accuracy_from_loss(loss: float) -> float:
    """Per-move accuracy in ``0..100`` from the winning chances given away.

    The curve published by Lichess for its accuracy metric (chess.com's is not
    public but behaves the same way): a perfect move scores 100, and accuracy falls
    off exponentially with the win-probability points lost.
    """
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * loss) - 3.1669))


@dataclass(frozen=True)
class ReviewThresholds:
    """Where one label stops and the next begins, in winning-chance points lost."""

    excellent: float = 2.0  # <= this and the move is Excellent
    good: float = 5.0
    inaccuracy: float = 10.0
    mistake: float = 20.0  # beyond this it is a Gaffe

    great_gap: float = 10.0  # the only move must beat the runner-up by this much
    sacrifice: float = 2.0  # pawns handed over before a move counts as a sacrifice
    winning: float = 0.90  # above this win probability, a big drop is a missed win
    crushing: float = 0.97  # already won: no brilliancy for a sac you did not need


class MoveJudge:
    """Turns "the move that was played" into a :class:`~kaisparov.insights.MoveVerdict`.

    ``lookahead`` is the number of plies searched *after* each candidate move;
    ``None`` takes the evaluator's own default. The cost of one verdict is roughly
    ``branching ** (lookahead + 1)`` evaluator calls, so keep it at 0 for a neural
    evaluator (one forward pass per legal move) and 1 for material.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        lookahead: int | None = None,
        thresholds: ReviewThresholds | None = None,
    ):
        self.evaluator = evaluator
        self.lookahead = evaluator.default_lookahead if lookahead is None else max(0, lookahead)
        self.thresholds = thresholds or ReviewThresholds()
        self.name = f"judge:{evaluator.name}"

    # ------------------------------------------------------------------ search

    def _negamax(self, game: ChessGame, plies: int) -> float:
        """Value of ``game`` for the side to move, ``plies`` deep."""
        if plies <= 0:
            return self.evaluator.evaluate(game)
        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        if not moves:
            return self.evaluator.evaluate(game)  # stuck: judge the position as it stands
        return max(self._value_after(game, move, plies - 1) for move in moves)

    def _value_after(self, game: ChessGame, move: Move, plies: int) -> float:
        """Value of playing ``move``, for the player who plays it."""
        undo = game.make(*move)
        try:
            if undo.captured is not None and undo.captured.type == PieceType.KING:
                return WIN  # the game ends here, nothing left to search
            # Hanging your own king loses on the spot in capture-the-king, and a
            # static evaluator (lookahead 0) would never see it. The test is a single
            # attack lookup, so run it at every depth and skip the subtree.
            if game.is_in_check(_mover(game)):
                return -WIN
            return -self._negamax(game, plies)
        finally:
            game.unmake(undo)

    def rank(self, game: ChessGame) -> list[tuple[Move, float]]:
        """Every legal move scored for the side to move, best first."""
        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        scored = [(move, self._value_after(game, move, self.lookahead)) for move in moves]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    # ------------------------------------------------------------- sacrifices

    def _material(self, game: ChessGame, player: Player) -> float:
        score = 0.0
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                piece = game.grid[col][row]
                if piece is None or piece.type == PieceType.KING:
                    continue
                value = get_piece_value(piece.type)
                score += value if piece.player == player else -value
        return score

    def _material_greed(self, game: ChessGame, player: Player, plies: int) -> float:
        """``player``'s material after ``plies`` of both sides grabbing greedily.

        Deliberately *naive*: it counts material and nothing else — no king-capture
        win, no idea that taking the piece loses on the spot. That blindness is the
        point (see :meth:`_sacrifice`). Written as an explicit minimax on ``player``
        rather than a negamax because ``make`` does not flip the turn when a king is
        captured, so "the side to move" is not a reliable sign to alternate on.
        """
        moves = all_moves(game.grid, game.turn, game.en_passant_target) if plies > 0 else []
        if not moves:
            return self._material(game, player)

        maximising = game.turn == player
        best = -WIN if maximising else WIN
        for move in moves:
            undo = game.make(*move)
            try:
                if undo.captured is not None and undo.captured.type == PieceType.KING:
                    value = self._material(game, player)  # the game ends here
                else:
                    value = self._material_greed(game, player, plies - 1)
            finally:
                game.unmake(undo)
            best = max(best, value) if maximising else min(best, value)
        return best

    def _sacrifice(self, game: ChessGame, move: Move) -> float:
        """Material (in pawns) the move offers the opponent, counting material only.

        Two plies of naive greed: the opponent grabs whatever the move left available
        and we take back what we can, with nobody noticing that the grab might lose
        the game. That is exactly the question a brilliancy asks — *does this look
        like free material?* — and it is why the search here must stay blind while
        :meth:`rank` (which decides whether the move is actually good) does not.

        Gated on the piece landing somewhere the opponent attacks, which is what a
        sacrifice looks like and what keeps the two-ply search off the hot path for
        the quiet moves that make up most of a game.
        """
        mover = game.turn
        before = self._material(game, mover)
        undo = game.make(*move)
        try:
            if undo.captured is not None and undo.captured.type == PieceType.KING:
                return 0.0
            if move[1] not in attacked_squares(game.grid, game.turn):
                return 0.0
            after = self._material_greed(game, mover, 2)
        finally:
            game.unmake(undo)
        return max(0.0, before - after)

    # ------------------------------------------------------------------ verdict

    def judge(self, game: ChessGame, move: Move) -> MoveVerdict | None:
        """Grade ``move`` in ``game`` — the position *before* the move is played."""
        ranked = self.rank(game)
        if not ranked:
            return None
        scores = dict(ranked)
        if move not in scores:
            return None  # not a move this position offers; nothing to say about it

        top_move, top = ranked[0]
        played = scores[move]
        slope = self.evaluator.slope
        wp_before = win_probability(top, slope)
        wp_after = win_probability(played, slope)
        loss = max(0.0, 100.0 * (wp_before - wp_after))

        # The runner-up is the best score that is *not* tied with the top one, so a
        # position with several equally good moves never produces an "only move".
        # Its winning chances also stand for "how well would this have gone anyway?",
        # which is what decides whether a sacrifice was needed or merely decorative.
        runner_up = next((value for _, value in ranked if value < top - EPS), None)
        wp_alt = wp_before if runner_up is None else win_probability(runner_up, slope)
        alone = runner_up is not None and 100.0 * (wp_before - wp_alt) >= self.thresholds.great_gap
        sacrificed = self._sacrifice(game, move) if loss <= self.thresholds.excellent else 0.0

        quality = self._classify(
            legal_count=len(ranked),
            is_top=played >= top - EPS,
            alone=alone,
            loss=loss,
            missed_win=top >= WIN and played < WIN,
            wp_alt=wp_alt,
            wp_after=wp_after,
            wp_before=wp_before,
            sacrificed=sacrificed,
        )
        return MoveVerdict(
            move=move,
            quality=quality,
            win_prob_before=wp_before,
            win_prob_after=wp_after,
            best_move=top_move,
            sacrificed=sacrificed,
            source=self.name,
        )

    def _classify(
        self,
        *,
        legal_count: int,
        is_top: bool,
        alone: bool,
        loss: float,
        missed_win: bool,
        wp_alt: float,
        wp_after: float,
        wp_before: float,
        sacrificed: float,
    ) -> MoveQuality:
        t = self.thresholds
        if legal_count == 1:
            return MoveQuality.FORCED
        # Walking past an immediate king capture is a missed win however comfortable
        # the position stays — a distinction winning chances alone cannot draw, since
        # "up two rooks" and "mate in one" both round to ~100%.
        if missed_win:
            return MoveQuality.MISS

        # A brilliancy is a sound sacrifice: near-best, hands over real material, and
        # was actually needed — if the second-best move already won, giving up a piece
        # is a flourish, not a brilliancy.
        if (
            sacrificed >= t.sacrifice
            and loss <= t.excellent
            and wp_alt < t.crushing
            and wp_after >= 0.5
        ):
            return MoveQuality.BRILLIANT
        if is_top:
            return MoveQuality.GREAT if alone else MoveQuality.BEST
        if loss <= t.excellent:
            return MoveQuality.EXCELLENT
        if loss <= t.good:
            return MoveQuality.GOOD
        if loss <= t.inaccuracy:
            return MoveQuality.INACCURACY
        # Throwing away a won position gets its own label, as on chess.com.
        if wp_before >= t.winning:
            return MoveQuality.MISS
        return MoveQuality.MISTAKE if loss <= t.mistake else MoveQuality.BLUNDER


def _mover(game: ChessGame) -> Player:
    """The side that just moved (``make`` has already flipped ``game.turn``)."""
    return Player.WHITE if game.turn == Player.BLACK else Player.BLACK


@dataclass
class GameReview:
    """Accumulates verdicts per side so a finished game can be summarised."""

    verdicts: dict[Player, list[MoveVerdict]] = field(
        default_factory=lambda: {Player.WHITE: [], Player.BLACK: []}
    )

    def add(self, player: Player, verdict: MoveVerdict) -> None:
        self.verdicts[player].append(verdict)

    def accuracy(self, player: Player) -> float | None:
        """Mean per-move accuracy in ``0..100``, or ``None`` if the side never moved."""
        moves = self.verdicts[player]
        if not moves:
            return None
        return sum(accuracy_from_loss(v.loss) for v in moves) / len(moves)

    def counts(self, player: Player) -> dict[MoveQuality, int]:
        """How many moves of each quality that side played, best label first."""
        tally = {quality: 0 for quality in MoveQuality}
        for verdict in self.verdicts[player]:
            tally[verdict.quality] += 1
        return tally


__all__ = [
    "EPS",
    "WIN",
    "GameReview",
    "MoveJudge",
    "ReviewThresholds",
    "accuracy_from_loss",
    "win_probability",
]
