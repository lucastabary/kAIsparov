"""Ground truth for generated problems: exact, exhaustive, and slow on purpose.

A generator proposes a position; the :class:`Oracle` decides whether it is a problem
and what its answer is. It must be *right*, not fast — a benchmark whose answer key is
wrong measures nothing — so it searches every line and applies the same end-of-game
rules as :class:`~kaisparov.envs.chess_env.ChessEnv`, draw rules included. That last
part matters more than it looks: capturing the last defender may leave an
"insufficient material" draw, and boxing the king in may be a stalemate. A search that
ignored them would call both a win, and a model that avoids them would be marked wrong.

It answers three kinds of question:

**Forced mates.** "Can the side to move force mate within ``depth`` of its own moves,
whatever the defence?" — mate in N. An AND/OR search, cut at the first refutation.

**Safety.** Which moves do not walk into a forced mate, and would the opponent have a
forced mate if the mover passed (a *threat*)?

**Material.** What a move wins or loses in pawns once both sides have played their
best ``plies`` replies, counting material only — an alpha-beta negamax with captures
searched first. Kings are not material: mating is a win (``±WIN``). Draw rules are
*not* applied here: material has no draw score, and within the three or so plies these
questions need they almost never decide anything. A stalemate found mid-search scores
0 — equal — or the search would read it as a win. Search an odd number of plies, so
the line ends on the opponent's reply rather than on a capture nobody answers.

Torch-free, and it leaves the game exactly as it found it.
"""

from __future__ import annotations

from kaisparov.bench.position import other
from kaisparov.core.coords import ALL_SQUARES
from kaisparov.core.draw import DEFAULT_RULES, DrawRules, draw_reason
from kaisparov.core.game import ChessGame
from kaisparov.core.move import Move
from kaisparov.core.pieces import PieceType, Player
from kaisparov.core.utils import get_piece_value

WIN = 1e6  # checkmate, in pawns
_INF = float("inf")


class Oracle:
    def __init__(self, draw_rules: DrawRules | None = DEFAULT_RULES):
        self.draw_rules = draw_rules

    # ---------------------------------------------------------- forced mates
    def winning_moves(self, game: ChessGame, depth: int) -> list[Move]:
        """Every move that forces mate within ``depth`` of the mover's moves."""
        return [move for move in game.legal_moves() if self._move_wins(game, move, depth)]

    def wins_within(self, game: ChessGame, depth: int) -> bool:
        """Whether the side to move can force mate within ``depth`` moves."""
        return self._attacker_wins(game, depth)

    def win_depth(self, game: ChessGame, max_depth: int) -> int | None:
        """The fewest moves in which the side to move forces mate, or ``None``."""
        for depth in range(1, max_depth + 1):
            if self._attacker_wins(game, depth):
                return depth
        return None

    def loses_within(self, game: ChessGame, depth: int) -> bool:
        """Whether *every* move of the side to move lets the opponent mate within ``depth``.

        ``False`` in a position that is already over (a draw, or no move at all).
        """
        if self.is_over(game):
            return False
        return self._defender_loses(game, depth)

    def mates_in_one(self, game: ChessGame) -> list[Move]:
        """The moves that deliver checkmate on the spot."""
        out: list[Move] = []
        for move in game.legal_moves():
            undo = game.make(*move)
            try:
                if game.is_checkmate():
                    out.append(move)
            finally:
                game.unmake(undo)
        return out

    def is_over(self, game: ChessGame) -> bool:
        """The env's end-of-step test: mate, no move at all, or a draw rule."""
        if not game.legal_moves():
            return True
        return draw_reason(game, self.draw_rules) is not None

    # ----------------------------------------------------------------- safety
    def safe_moves(self, game: ChessGame) -> list[Move]:
        """Moves after which the opponent cannot mate straight away.

        Under the old capture-the-king variant this filtered out the moves that hung
        your own king. Standard chess makes those illegal outright, so the question
        that survives is the next one up: which moves walk into a mate in one.
        """
        out: list[Move] = []
        for move in game.legal_moves():
            undo = game.make(*move)
            try:
                if not self._attacker_wins(game, 1):
                    out.append(move)
            finally:
                game.unmake(undo)
        return out

    def threatens(self, game: ChessGame, depth: int = 2) -> bool:
        """Would the opponent force mate within ``depth`` if the side to move passed?

        The *null-move* test: the same board with the other side to move, and no
        en-passant target (only the side that just moved could have granted one).
        """
        board = game.board.copy(stack=False)
        board.turn = other(game.turn) == Player.WHITE
        board.ep_square = None
        return self._attacker_wins(ChessGame(board=board), depth)

    def draws_after(self, game: ChessGame, move: Move) -> str | None:
        """Why the game would end drawn right after ``move``, or ``None``."""
        undo = game.make(*move)
        try:
            if game.is_checkmate():
                return None
            if not game.legal_moves():
                return "stalemate"
            return draw_reason(game, self.draw_rules)
        finally:
            game.unmake(undo)

    # --------------------------------------------------------------- material
    @staticmethod
    def material(game: ChessGame, player: Player) -> float:
        """``player``'s material minus the opponent's, in pawns, kings excluded."""
        score = 0.0
        grid = game.grid
        for col, row in ALL_SQUARES:
            piece = grid[col][row]
            if piece is None or piece.type == PieceType.KING:
                continue
            value = get_piece_value(piece.type)
            score += value if piece.player == player else -value
        return score

    def material_value(self, game: ChessGame, move: Move, plies: int) -> float:
        """The mover's material after ``move`` and ``plies`` replies of best play."""
        return self._material_search(game, move, plies, -_INF, _INF)

    def material_gain(self, game: ChessGame, move: Move, plies: int) -> float:
        """What ``move`` wins (positive) or loses, against the material right now."""
        value = self.material_value(game, move, plies)
        if abs(value) >= WIN:
            return value
        return value - self.material(game, game.turn)

    def material_gains(self, game: ChessGame, plies: int) -> dict[Move, float]:
        """:meth:`material_gain` for every move, exactly (no pruning across moves)."""
        return {move: self.material_gain(game, move, plies) for move in game.legal_moves()}

    # ----------------------------------------------------------------- search
    def _attacker_wins(self, game: ChessGame, depth: int) -> bool:
        if depth <= 0:
            return False
        return any(self._move_wins(game, move, depth) for move in game.legal_moves())

    def _move_wins(self, game: ChessGame, move: Move, depth: int) -> bool:
        undo = game.make(*move)
        try:
            if game.is_checkmate():
                return True
            if depth <= 1:
                return False
            if self.is_over(game):
                return False  # stalemate or a draw rule: not a win
            return self._defender_loses(game, depth - 1)
        finally:
            game.unmake(undo)

    def _defender_loses(self, game: ChessGame, depth: int) -> bool:
        """Every defence (the side to move's) still allows a mate within ``depth``."""
        replies = game.legal_moves()
        if not replies:
            return False  # already mate or stalemate; the caller decides which
        for reply in replies:
            undo = game.make(*reply)
            try:
                if self.is_over(game) or not self._attacker_wins(game, depth):
                    return False
            finally:
                game.unmake(undo)
        return True

    def _ordered_moves(self, game: ChessGame) -> list[Move]:
        """Captures first, biggest victim first — what makes alpha-beta prune."""
        grid = game.grid

        def victim(move: Move) -> float:
            target = grid[move[1][0]][move[1][1]]
            return get_piece_value(target.type) if target is not None else 0.0

        moves = game.legal_moves()
        moves.sort(key=victim, reverse=True)
        return moves

    def _material_search(
        self, game: ChessGame, move: Move, plies: int, alpha: float, beta: float
    ) -> float:
        """Value of ``move`` for its mover, inside the mover's ``(alpha, beta)`` window."""
        mover = game.turn
        undo = game.make(*move)
        try:
            if plies <= 0:
                # is_checkmate() short-circuits on is_check(), so the common quiet
                # leaf never generates a move list.
                return WIN if game.is_checkmate() else self.material(game, mover)
            replies = self._ordered_moves(game)
            if not replies:
                # Mate wins outright; a stalemate is equal, whatever is on the board.
                return WIN if game.is_checkmate() else 0.0
            reply_alpha, reply_beta = -beta, -alpha  # the replier's window
            best = -_INF
            for reply in replies:
                value = self._material_search(game, reply, plies - 1, reply_alpha, reply_beta)
                if value > best:
                    best = value
                    reply_alpha = max(reply_alpha, best)
                    if reply_alpha >= reply_beta:
                        break
            return -best
        finally:
            game.unmake(undo)


__all__ = ["WIN", "Oracle"]
