"""A position a problem starts from, written as FEN so it can be read and typed by hand.

:class:`Position` is the benchmark's immutable, serialisable view of a board. The
engine's :class:`~kaisparov.core.board.ChessGame` is mutable and carries per-piece
``has_moved`` flags; a problem needs neither, it needs a string that survives a JSONL
file and a code review. FEN is that string, with three conventions the variant needs:

- **Castling rights** decide the kings' and rooks' ``has_moved`` flags: a king on its
  home square with a right, and the rook of that right, are unmoved; every other king
  and rook counts as moved.
- **Pawns** are unmoved only on their starting rank, so a double push is offered
  exactly where standard chess offers one.
- **Clocks** (the last two FEN fields) are accepted and ignored: a problem starts a
  fresh game, whose draw bookkeeping begins at the position itself.

``moves`` are plies played from the FEN before the problem starts. They are what
gives a problem a *history* — an en-passant capture set up by the real double push,
or a position one move away from a threefold repetition — that FEN cannot hold.

Moves are written in UCI (``e2e4``), the plain ``from``/``to`` squares, which is
unambiguous without a position and needs no promotion letter in this variant.
"""

from __future__ import annotations

from dataclasses import dataclass

from kaisparov.core.board import ChessGame
from kaisparov.core.coords import ALL_SQUARES, Coord
from kaisparov.core.movegen import Move
from kaisparov.core.notation import square_name
from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType, Player

Grid = list[list["Piece | None"]]

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

_LETTERS: dict[PieceType, str] = {
    PieceType.KING: "k",
    PieceType.QUEEN: "q",
    PieceType.ROOK: "r",
    PieceType.BISHOP: "b",
    PieceType.KNIGHT: "n",
    PieceType.PAWN: "p",
}
_TYPES: dict[str, PieceType] = {letter: piece_type for piece_type, letter in _LETTERS.items()}

_KING_COL = BOARD_SIZE // 2
# Castling letter -> (player, home rank, rook file).
_CASTLING: dict[str, tuple[Player, int, int]] = {
    "K": (Player.WHITE, 0, BOARD_SIZE - 1),
    "Q": (Player.WHITE, 0, 0),
    "k": (Player.BLACK, BOARD_SIZE - 1, BOARD_SIZE - 1),
    "q": (Player.BLACK, BOARD_SIZE - 1, 0),
}


def other(player: Player) -> Player:
    return Player.BLACK if player == Player.WHITE else Player.WHITE


# ----------------------------------------------------------------------- squares


def parse_square(name: str) -> Coord:
    """``"e4"`` -> ``(4, 3)``."""
    if len(name) != 2 or not ("a" <= name[0] <= "h") or not ("1" <= name[1] <= "8"):
        raise ValueError(f"not a square: {name!r}")
    return (ord(name[0]) - ord("a"), int(name[1]) - 1)


def move_to_uci(move: Move) -> str:
    """``((4, 1), (4, 3))`` -> ``"e2e4"``."""
    return square_name(move[0]) + square_name(move[1])


def uci_to_move(text: str) -> Move:
    """``"e2e4"`` -> ``((4, 1), (4, 3))``."""
    if len(text) != 4:
        raise ValueError(f"not a UCI move: {text!r}")
    return (parse_square(text[:2]), parse_square(text[2:]))


# ---------------------------------------------------------------------- position


@dataclass(frozen=True)
class Position:
    """A FEN plus the plies played from it. Hashable, so suites can deduplicate."""

    fen: str
    moves: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _parse_fen(self.fen)  # fail at construction, not when a benchmark reaches it
        for move in self.moves:
            uci_to_move(move)

    # ------------------------------------------------------------- conversions
    @classmethod
    def start(cls) -> Position:
        return cls(START_FEN)

    @classmethod
    def from_game(cls, game: ChessGame) -> Position:
        """The FEN of ``game``'s current position (its history is not kept)."""
        return cls(to_fen(game.grid, game.turn, game.en_passant_target))

    def to_game(self) -> ChessGame:
        """A fresh :class:`ChessGame` at this position, ``moves`` already played."""
        grid, turn, en_passant = _parse_fen(self.fen)
        game = ChessGame(initial_board=grid, turn=turn, en_passant_target=en_passant)
        for text in self.moves:
            source, dest = uci_to_move(text)
            if not game.is_move_valid(source, dest):
                raise ValueError(f"setup move {text} is not legal in {self.fen}")
            game.make(source, dest)
        return game

    @property
    def turn(self) -> Player:
        """The side to move once the setup moves are played."""
        return self.to_game().turn if self.moves else _parse_fen(self.fen)[1]

    # ------------------------------------------------------------- transforms
    def mirrored(self) -> Position:
        """The same position with colours swapped and the board flipped top to bottom.

        A player who understands the position plays the mirrored move in it, which is
        what makes the mirror the natural way to test both colours with one problem.
        Setup moves are mirrored too, so the history (a repetition, an en-passant
        chance) survives the flip.
        """
        game = Position(self.fen).to_game()
        grid: Grid = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for col, row in ALL_SQUARES:
            piece = game.grid[col][row]
            if piece is not None:
                flipped = Piece(other(piece.player), piece.type)
                flipped.has_moved = piece.has_moved  # castling rights survive the flip
                grid[col][BOARD_SIZE - 1 - row] = flipped
        en_passant = game.en_passant_target
        if en_passant is not None:
            en_passant = (en_passant[0], BOARD_SIZE - 1 - en_passant[1])
        moves = tuple(move_to_uci(mirror_move(uci_to_move(move))) for move in self.moves)
        return Position(to_fen(grid, other(game.turn), en_passant), moves)


def mirror_move(move: Move) -> Move:
    """``move`` as it reads on the board :meth:`Position.mirrored` returns."""
    (sc, sr), (dc, dr) = move
    return ((sc, BOARD_SIZE - 1 - sr), (dc, BOARD_SIZE - 1 - dr))


# --------------------------------------------------------------------------- FEN


def to_fen(grid: Grid, turn: Player, en_passant: Coord | None = None) -> str:
    """Write a position as FEN. Castling rights are read off the ``has_moved`` flags."""
    ranks = []
    for row in range(BOARD_SIZE - 1, -1, -1):
        text, empty = "", 0
        for col in range(BOARD_SIZE):
            piece = grid[col][row]
            if piece is None:
                empty += 1
                continue
            if empty:
                text, empty = text + str(empty), 0
            letter = _LETTERS[piece.type]
            text += letter.upper() if piece.player == Player.WHITE else letter
        ranks.append(text + (str(empty) if empty else ""))

    def unmoved(col: int, row: int, player: Player, piece_type: PieceType) -> bool:
        piece = grid[col][row]
        return (
            piece is not None
            and piece.player == player
            and piece.type == piece_type
            and not piece.has_moved
        )

    rights = "".join(
        letter
        for letter, (player, rank, rook_col) in _CASTLING.items()
        if unmoved(_KING_COL, rank, player, PieceType.KING)
        and unmoved(rook_col, rank, player, PieceType.ROOK)
    )
    side = "w" if turn == Player.WHITE else "b"
    target = square_name(en_passant) if en_passant is not None else "-"
    return f"{'/'.join(ranks)} {side} {rights or '-'} {target} 0 1"


def _parse_fen(fen: str) -> tuple[Grid, Player, Coord | None]:
    fields = fen.split()
    if len(fields) < 2:
        raise ValueError(f"FEN needs at least a placement and a side to move: {fen!r}")
    placement, side = fields[0], fields[1]
    rights = fields[2] if len(fields) > 2 else "-"
    target = fields[3] if len(fields) > 3 else "-"

    ranks = placement.split("/")
    if len(ranks) != BOARD_SIZE:
        raise ValueError(f"FEN placement needs {BOARD_SIZE} ranks: {fen!r}")
    grid: Grid = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for i, rank in enumerate(ranks):
        row, col = BOARD_SIZE - 1 - i, 0
        for char in rank:
            if char.isdigit():
                col += int(char)
                continue
            if char.lower() not in _TYPES or col >= BOARD_SIZE:
                raise ValueError(f"bad FEN rank {rank!r} in {fen!r}")
            player = Player.WHITE if char.isupper() else Player.BLACK
            piece = Piece(player, _TYPES[char.lower()])
            # Everything counts as moved except what the conventions above free.
            piece.has_moved = piece.type in (PieceType.KING, PieceType.ROOK)
            if piece.type == PieceType.PAWN:
                start = 1 if player == Player.WHITE else BOARD_SIZE - 2
                piece.has_moved = row != start
            grid[col][row] = piece
            col += 1
        if col != BOARD_SIZE:
            raise ValueError(f"FEN rank {rank!r} does not fill {BOARD_SIZE} files in {fen!r}")

    if side not in ("w", "b"):
        raise ValueError(f"FEN side to move must be 'w' or 'b': {fen!r}")
    for letter in "" if rights == "-" else rights:
        if letter not in _CASTLING:
            raise ValueError(f"bad castling rights {rights!r} in {fen!r}")
        player, home_rank, rook_col = _CASTLING[letter]
        king, rook = grid[_KING_COL][home_rank], grid[rook_col][home_rank]
        if king is not None and king.player == player and king.type == PieceType.KING:
            king.has_moved = False
        if rook is not None and rook.player == player and rook.type == PieceType.ROOK:
            rook.has_moved = False

    en_passant = None if target == "-" else parse_square(target)
    return grid, Player.WHITE if side == "w" else Player.BLACK, en_passant


__all__ = [
    "START_FEN",
    "Position",
    "mirror_move",
    "move_to_uci",
    "other",
    "parse_square",
    "to_fen",
    "uci_to_move",
]
