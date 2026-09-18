"""Algebraic notation for the move history — the text a scoresheet would carry.

Standard short algebraic notation (SAN) with French piece letters, since the move
list is shown in the French UI: ``R`` roi, ``D`` dame, ``T`` tour, ``F`` fou, ``C``
cavalier, and nothing for a pawn. ``Cbd2``, ``exd5``, ``O-O``, ``Dh5+``.

Two twists come from the variant (see :mod:`kaisparov.core`):

- disambiguation looks at *pseudo-legal* moves, the ones the engine actually allows;
- there is no mate, the game ends when a king is taken, so ``#`` marks the capture
  of the king rather than a mating move. ``+`` still means "the king is attacked".

Pure Python, no torch and no pygame, so the UI only has to lay out strings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from kaisparov.core.coords import ALL_SQUARES, Coord
from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import PieceType, Player

PIECE_LETTERS: dict[PieceType, str] = {
    PieceType.KING: "R",
    PieceType.QUEEN: "D",
    PieceType.ROOK: "T",
    PieceType.BISHOP: "F",
    PieceType.KNIGHT: "C",
    PieceType.PAWN: "",
}


def square_name(coord: Coord) -> str:
    """``(4, 3)`` -> ``"e4"``."""
    return f"{chr(ord('a') + coord[0])}{coord[1] + 1}"


def move_to_san(game: ChessGame, source: Coord, dest: Coord) -> str:
    """Notation for ``source -> dest``, read off the position *before* it is played.

    The move is assumed legal, as for :meth:`ChessGame.make`. The check suffix needs
    the position after the move, so the move is made and unmade again: ``game`` comes
    back exactly as it was.
    """
    piece = game.grid[source[0]][source[1]]
    assert piece is not None, f"no piece to move on {source}"

    undo = game.make(source, dest)
    captured = undo.captured  # en passant included: the pawn taken is not on dest
    if captured is not None and captured.type == PieceType.KING:
        suffix = "#"
    elif game.is_in_check(game.turn):
        suffix = "+"
    else:
        suffix = ""
    game.unmake(undo)

    if piece.type == PieceType.KING and abs(dest[0] - source[0]) == 2:
        return ("O-O" if dest[0] > source[0] else "O-O-O") + suffix

    target = square_name(dest)
    if piece.type == PieceType.PAWN:
        # A pawn capture names the file it left: exd5.
        body = f"{square_name(source)[0]}x{target}" if captured is not None else target
        return body + suffix

    letter = PIECE_LETTERS[piece.type]
    take = "x" if captured is not None else ""
    return f"{letter}{_disambiguation(game, source, dest)}{take}{target}{suffix}"


def _disambiguation(game: ChessGame, source: Coord, dest: Coord) -> str:
    """The file, the rank, or both — whatever tells ``source`` apart from its twins.

    A twin is another piece of the same side and type that could also land on
    ``dest``. The file is preferred, then the rank, as in standard SAN.
    """
    piece = game.grid[source[0]][source[1]]
    assert piece is not None
    twins = [
        square
        for square in ALL_SQUARES
        if square != source
        and (other := game.grid[square[0]][square[1]]) is not None
        and other.player == piece.player
        and other.type == piece.type
        and dest in game.possible_moves(square)
    ]
    if not twins:
        return ""
    name = square_name(source)
    if all(square[0] != source[0] for square in twins):
        return name[0]
    if all(square[1] != source[1] for square in twins):
        return name[1]
    return name


class MoveRow(NamedTuple):
    """One line of the scoresheet: the move number, White's move, Black's reply."""

    number: int
    white: str
    black: str  # "" while Black has not replied yet


def numbered_moves(moves: Sequence[str], first: Player = Player.WHITE) -> list[MoveRow]:
    """Pair a flat list of plies into numbered rows, White's move then Black's.

    ``first`` is the side that made the first ply. When it is Black (a position set
    up with Black to move) the opening row leaves White's column as ``"..."``, the
    usual way to write it.
    """
    plies = list(moves)
    if first == Player.BLACK:
        plies.insert(0, "...")
    return [
        MoveRow(number=i // 2 + 1, white=plies[i], black=plies[i + 1] if i + 1 < len(plies) else "")
        for i in range(0, len(plies), 2)
    ]
