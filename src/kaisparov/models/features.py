"""Node feature sets: what each of the 64 square-nodes carries into a network.

The graph's *structure* — which edges exist, how an edge's score becomes a move —
belongs to a backend and lives in its processor. What sits on each node is a separate
experimental choice: the same features can feed several architectures (so comparing
them is fair), and one architecture can be trained on several feature sets (so the
features themselves can be compared). Hence one registry, shared by every backend,
chosen per run by the ``features:`` key of the training config.

**A name, once runs use it, never changes meaning.** A run records the name it was
trained with; to try different inputs, add a new set under a new name rather than
editing an existing one — or every checkpoint trained on the old meaning stops
loading, or worse, loads and reads garbage. ``tests/test_features.py`` pins the
output of each set on fixed positions so a change cannot slip through by accident.
Refactors that leave the output bit-identical (a faster ``encode_batch``) are fine.

Every set is laid out from the point of view of the side to move: "own" pieces are
the mover's. Each has two implementations that must agree exactly — :meth:`encode`,
a plain per-square reference, and :meth:`encode_batch`, the vectorised version the
self-play rollout calls every ply.

Torch-free: arrays are numpy ``float32``; the processor wraps them as tensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from kaisparov.core import bitboard_batch as bbb
from kaisparov.core.coords import ALL_SQUARES
from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import BOARD_SIZE, PieceType
from kaisparov.core.rules import attacked_squares
from kaisparov.core.utils import coord_to_index

NUM_NODES = BOARD_SIZE * BOARD_SIZE

# One-hot slot of each piece type, own pieces first (0-5) then the opponent's (6-11).
PIECE_ORDER = (
    PieceType.KING,
    PieceType.QUEEN,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.KNIGHT,
    PieceType.PAWN,
)
_PIECE_IDX = {piece_type: i for i, piece_type in enumerate(PIECE_ORDER)}
# chess.Board bitboard attributes, in PIECE_ORDER.
_CHESS_BITBOARDS = ("kings", "queens", "bishops", "rooks", "knights", "pawns")
_SQUARES = np.arange(NUM_NODES, dtype=np.uint64)
_ONE = np.uint64(1)


def _unpack(masks: np.ndarray) -> np.ndarray:
    """``(n,)`` uint64 bitboards -> ``(n, 64)`` float32 planes, square ``row*8+col``."""
    return ((masks[:, None] >> _SQUARES) & _ONE).astype(np.float32)


class FeatureSet(ABC):
    """A named, fixed-width encoding of a position as ``(64, dim)`` node features."""

    name: ClassVar[str]
    dim: ClassVar[int]
    description: ClassVar[str]

    @abstractmethod
    def encode(self, game: ChessGame) -> np.ndarray:
        """``(64, dim)`` features of one position — the readable reference."""

    @abstractmethod
    def encode_batch(self, games: list[ChessGame]) -> np.ndarray:
        """``(n, 64, dim)`` features of several positions, identical to :meth:`encode`."""


class Pieces(FeatureSet):
    """Piece type and owner, nothing else: 6 own one-hots, then 6 opponent ones.

    What the network had before the control flags existed (up to 2026-09-04): it has
    to work out on its own which lines are blocked and which squares are attacked.
    """

    name = "pieces"
    dim = 12
    description = "6 own + 6 opponent piece-type one-hots"

    # Both methods size their output with ``Pieces.dim`` rather than ``self.dim``: a
    # subclass calls them for its first twelve columns.

    def encode(self, game: ChessGame) -> np.ndarray:
        x = np.zeros((NUM_NODES, Pieces.dim), dtype=np.float32)
        grid, mover = game.grid, game.turn
        for col, row in ALL_SQUARES:
            piece = grid[col][row]
            if piece is None:
                continue
            offset = 0 if piece.player == mover else 6
            x[coord_to_index((col, row)), offset + _PIECE_IDX[piece.type]] = 1.0
        return x

    def encode_batch(self, games: list[ChessGame]) -> np.ndarray:
        x = np.zeros((len(games), NUM_NODES, Pieces.dim), dtype=np.float32)
        if not games:
            return x
        # python-chess holds one bitboard per piece type plus a per-colour occupancy,
        # already in this project's ``row * 8 + col`` order, so the planes unpack
        # straight out of the masks.
        own = np.zeros((6, len(games)), dtype=np.uint64)
        enemy = np.zeros((6, len(games)), dtype=np.uint64)
        for i, game in enumerate(games):
            board = game.board
            mine = board.occupied_co[board.turn]
            theirs = board.occupied_co[not board.turn]
            for k, attribute in enumerate(_CHESS_BITBOARDS):
                bb = getattr(board, attribute)
                own[k, i] = bb & mine
                enemy[k, i] = bb & theirs
        for k in range(6):
            x[:, :, k] = _unpack(own[k])
            x[:, :, 6 + k] = _unpack(enemy[k])
        return x


class PiecesControl(Pieces):
    """:class:`Pieces` plus two blocking-aware control flags (added 2026-09-04).

    Feature 12: the square is attacked by the opponent — set on the mover's king
    exactly when it is in check. Feature 13: the square is controlled by the mover.
    Sliders stop at the first blocker, which is what a static geometric graph cannot
    see by itself; these flags hand that reasoning over from the engine.
    """

    name = "pieces_control"
    dim = 14
    description = "pieces + attacked-by-opponent + controlled-by-mover"

    def encode(self, game: ChessGame) -> np.ndarray:
        x = np.zeros((NUM_NODES, self.dim), dtype=np.float32)
        x[:, : Pieces.dim] = super().encode(game)
        mover = game.turn
        for square in attacked_squares(game, mover.opponent):
            x[coord_to_index(square), 12] = 1.0
        for square in attacked_squares(game, mover):
            x[coord_to_index(square), 13] = 1.0
        return x

    def encode_batch(self, games: list[ChessGame]) -> np.ndarray:
        x = np.zeros((len(games), NUM_NODES, self.dim), dtype=np.float32)
        if not games:
            return x
        x[:, :, : Pieces.dim] = super().encode_batch(games)
        boards = [game.board for game in games]
        white_to_move = np.fromiter((b.turn for b in boards), dtype=bool, count=len(boards))
        atk_white = bbb.attacked_by_packed(bbb.pack_boards(boards, bbb.WHITE), bbb.WHITE)
        atk_black = bbb.attacked_by_packed(bbb.pack_boards(boards, bbb.BLACK), bbb.BLACK)
        x[:, :, 12] = _unpack(np.where(white_to_move, atk_black, atk_white))
        x[:, :, 13] = _unpack(np.where(white_to_move, atk_white, atk_black))
        return x


FEATURE_SETS: dict[str, FeatureSet] = {cls.name: cls() for cls in (Pieces, PiecesControl)}

# What a new run trains on unless its config names a set. Runs recorded before the
# key existed are *not* read as this: their set comes off their weights (see
# models.factory.resolve_architecture), so changing the default never relabels them.
DEFAULT_FEATURES = "pieces"


def get_feature_set(name: str) -> FeatureSet:
    try:
        return FEATURE_SETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown node feature set {name!r}. Known: {sorted(FEATURE_SETS)}."
        ) from None


def feature_set_for_dim(dim: int) -> str | None:
    """The set of this width, if exactly one has it.

    For checkpoints whose run predates the ``features`` key: the width of the input
    layer is then the only record of what the model was fed, and so far every set has
    a distinct width.
    """
    names = [name for name, fs in FEATURE_SETS.items() if fs.dim == dim]
    return names[0] if len(names) == 1 else None


__all__ = [
    "DEFAULT_FEATURES",
    "FEATURE_SETS",
    "FeatureSet",
    "Pieces",
    "PiecesControl",
    "PIECE_ORDER",
    "feature_set_for_dim",
    "get_feature_set",
]
