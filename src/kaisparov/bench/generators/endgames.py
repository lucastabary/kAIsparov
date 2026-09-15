"""Play-outs: can the model turn an advantage into a result, and hold a bad position?

One-move themes say whether a model *sees*; these say whether it can *plan* over tens
of moves against a real defence. The default opponent is ``material+safe`` — it grabs
material and never hangs its own king, the least a defender must do.

- ``conversion`` — heavy pieces against a bare king: capture it before the ply cap;
- ``material_edge`` — a full-ish middlegame a piece up: win it;
- ``hold`` — the same, a piece down: do not lose within the cap.
"""

from __future__ import annotations

import random

from kaisparov.bench.generators.base import BoardBuilder, SamplingGenerator, is_quiet
from kaisparov.bench.generators.draws import army
from kaisparov.bench.oracle import Oracle
from kaisparov.bench.problem import Problem
from kaisparov.bench.tasks import PlayOut
from kaisparov.core.pieces import BOARD_SIZE, PieceType, Player

_CONVERSION_DIFFICULTY = {"QQ": 1, "QR": 1, "Q": 2, "RR": 2, "R": 3}


class ConversionGenerator(SamplingGenerator):
    name = "conversion"
    theme = "conversion"
    description = "Heavy pieces against a bare king: capture it within the ply cap."

    def __init__(
        self,
        materials: tuple[str, ...] = ("Q", "RR", "R"),
        opponent: str = "material+safe",
        max_plies: int = 80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.materials = tuple(materials)
        for material in self.materials:
            army(material)
        self.opponent, self.max_plies = opponent, max_plies
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        material = rng.choice(self.materials)
        board = BoardBuilder.random_board(rng, army(material), [])
        if board is None:
            return None
        position = board.position(Player.WHITE)
        if not is_quiet(position.to_game(), self.oracle):
            return None
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=PlayOut(opponent=self.opponent, max_plies=self.max_plies, goal="win"),
            difficulty=_CONVERSION_DIFFICULTY.get(material, 2),
            tags=(material,),
            meta={"material": material},
        )


_ARMY_POOL = (
    PieceType.PAWN,
    PieceType.PAWN,
    PieceType.PAWN,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.QUEEN,
)
_EXTRA = {3: PieceType.KNIGHT, 5: PieceType.ROOK, 9: PieceType.QUEEN}


class _ImbalanceGenerator(SamplingGenerator):
    """Two mirrored-looking armies in their own halves, one of them a piece richer.

    ``edge`` is the hero's (White's) material lead in pawns: positive to convert,
    negative to hold. Pieces start in their own half, pawns off the back ranks, so the
    position reads like a game rather than a scatter of pieces.
    """

    goal = "win"

    def __init__(
        self,
        edges: tuple[int, ...] = (3, 5),
        min_pieces: int = 4,
        max_pieces: int = 7,
        opponent: str = "material+safe",
        max_plies: int = 80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edges = tuple(abs(int(edge)) for edge in edges)
        for edge in self.edges:
            if edge not in _EXTRA:
                raise ValueError(f"{self.name}: edge must be one of {sorted(_EXTRA)}")
        self.min_pieces, self.max_pieces = min_pieces, max_pieces
        self.opponent, self.max_plies = opponent, max_plies
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        edge = rng.choice(self.edges)
        common = [
            rng.choice(_ARMY_POOL) for _ in range(rng.randint(self.min_pieces, self.max_pieces))
        ]
        richer, poorer = [*common, _EXTRA[edge]], common
        white, black = (richer, poorer) if self.goal == "win" else (poorer, richer)

        board = BoardBuilder(rng)
        half = BOARD_SIZE // 2
        if (
            board.place(Player.WHITE, PieceType.KING, rows=range(0, 2)) is None
            or board.place(Player.BLACK, PieceType.KING, rows=range(BOARD_SIZE - 2, BOARD_SIZE))
            is None
            or not board.place_all(Player.WHITE, white, rows=range(0, half))
            or not board.place_all(Player.BLACK, black, rows=range(half, BOARD_SIZE))
        ):
            return None
        position = board.position(Player.WHITE)
        game = position.to_game()
        if not is_quiet(game, self.oracle):
            return None
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=PlayOut(opponent=self.opponent, max_plies=self.max_plies, goal=self.goal),
            difficulty=1 if edge >= 5 else 2,
            meta={"edge": edge if self.goal == "win" else -edge},
        )


class MaterialEdgeGenerator(_ImbalanceGenerator):
    name = "material_edge"
    theme = "material_edge"
    description = "A middlegame a piece up: win it within the ply cap."
    goal = "win"


class HoldGenerator(_ImbalanceGenerator):
    name = "hold"
    theme = "hold"
    description = "A middlegame a piece down: do not lose within the ply cap."
    goal = "not_lose"

    def __init__(self, max_plies: int = 40, **kwargs):
        super().__init__(max_plies=max_plies, **kwargs)


__all__ = ["ConversionGenerator", "HoldGenerator", "MaterialEdgeGenerator"]
