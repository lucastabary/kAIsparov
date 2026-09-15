"""Problem generators. Importing this package registers every built-in family.

A new family is a module here with a :class:`ProblemGenerator` subclass (usually a
:class:`SamplingGenerator`) and one import line below.
"""

from kaisparov.bench.generators.base import (
    PIECE_TYPES,
    BoardBuilder,
    GenerationError,
    ProblemGenerator,
    SamplingGenerator,
)
from kaisparov.bench.generators.fixed import FixedGenerator
from kaisparov.bench.generators.king_capture import KingCaptureGenerator

__all__ = [
    "PIECE_TYPES",
    "BoardBuilder",
    "FixedGenerator",
    "GenerationError",
    "KingCaptureGenerator",
    "ProblemGenerator",
    "SamplingGenerator",
]
