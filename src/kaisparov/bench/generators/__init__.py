"""Problem generators. Importing this package registers every built-in family.

A new family is a module here with a :class:`ProblemGenerator` subclass (usually a
:class:`SamplingGenerator`) and one import line below.
"""

from kaisparov.bench.generators.base import (
    EDGE_SQUARES,
    HEAVY_TYPES,
    PIECE_TYPES,
    BoardBuilder,
    GenerationError,
    ProblemGenerator,
    SamplingGenerator,
    is_quiet,
    near,
)
from kaisparov.bench.generators.draws import RepetitionTrapGenerator, StalemateTrapGenerator
from kaisparov.bench.generators.endgames import (
    ConversionGenerator,
    HoldGenerator,
    MaterialEdgeGenerator,
)
from kaisparov.bench.generators.fixed import FixedGenerator
from kaisparov.bench.generators.mate_in_one import MateInOneGenerator
from kaisparov.bench.generators.probes import PolicyRankGenerator, ValueSignGenerator
from kaisparov.bench.generators.robustness import (
    DistractorInvarianceGenerator,
    MirrorConsistencyGenerator,
)
from kaisparov.bench.generators.safety import (
    AvoidMateGenerator,
    AvoidPieceHangGenerator,
    EscapeCheckGenerator,
)
from kaisparov.bench.generators.special import SpecialRulesGenerator
from kaisparov.bench.generators.tactics import (
    ForkGenerator,
    FreeCaptureGenerator,
    ParryThreatGenerator,
    WinInNGenerator,
)

__all__ = [
    "EDGE_SQUARES",
    "HEAVY_TYPES",
    "PIECE_TYPES",
    "AvoidMateGenerator",
    "AvoidPieceHangGenerator",
    "BoardBuilder",
    "ConversionGenerator",
    "DistractorInvarianceGenerator",
    "EscapeCheckGenerator",
    "FixedGenerator",
    "ForkGenerator",
    "FreeCaptureGenerator",
    "GenerationError",
    "HoldGenerator",
    "MateInOneGenerator",
    "MaterialEdgeGenerator",
    "MirrorConsistencyGenerator",
    "ParryThreatGenerator",
    "PolicyRankGenerator",
    "ProblemGenerator",
    "RepetitionTrapGenerator",
    "SamplingGenerator",
    "SpecialRulesGenerator",
    "StalemateTrapGenerator",
    "ValueSignGenerator",
    "WinInNGenerator",
    "is_quiet",
    "near",
]
