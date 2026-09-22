"""Benchmarking: measure trained models on targeted chess problems.

Where :mod:`kaisparov.eval.arena` asks "who wins a whole game?", this package asks
"*which skills* does a model have?" — does it take a free king, see a two-move win,
keep its queen out of a pawn's reach, convert a won endgame. Each is a theme of
problems, and a model's profile across themes says far more about what training
taught it than one win rate does.

The pieces, each an extension point:

- :class:`Position` — a FEN (plus setup moves), the serialisable board;
- :class:`Task` — what to do there and how it is scored (:class:`FindMove`,
  :class:`AvoidMoves`, :class:`PlayOut`, ... subclass to add one);
- :class:`Problem` — a position, a task, a theme and a difficulty;
- :class:`ProblemGenerator` — produces problems of one theme, usually as a
  propose-and-verify :class:`SamplingGenerator` checked by the :class:`Oracle`;
- :class:`Suite` — a reproducible problem set, specced in YAML or frozen as JSONL;
- :class:`Contestant` — a player named by a spec (``material``, ``run:<id>@latest``);
- :class:`BenchmarkRunner` → :class:`BenchmarkReport` — outcomes and summaries.

Torch-free: only building a neural contestant imports torch.
CLI: ``kaisparov bench {generators|generate|run|show}``.
"""

from kaisparov.bench.contestants import BaselineContestant, Contestant, NeuralContestant
from kaisparov.bench.generators import (
    BoardBuilder,
    GenerationError,
    ProblemGenerator,
    SamplingGenerator,
)
from kaisparov.bench.oracle import Oracle
from kaisparov.bench.position import Position, move_to_uci, uci_to_move
from kaisparov.bench.problem import Problem
from kaisparov.bench.report import BenchmarkReport, ContestantResult, ProblemResult, Tally
from kaisparov.bench.runner import BenchmarkRunner
from kaisparov.bench.suite import Suite, SuiteEntry, SuiteSpec
from kaisparov.bench.tasks import (
    AvoidMoves,
    FindMove,
    MoveTask,
    Outcome,
    PlayOut,
    Task,
    TaskContext,
)

__all__ = [
    "AvoidMoves",
    "BaselineContestant",
    "BenchmarkReport",
    "BenchmarkRunner",
    "BoardBuilder",
    "Contestant",
    "ContestantResult",
    "FindMove",
    "GenerationError",
    "MoveTask",
    "NeuralContestant",
    "Oracle",
    "Outcome",
    "PlayOut",
    "Position",
    "Problem",
    "ProblemGenerator",
    "ProblemResult",
    "SamplingGenerator",
    "Suite",
    "SuiteEntry",
    "SuiteSpec",
    "Tally",
    "Task",
    "TaskContext",
    "move_to_uci",
    "uci_to_move",
]
