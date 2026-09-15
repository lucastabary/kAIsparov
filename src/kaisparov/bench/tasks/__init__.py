"""Tasks: what a contestant is asked to do in a problem's position, and how it is scored.

A :class:`Task` is the polymorphic half of a :class:`~kaisparov.bench.problem.Problem`
(the other half is the position). The runner never looks inside one: it hands the task
a fresh game, a policy and a :class:`TaskContext`, and gets an :class:`Outcome` back.
Adding a new kind of test is therefore one subclass — the generators, suites, runner
and reports all work with it unchanged.

The families:

- **one decision** (:mod:`.moves`) — :class:`FindMove` ("play one of these"),
  :class:`AvoidMoves` ("never play these"), :class:`WinMaterial` ("win a piece",
  graded by searching the move played);
- **play-outs** (:mod:`.playout`) — :class:`PlayOut`, "convert this", "hold this";
- **consistency** (:mod:`.consistency`) — :class:`SameMove`, the contestant compared
  with itself on a mirrored or lightly perturbed board;
- **probes** (:mod:`.probes`) — :class:`ValueSign`, :class:`PolicyRank`, graded on the
  contestant's analysis rather than on its move.

Every task serialises to a JSON-friendly dict tagged with its ``kind``; subclasses
register themselves on definition (importing this package registers the built-in
ones), so :meth:`Task.from_dict` finds them by name. Moves are UCI strings (``e2e4``).
"""

from kaisparov.bench.tasks.base import Outcome, Task, TaskContext, ask_move, legal_uci
from kaisparov.bench.tasks.consistency import SameMove
from kaisparov.bench.tasks.moves import AvoidMoves, FindMove, MoveTask, WinMaterial
from kaisparov.bench.tasks.playout import PlayOut
from kaisparov.bench.tasks.probes import PolicyRank, ProbeTask, ValueSign

__all__ = [
    "AvoidMoves",
    "FindMove",
    "MoveTask",
    "Outcome",
    "PlayOut",
    "PolicyRank",
    "ProbeTask",
    "SameMove",
    "Task",
    "TaskContext",
    "ValueSign",
    "WinMaterial",
    "ask_move",
    "legal_uci",
]
