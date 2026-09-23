---
name: new-bench-problem
description: Add a new kind of problem to the kAIsparov skill benchmark — a ProblemGenerator (and a Task if no existing one scores it), registered, tested against the Oracle, and added to the suites. Use when the user wants the benchmark to test a new chess skill or theme.
argument-hint: "<the skill to test, e.g. 'spot a skewer'>"
---

# /new-bench-problem — a new theme for `kaisparov bench`

Skill to test: `$ARGUMENTS`

## 0. Pin the problem down first

Before writing code, state in one or two sentences and get the user's agreement on:
- **the position family**: what is on the board, who moves;
- **the exact answer**: which moves count as right (or which must be avoided), and
  why the Oracle can decide it *exactly*;
- **the Task that scores it.** Existing ones: `FindMove`, `AvoidMoves`,
  `WinMaterial` (`bench/tasks/moves.py`), `PlayOut` (`playout.py`), `SameMove`
  (`consistency.py`), `ValueSign`, `PolicyRank` (`probes.py`). Reuse one when it fits.

## 1. Ground truth comes from the Oracle, never from a model

`bench/oracle.py` (or another exhaustive search) decides the answer, and it respects
the draw rules: a node with no legal move is mate (-WIN) or stalemate (0), never a
material count. If the Oracle cannot answer the question yet, extend it (with tests)
before writing the generator.

## 2. The generator

- A `SamplingGenerator` subclass (propose-and-verify) in the thematic module of
  `src/kaisparov/bench/generators/`: `tactics`, `safety`, `draws`, `endgames`,
  `special`, `probes`, `robustness`, or a new module if none fits.
  **`mate_in_one.py` is the reference implementation**: copy its shape.
- Class attributes `name` (the registry key, snake_case), `theme`, `description`
  (one line, shown by `kaisparov bench generators`).
- `__init__(self, <params>, **kwargs)`: validate the params, call `super().__init__(**kwargs)`.
- `propose(rng)` returns a `Problem(id="", theme=..., position=..., task=..., difficulty=..., meta=...)`
  or `None` to reject the draw. Build boards with `BoardBuilder`; reject early a
  position where the side *not* to move is in check (python-chess would hand every
  contestant a free king capture; `generate` rejects it too, but late).
- Register it: an import line and an `__all__` entry in `bench/generators/__init__.py`.

## 3. A new Task, only if none fits

A `Task` subclass in the matching `bench/tasks/` module, with its `kind`, `attempt`,
`params` / `from_params`, `mirrored` (colour-flipped twin), and `validate` when a
position can contradict it. Export it from `bench/tasks/__init__.py`, and add it to
the round-trip test in `tests/test_bench.py`.

## 4. Tests

- Add the generator to `CHEAP` in `tests/test_bench_themes.py`, with the cheapest
  params that still exercise it: `test_every_registered_generator_is_covered_here`
  fails otherwise. Generation is the slow part of the suite; keep it to a few seconds.
- A test that the accepted/forbidden moves are **exactly** what the Oracle says,
  like `test_escape_check_answers_are_exactly_the_safe_moves`.

## 5. Suites and a sanity run

- An entry in `config/benchmarks/standard.yaml`; in `smoke.yaml` too if it is a new
  task kind (smoke holds one example of each).
- `kaisparov bench generators` lists it;
  `kaisparov bench run config/benchmarks/smoke.yaml -a random -a material` runs it.
  On a tactical theme `material` should beat `random`; if not, say so: the problem
  may not test what it claims.

## 6. Finish

`lint-imports` must still pass (`bench/` stays torch-free). A `CHANGELOG.md` line
under `[Unreleased]` for the new theme. Then `/ship`, if the user wants it.
