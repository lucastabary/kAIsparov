# CLAUDE.md

Guidance for AI assistants (and humans) working in this repo.

## What this is

kAIsparov is a **research platform for GNNs that play chess**. A board is a graph
(64 square-nodes, piece movements as typed edges); the goal is to compare
different **GNN architectures** and **training methods** on the same task, sharing
one engine, one evaluation arena, and one experiment tracker. `rgcn` (a
relational GCN actor–critic trained with PPO self-play) is the first backend.

## Layout (`src/kaisparov/`)

| Package | Responsibility |
|---------|----------------|
| `core/` | Chess engine — a facade over **python-chess**, no torch. `coords` (single source of truth), `move` (`Move` with its promotion), `material` (what pieces and moves are worth, `WIN`), `game` (`ChessGame`: `make`/`unmake`, `legal_moves`, `grid` snapshot), `rules`, `draw`, `pieces`, `bitboard_batch` (vectorised control maps), `game_interface` (pygame). |
| `envs/` | `ChessEnv` — Gym-like `reset`/`step`/reward/terminal: when a game is over, and the raw material reward. Shaping (weights, mate and check bonuses) is `training/reward.py`; what pieces and moves are worth is `core/material.py`. |
| `models/` | Neural backends. Each `models/<name>/` exposes a `BACKEND_SPEC` (`backend_spec.py`). `features` — the named node feature sets, shared by every backend; `architecture` — `Architecture(model, hidden_dim, features)`, what a run records; `factory` — `build_agent` / `load_agent`, the only way to build or load a network. |
| `agents/` | Policies with `select_move(game)`: `RandomAgent`, `MaterialAgent`, `NeuralAgent`. |
| `analysis/` | Move review — `evaluators` (score a position), `judge` (grade a played move, chess.com-style labels + accuracy), `critic` (torch-backed evaluator). Vocabulary lives in `insights.py`. |
| `training/` | `config` (typed), `trainer`, `ppo` (buffer + negamax GAE), `rollout`, `curriculum`. |
| `eval/` | `arena` — play matches, win-rates, Elo. |
| `bench/` | Skill benchmark on targeted problems. `Position` (FEN), `Task` (how an answer is scored), `Problem`, `generators/` (`ProblemGenerator` registry, propose-and-verify `SamplingGenerator`), `oracle` (exact ground truth), `Suite` (YAML spec / frozen JSONL), `Contestant` specs, `BenchmarkRunner` → `BenchmarkReport`. Torch-free. |
| `tracking/` | `RunManager` writes `runs/<id>/`; `Registry` (torch-free) reads them. |
| `cli.py` | Unified entry: `kaisparov <train\|eval\|play\|runs>`. |

## Conventions

- **Coordinates**: `(col, row)` == `(x, y)`, origin bottom-left; `grid[col][row]`.
  White advances toward higher `row`. Node index = `row * 8 + col`. All of this
  lives in `core/coords.py` — use it, don't re-derive.
- **Rules**: standard chess, on [python-chess](https://python-chess.readthedocs.io/).
  Legal moves only, checkmate, stalemate, castling, en passant, promotion. The older
  capture-the-king variant is gone (tag `pre-python-chess`); runs from before it are
  not comparable. python-chess must not leak out of `core/` (`game`, `move`, `rules`) —
  everything above speaks `(col, row)` and `Piece`.
- **Architecture contracts** (`[tool.importlinter]` in `pyproject.toml`, run by
  `lint-imports`): python-chess stays in `core/`, `core/` imports nothing above it,
  `core/`, `bench/` and `tracking/registry` are torch-free, pygame stays in the UI. A
  new exception goes into the contract's `ignore_imports` with a comment saying why.
- **`Move` is a 3-tuple** `(source, dest, promotion)`. `game.make(*move)` and `move[0]`
  work as before, but `source, dest = move` does not, and a `Move` never matches a bare
  pair as a dict key — use `Move.coerce` on anything coming from outside. `make` with
  no named promotion queens; the model's action space is `(source, dest)` only, so
  underpromotion is not reachable by the policy yet.
- **`game.grid` is a cached snapshot**, rebuilt on demand and invalidated by every
  make/unmake. Writing into it does *not* move a piece — use `game.place(coord, piece)`
  to set a position up by hand. Hot paths read `game.board` and its bitboards
  (`board.pawns & board.occupied_co[chess.WHITE]`), already in the `row*8+col`
  convention; see `models/rgcn/processor.graphify_batch`.
- **Draws** live in `core/draw.py`: threefold repetition, 50 moves without a capture
  or a pawn move, insufficient material, and stalemate — each switchable via
  `DrawRules`. A draw is **never** a reward event: the drawing move scores what any
  quiet move scores. Searches (`MinimaxAgent`, `MoveJudge`, `Oracle`) must score a
  node with no legal move as mate (-WIN) or stalemate (0) — reading it off the
  material on the board hands the win to whoever is up a queen.
- **Style**: snake_case, English identifiers, ruff-formatted (line length 100).
- **Node features are named and frozen.** A run records the *name* of its feature set
  (`features:`), so a name never changes meaning once runs use it: to try other
  inputs, add a set under a new name in `models/features.py`, and pin it in
  `tests/test_features.py`. Build or load a network only through
  `factory.build_agent` / `factory.load_agent` — they keep the model and its
  processor on the same `Architecture`, and read an older run's features off its
  weights. A new shape-changing hyper-parameter goes into `Architecture`, not into
  each loader.
- **There is no "best" checkpoint.** A run is represented by its **latest** one
  (`resolve_checkpoint` takes `"latest"` or an epoch number, and nothing else). Which
  checkpoint is best is a research question, and the Elo against the baselines barely
  discriminates now that draws dominate — so the project does not answer it with one
  number. Do not reintroduce a metric that silently picks a checkpoint.
- **Experiment tracking** is the `runs/` registry. Do **not** reintroduce the old
  per-package `model_info.json` / `weights/` system — it was removed on purpose.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt   # CUDA 12.1 or /cpu index
pip install -e .

kaisparov train --config config/default.yaml   # or: python -m kaisparov.cli train
kaisparov eval  --games 60
kaisparov bench run config/benchmarks/smoke.yaml -a material -a run:<id>  # skill profile
kaisparov runs  list | show <id> | lineage <id> | graph  # graph = HTML lineage view
kaisparov bench show runs/benchmarks/<report>.json              # compare saved reports
kaisparov play  --vs-ai

ruff check . && ruff format --check .           # lint + format
mypy src/kaisparov                              # types (CI runs it; game_interface excluded)
lint-imports                                    # architecture contracts (see Conventions)
pytest                                          # tests (torch-free where possible)
```

## Git

- **Commits**: Conventional Commits — `type(scope): imperative subject`, lowercase and
  concise. Types in use: `feat`, `fix`, `perf`, `docs`, `tooling`; the scope is the
  package touched (`core`, `model`, `training`, `train`, `types`, …). One logical change
  per commit; add a body explaining the *why* when it isn't obvious from the subject.
- **Branches**: do the work on a short-lived branch off `main` (`fix/…`, `feat/…`), then
  merge back — fast-forward to keep history linear (no merge commit unless a real branch
  topology needs one). `main` is the integration branch and stays green.
- **Docs are part of the change.** A change that makes a README, `docs.md`,
  `config/README.md`, a model's `README.md` or `CLAUDE.md` wrong is not finished: fix
  them in the same commit. They are the showcase — a stale claim on the front page
  ("from-scratch engine", a parameter count, a sample output, a CLI that gained a
  command) costs more than the code it describes. `CHANGELOG.md` records *what it
  means*, not every commit: add an entry for a change someone would need explained.
- **Before committing / merging**: `ruff check . && ruff format --check .`, `mypy
  src/kaisparov`, `lint-imports` and `pytest` must pass — all five are CI steps
  (`.github/workflows/ci.yml`), and mypy is the easy one to forget. After code changes, also run `graphify update .`
  (see below).
- **Releases**: the version comes from the `vX.Y.Z` tag (setuptools-scm) — never write
  one into a file. Rename `[Unreleased]` in `CHANGELOG.md` to `[X.Y.Z] - <date>`, commit,
  push the tag; the Release workflow publishes that section.
- **Never commit** training artifacts — `runs/`, `data/`, `*.pth` are git-ignored on
  purpose (see Gotchas). Commit/push only when asked.

## Gotchas

- **CPU-first**: development runs on CPU (no suitable GPU). `device: auto` → CPU.
  Keep default configs light; **don't kick off long training** unless asked.
- `runs/`, `data/`, and `*.pth` are git-ignored — training artifacts never get
  committed.
- Tests avoid importing the pygame UI. It is not only about a display: CI installs the
  package with `pip install -e . --no-deps`, so **pygame is not there at all** and any
  test importing `kaisparov.play` or `core/game_interface.py` fails at collection. Test
  UI-adjacent logic through the torch- and pygame-free layer underneath it where you can
  (`core/game.py`, `insights.py`); when a test genuinely needs the UI modules, put it in
  a module guarded by `pytest.importorskip("pygame")` — see `tests/test_play_ui_wiring.py`.
  For a headless *manual* check, set `SDL_VIDEODRIVER=dummy`.
- Adding a backend = a new `models/<arch>/` folder (named by architecture, e.g.
  `rgcn`, `gat`) with a `BACKEND_SPEC` and a `README.md` describing the model (see
  `models/rgcn/README.md`), then a line in `models/factory.py`. Nothing else changes.
- Adding a benchmark test = a `ProblemGenerator` subclass in `bench/generators/` (plus
  an import line in its `__init__`) and, if no existing `Task` scores it, a `Task`
  subclass in the matching `bench/tasks/` module (moves, play-outs, consistency,
  probes). Ground truth must come from the `Oracle` (or another
  exact search) and respect the draw rules — never from a model. Suites live as YAML
  specs in `config/benchmarks/`; reports go to `runs/benchmarks/` (git-ignored).
- **Generated positions must be legal.** python-chess *will* generate the capture of a
  king left in check, so a position where the side not to move is in check hands every
  contestant a free "solution". `SamplingGenerator.generate` rejects those; a generator
  that builds positions another way has to do the same.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
