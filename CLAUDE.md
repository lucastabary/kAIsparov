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
| `core/` | Chess engine — pure Python, no torch. `coords` (single source of truth), `board` (`make`/`unmake`), `movegen`, `rules`, `attacks` (precomputed tables), `pieces`, `game_interface` (pygame). |
| `envs/` | `ChessEnv` — Gym-like `reset`/`step`/reward/terminal. The only place reward & game-over logic live. |
| `models/` | Neural backends. Each `models/<name>/` exposes a `BACKEND_SPEC` (`backend_spec.py`); `factory.py` loads by name. |
| `agents/` | Policies with `select_move(game)`: `RandomAgent`, `MaterialAgent`, `NeuralAgent`. |
| `training/` | `config` (typed), `trainer`, `ppo` (buffer + negamax GAE), `rollout`, `curriculum`. |
| `eval/` | `arena` — play matches, win-rates, Elo. |
| `tracking/` | `RunManager` writes `runs/<id>/`; `Registry` (torch-free) reads them. |
| `cli.py` | Unified entry: `kaisparov <train\|eval\|play\|runs>`. |

## Conventions

- **Coordinates**: `(col, row)` == `(x, y)`, origin bottom-left; `grid[col][row]`.
  White advances toward higher `row`. Node index = `row * 8 + col`. All of this
  lives in `core/coords.py` — use it, don't re-derive.
- **Variant**: capture-the-king. Moves are pseudo-legal (not filtered for leaving
  your own king in check); the game ends when a king is captured. Castling and en
  passant are implemented; no promotion or draws. This is intentional, not a bug.
- **Style**: snake_case, English identifiers, ruff-formatted (line length 100).
- **Experiment tracking** is the `runs/` registry. Do **not** reintroduce the old
  per-package `model_info.json` / `weights/` system — it was removed on purpose.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt   # CUDA 11.8 or /cpu index
pip install -e .

kaisparov train --config config/default.yaml   # or: python -m kaisparov.cli train
kaisparov eval  --games 60
kaisparov runs  list | show <id> | lineage <id> | best | graph  # graph = HTML lineage view
kaisparov play  --vs-ai

ruff check . && ruff format --check .           # lint + format
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
- **Before committing / merging**: `ruff check . && ruff format --check .` and `pytest`
  must pass. After code changes, also run `graphify update .` (see below).
- **Never commit** training artifacts — `runs/`, `data/`, `*.pth` are git-ignored on
  purpose (see Gotchas). Commit/push only when asked.

## Gotchas

- **CPU-first**: development runs on CPU (no suitable GPU). `device: auto` → CPU.
  Keep default configs light; **don't kick off long training** unless asked.
- `runs/`, `data/`, and `*.pth` are git-ignored — training artifacts never get
  committed.
- Tests avoid importing the pygame UI so they run headless; if you must import
  `kaisparov.play` in a headless check, set `SDL_VIDEODRIVER=dummy`.
- Adding a backend = a new `models/<arch>/` folder (named by architecture, e.g.
  `rgcn`, `gat`) with a `BACKEND_SPEC` and a `README.md` describing the model (see
  `models/rgcn/README.md`), then a line in `models/factory.py`. Nothing else changes.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
