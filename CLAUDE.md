# CLAUDE.md

Guidance for AI assistants (and humans) working in this repo.

## What this is

kAIsparov is a **research platform for GNNs that play chess**. A board is a graph
(64 square-nodes, piece movements as typed edges); the goal is to compare
different **GNN architectures** and **training methods** on the same task, sharing
one engine, one evaluation arena, and one experiment tracker. `gnn_v1` (a
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
  your own king in check); the game ends when a king is captured. No en passant,
  promotion, or draws. This is intentional, not a bug.
- **Style**: snake_case, English identifiers, ruff-formatted (line length 100).
- **Experiment tracking** is the `runs/` registry. Do **not** reintroduce the old
  per-package `model_info.json` / `weights/` system — it was removed on purpose.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt   # CUDA 11.8 or /cpu index
pip install -e .

kaisparov train --config config/default.yaml   # or: python -m kaisparov.cli train
kaisparov eval  --games 60
kaisparov runs  list | show <id> | best
kaisparov play  --vs-ai

ruff check . && ruff format --check .           # lint + format
pytest                                          # tests (torch-free where possible)
```

## Gotchas

- **CPU-first**: development runs on CPU (no suitable GPU). `device: auto` → CPU.
  Keep default configs light; **don't kick off long training** unless asked.
- `runs/`, `data/`, and `*.pth` are git-ignored — training artifacts never get
  committed.
- Tests avoid importing the pygame UI so they run headless; if you must import
  `kaisparov.play` in a headless check, set `SDL_VIDEODRIVER=dummy`.
- Adding a backend = a new `models/<name>/` folder with a `BACKEND_SPEC`, then a
  line in `models/factory.py`. Nothing else should need to change.
