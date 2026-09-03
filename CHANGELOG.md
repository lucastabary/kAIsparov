# Changelog

High-level milestones for kAIsparov. Commit history has the *what changed*;
this file keeps the *what it means* — the phases the project moved through and
the reasoning behind the big design decisions.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

Recent additions:

- **Run lineage view**: `kaisparov runs graph` renders the run registry as a
  self-contained `git log --graph`-style HTML page (one lane per resume family,
  forks where runs share a parent, status-coloured dots). Clicking a run shows its
  full config with the parameters that changed from its parent highlighted like a
  diff. Torch-free (`tracking/lineage_view.py`).
- **Full developer docs** (`docs.md`), `CLAUDE.md`, and this changelog.
- **Exact resume**: `--resume <run_id>` continues from a run's *latest* checkpoint,
  restoring optimizer moments and RNG state (a separate `*.state.pth` file), so a
  lineage of chunked runs equals one continuous run. Default checkpoint selection
  everywhere is now "latest" (use `--best` / `runs best` for best-Elo).
- **Batched self-play rollout** (~3–4× faster on CPU) and **negamax self-play GAE**.
- **mypy** added to CI; `notebooks/` workspace for analysis and interpretability.

The project was refactored from a bare game engine into a research platform in a
sequence of phases:

### Phase 5 — Unified CLI
- Single entry point `kaisparov <train|eval|play|runs>` (`cli.py`), lazy dispatch
  so read-only commands never import torch/pygame.
- Clean `play.py` (human-vs-human / `--vs-ai`) replacing the old `input()` menu.

### Phase 4 — Reproducible training & experiment tracking
- Config-driven `Trainer` (`training/config.py`, `config/default.yaml`), TensorBoard
  logging, seeding.
- **Negamax self-play credit assignment** in the PPO buffer (a move that lets the
  opponent win gets a negative advantage) — replaces the earlier single-perspective
  simplification.
- **Run tracking**: each run writes `runs/<id>/` (config, `metrics.jsonl`,
  TensorBoard, checkpoints, and a `run.json` with git commit, seed, device, param
  count, eval history, best checkpoint). Queryable via `kaisparov runs`.
- Agents are evaluated against baselines *during* training, logging an Elo curve.

### Phase 3 — Environment, baselines & evaluation
- `ChessEnv` (Gym-like) centralises reward / terminal / legal-move logic.
- `Policy` interface with `RandomAgent`, `MaterialAgent`, `NeuralAgent`.
- Evaluation arena: matches, win-rates, rough Elo. Sanity check: material beats
  random ~98%.

### Phase 2 — Fast, pure engine core
- Split `core/` into `coords` (single source of coordinate truth), `movegen`,
  `rules`, `attacks` (precomputed tables).
- `ChessGame.make`/`unmake` in O(1) (no per-move board cloning) + `copy()`.
- **Perft tests** matching standard chess (20 / 400 / 8902 from the start) lock the
  move generator and make/unmake.

### Phase 1 — Packaging & showcase hygiene
- `pyproject.toml`, pinned requirements, `README`, MIT license.
- Ruff (lint + format), pre-commit, GitHub Actions CI, first tests.

### Phase 0 — Working training loop
- Rebuilt the training package (PPO buffer + GAE, self-play rollout, curriculum)
  so the project trains end-to-end again.

### Big cleanup (pre-1.0)
- Removed the legacy `model_info.json`/`weights/` persistence, dead indirection
  (`scripts/`), unused code paths, and the package-internal model selector.
- `src/kaisparov/` layout; the `runs/` registry is now the single tracking system.

---

_When you cut a real release, tag it and move "Unreleased" items under a version
heading (e.g. `## [0.1.0] - 2026-08-29`)._
