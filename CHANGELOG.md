# Changelog

High-level milestones for kAIsparov. Commit history has the *what changed*;
this file keeps the *what it means* — the phases the project moved through and
the reasoning behind the big design decisions.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

Recent additions:

- **Standard chess, on python-chess** (tag `pre-python-chess` marks the last commit
  before it): the hand-written engine — pseudo-legal move generation, capture the king
  to win, no promotion — is gone. `core/game.py` is now a facade over `chess.Board`
  that keeps the project's own vocabulary (`(col, row)`, `Piece`, `make`/`unmake` with
  an `Undo`), and python-chess never leaks above it. Checkmate, stalemate, castling, en
  passant, promotion and the draw rules are the real ones. The reward's win term is a
  flat `checkmate` bonus instead of the value of a captured king. **Runs from before
  the tag are not comparable**, and the baselines stopped discriminating: `material vs
  random` fell from ~98% to ~62%, most games being draws.
- **Skill benchmark** (`bench/`, `kaisparov bench`): generated problems whose answer an
  exhaustive `Oracle` knows exactly, grouped by theme — mate in one, avoid mate, free
  capture, fork, parry a threat, convert an endgame, mirror consistency, value-sign and
  policy-rank probes. Where the arena says who wins, this says *what* a model can do.
  Suites are YAML (`config/benchmarks/`), reports JSON under `runs/benchmarks/`.
- **Node feature sets are named and frozen** (`models/features.py`): `pieces` (the 12
  piece-type one-hots) is the default for new runs; `pieces_control` adds the two
  blocking-aware control flags. A run records the *name*, `tests/test_features.py` pins
  each set's output, and a new encoding gets a new name rather than editing an old one.
- **A run is rebuildable from what it recorded**: `Architecture(model, hidden_dim,
  features)` plus `factory.build_agent` / `factory.load_agent`, the only way to build or
  load a network. A checkpoint older than the `features` key has its feature set read
  off the width of its input layer, so the 12-dim runs of early September load again.
- **Promotion picker in the UI**: a pawn reaching the last rank asks which piece it
  becomes, instead of always queening. The policy still queens — its action space is
  `(source, dest)` (see `todo.md`).
- **No "best" checkpoint any more**: `best.pth`, `runs best`, `play --vs-ai --best`
  and `run:<id>@best` are gone, and `resolve_checkpoint` takes `"latest"` or an epoch.
  Selecting on Elo against the baselines stopped meaning anything once draws started
  dominating, and picking a "best" model on a number nobody trusts is worse than not
  picking one. A run is represented by where it got to; use the benchmark when a
  comparison has to be made.
- **Faster where it was silly**: the repetition history keeps a cheap transposition key
  instead of a full Polyglot hash per move (make/unmake 2.1× faster), and the legal mask
  is a lookup instead of `torch.isin` (2.9×).

- **Second model backend — `shared_rgcn`**: `rgcn` with its message-passing steps
  unified. Instead of 4 distinct `RGCNConv` layers (one per-relation weight set per
  step), a single relational conv is applied `num_steps` times residually, on top of
  a linear encoder that lifts features to the working width. Message passing becomes
  an iterated operator rather than a stack: ~2.5× fewer parameters at equal width, and
  depth costs nothing. The board encoding is reused verbatim from `rgcn`, so an
  `rgcn` vs `shared_rgcn` run compares *only* the network.
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
