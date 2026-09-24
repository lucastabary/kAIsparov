# Changelog

High-level milestones for kAIsparov. Commit history has the *what changed*;
this file keeps the *what it means* — the phases the project moved through and
the reasoning behind the big design decisions.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). To release,
rename `[Unreleased]` to `[X.Y.Z] - <date>`, commit, and push a `vX.Y.Z` tag: the
package version is read off the tag, and the Release workflow publishes that section.

## [Unreleased]

- **Entropy thermostat**: `ppo.target_entropy` makes the entropy coefficient adapt after
  every epoch to hold the policy's entropy at a fraction of its maximum (`log(n_legal)`
  per position) — SAC's automatic temperature. A fixed coefficient meant something
  different in every phase: 0.08 drowned the checkmate-only signal, 0.003 froze the
  policy. The coefficient used and `entropy_norm` are logged each epoch, and the
  coefficient is saved with the checkpoint, so a chained phase carries on from it.

- **Pool weights are shares**: each opponent's `weight` in a pool is now its share of
  the games (`weight / sum`); the snapshot entry's weight is the share of the whole
  past-self stream. `group_weights` is gone (a config still using it is refused with a
  pointer); the shipped presets were converted to the same shares.
- **One minimax, any evaluator**: `MinimaxAgent(evaluator, depth)` is an alpha-beta
  search that scores its leaves with any `Evaluator` — material, the heuristic, or a
  network's critic (`MinimaxAgent.on_model`, what `MinimaxAgent(model, processor)`
  was). On material it needs no model: as a pool opponent (`{kind: minimax, params:
  {evaluator: material}}`, or `heuristic`) or a contestant (`material+minimax2`,
  `heuristic`), it keeps its pieces defended and refuses poisoned captures.
- **Fallible opponents**: any pool entry, snapshots included, takes
  `random_move_prob` (default 0) — the chance, on each move, that it plays a random
  legal move instead of its own. A strong but fallible opponent, e.g. a checkpoint's
  minimax at 0.1.
- **high_entropy, phases 2–3**: `entropy_coef` 0.048 → 0.015 and 0.024 → 0.01. At v4 x4
  the entropy term was 20–60x the policy loss: entropy rose, the win rate fell. The
  pools (`he_*` presets) add a heuristic minimax at the material baseline's weight from
  phase 0b, and past-selves that play a random move 10% (phase 2) / 5% (phase 3) of
  the time — about 3–4 slips a game.

- **Won-endgame curriculum**: `curriculum.defender_pieces` gives the side the learner
  plays against fewer pieces (`1` = a bare king), and seats the learner on the strong
  side. `high_entropy` opens with such a phase 0a (K + 3 majors vs K), then two
  bridges where the defender keeps pieces to capture with (K + 3 majors vs K + 1 major,
  K + 3 vs K + 2), and runs phases 0–1 at low entropy (0.01; 0.003 in 0a). Its first
  run, checkmate-only at `entropy_coef: 0.08`, mated in ~1.5% of games and learned
  nothing; phase 0a alone reached 92% wins, but went straight onto balanced boards
  it lost its pieces and drew.

- **Chain configs carry the shared settings**: besides its `stages:` list, a chain
  config may hold any training setting, applied to every stage under the stage's own
  values. The `high_entropy` phases now hold only what changes from one phase to the
  next; what the four have in common lives once in `high_entropy_all.yaml`.

- **Tooling**: the version comes from the git tag (setuptools-scm), and pushing a tag
  publishes a GitHub release. CI tests the pinned stack (it was on torch 2.0.1), and
  checks the architecture rules of `CLAUDE.md` as import-linter contracts.

## [1.0.0] - 2026-09-23

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
