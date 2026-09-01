# kAIsparov — Developer & Research Documentation

A guide to how the codebase is built and *why*, written for someone with an ML
background who wants to understand the system, run experiments, or extend it.

For a quick tour see [README.md](README.md); for repo conventions see
[CLAUDE.md](CLAUDE.md). This document goes deeper.

## Contents

1. [Philosophy & scope](#1-philosophy--scope)
2. [The pipeline at a glance](#2-the-pipeline-at-a-glance)
3. [The chess engine (`core/`)](#3-the-chess-engine-core)
4. [Board → graph, and the model (`models/`)](#4-board--graph-and-the-model-models)
5. [The RL layer: environment & agents (`envs/`, `agents/`)](#5-the-rl-layer-environment--agents-envs-agents)
6. [Training (`training/`)](#6-training-training)
7. [Evaluation (`eval/`)](#7-evaluation-eval)
8. [Experiment tracking & resume (`tracking/`)](#8-experiment-tracking--resume-tracking)
9. [Configuration](#9-configuration)
10. [The CLI](#10-the-cli)
11. [Extending: a new backend](#11-extending-a-new-backend)
12. [Interpretability — the research goal](#12-interpretability--the-research-goal)
13. [Developing: tests, lint, types](#13-developing-tests-lint-types)

---

## 1. Philosophy & scope

kAIsparov treats a chess position as a **graph** and studies how **Graph Neural
Networks** learn to play. It is deliberately a *platform*, not a single model: the
engine, environment, evaluation, and tracking are fixed, and the **architecture**
(how the network reasons) and the **training method** (how the policy learns) are
the variables you change and compare.

The **deeper goal is interpretability** — understanding *what* the GNN learns about
chess (piece relationships, board structure), not only how strong it gets.

**Rule simplification (important).** To keep the RL problem tractable the engine
plays a *capture-the-king* variant: moves are **pseudo-legal** (a side may leave its
own king attacked), and a game ends when a king is actually captured. Castling and
en passant are implemented; there is no check restriction, promotion, or
draw-by-rule. This is intentional.

---

## 2. The pipeline at a glance

```
ChessGame ──graphify──> PyG Data ──model──> (edge scores, state value)
   ▲   │                                          │
   │   └── legal mask (which edges are legal moves)│
   │                                               ▼
   │                           mask → Categorical → sampled move (an edge)
   │                                               │
   └────────────────── play(move) ────────────────┘
                     (self-play rollout collects transitions)
                                   │
                     PPOBuffer + negamax GAE
                                   │
                          train_one_epoch  ──▶ gradient step on the model
```

One **epoch** = collect a batch of self-play games (rollout) → compute advantages →
run several PPO update passes. Every few epochs the agent is evaluated against
baselines, and checkpoints + metrics are written to `runs/`.

---

## 3. The chess engine (`core/`)

Pure Python, no torch, no pygame — fast and unit-tested.

- **`coords.py`** — the single source of truth for geometry. A coordinate is
  `(col, row)` == `(x, y)`, origin bottom-left. White advances toward higher `row`.
  The linear index used by the graph is `row * 8 + col`. `in_bounds`,
  `coord_to_index`, `index_to_coord`, and the point-of-view transform
  `to_pov_coord` (an involution — applying it twice is identity) all live here.
- **`pieces.py`** — `PieceType`, `Player`, and `Piece` (with `__slots__` for speed;
  fields `player`, `type`, `has_moved`).
- **`attacks.py`** — movement geometry **precomputed once** at import: `KNIGHT_TARGETS`
  / `KING_TARGETS` (on-board hops) and `ORTHO_RAYS` / `DIAG_RAYS` (ordered rays for
  sliding pieces). Move generation is then table lookups, not recomputed ranges.
- **`movegen.py`** — `pseudo_legal_moves(grid, source)` and `all_moves(grid, player)`.
  Pure functions over a grid.
- **`rules.py`** — `is_in_check` / `find_king` (used by the `check` reward-shaping
  term; not required by the variant's move legality).
- **`board.py`** — `ChessGame`, the mutable state. Key methods:
  - `make(source, dest) -> Undo` and `unmake(undo)` — apply/reverse a move in **O(1)**
    (no board cloning). This is the throughput lever for rollouts and any future
    tree search. `Undo` captures everything needed to reverse, including castling.
  - `play(source, dest)` — validate then `make`; returns the captured piece or `None`.
  - `copy()` — a deep, independent clone.
  - `possible_moves`, `is_move_valid`, `to_pov_coord`/`get_pov_grid`, `is_in_check`.
- **`game_interface.py`** — the pygame board (rendering + mouse input). UI only.

**Correctness is locked by `tests/test_engine.py`**, including a *perft* count from
the start position that matches standard chess (20 / 400 / 8902), plus a make/unmake
round-trip invariant.

---

## 4. Board → graph, and the model (`models/`)

### Encoding (`gnn_v1/processor.py`, `base_processor.py`)

`GNN1Processor.graphify(game)` builds a PyG `Data`:

- **Nodes**: the 64 squares. Node features are **12-dim, relative to the side to
  move**: indices 0–5 mark an *ally* piece of a given type on that square, 6–11 an
  *enemy* piece. (Relative encoding means the network always "sees" from the mover's
  perspective, so no separate side-to-move plane is needed.)
- **Edges**: a **static** graph (same every position) built by
  `create_static_full_chess_graph()`. Each edge is one piece's possible movement, and
  carries a **relation type** (`edge_type`): `0` knight, `1` rook, `2` bishop, `3`
  king, `4` white pawn, `5` black pawn — 6 relations. (Castling is *not* an edge, so
  `gnn_v1` cannot emit castling moves; a documented limitation.)

### Network (`gnn_v1/model.py`)

`ChessRGCN` is a Relational GCN: 4 `RGCNConv` layers (with residual connections)
producing per-node embeddings. On top:

- **Actor**: for every edge, concatenate its two endpoint embeddings `[2·hidden]` and
  score it → one scalar per edge. The action space *is* the set of edges (moves).
- **Critic**: pool node embeddings with `AttentionalAggregation` → a single state
  value.

So `model(data)` returns `(action_scores [E], state_value [B])`.

### From scores to a move (`process_output`)

1. Build the **legal mask** over edges (`get_legal_mask`): which of the static edges
   are legal moves for the side to move right now.
2. `masked_logits = action_scores.masked_fill(~legal_mask, -inf)`.
3. `Categorical(logits=masked_logits)` → sample (training) or argmax (eval).
4. Map the chosen edge back to `(source, dest)` coordinates.

`process_output` accepts an optional precomputed `legal_mask` to avoid recomputing it.

### The backend contract (`backend_spec.py`, `factory.py`)

A backend is a package exposing `BACKEND_SPEC: BackendSpec` with `model_class`,
`processor_class`, `buffer_class`, `collect_data`, `train_one_epoch`. `factory.py`
loads backends by name (`load_backend`, `load_backend_spec`). `BaseModel`
(`base_model.py`) is a thin base: `create_agent`, `create_optimizer`,
`load_agent_for_inference` (loads a plain `state_dict`).

---

## 5. The RL layer: environment & agents (`envs/`, `agents/`)

### `ChessEnv` (`envs/chess_env.py`)

A small Gym-like wrapper and the **single place** reward/terminal logic lives:

- `reset(board=None)` → observation (the `ChessGame`).
- `step(move) -> StepResult(obs, reward, done, info)`. Reward is the **material value
  of the captured piece** (king = 1.0), always from the point of view of the player
  who just moved.
- Terminal on: king captured (→ `winner`), the side to move having no legal move
  (draw), or `max_plies` (draw).
- `legal_moves()` → all legal `(source, dest)` moves.

### Policies (`agents/`)

All agents implement `Policy.select_move(game) -> (source, dest) | None`:

- `RandomAgent` — uniform legal move.
- `MaterialAgent` — greedy 1-ply: capture the highest-value enemy piece (king if
  possible), else random. A meaningful baseline.
- `NeuralAgent` — wraps a model + processor; `select_move` graphifies, runs the model,
  masks, and returns the chosen move. (Imported lazily so baselines stay torch-free.)

---

## 6. Training (`training/`)

### PPO (`ppo.py`)

`PPOBuffer` stores per-step: the graph `Data`, the chosen action index, old log-prob,
old value, reward, done, and the legal mask (so the update never needs the engine).

**Advantages — negamax GAE (self-play).** Consecutive plies belong to *opposing*
players, so from the mover's viewpoint the next state is worth `-V(next)` and the
trailing advantage flips sign. Concretely, with `flip = -1` in self-play mode:

```
δ_t   = r_t + γ · (flip · V_{t+1}) − V_t          (0 bootstrap at episode end)
A_t   = δ_t + γ · λ · flip · A_{t+1}              (A reset to δ_t at episode end)
R_t   = A_t + V_t                                  (value target)
```

The consequence: a move that lets the opponent win gets a **negative** advantage.
(Set `self_play=False` for a plain single-agent trajectory.)

**`train_one_epoch`** re-runs the model on the stored states (batched), recomputes the
masked distribution for the stored actions, and optimizes the clipped PPO objective:

```
ratio   = exp(new_logπ − old_logπ)
L_policy = −E[min(ratio·A, clip(ratio, 1±ε)·A)]
L_value  = MSE(V, R)
loss     = L_policy + c_v·L_value − c_e·entropy      (+ gradient clipping)
```

Advantages are normalized per update; returns metrics (`policy_loss`, `value_loss`,
`entropy`, `loss`, `steps`).

### Rollout (`rollout.py`) — batched self-play

All episodes of an epoch are played **in parallel**: at each ply every still-running
game is graphified and the model runs **one batched forward pass** (`Batch.from_data_list`)
instead of one tiny forward per game. This is ~3–4× faster on CPU. Each episode's
transitions are buffered separately and flushed **contiguously** on termination, so
the negamax GAE sees clean episode boundaries. The agent plays both sides.

### Curriculum (`curriculum.py`)

`PieceCountCurriculum` generates random simplified starts (both kings + up to N pieces
per side, majors optionally disabled), so early training faces easier positions.

### Trainer (`trainer.py`)

Owns the loop: seeds torch/numpy/python, resolves the device (`auto`→CPU on this
machine), builds agent/optimizer/buffer/curriculum from the config, then per epoch:
`collect_data → train_one_epoch → log → (periodic) evaluate vs baselines → checkpoint`.
Everything is written through a `RunManager` (see §8).

---

## 7. Evaluation (`eval/`)

`arena.py` pits any two policies:

- `play_game(white, black)` → `GameResult(winner, plies, reason)`.
- `evaluate(a, b, games)` alternates colors (cancels first-move bias) → `MatchStats`
  with `score_a` (win=1, draw=0.5) and a rough `elo_diff = −400·log10(1/score − 1)`.

During training the neural agent is scored against `RandomAgent` and `MaterialAgent`,
producing `winrate_vs_*` and `elo_vs_*` metrics logged over epochs — the Elo-vs-random
curve is the headline learning signal.

---

## 8. Experiment tracking & resume (`tracking/`)

The **single** experiment-tracking system (there is no `model_info.json` anymore).

### What a run looks like

`RunManager` writes `runs/<run_id>/`:

```
config.yaml              # the exact resolved config
run.json                 # metadata + summary (see below)
metrics.jsonl            # one JSON line per logged step (train + eval)
tensorboard/             # TensorBoard scalars
checkpoints/
    epoch10.pth          # model weights only (inference-friendly)
    epoch10.state.pth    # optimizer + RNG state (for exact resume)
    best.pth             # copy of the best model checkpoint by elo_vs_random
```

`run.json` captures: status, timestamps, **git commit + dirty flag**, device, seed,
`num_params`, notes, the full config, `epochs_completed`, every checkpoint, the
`eval_history`, the `best_checkpoint`, and lineage (`parent_run_id`, `resumed_from`).

### Querying (`Registry`, torch-free)

`Registry` reads runs back without importing torch. CLI: `kaisparov runs list |
show <id> | lineage <id> | best --metric elo_vs_random`. `resolve_checkpoint(run_id,
which)` returns a checkpoint path where `which` defaults to **`"latest"`** (the most
recent epoch), or `"best"`, or an epoch number.

### Resume & lineage

`kaisparov train --resume <run_id>`:

1. Rebuilds the config from the parent run (so architecture matches the weights).
2. Loads the parent's **latest** checkpoint weights.
3. Loads the sibling `*.state.pth` to restore the **optimizer moments and RNG state**,
   so training continues *exactly* where it stopped (not a warm restart).
4. Continues epoch numbering from the parent's last epoch and records `parent_run_id`,
   forming a lineage you can inspect with `kaisparov runs lineage`.

This makes training in chunks on a laptop equivalent to one long run — important for
CPU-only work.

---

## 9. Configuration

`training/config.py` defines a typed `TrainConfig` with nested `PPOSettings`,
`RolloutSettings`, `CurriculumSettings`, `EvalSettings`. It loads/saves YAML (unknown
keys ignored) and is stored verbatim in each run for reproducibility.

- `config/default.yaml` — the documented defaults.
- `config/experiments/smoke.yaml` — a ~1-minute CPU sanity run.
- `config/experiments/resume_example.yaml` — how to continue a run from a config.
- Keep **one YAML per experiment** under `config/experiments/`.

**Documenting a run.** `title` and `description` fields let you record the intent of
each experiment; they're stored in `run.json` and shown by `kaisparov runs`.

**Reward shaping.** The `reward` field configures the self-play reward (mover's point
of view, per ply) — weighted terms `material`, `king_capture`, `check`, `step_penalty`.
Reference a named preset from `config/rewards.yaml` (`reward: aggressive`) or write the
terms inline. The resolved reward is stored in `run.json`, so every experiment records
its shaping. `training/reward.py` turns the settings into the function the rollout uses.

**Resuming from a config.** Set `resume_from_run: <run_id>` in the YAML to continue an
earlier run. You **don't redefine the architecture** — `model`/`hidden_dim` are
inherited from the parent (they must match its weights), as is any other field you
don't override. Only list what changes (e.g. more `epochs`, a new `learning_rate`, a
fresh `title`). This is equivalent to `--resume <run_id>` on the CLI, and restores the
optimizer + RNG state (see §8).

CLI flags (`--epochs`, `--seed`, `--episodes`, `--title`, `--description`, `--cpu`, …)
override individual fields on top of the config.

---

## 10. The CLI

One entry point, `kaisparov <command>` (or `python -m kaisparov.cli <command>`;
`python -m kaisparov.<train|play|eval|tracking>` also work):

| Command | What it does |
|---------|--------------|
| `kaisparov train` | Train (config-driven; `--resume <id>` to continue a run). |
| `kaisparov eval`  | Play matches between agents; win-rates + Elo. |
| `kaisparov play`  | Pygame board: human vs human, or `--vs-ai` (newest run's latest checkpoint by default; `--best` for best-Elo, `--checkpoint` for a path). |
| `kaisparov runs`  | `list` / `show` / `lineage` / `best` over recorded runs. |

`tensorboard --logdir runs/` watches loss + Elo curves live.

---

## 11. Extending: a new backend

To try a new architecture (e.g. GAT, a graph transformer) or a new learner:

1. Create `src/kaisparov/models/<name>/` with a model (`nn.Module`, ideally a
   `BaseModel` subclass), a processor (`graphify` + `process_output`), and a
   `BACKEND_SPEC`.
2. Add one line to `MODEL_MODULES` in `models/factory.py`.
3. Everything else — self-play, arena, Elo, tracking, resume — works unchanged:

```bash
kaisparov train --model <name>
kaisparov eval  --model <name> --checkpoint runs/<id>/checkpoints/best.pth
```

The engine, env, agents, and tracker never learn which model is running, which is
what makes architecture-vs-architecture comparison honest. To vary the *training
method* instead, provide different `collect_data` / `train_one_epoch` / `buffer_class`.

---

## 12. Interpretability — the research goal

The point of the platform is to ask *what the GNN learned*. Natural probes, using a
trained checkpoint (`runs/<id>/checkpoints/`):

- **Move (edge) scores** — which edges the actor rates highest on a position, and how
  that changes with small board perturbations.
- **Node embeddings** — cluster/PCA the per-square embeddings; do they encode piece
  type, mobility, threats, control? (`ChessRGCN` output before the heads.)
- **Relation flow** — since edges are typed (knight/rook/…​), how much does each
  relation contribute? (Ablate a relation, or inspect per-relation messages.)
- **Critic value landscape** — how the state value responds to material and structure.

The `notebooks/` folder is the workspace — start from `analyze.ipynb`, which already
loads runs, plots the Elo/loss curves, and dumps top edge scores + the critic value
for a position. Add one notebook per investigation.

---

## 13. Developing: tests, lint, types

```bash
pip install -e ".[dev]"          # ruff, mypy, pytest, pre-commit
pip install -e ".[notebooks]"    # jupyter, matplotlib, pandas

ruff check . && ruff format --check .   # lint + format (notebooks excluded)
mypy src/kaisparov                       # static types (game_interface excluded)
pytest                                    # engine, agents, env, arena, ppo, tracking, smoke
pre-commit install                        # run the above on commit
```

CI (`.github/workflows/ci.yml`) runs ruff + mypy + pytest on CPU wheels. Tests avoid
importing the pygame UI so they run headless.

**Conventions**: `(col, row)` coordinates via `core/coords.py`; snake_case, English,
ruff-formatted (line length 100); the `runs/` registry is the only tracking system;
keep training CPU-friendly and don't launch long runs unprompted.
