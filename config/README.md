# Training configuration reference

Training is driven by a YAML file passed to `kaisparov train --config <file>`. This
document lists **every parameter**, its type, default, and what it does.

- Anything you omit falls back to the default below — a minimal config is valid.
- CLI flags override the file (see [Overrides](#cli-overrides)).
- The **resolved** config (all defaults filled in) is saved to `runs/<id>/config.yaml`
  and `run.json`, so every run is fully reproducible.
- Start from [`default.yaml`](default.yaml); keep one file per experiment under
  [`experiments/`](experiments/).

```bash
kaisparov train --config config/default.yaml
```

## Contents

- [Top-level](#top-level)
- [`ppo:` — the learning algorithm](#ppo--the-learning-algorithm)
- [`rollout:` — self-play data collection](#rollout--self-play-data-collection)
- [`curriculum:` — starting positions](#curriculum--starting-positions)
- [`eval:` — evaluation during training](#eval--evaluation-during-training)
- [`reward:` — reward shaping](#reward--reward-shaping)
- [Resuming a run](#resuming-a-run)
- [CLI overrides](#cli-overrides)
- [Examples](#examples)

---

## Top-level

| Parameter | Type | Default | What it does |
|-----------|------|---------|--------------|
| `model` | str | `gnn_v1` | Which model backend to train (a folder under `src/kaisparov/models/`, loaded by name). |
| `hidden_dim` | int | `8` | Hidden size of the network — the main architecture knob. Must match the checkpoint when resuming. |
| `epochs` | int | `50` | Number of training epochs. One epoch = collect self-play data, then run the PPO update. |
| `seed` | int | `0` | Random seed for torch / numpy / python (reproducibility). |
| `device` | str | `auto` | `auto` (CUDA if available, else CPU), `cpu`, or `cuda`. On this project, `auto` → CPU. |
| `checkpoint_every` | int | `10` | Save a checkpoint every N epochs. `0` disables checkpointing. |
| `runs_dir` | str | `runs` | Directory where run artifacts are written (git-ignored). |
| `title` | str | `""` | Short label for the run — shown by `kaisparov runs list`. |
| `description` | str | `""` | Longer free text: your intent / hypothesis for this experiment. Stored in `run.json`. |
| `notes` | str | `""` | Extra free-text notes. Stored in `run.json`. |
| `resume_from_run` | str | `null` | A run id to continue (see [Resuming](#resuming-a-run)). `null` for a fresh run. |

> `resume_from` and `parent_run_id` also exist but are **filled automatically** on
> resume — you don't set them by hand.

---

## `ppo:` — the learning algorithm

Proximal Policy Optimization hyperparameters.

| Parameter | Type | Default | What it does |
|-----------|------|---------|--------------|
| `learning_rate` | float | `0.001` | Adam learning rate. |
| `gamma` | float | `0.99` | Discount factor for future rewards (0–1; higher = more far-sighted). |
| `gae_lambda` | float | `0.95` | GAE bias/variance trade-off (0–1; higher = lower bias, more variance). |
| `clip_eps` | float | `0.2` | PPO clipping range on the probability ratio — limits how far the policy moves per update. |
| `value_coef` | float | `0.5` | Weight of the value (critic) loss in the total loss. |
| `entropy_coef` | float | `0.01` | Weight of the entropy bonus — higher keeps the policy more exploratory. |
| `max_grad_norm` | float | `0.5` | Gradient-norm clipping (training stability). |
| `update_epochs` | int | `4` | How many optimization passes over the collected buffer each epoch. |
| `self_play` | bool | `true` | `true` = negamax self-play advantage (a move that lets the opponent win is penalized). `false` = plain single-agent GAE. |

---

## `rollout:` — self-play data collection

How much experience is gathered each epoch.

| Parameter | Type | Default | What it does |
|-----------|------|---------|--------------|
| `episodes_per_epoch` | int | `8` | Self-play games collected per epoch. Also the parallel batch width — all games step together through one batched forward pass. |
| `max_steps_per_episode` | int | `100` | Maximum plies (half-moves) before a game is truncated. |

---

## `curriculum:` — starting positions

**Optional.** Omit the whole `curriculum:` section (or set `curriculum: null`) and
training starts from the **normal chess position** (32 pieces). Include it to start
from randomised simplified positions (both kings always present) to make early
learning easier.

| Parameter | Type | Default | What it does |
|-----------|------|---------|--------------|
| `name` | str | `"Phase 1: 6 pieces"` | Label for the phase (recorded only). |
| `max_pieces_per_side` | int | `6` | Total pieces per side, **including the king**. |
| `allow_major` | bool | `false` | Allow queens and rooks to be placed. |
| `allow_minor` | bool | `true` | Allow bishops and knights. |
| `allow_pawns` | bool | `true` | Allow pawns. |

> Positions are **randomised**, not the standard opening. For denser, harder
> positions raise `max_pieces_per_side` and enable all piece types (e.g. `16` with
> `allow_major: true`) — you get a full but random board, not the exact chess start.

---

## `eval:` — evaluation during training

Periodically benchmarks the agent against the built-in baselines (random, material)
and logs win-rate + Elo. This is what draws the Elo-vs-random curve.

| Parameter | Type | Default | What it does |
|-----------|------|---------|--------------|
| `enabled` | bool | `true` | Turn periodic evaluation on/off. On CPU it's the slow part — disable it for quick runs. |
| `every` | int | `5` | Evaluate every N epochs. |
| `games` | int | `20` | Games played per opponent (colors alternate to cancel first-move bias). |
| `max_plies` | int | `200` | Ply cap per evaluation game. |

---

## `reward:` — reward shaping

The self-play reward, from the mover's point of view, applied each ply. Two ways to
set it:

**A named preset** from [`rewards.yaml`](rewards.yaml):

```yaml
reward: aggressive
```

**Or inline terms:**

```yaml
reward:
  material: 1.0
  check: 0.05
  step_penalty: 0.002
```

| Term | Type | Default | What it does |
|------|------|---------|--------------|
| `material` | float | `1.0` | Multiplier on the **value of the captured piece**. Piece values: pawn `0.01`, knight/bishop `0.03`, rook `0.05`, queen `0.09`, king `1.0`. |
| `king_capture` | float | `0.0` | Extra bonus added when the move captures the king (wins the game). |
| `check` | float | `0.0` | Bonus if the move leaves the opponent in check. |
| `step_penalty` | float | `0.0` | Subtracted every ply — rewards decisive (shorter) games. |

The default (`material: 1.0`, others `0`) is pure material. Presets live in
`rewards.yaml`; add your own there and reference them by name. The resolved reward is
saved in `run.json`.

---

## Resuming a run

Set `resume_from_run` to a run id (see `kaisparov runs list`) to **continue** it:

```yaml
resume_from_run: 20260901-120000_gnn_v1
title: "continue baseline, lower LR"
epochs: 40
ppo:
  learning_rate: 0.0005
```

- **The architecture is inherited** — you do *not* repeat `model` / `hidden_dim`
  (they must match the checkpoint). Any field you omit is inherited from the parent;
  list only what changes.
- It reloads the parent's **latest** checkpoint plus its **optimizer + RNG state**, so
  training continues exactly where it stopped.
- Epoch numbering continues from the parent, and the runs are linked
  (`kaisparov runs lineage <id>`).

---

## CLI overrides

Flags override the file (handy for quick tweaks without editing YAML):

```bash
kaisparov train --config config/default.yaml --epochs 100 --seed 1 --cpu
```

| Flag | Overrides |
|------|-----------|
| `--epochs N` | `epochs` |
| `--seed N` | `seed` |
| `--episodes N` | `rollout.episodes_per_epoch` |
| `--title "..."` | `title` |
| `--description "..."` | `description` |
| `--cpu` | forces `device: cpu` |
| `--model NAME` | `model` (fresh runs only — locked on resume) |
| `--hidden-dim N` | `hidden_dim` (fresh runs only — locked on resume) |
| `--resume <run_id>` | same as `resume_from_run` |
| `--runs-dir DIR` | `runs_dir` (also where `--resume` looks) |

---

## Examples

**Minimal** — everything else defaults:

```yaml
title: "quick test"
epochs: 10
```

**A real experiment:**

```yaml
model: gnn_v1
hidden_dim: 16
epochs: 200
seed: 0
device: auto
checkpoint_every: 20
title: "wider net, aggressive reward"
description: "hidden_dim 16 + check bonus; does the Elo curve climb faster?"

ppo:
  learning_rate: 0.0005
  entropy_coef: 0.02

rollout:
  episodes_per_epoch: 16
  max_steps_per_episode: 120

curriculum:
  max_pieces_per_side: 8
  allow_major: true

eval:
  every: 10
  games: 30

reward: aggressive
```

**Fast CPU sanity run:** see [`experiments/smoke.yaml`](experiments/smoke.yaml).
**Resuming:** see [`experiments/resume_example.yaml`](experiments/resume_example.yaml).
