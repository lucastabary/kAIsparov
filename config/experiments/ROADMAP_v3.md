# Training recipe v3 — "victory is the only thing that matters"

A ground-up restart on a **pure outcome** reward (king capture = ±1, *no* shaping).
The model must learn material value, piece defense and king safety on its own; the
pressure to not blunder comes from the **opponent pool**, not from the reward.

## Why pure outcome + a teaching pool

- Removing `king_safety`/`check`/`step_penalty` removes the asymmetric hand-crafting
  that biased v2 toward grabbing over defending. `step_penalty` is redundant with
  `gamma` (a fast win is already worth more, discounted).
- `material` is dropped too — but a pure-outcome signal is sparse, so it only becomes
  trainable because the pool contains opponents that *actually punish blunders*:
  `MaterialAgent` (captures a hung king/piece) and depth-1 **minimax past-selves** that
  refute one-move mistakes. Search — not reward shaping — injects the defense signal.
- This is deliberately *not* AlphaZero: the actor still proposes actions and the critic
  evaluates states. We only borrow the idea that search is a policy-improvement operator
  on the opponent side. See the distillation backlog below for the AlphaZero-lite path.

## The three phases

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Board | 4 pieces/side, no pawns | 12 pieces/side + pawns | standard start (real game) |
| Goal | capture + defend the king | material + pawns | full play |
| Reward | `king_capture_only` | `king_capture_only` | `king_capture_only` |
| LR | 5e-4 | 3e-4 | 1e-4 |
| entropy_coef | 0.02 | 0.012 | 0.006 |
| Pool tilt | teachers dominant (2:1) | teachers eased (1.5:1) | self-play dominant (1:1.5) |
| minimax depth | 1 | 1 | 1 (2 optional) |
| hidden_dim | **32 (fixed across all stages)** | 32 | 32 |
| epochs | 100 | 180 | 150 |

Config files: `scratch_v3_stage{1,2,3}.yaml`. Stages 2–3 resume the previous stage's
run id (paste it into `resume_from_run`). Architecture and reward are inherited; each
stage overrides curriculum / pool / LR / entropy. North-star metric:
`eval/winrate_vs_material` (best checkpoint is selected on `elo_vs_material`).

Reward preset lives in `config/rewards.yaml` as `king_capture_only`
(`material: 0.0`, `king_capture: 1.0`).

## Backlog — variants to try and compare

- [ ] **Minimax distillation (AlphaZero-lite / expert iteration).** Instead of only
      using the model+minimax as an *opponent*, let the learner **act** with a
      depth-2 minimax and train the policy by **distillation** toward the minimax-chosen
      move (cross-entropy `-log π(a_minimax | s)`), with the value head still regressed
      toward the game outcome. This is the sound way to train on model+minimax actions:
      **do NOT** feed minimax actions through the PPO ratio — the behavior policy isn't
      π, so the importance ratio is biased and blows up (this is the likely source of
      v2's `non-finite loss` skips). Distillation sidesteps the sparse-reward credit
      assignment and injects piece defense directly via search. Expected to help most in
      stages 2–3, where the pure +/-1 signal is sparsest. Build as a variant so it can be
      A/B'd against the plain-PPO v3 baseline above.
- [ ] Try `snapshot_search_depth: 2` in stage 3 (stronger self-play, ~branching-factor
      cost per opponent move) once CPU budget allows.
- [ ] Optional LR anneal *within* a phase rather than the per-phase step schedule.
