# `shared_rgcn` — Weight-tied Relational GCN actor–critic

`rgcn`'s twin, with **one** set of relational weights shared by **all** message-passing
steps. Everything else — board encoding, static graph, heads, PPO training — is
identical, so the two backends isolate a single question: *does each round of message
passing need its own transform?*

## The idea

`rgcn` stacks 4 distinct `RGCNConv` layers: each step of message passing has its own
per-relation MLPs, so the network learns a *different* transform for the 1st, 2nd, 3rd
and 4th hop.

> **`shared_rgcn` unifies those steps: one relational conv is applied repeatedly.**
> The 1st, 2nd, 3rd … hop all use the same per-relation weights.

Message passing stops being a stack of layers and becomes an **iterated operator** —
the same local rule ("how a rook-relation propagates information") applied over and
over, like a recurrent GNN or an unrolled fixed-point iteration.

Why that's a reasonable bias for chess: the relation "these two squares are a rook
move apart" means the same thing at every hop. A shared operator says so explicitly,
which gives:

- **Far fewer parameters** — the backbone's cost no longer scales with depth.
- **Free depth** — `num_steps` is a pure inference-time knob: 2, 4 or 8 rounds of
  propagation, same weights, same checkpoint. Deeper = longer-range reasoning
  (a 4-hop tactic) at no parameter cost.
- **A cleaner interpretability target** — there is *one* per-relation transform to
  inspect instead of four, and you can watch the node embeddings evolve step by step
  under a single operator.

The trade-off is capacity: a tied operator cannot specialise "first look at direct
attacks, then at defenders" across steps the way distinct layers can.

## Board → graph (`processor.py`)

Unchanged from `rgcn`: `SharedRGCNProcessor` **is** `RGCNProcessor` (subclassed for a
backend-specific name). Same 64 square-nodes, same 14-dim ally/enemy + control-flag
node features, same static 2536-edge graph with 6 relation types, same legal masking
and edge→move decoding. See [`models/rgcn/README.md`](../rgcn/README.md) for the full
description — reusing the encoding rather than copying it is what keeps an
`rgcn` vs `shared_rgcn` comparison a comparison *of the network only*.

## The network (`model.py`)

`SharedChessRGCN` — the weight-tied backbone:

1. **Encoder** — `Linear(14 → hidden)` + ReLU. Lifts node features to the working
   width so the shared conv is always a `hidden → hidden` map (this is what makes
   tying possible at all — `rgcn`'s first layer has a different shape from the rest).
2. **Shared step** — one `RGCNConv(hidden → hidden, 6 relations)` applied `num_steps`
   times, each time residually: `x ← ReLU(x + conv(x))`.
3. **Decoder** — `Linear(hidden → hidden)` → the per-node embeddings.

The heads are `rgcn`'s, unchanged:

- **Actor** — concatenate an edge's endpoint embeddings `[2·hidden]` → MLP → one score
  per edge, `action_scores [E]`.
- **Critic** — `AttentionalAggregation` over the node embeddings → MLP → `state_value [B]`.

`num_steps` defaults to **4**, matching `rgcn`'s depth, and is a constructor argument
(`SharedRGCNModel(hidden_dim=..., num_steps=...)`); training and evaluation pass only
`hidden_dim`, so they use the default.

## Sizes

| `hidden_dim` | `shared_rgcn` | `rgcn` (same width) |
|--------------|---------------|---------------------|
| 8 (default)  | 955           | 2,467               |
| 16           | 3,443         | 8,131               |

Parameter counts are **independent of `num_steps`** — 2 steps or 8 steps, same 955
parameters at `hidden_dim=8`.

## Training

Same stack as `rgcn` via `BACKEND_SPEC`: PPO self-play with negamax advantage
(`training/rollout.py` + `training/ppo.py`), `PPOBuffer`, configurable reward.

```bash
kaisparov train --config config/default.yaml --model shared_rgcn   # or `model: shared_rgcn` in the YAML
kaisparov eval  --model shared_rgcn --checkpoint runs/<id>/checkpoints/best.pth
```

## Strengths & limitations

- ✅ ~2.5× fewer parameters than `rgcn` at equal width; depth costs nothing.
- ✅ One relational operator to interpret, applied iteratively.
- ✅ Drop-in comparison against `rgcn` — only the network differs.
- ⚠️ Less capacity: no step-specific specialisation.
- ⚠️ Inherits every `rgcn` representation limitation — **no castling moves** (the king
  has only 1-step edges), compact 14-dim features, dense static graph.

## Files

- `model.py` — `SharedChessRGCN` (tied backbone) + `SharedRGCNModel` (heads, `forward`).
- `processor.py` — `SharedRGCNProcessor`, a re-export of `rgcn`'s encoding.
- `__init__.py` — assembles `BACKEND_SPEC`.
