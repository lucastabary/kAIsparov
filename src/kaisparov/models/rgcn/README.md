# `rgcn` — Relational GCN actor–critic

The first model backend of kAIsparov. This document explains **how it works** and
**the idea behind it**. (Every model backend ships a README like this — it's the
template for describing a model in this project.)

## The idea

Chess positions have a rich *relational* structure: a rook relates to every square
on its rank and file, a knight to its L-shaped targets, and so on. Convolutional or
flat encodings blur that structure. `rgcn` keeps it explicit:

> **The board is a graph. Squares are nodes. A move is an edge. Each edge carries the
> *type of piece motion* it represents (knight / rook / bishop / king / pawn) as a
> relation.**

A **Relational Graph Convolutional Network** (R-GCN) then reasons over that graph
with **per-relation** weights — knight-relations and rook-relations are transformed
differently — which is exactly the inductive bias chess asks for. Because moves *are*
edges, the policy scores edges directly, so the action space is native to the graph.

It is an **actor–critic**: one head scores moves (the policy), another scores the
position (the value), sharing the same node embeddings.

## Board → graph (`processor.py`)

`RGCNProcessor.graphify(game)` turns a `ChessGame` into a PyTorch-Geometric `Data`:

**Nodes** — the 64 squares (node index = `row * 8 + col`, see `core/coords.py`).

**Node features** — a 12-dim one-hot per square, **relative to the side to move**:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6–11 |
|-------|---|---|---|---|---|---|------|
| means | ally king | ally queen | ally bishop | ally rook | ally knight | ally pawn | same six, but **enemy** |

Encoding pieces as *ally/enemy* rather than *white/black* means the network always
sees the position from the mover's perspective — no separate side-to-move plane, and
White and Black share weights.

**Edges** — a **static** graph (identical for every position), built once by
`create_static_full_chess_graph()`. Every geometrically possible piece motion on an
empty board is an edge, tagged with a **relation type** (`edge_type`):

| relation | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| motion | knight | rook (rank/file rays) | bishop (diagonal rays) | king (1 step) | white pawn (push + double + captures) | black pawn |
| # edges | 336 | 896 | 560 | 420 | 162 | 162 |

Total **E = 2536** directed edges. The graph is static; what changes per position is
only the node features (and which edges are *legal*, applied later as a mask).

> **Representable moves.** Because pawn-capture edges point to the diagonal squares
> regardless of occupancy, **en passant is representable** (it's a legal diagonal
> pawn move). **Castling is *not*** — the king only has 1-step edges, so `rgcn`
> cannot emit a castling move even though the engine supports it. A future backend
> could add castling edges.

## The network (`model.py`)

`ChessRGCN` — the shared backbone:

- 4 × `RGCNConv` layers (`in=12 → hidden → hidden → hidden → hidden`), each with
  **6 relation-specific weight sets**.
- ReLU + **residual connections** on the two middle layers.
- Output: a `hidden`-dim embedding per node.

Two heads on top (`RGCNModel`):

- **Actor** — for each edge, concatenate its endpoint embeddings `[2·hidden]` and pass
  through an MLP → **one score per edge** → `action_scores` of shape `[E]`.
- **Critic** — pool all node embeddings with `AttentionalAggregation` (a learned,
  attention-weighted sum) → an MLP → **one scalar** `state_value` per board `[B]`.

`model(data)` returns `(action_scores [E], state_value [B])`. Batches of positions are
handled natively by PyG (`Batch`), which the rollout and PPO update rely on.

## From scores to a move (`process_output`)

1. Build a boolean **legal mask** over the `E` edges (`get_legal_mask`) — which static
   edges are legal moves in *this* position for the side to move.
2. `masked_logits = action_scores.masked_fill(~legal_mask, -inf)`.
3. `Categorical(logits=masked_logits)` → **sample** (training) or **argmax** (eval).
4. Decode the chosen edge back to a `(source, dest)` move.

The masking is what ties the fixed graph to the live position: the network proposes
scores for all conceivable edges, and only the legal ones can be chosen.

## Training (`__init__.py` → `BACKEND_SPEC`)

`rgcn` plugs into the shared training stack via its `BACKEND_SPEC`:

| field | value |
|-------|-------|
| `model_class` | `RGCNModel` (actor–critic) |
| `processor_class` | `RGCNProcessor` (graphify + decode) |
| `buffer_class` | `PPOBuffer` |
| `collect_data` | self-play rollout (`training/rollout.py`) |
| `train_one_epoch` | PPO update (`training/ppo.py`) |

It is trained by **PPO self-play** with negamax advantage (see `docs.md` §6). The
reward is configurable (`config/rewards.yaml`); the default is material captured.

## Sizes

| `hidden_dim` | parameters |
|--------------|------------|
| 8 (default) | 2,355 |
| 16 | 7,907 |

Deliberately tiny — this is a research testbed, and a small model is easier to probe
for **interpretability** (the project's main goal): node embeddings, per-relation
message flow, and edge scores are all small enough to inspect directly (see
`notebooks/`).

## Strengths & limitations

- ✅ Relational bias matched to chess; White/Black weight sharing; native edge actions.
- ✅ Small and interpretable.
- ⚠️ **Cannot castle** (no castling edges).
- ⚠️ Minimal 12-dim features (piece type + side only) — no positional/rank features,
  no move history.
- ⚠️ The static full-move graph is dense (`E = 2536`); most edges are illegal in any
  given position and get masked out.

## Files

- `model.py` — `ChessRGCN` backbone + `RGCNModel` (heads, `forward`).
- `processor.py` — `RGCNProcessor` (`graphify`, `process_output`), `get_legal_mask`,
  `compute_reward`.
- `__init__.py` — assembles `BACKEND_SPEC`.
