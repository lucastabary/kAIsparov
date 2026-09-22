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

**Node features** — a named set from `models/features.py`, chosen per run by the
`features:` config key and shared with every backend. Each is a vector per square,
**relative to the side to move**:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6–11 | 12 | 13 |
|-------|---|---|---|---|---|---|------|----|----|
| means | ally king | ally queen | ally bishop | ally rook | ally knight | ally pawn | same six, but **enemy** | attacked by enemy | defended by ally |

- `pieces` (12 dims, indices 0–11) — the default for new runs.
- `pieces_control` (14 dims) — adds the two control flags, 12 and 13. Runs from
  2026-09-04 to the switch to `pieces` trained on it.

The model sizes its input layer from the set (`RGCNModel(features=...)`), and the
processor encodes with the same one, so the two cannot disagree.

Encoding pieces as *ally/enemy* rather than *white/black* means the network always
sees the position from the mover's perspective — no separate side-to-move plane, and
White and Black share weights.

With `pieces_control`, indices 12–13 are a **blocking-aware control map**: the batched path computes them
for every position at once with the vectorised Kogge-Stone fills in
`core/bitboard_batch.py`, packed straight from python-chess's own bitboards
(`core/rules.attacked_squares` is the single-position reference the tests compare
against):
whether each square is attacked by the opponent / defended by the mover, computed
from the *actual* position (sliders stop at the first blocker; pawns control their
diagonals). This is deliberate. The static edge set (below) encodes moves *on an
empty board*, so message passing alone cannot tell a real check from one blocked by
an intervening piece — the net could learn to attack (a 1-hop, edge-local pattern:
"my piece → enemy king") but never to perceive its *own* king in check, and so never
learned to defend it. Feature 12 on the ally king's square is exactly "in check";
on empty squares it marks unsafe destinations (e.g. a king's escape squares). The
legal mask already does this blocking-aware work for the mover's own moves, but
nothing exposed the *opponent's* threats until these two features.

**Edges** — a **static** graph (identical for every position), built once by
`create_static_full_chess_graph()` (`graph.py`). Every geometrically possible piece motion on an
empty board is an edge, tagged with a **relation type** (`edge_type`):

| relation | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| motion | knight | rook (rank/file rays) | bishop (diagonal rays) | king (1 step) | white pawn (push + double + captures) | black pawn |
| # edges | 336 | 896 | 560 | 420 | 162 | 162 |

Total **E = 2536** directed edges. The graph is static; what changes per position is
only the node features (and which edges are *legal*, applied later as a mask).

> **Representable moves.** Because pawn-capture edges point to the diagonal squares
> regardless of occupancy, **en passant is representable** (it's a legal diagonal
> pawn move). **Castling is too**, though by accident rather than by design: a king
> castles from `e1` to `g1` or `c1`, two squares along the rank, which the *rook*
> relation already has as an edge. The move is legal, so the mask keeps it, and
> playing it castles. There is no king-relation or castling-relation edge for it,
> so the network has to learn the move through the rook edge that carries it —
> a dedicated relation would be a reasonable thing to try.
>
> **Underpromotion is not either.** The action space is `(source, dest)`, and the four
> promotions of one pawn push share that pair, so playing the edge queens. Adding it
> is a natural fit for a *relational* model — three extra relations
> (promote-to-rook/bishop/knight) and a promotion component in the move key — and is
> written up in `todo.md`.

## The network (`model.py`)

`ChessRGCN` — the shared backbone:

- 4 × `RGCNConv` layers (`in=features → hidden → hidden → hidden → hidden`, so 12
  inputs with the default `pieces` set, 14 with `pieces_control`), each with
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

1. Build a boolean **legal mask** over the `E` edges (`legal_mask`) — which static
   edges are legal moves in *this* position for the side to move.
2. **Combine the edges that mean the same move.** One `(source, dest)` pair can be
   several edges: a one-square king step is also a rook or bishop edge, while a knight
   jump is a single edge. `aggregate_edge_logits_to_moves` sums each move's probability
   mass across its edges (a `logsumexp` over the group), giving one logit per *move*.
   Without it a move spread over three edges has its probability split three ways, and
   a greedy `argmax` is biased against king moves — exactly the moves that escape check.
3. `Categorical(logits=move_logits)` → **sample** (training) or **argmax** (eval).
4. Decode the chosen move key `src * 64 + dst` back to `(source, dest)` (a pawn reaching
   the last rank queens — see the note on underpromotion above).

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
reward is configurable (`config/rewards.yaml`); the default counts material captured
plus what a promotion gains, with a flat bonus for checkmate.

## Sizes

With the default `pieces` features (12 inputs; `pieces_control` adds ~100):

| `hidden_dim` | parameters |
|--------------|------------|
| 8 (default) | 2,355 |
| 16 | 7,907 |
| 64 | 108,419 |

Deliberately tiny — this is a research testbed, and a small model is easier to probe
for **interpretability** (the project's main goal): node embeddings, per-relation
message flow, and edge scores are all small enough to inspect directly (see
`notebooks/`).

## Strengths & limitations

- ✅ Relational bias matched to chess; White/Black weight sharing; native edge actions.
- ✅ Small and interpretable.
- ⚠️ Castling rides on the rook relation's 2-square edge rather than a relation of
  its own (see above) — reachable, but not typed as what it is.
- ⚠️ Compact features (piece type + side, optionally the control flags) — no
  positional/rank features, no move history.
- ⚠️ The static full-move graph is dense (`E = 2536`); most edges are illegal in any
  given position and get masked out.

## Files

- `model.py` — `ChessRGCN` backbone + `RGCNModel` (heads, `forward`).
- `graph.py` — the static edge set, `create_static_full_chess_graph()`, and its 6
  relations.
- `processor.py` — `RGCNProcessor` (`graphify`, `legal_mask`, `process_output`) and
  `get_legal_mask`. Node features come from `models/features.py`.
- `__init__.py` — assembles `BACKEND_SPEC`.
