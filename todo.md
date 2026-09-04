# TODO / Future work

Ideas deferred on purpose. Not bugs — things worth doing when the time is right.

## Engine performance — bitboard representation

**When:** only once a tree search (MCTS / AlphaZero-style) is added, where move
generation is called millions of times. Today the training bottleneck is the GNN
forward/backward on CPU, not the engine, so this has near-zero payoff yet.

**Idea:** replace the `list[list[Piece | None]]` board of Python objects with an
integer/bitboard representation:

- Board state as bitboards (one 64-bit int per piece type × color), or a flat
  `numpy` int8 array of 64 squares.
- Move generation via precomputed masks — knight/king lookup tables (already the
  shape of `core/attacks.py`), and *magic bitboards* for sliding pieces.
- Expect orders-of-magnitude faster `pseudo_legal_moves` / `all_moves`.

**Cost / caveats:**
- Significant rewrite of `core/board.py` + `core/movegen.py`; keep `core/coords.py`
  as the geometry source of truth.
- Ripples into `models/rgcn/processor.py` (`graphify` reads `grid[col][row]`) and
  the pygame UI — introduce an accessor so callers don't touch the raw representation.
- Keep the perft tests green (they're the safety net for any engine rewrite).

**Prereq to justify it:** profile first (`cProfile` on a rollout) to confirm the
engine is actually the bottleneck.

## Done since

- **Opponent pool (league)** — `rollout.opponent: pool`, trains against frozen past
  snapshots (`training/opponents.py`, `rollout_vs.py`).
- **MinimaxAgent** — negamax alpha-beta on the critic, ordered by the actor
  (`agents/minimax_agent.py`); usable via `--minimax-depth`. Natural bridge to a full
  MCTS / AlphaZero (expert-iteration) setup, which remains the next big step.

## Other deferred ideas

- **AlphaZero-style expert iteration** — use the search (MinimaxAgent / a future MCTS)
  to pick moves for both sides and distill the policy toward the search's choice + value
  regression on outcomes. Replaces PPO's on-policy self-play for much stronger play.
- **`gnn_v2` backend** — a second architecture (GAT / message-passing / graph
  transformer) to exercise the multi-backend design and start real comparisons.
- **Batched evaluation** — the arena plays games sequentially; parallelising neural
  games with batched inference would speed up periodic eval during training.
- **Longer training runs** — actually beat the baselines and produce the Elo curve
  for the README (deferred: currently finishing the scaffolding, not training).
- **Interpretability notebooks** — node-embedding probes, per-relation message flow,
  attention analysis (the project's main research goal; see `notebooks/`).
