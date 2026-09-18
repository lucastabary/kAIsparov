# TODO / Future work

Ideas deferred on purpose. Not bugs — things worth doing when the time is right.

## Engine performance

**Status: largely settled.** The hand-written engine was replaced by a facade over
python-chess (`core/game.py`), which already holds the position as bitboards and does
legal move generation with real pin detection — measurably faster than filtering
pseudo-legal moves with a make/unmake per candidate. The feature pipeline reads those
bitboards directly (`models/rgcn/processor.graphify_batch`), which is where the
throughput actually was.

**What is left, and when:** only once a tree search (MCTS / AlphaZero-style) makes
move generation the bottleneck. python-chess is pure Python, so at that point the
options are a C/Rust-backed engine behind the same facade, or caching move lists
across a search. Profile first (`cProfile` on a rollout) — today the bottleneck is
the GNN forward/backward on CPU.

## Underpromotion in the action space

The engine generates all four promotions, but the policy is a distribution over
`(source, dest)` pairs, so playing a promotion edge queens and underpromotion is
unreachable. The fix fits the architecture: make the move key
`(src * 64 + dst) * 4 + promo_idx` in `aggregate_edge_logits_to_moves`, and add three
edge relations (promote-to-rook/bishop/knight) to the static graph, with the queen
promotion staying on the existing pawn edge as `promo_idx = 0`.

Two things to get right: those relations receive gradient only on the rare ply that
promotes, so initialise them from the pawn relation rather than at random, and add a
learnable bias per promotion piece as a cheap prior. And `num_relations` going 6 → 9
changes the shape of `RGCNConv.weight`, so existing checkpoints need a migration that
copies relations 0-5 and seeds 6-8.

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
