# Notebooks

Data-science workspace for kAIsparov — analysing runs, model outputs, and above all
**model interpretability** (the project's main research goal).

Install the extras first:

```bash
pip install -e ".[notebooks]"
```

- **`analyze.ipynb`** — load runs from `runs/`, plot learning curves and Elo-vs-baseline
  curves (lineage-aware), and a starter cell that loads a checkpoint and inspects the
  policy's per-move (edge) scores and the critic's value on a position.

Add one notebook per investigation (e.g. `embeddings.ipynb`, `attention.ipynb`,
`piece_representations.ipynb`). Keep them runnable top-to-bottom; heavy outputs are
git-ignored via `.ipynb_checkpoints/`.
