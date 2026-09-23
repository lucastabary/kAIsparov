---
name: new-backend
description: Add a new GNN model backend to kAIsparov (models/<arch>/ with BACKEND_SPEC, model, processor, README), registered in the factory, tested and documented. Use when the user wants to try a new architecture (GAT, graph transformer, ...).
argument-hint: "<architecture, e.g. 'gat'>"
---

# /new-backend — a new architecture under `models/`

Architecture: `$ARGUMENTS`

The project exists to compare architectures on the same task, so a backend must
change **only what it studies**. `models/shared_rgcn/` is the template: it isolates
one question (weight tying) and reuses everything else.

## 0. Agree on the question first

State in two sentences what the backend changes compared to `rgcn` and what it keeps
identical (graph encoding, node features, heads, PPO training), and get the user's
agreement. Name the folder after the architecture (`gat`, `graph_transformer`, …).

## 1. `src/kaisparov/models/<arch>/`

- `model.py`: a `BaseModel` subclass with `MODEL_NAME = "<arch>"`, an `__init__`
  that takes `hidden_dim` and `features=DEFAULT_FEATURES` (from `models/features.py`),
  and a `forward(data)` returning `(action_scores over edges, value)` like `rgcn`.
- `processor.py`: when the encoding is unchanged, subclass `RGCNProcessor` with no
  body, like `shared_rgcn/processor.py`: copying it would let the two drift apart.
- `__init__.py`: `PROCESSOR_CLASS` and `BACKEND_SPEC = BackendSpec(...)`, copied from
  `shared_rgcn/__init__.py`.
- `README.md`: the idea, the architecture, what it keeps from `rgcn`, the parameter
  count at the default width, and the question it answers (see `rgcn/README.md`,
  `shared_rgcn/README.md`).

## 2. Registration: one line

`MODEL_MODULES` in `models/factory.py`. Build and load only through
`factory.build_agent` / `factory.load_agent`. If the state dict names its layers
differently, extend `_HIDDEN_DIM_KEYS` / `_INPUT_DIM_KEYS` there only if it has to
read runs that do not record their architecture (new runs record it).

## 3. Rules that must hold

- **Node features are frozen by name**: use a set from `models/features.py`; new
  inputs mean a *new* set name, pinned in `tests/test_features.py`.
- A new **shape-changing hyper-parameter** goes into `Architecture`
  (`models/architecture.py`), not into each loader.
- `lint-imports` must pass: python-chess stays in `core/`, and the new package
  reaches the board only through the processor.

## 4. Tests

- Add `<arch>` to the `parametrize` list in `tests/test_architecture.py`.
- `tests/test_<arch>.py` on the model of `tests/test_shared_rgcn.py`: the spec is
  registered, a forward pass has the right shapes and decodes to a legal move, a
  build → save → `load_agent` round trip gives the same outputs, and one test for what
  makes this architecture *itself* (the claim its README makes).

## 5. Docs and config

- `config/README.md`: the `model` row lists the backends.
- `README.md`: the backends in the intro and in the layout tree, and a paragraph in
  the architectures section with the link to the model README.
- `CLAUDE.md` if the layout table changes; a `CHANGELOG.md` entry under `[Unreleased]`.
- An experiment config in `config/experiments/` if the user wants one. Check it with
  a 1-epoch run at most: training is CPU-bound here, **never start a long run unasked**.

Then `/ship`, if the user wants it.
