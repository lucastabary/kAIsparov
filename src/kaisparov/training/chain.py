"""Chain configs: a YAML file that names a *suite* of training configs.

A normal config describes one run. A **chain config** describes a recipe: it holds
nothing but a ``stages:`` list of other config files, so a multi-phase curriculum
gets one memorable entry point instead of a long command line::

    # config/experiments/high_entropy_all.yaml
    title: "high_entropy v4 — full curriculum"
    stages:
      - high_entropy_phase1-1.yaml
      - high_entropy_phase1-2.yaml
      - high_entropy_phase2.yaml
      - high_entropy_phase3.yaml

    kaisparov train --config config/experiments/high_entropy_all.yaml

which is exactly equivalent to passing the four files by hand: each stage after the
first resumes from the run the previous one produced (see :mod:`kaisparov.train`).

Rules:

- Entries are resolved **relative to the chain file** first, then to the working
  directory, so a recipe can sit next to its stages and still be run from anywhere.
- An entry may be a glob (``high_entropy_phase*.yaml``); matches are expanded in
  sorted order. Prefer an explicit list when the order matters and doesn't sort.
- A stage may itself be a chain — it is expanded in place (loops are refused).
- A chain file carries no training settings: those belong in the stage files.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import yaml

CHAIN_KEY = "stages"
# The only other keys a chain file may carry: they document the recipe, they don't
# configure a run (a chain never becomes a TrainConfig).
_DOC_KEYS = frozenset({"title", "description", "notes"})
_GLOB_CHARS = "*?["


def read_config_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML config as a mapping (empty file -> ``{}``)."""
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"'{path}': a config must be a YAML mapping, got {type(data).__name__}.")
    return data


def is_chain_config(data: dict[str, Any]) -> bool:
    """True if this raw config is an entry point listing other configs."""
    return CHAIN_KEY in data


def ensure_not_chain(path: str | Path, data: dict[str, Any]) -> None:
    """Refuse a chain file where a single training config is expected."""
    if is_chain_config(data):
        raise SystemExit(
            f"'{path}' is a chain config (it only lists `{CHAIN_KEY}:`), not a training "
            "config. Run it with `kaisparov train --config <that file>`, which expands "
            "it into its stages."
        )


def _check_chain_keys(path: Path, data: dict[str, Any]) -> None:
    extra = sorted(set(data) - _DOC_KEYS - {CHAIN_KEY})
    if extra:
        raise SystemExit(
            f"'{path}' is a chain config, so it may only hold `{CHAIN_KEY}:` plus "
            f"{sorted(_DOC_KEYS)}; found {extra}. Per-stage settings belong in the "
            "stage files it points to."
        )


def _resolve_entry(entry: str, base: Path, source: Path) -> list[Path]:
    """Resolve one ``stages:`` entry — a path or a glob — to existing files."""
    if any(char in entry for char in _GLOB_CHARS):
        matches = sorted(glob.glob(str(base / entry))) or sorted(glob.glob(entry))
        if not matches:
            raise SystemExit(f"'{source}': pattern '{entry}' matches no config file.")
        return [Path(match) for match in matches]
    for candidate in (base / entry, Path(entry)):
        if candidate.is_file():
            return [candidate]
    raise SystemExit(f"'{source}': stage '{entry}' not found (looked in {base} and {Path.cwd()}).")


def _expand_one(path: Path, label: str, stack: tuple[Path, ...]) -> list[str]:
    """Expand ``path`` to the stages it stands for (itself, if it isn't a chain)."""
    if not path.is_file():
        # Not ours to report: pass it through and let the stage loader raise where it
        # normally would. (Stages named *inside* a chain are checked when resolved.)
        return [label]
    data = read_config_mapping(path)
    if not is_chain_config(data):
        return [label]

    resolved = path.resolve()
    if resolved in stack:
        loop = " -> ".join(p.name for p in (*stack, resolved))
        raise SystemExit(f"Config chain loops back on itself: {loop}")
    _check_chain_keys(path, data)

    entries = data[CHAIN_KEY]
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"'{path}': `{CHAIN_KEY}:` must be a non-empty list of config paths.")

    stages: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            raise SystemExit(f"'{path}': every `{CHAIN_KEY}:` entry must be a path, got {entry!r}.")
        for stage in _resolve_entry(entry, path.parent, path):
            stages.extend(_expand_one(stage, str(stage), (*stack, resolved)))
    return stages


def expand_config_chain(paths: list[str]) -> list[str]:
    """Expand any chain config in ``paths`` into the stages it names.

    Plain configs pass through untouched (and keep the spelling you typed), so this
    is a no-op for the usual ``--config one.yaml`` and for a hand-written chain of
    several files.
    """
    stages: list[str] = []
    for path in paths:
        stages.extend(_expand_one(Path(path), path, ()))
    return stages
