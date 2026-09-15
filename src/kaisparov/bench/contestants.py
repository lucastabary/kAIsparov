"""Contestants: who sits the benchmark, named by a one-line spec.

A :class:`Contestant` is a recipe for a :class:`~kaisparov.agents.base.Policy` — it
builds one on demand, so the runner can hand every contestant a freshly seeded policy
and the neural ones pay for loading their weights once. The spec string is how the
command line and a play-out's ``opponent`` name one::

    random                          the baselines ...
    material+safe                   ... here refusing moves that hang their own king
    run:20260903-155710_rgcn        a tracked run, latest checkpoint
    run:20260903-155710_rgcn@best   ... its best checkpoint (or @latest, @40 for epoch 40)
    ckpt:path/to/weights.pth        a raw checkpoint (backend rgcn, width inferred)
    run:<id>@best+minimax2          any neural source wrapped in a depth-2 search
    v2=run:<id>@best                ``label=`` renames it in the reports

Modifiers after ``+``: ``safe`` (avoid king suicide), ``sample`` (sample the policy
instead of taking its argmax), ``minimax<N>`` (search ``N`` plies with the critic).

Sources register by prefix (:class:`Contestant` subclasses with ``prefixes``), so a
new kind of player — an external engine, a batch of checkpoints — is one class.
Torch is only imported when a neural contestant is actually built.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from kaisparov.agents.base import Policy
from kaisparov.insights import Analyzer

_MINIMAX = re.compile(r"minimax(\d+)")


@dataclass(frozen=True)
class Modifiers:
    safe: bool = False
    sample: bool = False
    search_depth: int = 0

    @classmethod
    def parse(cls, words: list[str]) -> Modifiers:
        safe = sample = False
        depth = 0
        for word in words:
            if word == "safe":
                safe = True
            elif word == "sample":
                sample = True
            elif match := _MINIMAX.fullmatch(word):
                depth = int(match.group(1))
            else:
                raise ValueError(f"unknown contestant modifier {word!r} (safe, sample, minimax<N>)")
        return cls(safe=safe, sample=sample, search_depth=depth)

    def suffix(self) -> str:
        words = (["safe"] if self.safe else []) + (["sample"] if self.sample else [])
        if self.search_depth:
            words.append(f"minimax{self.search_depth}")
        return "".join(f"+{word}" for word in words)


class Contestant(ABC):
    """A named recipe for a policy. Subclasses list the spec ``prefixes`` they parse."""

    prefixes: ClassVar[tuple[str, ...]] = ()
    _sources: ClassVar[dict[str, type[Contestant]]] = {}

    name: str
    spec: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for prefix in cls.__dict__.get("prefixes", ()):
            if prefix in Contestant._sources:
                raise TypeError(f"contestant prefix {prefix!r} is already registered")
            Contestant._sources[prefix] = cls

    @abstractmethod
    def build(self, seed: int) -> Policy:
        """A fresh policy; building twice with one seed must give the same player."""

    def analyzer(self) -> Analyzer | None:
        """How the contestant explains a position, for the probe tasks. Default: none."""
        return None

    @classmethod
    @abstractmethod
    def from_parts(
        cls, prefix: str, body: str, modifiers: Modifiers, *, runs_dir: str | Path
    ) -> Contestant:
        """Build from a parsed spec: ``prefix[:body]`` plus its modifiers."""

    @staticmethod
    def parse(spec: str, *, runs_dir: str | Path = "runs") -> Contestant:
        text = spec.strip()
        label = None
        head = text.split(":", 1)[0]
        if "=" in head:
            label, text = (part.strip() for part in text.split("=", 1))

        source, *words = text.split("+")
        prefix, _, body = source.partition(":")
        if prefix not in Contestant._sources:
            known = ", ".join(sorted(Contestant._sources))
            raise ValueError(f"unknown contestant {spec!r} (known sources: {known})")
        contestant = Contestant._sources[prefix].from_parts(
            prefix, body, Modifiers.parse(words), runs_dir=runs_dir
        )
        if label:
            contestant.name = label
        contestant.spec = spec
        return contestant


# --------------------------------------------------------------------- baselines


@dataclass
class BaselineContestant(Contestant):
    """The torch-free reference players."""

    prefixes: ClassVar[tuple[str, ...]] = ("random", "material")

    kind: str
    modifiers: Modifiers = Modifiers()
    name: str = ""
    spec: str = ""

    def __post_init__(self) -> None:
        if self.modifiers.search_depth or self.modifiers.sample:
            raise ValueError(f"{self.kind}: only the 'safe' modifier applies to a baseline")
        self.name = self.name or self.kind + self.modifiers.suffix()
        self.spec = self.spec or self.name

    @classmethod
    def from_parts(
        cls, prefix: str, body: str, modifiers: Modifiers, *, runs_dir: str | Path
    ) -> BaselineContestant:
        if body:
            raise ValueError(f"baseline {prefix!r} takes no argument (got {body!r})")
        return cls(kind=prefix, modifiers=modifiers)

    def build(self, seed: int) -> Policy:
        from kaisparov.agents.material_agent import MaterialAgent
        from kaisparov.agents.random_agent import RandomAgent

        agent_class = RandomAgent if self.kind == "random" else MaterialAgent
        return agent_class(seed=seed, avoid_king_suicide=self.modifiers.safe)

    def analyzer(self) -> Analyzer | None:
        """Material explains itself by counting; random has nothing to explain."""
        if self.kind != "material":
            return None
        from kaisparov.bench.analyzers import MaterialAnalyzer

        return MaterialAnalyzer()


# ------------------------------------------------------------------------ neural


@dataclass
class NeuralContestant(Contestant):
    """A trained backend, from a tracked run or a raw checkpoint file."""

    prefixes: ClassVar[tuple[str, ...]] = ("run", "ckpt")

    checkpoint: Path
    model: str | None = None  # backend name; None = the default backend
    hidden_dim: int | None = None  # None = read it off the checkpoint
    modifiers: Modifiers = Modifiers()
    name: str = ""
    spec: str = ""
    _loaded: Any = field(default=None, init=False, repr=False)
    _policy: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = self.name or Path(self.checkpoint).stem + self.modifiers.suffix()
        self.spec = self.spec or self.name

    @classmethod
    def from_parts(
        cls, prefix: str, body: str, modifiers: Modifiers, *, runs_dir: str | Path
    ) -> NeuralContestant:
        if not body:
            raise ValueError(f"{prefix}: needs an argument, e.g. {prefix}:<id>")
        if prefix == "ckpt":
            return cls(checkpoint=Path(body), modifiers=modifiers)

        from kaisparov.tracking.registry import Registry

        run_id, _, which = body.partition("@")
        registry = Registry(runs_dir)
        run = registry.get(run_id)
        selector: int | str = int(which) if which.isdigit() else (which or "latest")
        config = run.get("config") or {}
        return cls(
            checkpoint=registry.resolve_checkpoint(run_id, selector),
            model=run.get("model") or config.get("model"),
            hidden_dim=config.get("hidden_dim"),
            modifiers=modifiers,
            name=f"{run_id}@{selector}{modifiers.suffix()}",
        )

    def _load(self):
        """``(model, processor)``, loaded on first use and shared by every build."""
        if self._loaded is None:
            import torch

            from kaisparov.models.factory import infer_hidden_dim, load_backend_spec

            if not Path(self.checkpoint).exists():
                raise FileNotFoundError(f"{self.name}: no checkpoint at {self.checkpoint}")
            device = torch.device("cpu")
            spec = load_backend_spec(self.model)
            state_dict = torch.load(self.checkpoint, map_location=device, weights_only=True)
            hidden_dim = self.hidden_dim or infer_hidden_dim(state_dict) or 8
            model = spec.model_class.create_agent(device=device, hidden_dim=hidden_dim)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{self.name}: {self.checkpoint} does not fit the current {spec.name} "
                    "backend — most likely trained before a change to its input features"
                ) from exc
            model.eval()
            self._loaded = (model, spec.processor_class())
        return self._loaded

    def build(self, seed: int) -> Policy:
        import torch

        # The agents are stateless — only +sample draws from torch's global generator,
        # which reseeding covers — so one agent serves every problem.
        torch.manual_seed(seed)
        if self._policy is None:
            model, processor = self._load()
            if self.modifiers.search_depth:
                from kaisparov.agents.minimax_agent import MinimaxAgent

                self._policy = MinimaxAgent(
                    model,
                    processor,
                    depth=self.modifiers.search_depth,
                    avoid_king_suicide=self.modifiers.safe,
                )
            else:
                from kaisparov.agents.neural_agent import NeuralAgent

                self._policy = NeuralAgent(
                    model,
                    processor,
                    deterministic=not self.modifiers.sample,
                    avoid_king_suicide=self.modifiers.safe,
                )
        return self._policy

    def analyzer(self) -> Analyzer:
        """The raw network's view (value head, full policy ranking), search or not."""
        from kaisparov.agents.neural_analyzer import NeuralAnalyzer

        model, processor = self._load()
        return NeuralAnalyzer(model, processor, top_k=4096)


__all__ = ["BaselineContestant", "Contestant", "Modifiers", "NeuralContestant"]
