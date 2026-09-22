"""Neural backends.

Lazy on purpose: ``models.features`` and ``models.architecture`` are torch-free, and
importing them must not pull torch in through this package.
"""

from typing import Any

__all__ = ["load_backend", "load_backend_spec"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from kaisparov.models import factory

        return getattr(factory, name)
    raise AttributeError(f"module 'kaisparov.models' has no attribute {name!r}")
