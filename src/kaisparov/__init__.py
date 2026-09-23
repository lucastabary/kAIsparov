"""kAIsparov — a GNN that learns to play chess via PPO self-play."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kaisparov")  # set from the git tag at install time
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"
