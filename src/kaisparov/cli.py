"""Unified command-line interface: ``kaisparov <command> [options]``.

Commands (each has its own ``-h``):

    kaisparov train [--config ...] [--epochs N] [--cpu] ...
    kaisparov eval  [--games N] [--model gnn_v1 --checkpoint ...] ...
    kaisparov play  [--vs-ai --checkpoint ...] ...
    kaisparov runs  list | show <id> | best [--metric ...]

Sub-commands are imported lazily, so ``kaisparov runs`` stays fast and never
loads torch or pygame.
"""

from __future__ import annotations

import sys

COMMANDS = ("train", "eval", "play", "runs")

_USAGE = "usage: kaisparov {train|eval|play|runs} [options]  (try: kaisparov <command> -h)"


def _dispatch(command: str, rest: list[str]) -> None:
    if command == "train":
        from kaisparov.train import main
    elif command == "eval":
        from kaisparov.eval.arena import main
    elif command == "play":
        from kaisparov.play import main
    else:  # runs
        from kaisparov.tracking.registry import main
    main(rest)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"Unknown command '{command}'.\n{_USAGE}")
        raise SystemExit(2)

    _dispatch(command, rest)


if __name__ == "__main__":
    main()
