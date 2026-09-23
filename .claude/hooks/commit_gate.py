"""PreToolUse hook: run the CI checks before Claude commits, and block the commit if one fails.

The steps are those of .github/workflows/ci.yml, fastest first, stopping at the first
failure; Claude gets the tail of its output. A change that touches only Markdown skips
them, and so does a commit with nothing changed since HEAD (HEAD passed already).

The checks run on the tree being committed — the git checkout around the shell's
current directory, which is a worktree when several sessions share the repo — with
that tree's `src/` first on the path: the editable install points at the main checkout.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(sys.executable).parent  # console scripts sit next to the interpreter
PY = [sys.executable, "-m"]
STEPS = [
    ("ruff check", [*PY, "ruff", "check", "."]),
    ("ruff format", [*PY, "ruff", "format", "--check", "."]),
    ("mypy", [*PY, "mypy", "src/kaisparov"]),
    ("import contracts", [str(SCRIPTS / "lint-imports")]),
    ("pytest", [*PY, "pytest", "-q", "-p", "no:warnings"]),
]
TAIL = 40  # lines of a failing step's output shown to Claude


def git(tree: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=tree, capture_output=True, text=True).stdout


def tree_being_committed(event: dict) -> Path:
    cwd = event.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR", ".")
    top = git(Path(cwd), "rev-parse", "--show-toplevel").strip()
    return Path(top or cwd).resolve()


def main() -> int:
    tree = tree_being_committed(json.load(sys.stdin))
    changed = git(tree, "diff", "--name-only", "HEAD").split()
    changed += git(tree, "ls-files", "--others", "--exclude-standard").split()
    if all(path.endswith(".md") for path in changed):
        return 0
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tree / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    for name, command in STEPS:
        run = subprocess.run(
            command,
            cwd=tree,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if run.returncode != 0:
            output = (run.stdout + run.stderr).strip().splitlines()
            print(
                f"Commit blocked in {tree}: `{name}` failed (the same check runs in CI).\n"
                + "\n".join(output[-TAIL:]),
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
