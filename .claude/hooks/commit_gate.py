"""PreToolUse hook: run the CI checks before Claude commits, and block the commit if one fails.

The steps are those of .github/workflows/ci.yml, fastest first, stopping at the first
failure; Claude gets the tail of its output. A change that touches only Markdown skips
them, and so does a commit with nothing changed since HEAD (HEAD passed already).
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")).resolve()
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


def changed_paths() -> list[str]:
    def git(*args: str) -> list[str]:
        out = subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True)
        return out.stdout.split()

    return git("diff", "--name-only", "HEAD") + git("ls-files", "--others", "--exclude-standard")


def main() -> int:
    sys.stdin.read()  # the tool call; the `if` rule in settings.json already matched it
    changed = changed_paths()
    if all(path.endswith(".md") for path in changed):
        return 0
    for name, command in STEPS:
        run = subprocess.run(
            command,
            cwd=PROJECT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if run.returncode != 0:
            output = (run.stdout + run.stderr).strip().splitlines()
            print(
                f"Commit blocked: `{name}` failed (the same check runs in CI).\n"
                + "\n".join(output[-TAIL:]),
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
