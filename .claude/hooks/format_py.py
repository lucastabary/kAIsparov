"""PostToolUse hook: ruff-format a Python file right after Claude edits it.

Also applies ruff's safe fixes, except removing unused imports: mid-change, an import
often lands one edit before the code that uses it. Never blocks; the commit gate
(`commit_gate.py`) is what catches whatever ruff cannot fix.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    event = json.load(sys.stdin)
    path = Path(event.get("tool_input", {}).get("file_path", ""))
    project = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")).resolve()
    if path.suffix not in (".py", ".pyi") or not path.is_file():
        return
    if not path.resolve().is_relative_to(project):
        return
    ruff = [sys.executable, "-m", "ruff"]
    quiet = {"cwd": project, "capture_output": True}
    subprocess.run(
        [*ruff, "check", "--fix", "--unfixable", "F401", "--force-exclude", "-q", str(path)],
        **quiet,
    )
    subprocess.run([*ruff, "format", "--force-exclude", "-q", str(path)], **quiet)


if __name__ == "__main__":
    main()
