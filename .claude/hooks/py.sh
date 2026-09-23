#!/bin/sh
# Run a hook script with the project's Python: the venv the `kaisparov` command uses when
# there is one (.env on Windows, .venv elsewhere), the first python3 on PATH otherwise.
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
for py in "$root/.env/Scripts/python.exe" "$root/.venv/Scripts/python.exe" "$root/.venv/bin/python"; do
  if [ -x "$py" ]; then exec "$py" "$@"; fi
done
exec python3 "$@"
