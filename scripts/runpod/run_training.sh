#!/usr/bin/env bash
# Per-session launcher, run ON the RunPod pod. Refreshes the code and starts the
# training curriculum on the GPU. Runs inside tmux so training survives a dropped SSH
# connection (unless already inside one). Artifacts land in runs/ on the volume.
#
# Usage (over SSH, in the pod terminal, or via manage_pod.py run):
#   bash scripts/runpod/run_training.sh                        # default: v4 curriculum
#   bash scripts/runpod/run_training.sh config/experiments/a.yaml [b.yaml ...]   # custom
# then detach from tmux with:  Ctrl-b  d      (reattach later: tmux attach -t train)
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/kAIsparov"
SESSION="train"

# Configs chained through `kaisparov train`: those passed as arguments, else the
# default v4 curriculum (stage 1 -> 2 -> 3).
if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=(
    config/experiments/scratch_v4_stage1.yaml
    config/experiments/scratch_v4_stage2.yaml
    config/experiments/scratch_v4_stage3.yaml
  )
fi

cd "$REPO_DIR"
echo ">> git pull"
git pull --ff-only
source .venv/bin/activate

LOG="runs/last_run_$(date +%Y%m%d_%H%M%S).log"
CMD="kaisparov train --config ${CONFIGS[*]} 2>&1 | tee \"$LOG\""

if [ -n "${TMUX:-}" ]; then
  # Already inside a tmux (e.g. launched by manage_pod.py run) — don't nest, run here.
  echo ">> already inside tmux; running directly."
  eval "$CMD"
elif command -v tmux >/dev/null 2>&1; then
  echo ">> launching in tmux session '$SESSION' (detach: Ctrl-b d)"
  tmux new-session -A -s "$SESSION" "source .venv/bin/activate && $CMD"
else
  echo ">> tmux not found, running in the foreground"
  eval "$CMD"
fi
