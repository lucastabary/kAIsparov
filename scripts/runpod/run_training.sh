#!/usr/bin/env bash
# Per-session launcher, run ON the RunPod pod. Refreshes the code and starts the
# scratch-v3 curriculum on the GPU. Runs inside tmux so training survives an SSH drop.
# Artifacts land in runs/ on the persistent volume; pull them home before terminating.
#
# Usage (over SSH or in the pod terminal):
#   bash scripts/runpod/run_training.sh
# then detach from tmux with:  Ctrl-b  d      (reattach later: tmux attach -t train)
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/kAIsparov"
SESSION="train"

cd "$REPO_DIR"
echo ">> git pull"
git pull --ff-only
source .venv/bin/activate

CMD='kaisparov train --config \
  config/experiments/scratch_v3_stage1.yaml \
  config/experiments/scratch_v3_stage2.yaml \
  config/experiments/scratch_v3_stage3.yaml \
  2>&1 | tee "runs/last_run_$(date +%Y%m%d_%H%M%S).log"'

if command -v tmux >/dev/null 2>&1; then
  echo ">> launching in tmux session '$SESSION' (detach: Ctrl-b d)"
  tmux new-session -A -s "$SESSION" "source .venv/bin/activate && $CMD"
else
  echo ">> tmux not found, running in the foreground"
  eval "$CMD"
fi
