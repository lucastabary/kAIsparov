#!/usr/bin/env bash
# One-time setup, run ON the RunPod pod, on a pod backed by a persistent /workspace
# network volume. It puts the repo checkout AND the venv on the volume, so every later
# session skips the slow reinstall and only does a `git pull`.
#
# Usage (in the pod's web terminal or over SSH):
#   bash <(curl -sSL https://raw.githubusercontent.com/lucastabary/kAIsparov/main/scripts/runpod/setup_pod.sh)
# or, if the repo is already cloned:
#   bash scripts/runpod/setup_pod.sh
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"          # network-volume mount point on RunPod
REPO_URL="${REPO_URL:-https://github.com/lucastabary/kAIsparov.git}"
REPO_DIR="$WORKSPACE/kAIsparov"

mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo ">> cloning $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo ">> creating venv on the volume"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# GPU wheels: PyTorch built against CUDA 11.8 (matches requirements.txt torch==2.0.1).
echo ">> installing GPU dependencies (cu118)"
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-dev.txt
pip install -e .

echo ">> sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
echo ">> setup done. Next sessions: bash scripts/runpod/run_training.sh"
