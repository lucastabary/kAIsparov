# Remote training on RunPod

Repeatable loop for training kAIsparov on a rented GPU pod, paying only while a pod is
running. State (repo checkout + venv + `runs/`) lives on a **persistent network volume**,
so each session is just *start pod → pull → train → download artifacts → terminate pod*.

## Is an RTX 4090 the right pick?

Yes — it works and it's the best price/perf card RunPod offers (24 GB VRAM, far more than
this tiny `hidden_dim=32` RGCN needs). One honest caveat: **this workload is CPU-bound, not
GPU-bound.** The chess engine (movegen, rollouts, depth-1 minimax opponents) is pure Python
on CPU; the GNN forward/backward is a rounding error next to it. So:

- Don't expect a 10× speedup over a fast CPU — the GPU sits mostly idle between moves.
- When choosing a pod, favour **high per-core CPU clock and enough vCPUs** over the biggest
  GPU. A 4090 with a decent CPU allocation is a fine, cheap choice; a pricier A100 would be
  wasted money here.
- Ballpark cost: 4090 community cloud ≈ \$0.35–0.45/hr, secure cloud ≈ \$0.70/hr; a network
  volume ≈ \$0.05–0.07/GB/month (check current RunPod prices). Between sessions you pay only
  the volume (a few cents/day), never the GPU.

## One-time setup

1. **Create a Network Volume** (RunPod → Storage). ~20 GB is plenty, in a region that has
   RTX 4090s. This is what persists between sessions.
2. **Deploy a pod** on that volume: pick an **RTX 4090**, a **PyTorch 2.x / CUDA 11.8**
   template (or any CUDA base image), and attach the volume at `/workspace`.
3. In the pod's web terminal, run the setup once:
   ```bash
   bash <(curl -sSL https://raw.githubusercontent.com/lucastabary/kAIsparov/main/scripts/runpod/setup_pod.sh)
   ```
   This clones the repo and builds the venv **on the volume** (`/workspace/kAIsparov`), so
   it survives pod termination.
4. (Optional but recommended) Install `runpodctl` locally on Windows to start/stop pods and
   move files from the terminal: https://github.com/runpod/runpodctl
5. (Optional) Add your SSH public key in RunPod → Settings, so you can `ssh` / `scp` into
   pods instead of using the web terminal.

## Each training session

1. **Start** a pod on the network volume (RunPod web UI, or `runpodctl`).
2. **Launch** the curriculum:
   ```bash
   cd /workspace/kAIsparov && bash scripts/runpod/run_training.sh
   ```
   It `git pull`s, then runs the 3-stage v3 command inside `tmux` (detach with `Ctrl-b d`;
   reattach with `tmux attach -t train`). Training keeps going if your SSH/browser drops.
3. **Watch** (optional): in a second shell on the pod,
   ```bash
   source /workspace/kAIsparov/.venv/bin/activate
   tensorboard --logdir /workspace/kAIsparov/runs --host 0.0.0.0 --port 6006
   ```
   Expose port 6006 on the pod to open TensorBoard in your browser.
4. **Terminate the pod** (not just stop) as soon as training finishes — GPU billing stops.
   The network volume (repo, venv, `runs/`) persists.
5. **Pull the artifacts home** — via the volume's S3 API, so *no pod needs to be running*
   (see below). `runs/` is git-ignored on purpose; S3 is how it comes back.

## Getting artifacts back with the volume's S3 API (recommended)

RunPod network volumes are S3-compatible: the **bucket = the volume**, so `/workspace/...`
on the pod is the same bytes you read over S3 — even with every pod terminated. Use S3 for
`runs/` / `*.pth`; keep **git** for code (the pod still `git pull`s).

One-time, on your Windows machine:

1. Create **S3 API keys** in RunPod → Settings → *S3 API Keys* (distinct from your RunPod
   API key). You get an access key + secret.
2. Configure an AWS CLI profile:
   ```powershell
   aws configure --profile runpod   # region: eu-ro-1, output: json
   ```

Every time you want the latest results (pod can be off):
```powershell
scripts\runpod\pull_runs.ps1
# or directly:
aws s3 sync s3://9v22kl54a0/kAIsparov/runs .\runs `
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io --profile runpod
```

Notes on the RunPod S3 API:
- The volume is **region-locked to EU-RO-1** — the pod must be deployed in that datacenter
  to mount it, and you need a 4090 available there.
- It supports the standard `aws s3 cp` / `sync` / `ls` / `rm`; stick to those.
- Bucket id (`9v22kl54a0`), region (`eu-ro-1`), and endpoint are set in `pull_runs.ps1` —
  edit them there if the volume ever changes.
- SSH `scp -r -P <port> root@<host>:/workspace/kAIsparov/runs .\runs` still works as a
  fallback while a pod is running.

## Notes

- `device: auto` in the configs resolves to CUDA when a GPU is present, so nothing in the
  YAML needs changing between local (CPU) and pod (GPU).
- To resume/extend a run later, the `runs/` history is already on the volume — the trainer
  can pick it up on the next session.
- Never commit `runs/`, `data/`, or `*.pth` (git-ignored). These scripts don't.
