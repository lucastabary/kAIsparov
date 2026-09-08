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
2. **Deploy a pod** on that volume: pick an **RTX 4090**, a **PyTorch 2.4 (Python 3.11)**
   template, and attach the volume at `/workspace`. The image's CUDA version doesn't matter
   (the torch wheel ships its own cu121 libs; the host driver is recent enough).
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

## Driving the pod from your machine (`manage_pod.py`)

`manage_pod.py` wraps the official [`runpod` Python SDK](https://pypi.org/project/runpod/)
plus your system `ssh`/`tmux`, so you can start, stop, shell into, and run jobs on the pod
without touching the RunPod web UI. Its headline trick: **run one command and have the pod
power itself off the moment the command finishes** (starting the pod first if it was
stopped), so you never pay for idle GPU time.

```bash
pip install -r scripts/runpod/requirements.txt   # installs the runpod SDK
```

Then give it your **RunPod API key** (RunPod → Settings → API Keys). Easiest: drop it in
a gitignored **`.env.local`** at the repo root — `manage_pod.py` loads it automatically:

```ini
# .env.local  (never committed)
RUNPOD_API_KEY=your-key-here
AWS_PROFILE=runpods3
```

> It's `.env.local`, **not** `.env`, because in this repo `.env` is the Python virtualenv
> *directory*. Alternatively set a persistent user variable (needs a fresh terminal after):
> `[Environment]::SetEnvironmentVariable("RUNPOD_API_KEY", "<key>", "User")`.

The command reads its config from the environment (all optional except the API key):

| Variable | Meaning | Default |
|----------|---------|---------|
| `RUNPOD_API_KEY` | RunPod API key (**required**) | — |
| `RUNPOD_POD_ID` | which pod to manage | the only pod on the account |
| `RUNPOD_GPU_COUNT` | GPUs to attach on start | `1` |
| `RUNPOD_SSH_USER` | SSH user on the pod | `root` |
| `RUNPOD_SSH_KEY` | private SSH key path | `~/.ssh/id_ed25519` |
| `RUNPOD_REPO_DIR` | repo checkout on the pod | `/workspace/kAIsparov` |

Each has a matching flag (`--pod-id`, `--gpu-count`, `--ssh-key`). SSH uses the pod's
directly-exposed TCP port for private port 22, so make sure the pod **exposes TCP port 22**
and your **public key is registered** in RunPod → Settings → SSH Public Keys.

```bash
python scripts/runpod/manage_pod.py list             # all pods on the account
python scripts/runpod/manage_pod.py status           # status + SSH command + tmux sessions
python scripts/runpod/manage_pod.py start            # resume the pod, wait for SSH, git pull
python scripts/runpod/manage_pod.py pull             # git pull the repo on the running pod
python scripts/runpod/manage_pod.py stop             # stop it (GPU billing ends; volume persists)
python scripts/runpod/manage_pod.py ssh              # interactive shell on the pod
python scripts/runpod/manage_pod.py ssh -- nvidia-smi  # or a one-off command
python scripts/runpod/manage_pod.py tmux list        # the pod's tmux sessions
python scripts/runpod/manage_pod.py tmux attach train  # attach to one (add --create to make it)

# Start (if needed) → git pull → run → power off at the end. The job runs inside tmux
# on the pod (so it survives an SSH drop) and its output is streamed here live:
python scripts/runpod/manage_pod.py run -- bash scripts/runpod/run_training.sh
python scripts/runpod/manage_pod.py run -- bash scripts/runpod/run_training.sh config/experiments/x.yaml
python scripts/runpod/manage_pod.py run --keep -- kaisparov eval --games 60   # don't stop after
```

`run` executes from `RUNPOD_REPO_DIR` (`/workspace/kAIsparov`) with the repo's `.venv`
activated, so relative paths and `kaisparov` work directly. `run_training.sh` chains the
**v4 curriculum** by default, or the config files you pass it as arguments.

`start` and `run` **`git pull --ff-only` the pod's repo by default** so a session always
runs fresh code (a failed pull warns but doesn't abort); pass `--no-pull` to skip it, or use
the standalone `pull` command. For `run`, `Ctrl-C` only detaches your local viewer — the
command keeps running on the pod; reattach with `tmux attach`. The automatic power-off fires
from *this* process once the command exits, so if you kill it you'll need to `stop` the pod
yourself.

## Each training session

1. **Start** a pod on the network volume (RunPod web UI, or `runpodctl`).
2. **Launch** the curriculum:
   ```bash
   cd /workspace/kAIsparov && bash scripts/runpod/run_training.sh
   ```
   It `git pull`s, then runs the default 3-stage v4 curriculum inside `tmux` (detach with
   `Ctrl-b d`; reattach with `tmux attach -t train`). Pass config paths as arguments to
   run a different set. Training keeps going if your SSH/browser drops.
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
2. Configure an AWS CLI profile named `runpods3`:
   ```powershell
   aws configure --profile runpods3   # region: eu-ro-1, output: json
   ```
   Then add the volume's endpoint to that profile so no per-command flags are needed —
   in `~/.aws/config` under `[profile runpods3]`:
   ```ini
   endpoint_url = https://s3api-eu-ro-1.runpod.io
   ```
   `.vscode/settings.json` sets `AWS_PROFILE=runpods3` for this project's terminals, so
   inside the project `aws s3 ...` picks the profile, region and endpoint automatically.

Every time you want the latest results (pod can be off):
```powershell
scripts\runpod\pull_runs.ps1              # sync runs/ home, THEN delete them from the volume
scripts\runpod\pull_runs.ps1 -KeepRemote  # sync but leave them on the volume
scripts\runpod\pull_runs.ps1 -List        # just list what's on the volume
```
By default `pull_runs.ps1` **moves**: it syncs `runs/` home and then deletes them from the
volume (the bucket *is* the volume, so this frees the storage you pay for). The delete only
runs if the sync succeeded. Since it also removes them from the volume, a later cross-session
`resume` from one of those runs would need it re-uploaded — pass `-KeepRemote` when you plan
to extend a run.

Notes on the RunPod S3 API:
- The volume is **region-locked to EU-RO-1** — the pod must be deployed in that datacenter
  to mount it, and you need a 4090 available there.
- It supports the standard `aws s3 cp` / `sync` / `ls` / `rm`; stick to those.
- Bucket id (`pjg2ftia6g`), region (`eu-ro-1`), and endpoint are set in `pull_runs.ps1` —
  edit them there if the volume ever changes.
- SSH `scp -r -P <port> root@<host>:/workspace/kAIsparov/runs .\runs` still works as a
  fallback while a pod is running.

## Notes

- `device: auto` in the configs resolves to CUDA when a GPU is present, so nothing in the
  YAML needs changing between local (CPU) and pod (GPU).
- To resume/extend a run later, the `runs/` history is already on the volume — the trainer
  can pick it up on the next session.
- Never commit `runs/`, `data/`, or `*.pth` (git-ignored). These scripts don't.
