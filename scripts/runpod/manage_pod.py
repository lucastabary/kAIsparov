#!/usr/bin/env python3
"""Manage the kAIsparov RunPod pod from your local machine.

A thin, dependency-light wrapper around the official ``runpod`` Python SDK plus
your system ``ssh``/``tmux``. It lets you start/stop the pod, open a shell, list
and attach to the ``tmux`` sessions running on it, and — the headline feature —
run one command on the pod and have the pod power off the moment it finishes
(starting the pod first if it was stopped), so you never pay for idle GPU time.

Setup
-----
    pip install runpod            # or: pip install -r scripts/runpod/requirements.txt

The API key is read from the environment (never hard-code it):

    export RUNPOD_API_KEY=...          # RunPod -> Settings -> API Keys

A ``.env`` file at the repo root (or the current directory) is loaded
automatically, so you can also just drop ``RUNPOD_API_KEY=...`` in there.

Configuration (all optional, env or flags)
------------------------------------------
    RUNPOD_POD_ID     which pod to manage (else: the only pod on the account)
    RUNPOD_GPU_COUNT  GPUs to attach on start/resume (default: 1)
    RUNPOD_SSH_USER   SSH user on the pod (default: root)
    RUNPOD_SSH_KEY    path to the private SSH key (default: ~/.ssh/id_ed25519)

Examples
--------
    python scripts/runpod/manage_pod.py list           # pods on the account
    python scripts/runpod/manage_pod.py status
    python scripts/runpod/manage_pod.py start
    python scripts/runpod/manage_pod.py ssh             # interactive shell
    python scripts/runpod/manage_pod.py tmux list
    python scripts/runpod/manage_pod.py tmux attach train
    python scripts/runpod/manage_pod.py stop

    # Start (if needed), run the curriculum, then power the pod off at the end:
    python scripts/runpod/manage_pod.py run -- \
        bash scripts/runpod/run_training.sh

Notes
-----
* ``run`` launches the command inside a ``tmux`` session on the pod, so the work
  survives a dropped SSH connection; this script tails its output and, once the
  command exits, stops the pod (unless ``--keep``). If *this* process is killed
  the remote command keeps running, but the automatic power-off won't fire —
  reconnect with ``tmux attach`` and stop the pod yourself.
* SSH uses the pod's directly-exposed TCP port for private port 22, which RunPod
  maps to a public ``ip:port``. Make sure your key is registered in RunPod
  (Settings -> SSH Public Keys) and that the pod exposes TCP port 22.
"""

from __future__ import annotations

import argparse
import base64
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

try:
    import runpod
except ImportError:  # pragma: no cover - guidance only
    sys.exit(
        "The 'runpod' SDK is not installed.\n"
        "  pip install runpod   (or: pip install -r scripts/runpod/requirements.txt)"
    )

# Remote directory where `run` keeps its per-session log and exit-code marker.
REMOTE_RUN_DIR = "$HOME/.kaisparov_runs"

# How long to wait, in seconds, for the pod to come up / SSH to answer.
START_TIMEOUT = 300
SSH_TIMEOUT = 180


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_dotenv() -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file without overriding real env.

    Looks at the repo root (two levels up from this file) and the current
    working directory. Keeps things dependency-free — no python-dotenv needed.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / ".env",
        Path.cwd() / ".env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class Config:
    """Resolved settings, from CLI flags with env fallbacks."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
        if not self.api_key:
            sys.exit(
                "RUNPOD_API_KEY is not set. Export it or add it to a .env file:\n"
                "  export RUNPOD_API_KEY=..."
            )
        runpod.api_key = self.api_key

        self.pod_id: str | None = getattr(args, "pod_id", None) or os.environ.get("RUNPOD_POD_ID")
        self.gpu_count = int(
            getattr(args, "gpu_count", None) or os.environ.get("RUNPOD_GPU_COUNT") or 1
        )
        self.ssh_user = os.environ.get("RUNPOD_SSH_USER", "root")
        default_key = str(Path.home() / ".ssh" / "id_ed25519")
        self.ssh_key = getattr(args, "ssh_key", None) or os.environ.get(
            "RUNPOD_SSH_KEY", default_key
        )


# --------------------------------------------------------------------------- #
# Pod helpers
# --------------------------------------------------------------------------- #
def resolve_pod_id(cfg: Config) -> str:
    """Return the pod id to act on, auto-selecting when the account has one pod."""
    if cfg.pod_id:
        return cfg.pod_id
    pods = runpod.get_pods()
    if not pods:
        sys.exit("No pods found on this account. Create one in the RunPod dashboard.")
    if len(pods) == 1:
        return pods[0]["id"]
    print("Multiple pods found — pick one with --pod-id or set RUNPOD_POD_ID:\n")
    for pod in pods:
        print(f"  {pod['id']}  {pod.get('name', '?')}  [{pod.get('desiredStatus', '?')}]")
    sys.exit(1)


def get_pod(pod_id: str) -> dict:
    pod = runpod.get_pod(pod_id)
    if not pod:
        sys.exit(f"Pod {pod_id} not found (check RUNPOD_POD_ID / --pod-id).")
    return pod


def is_running(pod: dict) -> bool:
    return pod.get("desiredStatus") == "RUNNING"


def ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    """Public ``(ip, port)`` mapped to the pod's private TCP port 22, if any."""
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port.get("privatePort") == 22 and port.get("type") == "tcp" and port.get("ip"):
            return port["ip"], int(port["publicPort"])
    return None


# --------------------------------------------------------------------------- #
# SSH helpers
# --------------------------------------------------------------------------- #
def ssh_base(cfg: Config, ip: str, port: int, *, tty: bool = False) -> list[str]:
    """Base ``ssh`` argv for the pod (key, port, host, sane connection options)."""
    argv = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if tty:
        argv.append("-t")
    if cfg.ssh_key and Path(cfg.ssh_key).expanduser().is_file():
        argv += ["-i", str(Path(cfg.ssh_key).expanduser())]
    argv.append(f"{cfg.ssh_user}@{ip}")
    return argv


def ssh_run(
    cfg: Config,
    ip: str,
    port: int,
    remote_cmd: str,
    *,
    tty: bool = False,
    check: bool = False,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run one remote command over SSH."""
    argv = ssh_base(cfg, ip, port, tty=tty) + [remote_cmd]
    return subprocess.run(
        argv,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def require_endpoint(cfg: Config, pod_id: str) -> tuple[dict, str, int]:
    """Fetch the pod and its SSH endpoint, erroring out if it isn't reachable."""
    pod = get_pod(pod_id)
    if not is_running(pod):
        sys.exit(
            f"Pod {pod_id} is not running (status: {pod.get('desiredStatus')}). Start it first."
        )
    endpoint = ssh_endpoint(pod)
    if not endpoint:
        sys.exit(
            "Pod is running but exposes no public TCP port for SSH (private port 22).\n"
            "Add TCP port 22 to the pod's exposed ports in the RunPod dashboard."
        )
    return pod, endpoint[0], endpoint[1]


def wait_for_ssh(cfg: Config, ip: str, port: int, timeout: int = SSH_TIMEOUT) -> None:
    """Block until the pod answers SSH (sshd can lag a few seconds behind RUNNING)."""
    print(f">> waiting for SSH on {ip}:{port} ...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = ssh_base(cfg, ip, port) + [
            "-o",
            "ConnectTimeout=5",
            "-o",
            "BatchMode=yes",
            "true",
        ]
        result = subprocess.run(probe, capture_output=True, text=True)
        if result.returncode == 0:
            print(">> SSH is up.")
            return
        time.sleep(4)
    sys.exit("Timed out waiting for SSH. Check your key (RunPod -> Settings -> SSH Public Keys).")


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def ensure_running(cfg: Config, pod_id: str) -> tuple[dict, str, int]:
    """Start the pod if needed and return it once SSH is reachable."""
    pod = get_pod(pod_id)
    if not is_running(pod):
        start_pod(cfg, pod_id)
        pod = get_pod(pod_id)
    endpoint = ssh_endpoint(pod)
    if not endpoint:
        sys.exit("Pod is running but has no SSH TCP port (expose private port 22).")
    ip, port = endpoint
    wait_for_ssh(cfg, ip, port)
    return pod, ip, port


def start_pod(cfg: Config, pod_id: str) -> dict:
    """Resume the pod and wait until it reports RUNNING with an SSH endpoint."""
    pod = get_pod(pod_id)
    if is_running(pod):
        print(f">> pod {pod_id} is already running.")
        return pod
    print(f">> resuming pod {pod_id} with {cfg.gpu_count} GPU(s) ...")
    runpod.resume_pod(pod_id, cfg.gpu_count)
    deadline = time.time() + START_TIMEOUT
    while time.time() < deadline:
        pod = get_pod(pod_id)
        if is_running(pod) and ssh_endpoint(pod):
            print(">> pod is RUNNING.")
            return pod
        time.sleep(5)
    sys.exit("Timed out waiting for the pod to start.")


def stop_pod(cfg: Config, pod_id: str) -> None:
    pod = get_pod(pod_id)
    if not is_running(pod):
        print(f">> pod {pod_id} is already stopped (status: {pod.get('desiredStatus')}).")
        return
    print(f">> stopping pod {pod_id} ...")
    runpod.stop_pod(pod_id)
    print(">> stop requested. GPU billing ends once it exits; the volume persists.")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_list(cfg: Config, _args: argparse.Namespace) -> int:
    pods = runpod.get_pods()
    if not pods:
        print("No pods on this account.")
        return 0
    for pod in pods:
        gpu = (pod.get("machine") or {}).get("gpuDisplayName", "?")
        print(
            f"{pod['id']}  {pod.get('name', '?'):<24}  "
            f"{pod.get('desiredStatus', '?'):<9}  "
            f"{pod.get('gpuCount', 0)}x {gpu}  ${pod.get('costPerHr', 0)}/hr"
        )
    return 0


def cmd_status(cfg: Config, _args: argparse.Namespace) -> int:
    pod_id = resolve_pod_id(cfg)
    pod = get_pod(pod_id)
    gpu = (pod.get("machine") or {}).get("gpuDisplayName", "?")
    print(f"id:       {pod['id']}")
    print(f"name:     {pod.get('name', '?')}")
    print(f"status:   {pod.get('desiredStatus', '?')}")
    print(f"gpu:      {pod.get('gpuCount', 0)}x {gpu}")
    print(f"cost:     ${pod.get('costPerHr', 0)}/hr")
    uptime = pod.get("uptimeSeconds")
    if uptime:
        print(f"uptime:   {uptime // 3600}h {(uptime % 3600) // 60}m")
    endpoint = ssh_endpoint(pod)
    if endpoint:
        ip, port = endpoint
        print(f"ssh:      ssh -p {port} {cfg.ssh_user}@{ip}")
        result = ssh_run(cfg, ip, port, "tmux ls 2>/dev/null || true", capture=True)
        listing = (result.stdout or "").strip()
        print(
            "tmux:     " + (listing.replace("\n", "\n          ") if listing else "(no sessions)")
        )
    else:
        print("ssh:      (pod not running / no TCP port 22 exposed)")
    return 0


def cmd_start(cfg: Config, _args: argparse.Namespace) -> int:
    start_pod(cfg, resolve_pod_id(cfg))
    return 0


def cmd_stop(cfg: Config, _args: argparse.Namespace) -> int:
    stop_pod(cfg, resolve_pod_id(cfg))
    return 0


def cmd_ssh(cfg: Config, args: argparse.Namespace) -> int:
    pod_id = resolve_pod_id(cfg)
    _pod, ip, port = require_endpoint(cfg, pod_id)
    if args.command:
        # Run a one-off command (non-interactive) and forward its exit code.
        remote = " ".join(shlex.quote(part) for part in args.command)
        return ssh_run(cfg, ip, port, remote).returncode
    # Interactive login shell.
    return subprocess.run(ssh_base(cfg, ip, port, tty=True)).returncode


def cmd_tmux(cfg: Config, args: argparse.Namespace) -> int:
    pod_id = resolve_pod_id(cfg)
    _pod, ip, port = require_endpoint(cfg, pod_id)
    if args.tmux_action == "list":
        result = ssh_run(cfg, ip, port, "tmux ls", capture=True)
        out = (result.stdout or "").strip()
        print(out if out else "(no tmux sessions)")
        return 0
    # attach
    name = shlex.quote(args.session)
    remote = f"tmux new-session -A -s {name}" if args.create else f"tmux attach -t {name}"
    return subprocess.run(ssh_base(cfg, ip, port, tty=True) + [remote]).returncode


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    """Start the pod if needed, run a command in tmux, then stop the pod at the end."""
    if not args.command:
        sys.exit("Nothing to run. Usage: run [--session NAME] [--keep] -- <command...>")
    command = " ".join(args.command)
    session = args.session

    pod_id = resolve_pod_id(cfg)
    _pod, ip, port = ensure_running(cfg, pod_id)

    launch_remote_command(cfg, ip, port, session, command)
    print(
        f">> launched in tmux session '{session}'. Streaming output "
        "(Ctrl-C detaches; the command keeps running):\n"
    )
    try:
        stream_until_done(cfg, ip, port, session)
    except KeyboardInterrupt:
        print(
            f"\n>> detached. The command is still running on the pod.\n"
            f">> reattach: python {Path(__file__).name} tmux attach {session}\n"
            f">> the pod was NOT stopped."
        )
        return 130

    exit_code = read_exit_code(cfg, ip, port, session)
    print(f"\n>> command finished with exit code {exit_code}.")

    if args.keep:
        print(">> --keep set: leaving the pod running.")
    else:
        stop_pod(cfg, pod_id)
    return exit_code


# --------------------------------------------------------------------------- #
# `run` internals
# --------------------------------------------------------------------------- #
def _runner_script(session: str, command: str) -> str:
    """Bash script (run on the pod, in tmux) that logs output and records the exit code."""
    return (
        "#!/usr/bin/env bash\n"
        "set -o pipefail\n"
        f'RUN_DIR="{REMOTE_RUN_DIR}"\n'
        'mkdir -p "$RUN_DIR"\n'
        f'LOG="$RUN_DIR/{session}.log"\n'
        f'EXIT_FILE="$RUN_DIR/{session}.exit"\n'
        'rm -f "$EXIT_FILE"\n'
        ': > "$LOG"\n'
        # A subshell (not a brace group) so an explicit `exit` in the user's
        # command only ends the subshell. tee mirrors output to the log (which we
        # tail) and to the tmux pane (so a direct `tmux attach` also shows it);
        # PIPESTATUS[0] is the command's own status, not tee's.
        "(\n"
        f"{command}\n"
        ') 2>&1 | tee "$LOG"\n'
        'echo "${PIPESTATUS[0]}" > "$EXIT_FILE"\n'
    )


def launch_remote_command(cfg: Config, ip: str, port: int, session: str, command: str) -> None:
    """Write the runner script to the pod and launch it detached in tmux."""
    script = _runner_script(session, command)
    encoded = base64.b64encode(script.encode()).decode()
    script_path = f"{REMOTE_RUN_DIR}/{session}.sh"
    bootstrap = (
        f'mkdir -p "{REMOTE_RUN_DIR}" && '
        f"printf '%s' '{encoded}' | base64 -d > \"{script_path}\" && "
        f"if tmux has-session -t {shlex.quote(session)} 2>/dev/null; then "
        f'echo "tmux session {session} already exists" >&2; exit 3; fi && '
        f'tmux new-session -d -s {shlex.quote(session)} "bash {script_path}"'
    )
    result = ssh_run(cfg, ip, port, bootstrap, capture=True)
    if result.returncode != 0:
        sys.exit(
            f"Failed to launch the command on the pod:\n{(result.stdout or '').strip()}\n"
            f"(If the session already exists, pick another with --session.)"
        )


def stream_until_done(cfg: Config, ip: str, port: int, session: str) -> None:
    """Tail the remote log live until the exit marker appears; reconnect if SSH drops."""
    log = f"{REMOTE_RUN_DIR}/{session}.log"
    marker = f"{REMOTE_RUN_DIR}/{session}.exit"
    # tail -f the log, stopping as soon as the exit-code file lands.
    remote = (
        f'touch "{log}"; tail -n +1 -f "{log}" & tp=$!; '
        f'while [ ! -f "{marker}" ]; do sleep 2; done; '
        "sleep 1; kill $tp 2>/dev/null; exit 0"
    )
    while True:
        subprocess.run(ssh_base(cfg, ip, port, tty=True) + [remote])
        # If the marker exists, we're genuinely done; otherwise SSH dropped — retry.
        check = ssh_run(cfg, ip, port, f'test -f "{marker}"')
        if check.returncode == 0:
            return
        print(">> SSH connection dropped; reconnecting to keep watching ...", flush=True)
        time.sleep(3)


def read_exit_code(cfg: Config, ip: str, port: int, session: str) -> int:
    marker = f"{REMOTE_RUN_DIR}/{session}.exit"
    result = ssh_run(cfg, ip, port, f'cat "{marker}" 2>/dev/null', capture=True)
    text = (result.stdout or "").strip()
    try:
        return int(text)
    except ValueError:
        return 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage_pod.py",
        description="Manage the kAIsparov RunPod pod (start/stop/ssh/tmux/run).",
    )
    parser.add_argument("--pod-id", help="Pod id (default: RUNPOD_POD_ID, or the only pod).")
    parser.add_argument(
        "--gpu-count", type=int, help="GPUs to attach on start (default: RUNPOD_GPU_COUNT or 1)."
    )
    parser.add_argument(
        "--ssh-key", help="Private SSH key path (default: RUNPOD_SSH_KEY or ~/.ssh/id_ed25519)."
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all pods on the account.")
    sub.add_parser("status", help="Show the pod's status, SSH command, and tmux sessions.")
    sub.add_parser("start", help="Start (resume) the pod and wait for SSH.")
    sub.add_parser("stop", help="Stop the pod (GPU billing ends; the volume persists).")

    p_ssh = sub.add_parser("ssh", help="Open an interactive SSH shell (or run a one-off command).")
    p_ssh.add_argument(
        "command", nargs=argparse.REMAINDER, help="Optional command to run instead of a shell."
    )

    p_tmux = sub.add_parser("tmux", help="List or attach to tmux sessions on the pod.")
    tmux_sub = p_tmux.add_subparsers(dest="tmux_action", required=True)
    tmux_sub.add_parser("list", help="List tmux sessions.")
    p_attach = tmux_sub.add_parser("attach", help="Attach to a tmux session.")
    p_attach.add_argument("session", help="tmux session name.")
    p_attach.add_argument(
        "--create", action="store_true", help="Create the session if it doesn't exist."
    )

    p_run = sub.add_parser(
        "run",
        help="Start pod if needed, run a command in tmux, then stop the pod at the end.",
    )
    p_run.add_argument("--session", default="run", help="tmux session name (default: run).")
    p_run.add_argument(
        "--keep", action="store_true", help="Leave the pod running after the command finishes."
    )
    p_run.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command to run (put it after --)."
    )
    return parser


COMMANDS = {
    "list": cmd_list,
    "status": cmd_status,
    "start": cmd_start,
    "stop": cmd_stop,
    "ssh": cmd_ssh,
    "tmux": cmd_tmux,
    "run": cmd_run,
}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    # argparse.REMAINDER keeps a leading "--"; drop it so it isn't run as a command.
    command = getattr(args, "command", None)
    if command and command[0] == "--":
        args.command = command[1:]
    cfg = Config(args)
    return COMMANDS[args.cmd](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
