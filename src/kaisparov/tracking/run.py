"""RunManager — records everything about a single training run.

Each run gets its own directory under ``runs/<run_id>/``::

    config.yaml          # the exact resolved config
    run.json             # metadata + summary (status, git, params, best, history)
    metrics.jsonl        # one JSON line per logged step (append-only)
    tensorboard/         # TensorBoard event files
    checkpoints/
        epoch10.pth
        best.pth         # copy of the best checkpoint by the tracked metric

The paired :class:`~kaisparov.tracking.registry.Registry` reads these back without
needing torch.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retry_io(action: Callable[[], object], attempts: int = 12, delay: float = 0.2) -> bool:
    """Run a file op, retrying transient Windows locks (OneDrive / antivirus / editors).

    Returns True on success. Training must never crash because a synced folder briefly
    held a file open, so callers treat a False as "skip this write, retry next time".
    """
    for i in range(attempts):
        try:
            action()
            return True
        except PermissionError:
            if i < attempts - 1:
                time.sleep(delay)
    return False


def _git_info() -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


class RunManager:
    def __init__(
        self,
        *,
        root: str | Path = "runs",
        model: str,
        config: dict[str, Any],
        seed: int,
        device: str,
        num_params: int | None = None,
        title: str = "",
        description: str = "",
        notes: str = "",
        parent_run_id: str | None = None,
        resumed_from: str | None = None,
    ):
        self.root = Path(root)
        self.run_id = self._make_run_id(model)
        self.dir = self.root / self.run_id
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(log_dir=str(self.dir / "tensorboard"))
        self.metrics_path = self.dir / "metrics.jsonl"
        self.run_path = self.dir / "run.json"

        self._meta: dict[str, Any] = {
            "run_id": self.run_id,
            "model": model,
            "status": "running",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "finished_at": None,
            "git": _git_info(),
            "device": device,
            "seed": seed,
            "num_params": num_params,
            "title": title,
            "description": description,
            "notes": notes,
            "parent_run_id": parent_run_id,
            "resumed_from": resumed_from,
            "config": config,
            "epochs_completed": 0,
            "checkpoints": [],
            "eval_history": [],
            "best_checkpoint": None,
            "final_metrics": None,
        }
        with (self.dir / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        self._save_meta()

    def _make_run_id(self, model: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}_{model}"
        run_id, n = base, 1
        while (self.root / run_id).exists():
            run_id = f"{base}_{n}"
            n += 1
        return run_id

    def _save_meta(self) -> None:
        self._meta["updated_at"] = _utc_now()
        payload = json.dumps(self._meta, indent=2, ensure_ascii=False)
        tmp = self.run_path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        if not _retry_io(lambda: tmp.replace(self.run_path)):
            # Persistent lock on the atomic swap — write in place instead (loses
            # atomicity once), and if even that is locked, skip: next save rewrites it.
            _retry_io(lambda: self.run_path.write_text(payload, encoding="utf-8"))
            tmp.unlink(missing_ok=True)

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"

        def _append() -> None:
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(line)

        _retry_io(_append)

    # ------------------------------------------------------------------- logging
    def log_metrics(self, epoch: int, metrics: dict[str, float], section: str = "train") -> None:
        self._append_jsonl({"epoch": epoch, "section": section, "time": _utc_now(), **metrics})
        for key, value in metrics.items():
            self._writer.add_scalar(f"{section}/{key}", value, epoch)
        self._meta["epochs_completed"] = max(self._meta["epochs_completed"], epoch)
        self._save_meta()

    def log_eval(self, epoch: int, eval_metrics: dict[str, float]) -> None:
        self._append_jsonl({"epoch": epoch, "section": "eval", "time": _utc_now(), **eval_metrics})
        for key, value in eval_metrics.items():
            self._writer.add_scalar(f"eval/{key}", value, epoch)
        self._meta["eval_history"].append({"epoch": epoch, **eval_metrics})
        self._save_meta()

    # --------------------------------------------------------------- checkpoints
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: dict[str, float] | None = None,
        *,
        best_metric: str | None = None,
        best_mode: str = "max",
        trainer_state: dict[str, Any] | None = None,
    ) -> Path:
        path = self.dir / "checkpoints" / f"epoch{epoch}.pth"
        _retry_io(lambda: torch.save(model.state_dict(), path))  # pure weights (inference-friendly)

        record: dict[str, Any] = {
            "epoch": epoch,
            "file": path.name,
            "metrics": metrics or {},
            "created_at": _utc_now(),
        }
        # Optimizer/RNG go in a sibling file so resume can continue *exactly*, while
        # the model checkpoint stays a plain state_dict for inference.
        if trainer_state is not None:
            state_path = self.dir / "checkpoints" / f"epoch{epoch}.state.pth"
            _retry_io(lambda: torch.save(trainer_state, state_path))
            record["state_file"] = state_path.name
        self._meta["checkpoints"].append(record)

        if best_metric is not None and metrics is not None and best_metric in metrics:
            self._maybe_update_best(record, best_metric, best_mode, metrics[best_metric])

        self._save_meta()
        return path

    def _maybe_update_best(self, record, metric, mode, value) -> None:
        current = self._meta["best_checkpoint"]
        better = (
            current is None
            or (mode == "max" and value > current["value"])
            or (mode == "min" and value < current["value"])
        )
        if better:
            best_path = self.dir / "checkpoints" / "best.pth"
            src = self.dir / "checkpoints" / record["file"]
            _retry_io(lambda: shutil.copyfile(src, best_path))
            self._meta["best_checkpoint"] = {
                "epoch": record["epoch"],
                "file": record["file"],
                "metric": metric,
                "value": value,
            }

    # -------------------------------------------------------------------- finish
    def finish(
        self, status: str = "completed", final_metrics: dict[str, float] | None = None
    ) -> None:
        self._meta["status"] = status
        self._meta["finished_at"] = _utc_now()
        if final_metrics is not None:
            self._meta["final_metrics"] = final_metrics
        self._save_meta()
        self._writer.flush()
        self._writer.close()

    def __enter__(self) -> RunManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.finish(status="failed")
        elif self._meta["status"] == "running":
            self.finish(status="completed")
