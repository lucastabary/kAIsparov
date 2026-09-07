"""Config-driven PPO self-play trainer with tracking and periodic evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.core.board import ChessGame
from kaisparov.eval.arena import evaluate
from kaisparov.models.factory import load_backend, load_backend_spec
from kaisparov.tracking.run import RunManager
from kaisparov.training.config import OpponentSpec, TrainConfig, build_pool_spec
from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum
from kaisparov.training.reward import make_reward_fn

# vs_random saturates at 100% (elo capped) once the model beats a random mover, so
# it carries no gradient for "best checkpoint" selection. vs_material is the metric
# that actually discriminates skill here.
BEST_METRIC = "elo_vs_material"


def _build_baseline(name: str, seed: int, avoid_king_suicide: bool = False):
    """Instantiate a fixed baseline opponent by name (legacy flat-config path)."""
    if name == "random":
        return RandomAgent(seed=seed, avoid_king_suicide=avoid_king_suicide)
    if name == "material":
        return MaterialAgent(seed=seed, avoid_king_suicide=avoid_king_suicide)
    raise ValueError(f"Unknown baseline opponent '{name}' (expected 'random' or 'material').")


def _load_frozen_model(spec, device, hidden_dim, checkpoint: str):
    """Load a frozen (eval, no-grad) model of this run's architecture from a checkpoint."""
    model = spec.model_class.create_agent(device=device, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _build_pool_baseline(spec, device, hidden_dim, opp: OpponentSpec, seed: int):
    """Build one fixed baseline agent from a pool-preset entry and its ``params``.

    ``random``/``material`` are model-free; ``neural``/``minimax`` load a frozen model
    from ``params.checkpoint`` (a past run's weights) — a strong, fixed teacher.
    """
    params = opp.params
    kind = opp.kind
    if kind == "random":
        return RandomAgent(
            seed=params.get("seed", seed),
            avoid_king_suicide=params.get("avoid_king_suicide", False),
        )
    if kind == "material":
        return MaterialAgent(
            seed=params.get("seed", seed),
            avoid_king_suicide=params.get("avoid_king_suicide", False),
        )
    if kind in ("neural", "minimax"):
        checkpoint = params.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                f"A '{kind}' baseline needs params.checkpoint (path to frozen weights)."
            )
        model = _load_frozen_model(spec, device, hidden_dim, checkpoint)
        processor = spec.processor_class()
        if kind == "minimax":
            from kaisparov.agents.minimax_agent import MinimaxAgent

            return MinimaxAgent(
                model,
                processor,
                depth=int(params.get("depth", 2)),
                avoid_king_suicide=params.get("avoid_king_suicide", False),
            )
        from kaisparov.agents.neural_agent import NeuralAgent

        return NeuralAgent(
            model,
            processor,
            deterministic=params.get("deterministic", False),
            avoid_king_suicide=params.get("avoid_king_suicide", False),
        )
    raise ValueError(f"Kind {kind!r} is not valid for a fixed baseline (group: baseline).")


def _resolve_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        _seed_everything(config.seed)
        self.device = _resolve_device(config.device)

        self.module = load_backend(config.model)
        self.spec = load_backend_spec(config.model)

        self.agent = self.spec.model_class.create_agent(
            device=self.device, hidden_dim=config.hidden_dim
        )
        self.optimizer = self.spec.model_class.create_optimizer(
            self.agent, learning_rate=config.ppo.learning_rate
        )
        self.buffer = self.spec.buffer_class(
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            self_play=config.ppo.self_play,
        )
        # No curriculum -> games start from the normal chess position.
        self.curriculum: PieceCountCurriculum | None = None
        if config.curriculum is not None:
            self.curriculum = PieceCountCurriculum(
                PhaseConfig(
                    name=config.curriculum.name,
                    max_pieces_per_side=config.curriculum.max_pieces_per_side,
                    allow_major=config.curriculum.allow_major,
                    allow_minor=config.curriculum.allow_minor,
                    allow_pawns=config.curriculum.allow_pawns,
                    ensure_kings_safe=config.curriculum.ensure_kings_safe,
                ),
                seed=config.seed,
            )
        self._processor = self.spec.processor_class()
        self._num_params = sum(p.numel() for p in self.agent.parameters())
        self.reward_fn = make_reward_fn(config.reward)

        # Opponent pool (league). Seeded baselines make it non-empty from epoch 1;
        # without baselines it falls back to self-play until the first snapshot.
        self.pool = None
        # Epochs between adding a frozen past-self; set from the pool spec / flat field.
        self.snapshot_every = config.rollout.snapshot_every
        # Whether the pool grows past-selves at all (a baseline-only preset does not).
        self.take_snapshots = True
        if config.rollout.opponent == "pool":
            from kaisparov.training.opponents import OpponentPool

            if config.rollout.pool is not None:
                self.pool = self._build_pool_from_spec(config)
            else:
                avoid_suicide = config.rollout.opponent_avoid_king_suicide
                baselines = [
                    _build_baseline(n, config.seed, avoid_king_suicide=avoid_suicide)
                    for n in config.rollout.baselines
                ]
                self.pool = OpponentPool(
                    self.spec,
                    self.device,
                    config.hidden_dim,
                    max_size=config.rollout.pool_size,
                    seed=config.seed,
                    baselines=baselines,
                    baseline_weight=config.rollout.baseline_weight,
                    snapshot_weight=config.rollout.snapshot_weight,
                    baseline_weights=config.rollout.baseline_weights,
                    search_depth=config.rollout.snapshot_search_depth,
                    avoid_king_suicide=avoid_suicide,
                )

        # Resume: load weights, optimizer + RNG state, and continue epoch numbering.
        self.start_epoch = 0
        if config.resume_from is not None:
            self.agent.load_state_dict(
                torch.load(config.resume_from, map_location=self.device, weights_only=True)
            )
            print(f"Resumed weights from {config.resume_from}")
            self._restore_trainer_state(config.resume_from)
            if config.parent_run_id is not None:
                from kaisparov.tracking.registry import Registry

                parent = Registry(config.runs_dir).get(config.parent_run_id)
                self.start_epoch = int(parent.get("epochs_completed", 0))

        self.run = RunManager(
            root=config.runs_dir,
            model=config.model,
            config=config.to_dict(),
            seed=config.seed,
            device=str(self.device),
            num_params=self._num_params,
            title=config.title,
            description=config.description,
            notes=config.notes,
            parent_run_id=config.parent_run_id,
            resumed_from=config.resume_from,
        )

    # --------------------------------------------------------------- opponent pool
    def _build_pool_from_spec(self, config: TrainConfig):
        """Build the opponent pool from a resolved pool preset (``rollout.pool``).

        The flat ``opponents`` list is split into fixed baselines (built now, each with
        its own ``params``) and the single ``snapshot`` stream (its ``params`` drive the
        past-selves: ``count`` -> pool size, ``every`` -> cadence, ``depth`` ->
        minimax/neural, plus ``avoid_king_suicide`` / ``deterministic``).
        """
        from kaisparov.training.opponents import OpponentPool

        pspec = build_pool_spec(config.rollout.pool)
        baselines, baseline_weights = [], []
        for opp in pspec.baselines:
            baselines.append(
                _build_pool_baseline(self.spec, self.device, config.hidden_dim, opp, config.seed)
            )
            baseline_weights.append(opp.weight)

        snap = pspec.snapshot
        self.take_snapshots = snap is not None
        self.snapshot_every = snap.every if snap else config.rollout.snapshot_every
        gw = pspec.group_weights
        sp = snap.params if snap else {}
        return OpponentPool(
            self.spec,
            self.device,
            config.hidden_dim,
            max_size=snap.count if snap else 0,
            seed=config.seed,
            baselines=baselines,
            baseline_weight=gw.get("baseline"),
            snapshot_weight=gw.get("snapshot"),
            baseline_weights=baseline_weights or None,
            search_depth=int(sp.get("depth", 0)),
            avoid_king_suicide=bool(sp.get("avoid_king_suicide", False)),
            snapshot_deterministic=bool(sp.get("deterministic", False)),
        )

    # -------------------------------------------------------------- resume state
    def _trainer_state(self, epoch: int) -> dict:
        """Optimizer + RNG snapshot, saved beside a checkpoint for exact resume."""
        return {
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

    def _restore_trainer_state(self, model_path: str) -> None:
        state_path = Path(model_path).with_name(Path(model_path).stem + ".state.pth")
        if not state_path.exists():
            print("(no optimizer/RNG state beside the checkpoint — optimizer starts fresh)")
            return
        # Our own trusted snapshot; it holds pickled RNG/optimizer state (numpy & python
        # RNG tuples) that weights_only=True would reject, so load the full pickle.
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        for opt_state in self.optimizer.state.values():
            for key, value in opt_state.items():
                if isinstance(value, torch.Tensor):
                    opt_state[key] = value.to(self.device)
        rng = state.get("rng", {})
        if rng:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        print(f"Restored optimizer + RNG state from {state_path.name}")

    # ------------------------------------------------------------------ collection
    def _collect(self, epoch: int) -> dict[str, float]:
        """Collect an epoch of experience: self-play, or vs a pooled opponent."""
        cfg = self.config
        if self.pool is not None and len(self.pool) > 0:
            from kaisparov.training.rollout_vs import collect_vs_opponent

            self.buffer.self_play = False  # single-agent: opponent is the environment
            return collect_vs_opponent(
                self.agent,
                self.buffer,
                num_episodes=cfg.rollout.episodes_per_epoch,
                max_steps_per_episode=cfg.rollout.max_steps_per_episode,
                model_module=self.module,
                curriculum=self.curriculum,
                reward_settings=cfg.reward,
                # Draw a fresh opponent per episode (not one for the whole epoch), so a
                # single epoch's batch mixes baselines and snapshots instead of being
                # all-vs-one — which was making game length and the critic target swing
                # wildly from epoch to epoch.
                sample_opponent=self.pool.sample,
                seed=cfg.seed + epoch,
            )

        self.buffer.self_play = cfg.ppo.self_play

        from kaisparov.training.parallel_rollout import resolve_num_workers

        if resolve_num_workers(cfg.rollout.num_workers) > 1 and cfg.rollout.episodes_per_epoch > 1:
            from kaisparov.training.parallel_rollout import collect_data_parallel

            return collect_data_parallel(
                self.agent,
                self.buffer,
                num_workers=cfg.rollout.num_workers,
                num_episodes=cfg.rollout.episodes_per_epoch,
                max_steps_per_episode=cfg.rollout.max_steps_per_episode,
                model_name=cfg.model,
                hidden_dim=cfg.hidden_dim,
                reward_settings=cfg.reward,
                curriculum_settings=cfg.curriculum,
                gamma=cfg.ppo.gamma,
                gae_lambda=cfg.ppo.gae_lambda,
                self_play=cfg.ppo.self_play,
                base_seed=cfg.seed + epoch,
            )

        return self.spec.collect_data(
            self.agent,
            ChessGame(),
            self.buffer,
            num_episodes=cfg.rollout.episodes_per_epoch,
            max_steps_per_episode=cfg.rollout.max_steps_per_episode,
            model_module=self.module,
            curriculum=self.curriculum,
            reward_fn=self.reward_fn,
        )

    # ------------------------------------------------------------------ training
    def train(self) -> str:
        cfg = self.config
        start = "curriculum" if self.curriculum is not None else "standard position"
        print(
            f"Run {self.run.run_id} | device={self.device} | params={self._num_params} | "
            f"start={start}"
        )
        if cfg.title:
            print(f"  {cfg.title}")
        last_eval: dict[str, float] = {}
        metrics: dict[str, float] = {}

        with self.run:
            for epoch in range(self.start_epoch + 1, self.start_epoch + cfg.epochs + 1):
                self.buffer.clear()
                rollout_stats = self._collect(epoch)
                metrics = self.spec.train_one_epoch(
                    self.agent,
                    self.buffer,
                    self.optimizer,
                    device=self.device,
                    update_epochs=cfg.ppo.update_epochs,
                    clip_eps=cfg.ppo.clip_eps,
                    value_coef=cfg.ppo.value_coef,
                    entropy_coef=cfg.ppo.entropy_coef,
                    max_grad_norm=cfg.ppo.max_grad_norm,
                )
                self.run.log_metrics(epoch, metrics, section="train")
                if rollout_stats:
                    self.run.log_metrics(epoch, rollout_stats, section="rollout")

                if cfg.eval.enabled and epoch % cfg.eval.every == 0:
                    last_eval = self.evaluate()
                    self.run.log_eval(epoch, last_eval)

                self._print_progress(epoch, metrics, rollout_stats, last_eval)

                if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0:
                    self.run.save_checkpoint(
                        self.agent,
                        epoch,
                        {**metrics, **last_eval},
                        best_metric=BEST_METRIC,
                        best_mode="max",
                        trainer_state=self._trainer_state(epoch),
                    )

                if (
                    self.pool is not None
                    and self.take_snapshots
                    and self.snapshot_every
                    and epoch % self.snapshot_every == 0
                ):
                    self.pool.snapshot(self.agent)

            self.run.finish(final_metrics={**metrics, **last_eval})

        print(f"Done. Artifacts in {self.run.dir}")
        return self.run.run_id

    # ---------------------------------------------------------------- evaluation
    def evaluate(self) -> dict[str, float]:
        from kaisparov.agents.neural_agent import NeuralAgent

        neural = NeuralAgent(self.agent, self._processor, deterministic=True)
        cfg = self.config.eval
        vs_random = evaluate(
            neural, RandomAgent(seed=self.config.seed), cfg.games, max_plies=cfg.max_plies
        )
        vs_material = evaluate(
            neural, MaterialAgent(seed=self.config.seed), cfg.games, max_plies=cfg.max_plies
        )
        return {
            "winrate_vs_random": vs_random.score_a,
            "elo_vs_random": vs_random.elo_diff,
            "winrate_vs_material": vs_material.score_a,
            "elo_vs_material": vs_material.elo_diff,
        }

    def _print_progress(
        self, epoch: int, metrics: dict, rollout_stats: dict, last_eval: dict
    ) -> None:
        line = (
            f"[epoch {epoch:>3}] loss={metrics.get('loss', 0):+.4f} "
            f"policy={metrics.get('policy_loss', 0):+.4f} "
            f"value={metrics.get('value_loss', 0):.4f} "
            f"entropy={metrics.get('entropy', 0):.3f} "
            f"| king_capture={rollout_stats.get('king_capture_rate', 0):.0%} "
            f"plies={rollout_stats.get('avg_plies', 0):.0f}"
        )
        # In league (pool) mode the learner's own win/loss split is the informative
        # signal; the self-play rollout doesn't report one, hence the guard.
        if "winrate" in rollout_stats:
            line += f" (w{rollout_stats['winrate']:.0%}/l{rollout_stats.get('lossrate', 0):.0%})"
        # An epoch whose PPO passes were all skipped for a non-finite loss reports
        # zeros above — flag it so it isn't mistaken for a genuine zero-loss epoch.
        skips = metrics.get("nonfinite_skips", 0)
        if skips:
            line += f" | SKIPPED (non-finite loss ×{skips:.0f})"
        if last_eval and self.config.eval.every and epoch % self.config.eval.every == 0:
            # vs_material is the discriminating eval (vs_random saturates at 100%); show
            # both, and it's what BEST_METRIC selects checkpoints on.
            line += (
                f" | vs_random={last_eval['winrate_vs_random']:.0%} "
                f"vs_material={last_eval['winrate_vs_material']:.0%} "
                f"(elo {last_eval['elo_vs_material']:+.0f})"
            )
        print(line)
