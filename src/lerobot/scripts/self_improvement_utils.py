"""Self-improvement loop for ACT Simple + AWM head policy.

Provides building blocks for an autoresearch agent to iteratively evaluate,
collect trajectories, finetune, and re-evaluate a BC + world-model policy.

Building blocks
───────────────
  eval_and_collect()       Local GPU eval → (metrics, episodes)  [inner loop]
  evaluate_final()         Submit SLURM multi-seed eval jobs (non-blocking)
  collect_eval_results()   Aggregate results after eval jobs complete
  TrajectoryBuffer         Stores trajectories, filters by success/failure
  finetune()               End-to-end training on success data + pretrain replay
  finetune_wm()            WM-only training on all/failure data + pretrain replay

Design
──────
``finetune`` trains the full model (encoder + policy + WM) on successful online
trajectories mixed with pretrain replay data.  ``finetune_wm`` freezes everything
except WM decoder internals (including ``wm_cross_attn_proj``) and trains on
suboptimal online trajectories mixed with pretrain replay.  Both functions use
a fixed pretrain/online batch ratio to prevent catastrophic forgetting.

The intended pipeline is::

    eval_and_collect → finetune → finetune_wm → eval_and_collect → ...

Both finetuning functions accept a ``wandb_run`` and ``global_step`` to enable
continuous WandB logging across the full self-improvement loop.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. Inner-loop eval — local GPU, returns metrics + trajectories
# ═══════════════════════════════════════════════════════════════════


def eval_and_collect(
    policy_path: str,
    env_type: str = "pusht",
    n_episodes: int = 50,
    seed: int = 42,
    device: str = "cuda",
    use_planning: bool = False,
    planning_algorithm: str = "gcp",
) -> tuple[dict, list[dict]]:
    """Evaluate on the current GPU and return metrics + trajectory data.

    This is the inner-loop workhorse: no SLURM, no waiting.  Runs
    *n_episodes* rollouts on the current device, computes success rate /
    rewards, and returns the full episode data for finetuning.

    Args:
        policy_path: Path to pretrained_model directory.
        env_type: Environment type (default ``"pusht"``).
        n_episodes: Number of rollout episodes.
        seed: Random seed for env resets.
        device: Torch device for inference.
        use_planning: Enable latent-space planning at test time.
        planning_algorithm: ``"mppi"`` or ``"gcp"``.

    Returns:
        (metrics, episodes):
            *metrics* — dict with ``pc_success``, ``avg_sum_reward``,
            ``avg_max_reward``, ``n_episodes``, and ``per_episode`` list.
            *episodes* — list of episode dicts, each containing::

                {
                    "observations": {"observation.image": (T, C, H, W), ...},
                    "actions":  (T-1, action_dim),
                    "rewards":  (T-1,),
                    "success":  bool,
                }
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.envs.goal_provider import make_goal_provider
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy

    # ── env ──────────────────────────────────────────────────────
    env_cfg = make_env_config(env_type)
    envs_dict = make_env(env_cfg, n_envs=n_episodes)
    vec_env = next(iter(next(iter(envs_dict.values())).values()))

    # ── policy ───────────────────────────────────────────────────
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = device

    if use_planning:
        policy_cfg.use_planning = True
        policy_cfg.planning.algorithm = planning_algorithm

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    if use_planning and (not hasattr(policy, "_planner") or policy._planner is None):
        from lerobot.policies.act_simple_with_awm_head.planning import make_planner

        policy._planner = make_planner(policy.config.planning)

    # ── processors ───────────────────────────────────────────────
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg,
    )

    # ── goal provider (for planning) ─────────────────────────────
    goal_provider = None
    if use_planning:
        try:
            goal_provider = make_goal_provider(env_type)
        except ValueError:
            logger.warning("No goal provider for %s; falling back to BC.", env_type)

    # ── run evaluation ───────────────────────────────────────────
    with torch.no_grad():
        info = eval_policy(
            env=vec_env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            return_episode_data=True,
            start_seed=seed,
            goal_provider=goal_provider,
        )

    vec_env.close()

    # ── build return values ──────────────────────────────────────
    metrics = {
        **info["aggregated"],
        "n_episodes": n_episodes,
        "per_episode": info["per_episode"],
    }
    episodes = _episodes_from_eval_info(info)
    return metrics, episodes


# ═══════════════════════════════════════════════════════════════════
# 2. Decisive multi-seed eval — SLURM parallel
# ═══════════════════════════════════════════════════════════════════


def evaluate_final(
    policy_path: str,
    env_type: str = "pusht",
    seeds: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000),
    n_episodes_per_seed: int = 50,
    compute_script: str = "compute_inference.sh",
    output_dir: str | None = None,
    policy_overrides: list[str] | None = None,
) -> tuple[dict[int, str], str]:
    """Submit parallel SLURM eval jobs (non-blocking).

    Submits one ``lerobot-eval`` job per seed and **returns immediately**
    with the SLURM job IDs.  The caller (autoresearch agent) is responsible
    for babysitting the jobs via ``squeue`` and calling
    :func:`collect_eval_results` once they complete.

    5 seeds × 50 episodes → 250 total episodes, SE ≈ 3.2%, CI ≈ ±6.3%.

    Args:
        policy_path: Path to pretrained_model directory.
        env_type: Gymnasium environment identifier (default ``"pusht"``).
        seeds: Random seeds — one SLURM job per seed.
        n_episodes_per_seed: Episodes evaluated per seed.
        compute_script: SLURM wrapper script (e.g. ``"compute_inference.sh"``).
        output_dir: Root directory for per-seed results.  Auto-generated from
            *policy_path* if ``None``.
        policy_overrides: Extra ``lerobot-eval`` CLI flags, e.g.
            ``["--policy.use_planning=true"]``.

    Returns:
        (job_ids, eval_dir):
            *job_ids* — dict mapping seed → SLURM job ID string.
            *eval_dir* — path to the directory containing per-seed outputs.
            Pass *eval_dir* to :func:`collect_eval_results` after jobs finish.
    """
    if output_dir is None:
        parent = Path(policy_path).parent
        output_dir = str(parent / "eval" / f"eval_{int(time.time())}")
    eval_dir = Path(output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── clear inherited SLURM env vars to avoid CPU-binding errors ─
    for key in list(os.environ.keys()):
        if key.startswith("SLURM_"):
            del os.environ[key]

    # ── submit one job per seed ──────────────────────────────────
    job_ids: dict[int, str] = {}
    for seed in seeds:
        seed_dir = eval_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "sbatch", compute_script,
            "lerobot-eval",
            f"--policy.path={policy_path}",
            f"--env.type={env_type}",
            f"--eval.n_episodes={n_episodes_per_seed}",
            f"--eval.batch_size={n_episodes_per_seed}",
            f"--seed={seed}",
            f"--output_dir={seed_dir}",
            "--policy.device=cuda",
        ]
        if policy_overrides:
            cmd.extend(policy_overrides)

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids[seed] = job_id
        logger.info("Submitted eval seed=%d → job %s", seed, job_id)

    return job_ids, str(eval_dir)


def collect_eval_results(
    eval_dir: str,
    seeds: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000),
) -> dict:
    """Aggregate results from completed multi-seed eval jobs.

    Call this **after** all SLURM jobs from :func:`evaluate_final` have
    finished.  Reads ``eval_info.json`` from each seed directory and
    computes mean ± std success rate.

    Args:
        eval_dir: Path returned by :func:`evaluate_final`.
        seeds: The same seeds that were passed to :func:`evaluate_final`.

    Returns:
        dict with ``mean_success``, ``std_success``, ``mean_eval_ep_s``,
        ``per_seed`` breakdown, and ``n_seeds``.
    """
    eval_path = Path(eval_dir)
    per_seed: dict[int, dict] = {}
    for seed in seeds:
        metrics_file = eval_path / f"seed_{seed}" / "eval_info.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                per_seed[seed] = json.load(f)
        else:
            logger.warning("No results for seed %d at %s", seed, metrics_file)

    success_rates = []
    ep_speeds = []
    for info in per_seed.values():
        overall = info.get("overall", info.get("aggregated", {}))
        if "pc_success" in overall:
            success_rates.append(overall["pc_success"])
        if "eval_ep_s" in overall:
            ep_speeds.append(overall["eval_ep_s"])

    return {
        "mean_success": float(np.mean(success_rates)) if success_rates else 0.0,
        "std_success": float(np.std(success_rates)) if success_rates else 0.0,
        "mean_eval_ep_s": float(np.mean(ep_speeds)) if ep_speeds else 0.0,
        "per_seed": {
            str(s): info.get("overall", info.get("aggregated", {}))
            for s, info in per_seed.items()
        },
        "n_seeds": len(success_rates),
    }


# ═══════════════════════════════════════════════════════════════════
# Helpers — episode extraction
# ═══════════════════════════════════════════════════════════════════


def _episodes_from_eval_info(info: dict) -> list[dict]:
    """Split flat eval_policy output into per-episode dicts."""
    ep_data = info["episodes"]
    per_ep = info["per_episode"]

    episodes: list[dict] = []
    for ep_info in per_ep:
        ep_idx = ep_info["episode_ix"]
        mask = ep_data["episode_index"] == ep_idx

        obs: dict[str, Tensor] = {}
        for key in ep_data:
            if key.startswith("observation.") and not key.endswith("_is_pad"):
                obs[key] = ep_data[key][mask]

        actions = ep_data["action"][mask]
        rewards = ep_data["reward"][mask] if "reward" in ep_data else torch.zeros(mask.sum())

        episodes.append({
            "observations": obs,
            "actions": actions,       # (N,) — last frame is copy-padded
            "rewards": rewards,
            "success": bool(ep_info["success"]),
        })
    return episodes


# ═══════════════════════════════════════════════════════════════════
# 3. Trajectory buffer
# ═══════════════════════════════════════════════════════════════════


class TrajectoryBuffer:
    """Accumulates evaluation episodes for finetuning.

    Episodes are stored as dicts returned by :func:`eval_and_collect`.
    Call :meth:`add` to append episodes and :meth:`as_dataset` to build a
    PyTorch ``Dataset`` for the training loop.
    """

    def __init__(self) -> None:
        self.episodes: list[dict] = []

    # ── mutators ─────────────────────────────────────────────────

    def add(self, episodes: list[dict]) -> None:
        """Append a batch of episode dicts to the buffer."""
        self.episodes.extend(episodes)

    def clear(self) -> None:
        self.episodes.clear()

    # ── accessors ────────────────────────────────────────────────

    @property
    def n_total(self) -> int:
        return len(self.episodes)

    @property
    def n_success(self) -> int:
        return sum(1 for ep in self.episodes if ep["success"])

    @property
    def n_fail(self) -> int:
        return self.n_total - self.n_success

    def as_dataset(self, mode: str, chunk_size: int) -> Dataset:
        """Build a :class:`_FinetuneDataset` from buffered episodes.

        Args:
            mode: ``"all"`` — every episode,
                  ``"success_only"`` — only successful episodes, or
                  ``"failure_only"`` — only failed episodes.
            chunk_size: Action chunk length (matches model config).

        Returns:
            Iterable :class:`torch.utils.data.Dataset`.
        """
        if mode == "success_only":
            eps = [ep for ep in self.episodes if ep["success"]]
        elif mode == "failure_only":
            eps = [ep for ep in self.episodes if not ep["success"]]
        elif mode == "all":
            eps = list(self.episodes)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Choose 'all', 'success_only', or 'failure_only'.")
        if not eps:
            raise RuntimeError(
                f"No episodes match mode={mode!r} "
                f"(buffer has {self.n_success} success, {self.n_fail} fail)."
            )
        return _FinetuneDataset(eps, chunk_size)

    def __repr__(self) -> str:
        return (
            f"TrajectoryBuffer(n_total={self.n_total}, "
            f"n_success={self.n_success}, n_fail={self.n_fail})"
        )


class _FinetuneDataset(Dataset):
    """Training samples: ``(current_obs, future_obs, actions)`` tuples.

    Each sample at starting timestep *t* within an episode provides:

    - Observations stacked as ``(2, *shape)`` — current at *t*, future at
      *t + chunk_size*.
    - Action chunk ``(chunk_size, action_dim)`` — zero-padded if the
      episode is shorter than the chunk.
    - Padding masks for actions and observations (for WM target validity).

    The format matches the batch layout expected by
    ``ACTSimpleWithAWMHeadPolicy.forward()``.
    """

    def __init__(self, episodes: list[dict], chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.samples: list[tuple[dict, int, int]] = []

        for ep in episodes:
            # N frames total (last frame copy-padded by eval); real actions = N-1
            N = ep["actions"].shape[0]
            T_actions = N - 1  # number of real action steps

            if T_actions < 1:
                continue

            # Valid starting indices: need a full chunk of actions + future obs
            n_valid = max(1, T_actions - chunk_size + 1)
            for t in range(n_valid):
                self.samples.append((ep, t, T_actions))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ep, t, T_actions = self.samples[idx]
        C = self.chunk_size

        batch: dict[str, Tensor] = {}

        # ── observations: stack current (t) and future (t + C) ──
        for key, obs_tensor in ep["observations"].items():
            current = obs_tensor[t]
            future_idx = min(t + C, T_actions)  # T_actions is index of last obs
            future = obs_tensor[future_idx]
            batch[key] = torch.stack([current, future])  # (2, *)

        # ── actions: (chunk_size, action_dim), zero-padded ───────
        action_end = min(t + C, T_actions)
        actions = ep["actions"][t:action_end]
        pad_len = C - actions.shape[0]
        if pad_len > 0:
            actions = torch.cat([actions, torch.zeros(pad_len, actions.shape[-1])])
        batch["action"] = actions

        # ── padding masks ────────────────────────────────────────
        action_is_pad = torch.zeros(C, dtype=torch.bool)
        if pad_len > 0:
            action_is_pad[C - pad_len :] = True
        batch["action_is_pad"] = action_is_pad

        obs_is_pad = torch.zeros(2, dtype=torch.bool)
        if t + C > T_actions:
            obs_is_pad[1] = True  # future observation beyond episode end
        batch["observation.state_is_pad"] = obs_is_pad

        return batch


# ═══════════════════════════════════════════════════════════════════
# 4. Pretrain replay dataset
# ═══════════════════════════════════════════════════════════════════


_pretrain_cache: dict[str, tuple] = {}


def load_pretrain_datasets(
    repo_id: str = "lerobot/pusht",
    chunk_size: int = 16,
    val_ratio: float = 0.1,
) -> tuple[Dataset, Dataset]:
    """Load pretrain dataset split into replay and validation portions.

    Uses LeRobotDataset with ``delta_timestamps`` to produce samples in the
    same (obs_t, obs_{t+C}, action_chunk) format as :class:`_FinetuneDataset`.

    Returns:
        (train_dataset, val_dataset)
    """
    cache_key = f"{repo_id}_{chunk_size}_{val_ratio}"
    if cache_key in _pretrain_cache:
        return _pretrain_cache[cache_key]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Load metadata to get fps and episode count
    meta_ds = LeRobotDataset(repo_id)
    fps = meta_ds.fps
    n_episodes = meta_ds.num_episodes
    del meta_ds

    # Compute delta_timestamps from chunk_size and fps
    obs_deltas = [0.0, chunk_size / fps]
    action_deltas = [i / fps for i in range(chunk_size)]
    delta_timestamps = {
        "observation.image": obs_deltas,
        "observation.state": obs_deltas,
        "observation.environment_state": obs_deltas,
        "action": action_deltas,
    }

    # Split episodes: last val_ratio for validation
    n_val = max(1, int(n_episodes * val_ratio))
    train_episodes = list(range(n_episodes - n_val))
    val_episodes = list(range(n_episodes - n_val, n_episodes))

    logger.info(
        "Pretrain replay: %d train episodes, %d val episodes (fps=%d, chunk=%d)",
        len(train_episodes), len(val_episodes), fps, chunk_size,
    )

    train_ds = LeRobotDataset(repo_id, episodes=train_episodes, delta_timestamps=delta_timestamps)
    val_ds = LeRobotDataset(repo_id, episodes=val_episodes, delta_timestamps=delta_timestamps)

    _pretrain_cache[cache_key] = (train_ds, val_ds)
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════════
# 5. Diagnostics — representation health & validation
# ═══════════════════════════════════════════════════════════════════


def _compute_effective_rank(z: Tensor) -> float:
    """Compute effective rank of representation matrix via Shannon entropy of SVs.

    Args:
        z: Tensor of shape (S, B, D) or (N, D).

    Returns:
        Effective rank (scalar).  Higher = more dimensions used.
    """
    if z.ndim == 3:
        z = z.permute(1, 0, 2).reshape(-1, z.shape[-1])  # (S*B, D)
    # Clamp to reasonable size to avoid OOM on SVD
    if z.shape[0] > 512:
        z = z[:512]
    try:
        s = torch.linalg.svdvals(z.float())
        s = s / s.sum().clamp(min=1e-10)
        log_s = torch.log(s.clamp(min=1e-10))
        eff_rank = torch.exp(-(s * log_s).sum())
        return eff_rank.item()
    except Exception:
        return -1.0


def _wm_forward_with_diagnostics(
    policy: torch.nn.Module,
    batch: dict[str, Tensor],
) -> tuple[Tensor, dict]:
    """WM-only forward pass with comprehensive diagnostics.

    Returns the same (loss, info) as :func:`_wm_forward` but with additional
    metrics for representation health monitoring.
    """
    from lerobot.policies.act_simple_with_awm_head.modeling_act_simple_with_awm_head import (
        _compute_wm_loss,
        _slice_obs_batch,
    )
    from lerobot.utils.constants import OBS_IMAGES

    config = policy.config
    model = policy.model

    curr_batch = _slice_obs_batch(batch, 0)
    next_batch = _slice_obs_batch(batch, 1)

    if config.image_features:
        curr_batch = dict(curr_batch)
        curr_batch[OBS_IMAGES] = [curr_batch[key] for key in config.image_features]
        next_batch = dict(next_batch)
        next_batch[OBS_IMAGES] = [next_batch[key] for key in config.image_features]

    # ── encode current observation ───────────────────────────────
    batch_size, encoder_out, encoder_pos, encoder_in = model._encode(curr_batch)

    # ── encode next observation (target, stop-gradient) ──────────
    if config.use_ema_target:
        z_target = model._encode_ema(next_batch)
    else:
        with torch.no_grad():
            _, _, _, next_encoder_in = model._encode(next_batch)
        z_target = next_encoder_in.detach()
    if config.normalize_wm_representations:
        z_target = F.normalize(z_target, dim=-1)

    # ── world-model decoder ──────────────────────────────────────
    actions = batch["action"]  # (B, T, action_dim)
    T = actions.shape[1]
    action_embeds = model.wm_action_proj(actions).transpose(0, 1)
    wm_action_pos = model.wm_action_pos_embed.weight[:T].unsqueeze(1)

    S = model.n_encoder_tokens
    query_pos = model.wm_query_pos_embed.weight.unsqueeze(1)
    queries = (model.wm_query_tokens + query_pos).expand(-1, batch_size, -1)
    wm_in = torch.cat([queries, action_embeds + wm_action_pos], dim=0)

    wm_encoder_in = encoder_in.detach() if config.detach_encoder_from_wm else encoder_in
    wm_cross_kv = model.wm_cross_attn_proj(wm_encoder_in)
    wm_cross_pos = encoder_pos
    wm_out = model.wm_decoder(wm_in, wm_cross_kv, wm_cross_pos)
    z_pred = model.wm_proj_head(wm_out[:S])
    if config.normalize_wm_representations:
        z_pred = F.normalize(z_pred, dim=-1)

    # ── loss ─────────────────────────────────────────────────────
    next_obs_is_pad = batch.get(
        "observation.state_is_pad",
        batch.get("observation.environment_state_is_pad"),
    )
    valid_wm = ~next_obs_is_pad[:, 1]
    wm_loss = _compute_wm_loss(z_pred, z_target, valid_wm)

    # ── diagnostics ──────────────────────────────────────────────
    z_pred_d = z_pred.detach()
    z_target_d = z_target.detach()

    z_pred_norm = z_pred_d.norm(dim=-1).mean().item()
    z_target_norm = z_target_d.norm(dim=-1).mean().item()
    z_pred_batch_std = z_pred_d.std(dim=1).mean().item()
    z_target_batch_std = z_target_d.std(dim=1).mean().item()

    info = {
        "loss": wm_loss.item(),
        "wm_loss": wm_loss.item(),
        "wm_cosine_sim": F.cosine_similarity(z_pred_d, z_target_d, dim=-1).mean().item(),
        "z_pred_norm": z_pred_norm,
        "z_target_norm": z_target_norm,
        "z_pred_target_norm_ratio": z_pred_norm / max(z_target_norm, 1e-8),
        "z_pred_batch_std": z_pred_batch_std,
        "z_target_batch_std": z_target_batch_std,
    }
    return wm_loss, info


@torch.no_grad()
def _compute_validation_metrics(
    policy: torch.nn.Module,
    val_loader_iter,
    preprocessor,
    device: str = "cuda",
) -> dict:
    """Run one forward pass on pretrain validation data and return metrics.

    Computes both WM-only metrics and action loss (even during WM-only mode)
    to detect unexpected drift in the frozen policy.
    """
    policy.eval()
    try:
        val_batch = next(val_loader_iter)
    except StopIteration:
        return {}

    val_batch = preprocessor(val_batch)

    # Full forward for action_loss + wm metrics
    try:
        loss, info = policy.forward(val_batch)
        metrics = {
            "val/wm_loss": info.get("wm_loss", 0.0),
            "val/action_loss": info.get("action_loss", 0.0),
            "val/wm_cosine_sim": info.get("wm_cosine_sim", 0.0),
            "val/z_pred_norm": info.get("z_pred_norm", 0.0),
            "val/z_target_norm": info.get("z_target_norm", 0.0),
            "val/z_pred_target_norm_ratio": info.get("z_pred_target_norm_ratio", 0.0),
            "val/z_pred_batch_std": info.get("z_pred_batch_std", 0.0),
            "val/z_target_batch_std": info.get("z_target_batch_std", 0.0),
            "val/loss": info.get("loss", 0.0),
        }
    except Exception as e:
        logger.warning("Validation forward failed: %s", e)
        metrics = {}

    policy.train()
    return metrics


# ═══════════════════════════════════════════════════════════════════
# 6. Finetuning — with pretrain replay mixing & WandB logging
# ═══════════════════════════════════════════════════════════════════

# WM parameters INCLUDING wm_cross_attn_proj (used by finetune for full model)
_WM_PREFIXES = (
    "model.wm_decoder",
    "model.wm_action_proj",
    "model.wm_action_pos_embed",
    "model.wm_query_tokens",
    "model.wm_query_pos_embed",
    "model.wm_proj_head",
    "model.wm_cross_attn_proj",
    "model.wm_image_decoder",
)

# WM parameters EXCLUDING wm_cross_attn_proj (used by finetune_wm)
_WM_STRICT_PREFIXES = (
    "model.wm_decoder",
    "model.wm_action_proj",
    "model.wm_action_pos_embed",
    "model.wm_query_tokens",
    "model.wm_query_pos_embed",
    "model.wm_proj_head",
    "model.wm_image_decoder",
)


def _cat_batches(b1: dict[str, Tensor], b2: dict[str, Tensor]) -> dict[str, Tensor]:
    """Concatenate two batch dicts along the batch dimension."""
    out: dict[str, Tensor] = {}
    for key in b1:
        if key in b2 and isinstance(b1[key], Tensor) and isinstance(b2[key], Tensor):
            out[key] = torch.cat([b1[key], b2[key]], dim=0)
        else:
            out[key] = b1[key]
    return out


def _load_model_and_preprocessors(
    policy_path: str,
    device: str,
    detach_encoder_from_wm: bool = False,
):
    """Load the policy, preprocessor, and postprocessor."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env_config
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    env_cfg = make_env_config("pusht")
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = device
    policy_cfg.detach_encoder_from_wm = detach_encoder_from_wm
    policy_cfg.wm_warmup_steps = 0

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, policy_cfg


def finetune(
    policy_path: str,
    buffer: TrajectoryBuffer,
    commit_hash: str,
    n_steps: int = 500,
    lr: float = 5e-6,
    batch_size: int = 8,
    pretrain_ratio: float = 0.5,
    grad_clip_norm: float = 10.0,
    device: str = "cuda",
    log_interval: int = 10,
    val_interval: int = 50,
    health_interval: int = 100,
    wandb_run=None,
    global_step: int = 0,
) -> tuple[str, int]:
    """End-to-end finetuning on successful online trajectories + pretrain replay.

    Trains the **full model** (BC head + WM) on successful online data mixed
    with pretrain replay.  Intended to settle the encoder representations
    before WM-only adaptation via :func:`finetune_wm`.

    Batch composition: ``pretrain_ratio`` of each batch comes from the pretrain
    dataset, the rest from successful online episodes.

    Args:
        policy_path: Path to pretrained_model directory.
        buffer: :class:`TrajectoryBuffer` — successes are extracted automatically.
        commit_hash: Git commit hash for checkpoint namespacing.
        n_steps: Number of gradient update steps.
        lr: Learning rate (default 5e-6, slightly aggressive for success data).
        batch_size: Total batch size (split between pretrain and online).
        pretrain_ratio: Fraction of each batch from pretrain data (default 0.5).
        grad_clip_norm: Max gradient norm for clipping.
        device: Torch device.
        log_interval: Log metrics to wandb every N steps.
        val_interval: Compute validation metrics every N steps.
        health_interval: Compute representation health every N steps.
        wandb_run: Active wandb run object (or None to skip logging).
        global_step: Starting step counter for wandb logging.

    Returns:
        (checkpoint_path, new_global_step)
    """
    from lerobot.datasets.utils import cycle

    policy, preprocessor, postprocessor, policy_cfg = _load_model_and_preprocessors(
        policy_path, device, detach_encoder_from_wm=False,
    )
    policy.train()
    chunk_size = policy_cfg.chunk_size

    # ── online dataset (success-only) ────────────────────────────
    online_ds = buffer.as_dataset(mode="success_only", chunk_size=chunk_size)
    pretrain_bs = max(1, round(batch_size * pretrain_ratio))
    online_bs = max(1, batch_size - pretrain_bs)

    online_loader = DataLoader(
        online_ds, batch_size=online_bs, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"), drop_last=True,
    )

    # ── pretrain replay + validation ─────────────────────────────
    pretrain_train_ds, pretrain_val_ds = load_pretrain_datasets(chunk_size=chunk_size)
    pretrain_loader = DataLoader(
        pretrain_train_ds, batch_size=pretrain_bs, shuffle=True, num_workers=2,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        pretrain_val_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"), drop_last=True,
    )

    online_iter = cycle(online_loader)
    pretrain_iter = cycle(pretrain_loader)
    val_iter = cycle(val_loader)

    logger.info(
        "finetune: %d online success samples, %d pretrain samples, "
        "batch=%d (pretrain=%d + online=%d), lr=%.1e, steps=%d",
        len(online_ds), len(pretrain_train_ds), batch_size, pretrain_bs, online_bs, lr, n_steps,
    )

    # ── optimizer (all params trainable) ─────────────────────────
    trainable_params = list(policy.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info("finetune: %d trainable params (full model)", n_trainable)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # ── training loop ────────────────────────────────────────────
    for step in range(1, n_steps + 1):
        pretrain_batch = next(pretrain_iter)
        online_batch = next(online_iter)
        batch = _cat_batches(pretrain_batch, online_batch)
        batch = preprocessor(batch)

        loss, info = policy.forward(batch)

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()

        if hasattr(policy, "update"):
            policy.update()

        # ── logging ──────────────────────────────────────────────
        gs = global_step + step
        if wandb_run is not None and step % log_interval == 0:
            log_dict = {f"finetune/{k}": v for k, v in info.items()}
            log_dict["finetune/lr"] = lr
            wandb_run.log(log_dict, step=gs)

        if wandb_run is not None and step % val_interval == 0:
            val_metrics = _compute_validation_metrics(policy, val_iter, preprocessor, device)
            if val_metrics:
                wandb_run.log({f"finetune/{k}": v for k, v in val_metrics.items()}, step=gs)
            policy.train()

        if wandb_run is not None and step % health_interval == 0:
            # Effective rank on a fresh batch
            with torch.no_grad():
                health_batch = next(online_iter)
                health_batch = preprocessor(health_batch)
                _, h_info = _wm_forward_with_diagnostics(policy, health_batch)
            wandb_run.log({
                "finetune/health/z_pred_effective_rank": _compute_effective_rank(
                    torch.zeros(1)  # placeholder — computed inside forward
                ),
                **{f"finetune/health/{k}": v for k, v in h_info.items()},
            }, step=gs)

        if step % (log_interval * 5) == 0 or step == 1:
            parts = [f"[finetune] step {step}/{n_steps}"]
            for k in ("loss", "action_loss", "wm_loss", "wm_cosine_sim"):
                if k in info:
                    parts.append(f"{k}={info[k]:.4f}")
            logger.info("  ".join(parts))

    # ── save checkpoint ──────────────────────────────────────────
    new_global_step = global_step + n_steps
    save_dir = (
        Path(policy_path).parent / "self_improvement" / commit_hash / f"step_{new_global_step}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)
    logger.info("Saved finetuned model → %s", save_dir)

    return str(save_dir), new_global_step


def finetune_wm(
    policy_path: str,
    buffer: TrajectoryBuffer,
    commit_hash: str,
    n_steps: int = 500,
    lr: float = 1e-6,
    batch_size: int = 8,
    pretrain_ratio: float = 0.6,
    online_mode: str = "all",
    grad_clip_norm: float = 10.0,
    device: str = "cuda",
    log_interval: int = 10,
    val_interval: int = 50,
    health_interval: int = 100,
    wandb_run=None,
    global_step: int = 0,
) -> tuple[str, int]:
    """WM-only finetuning with frozen cross-attention projection + pretrain replay.

    Freezes **all** parameters except WM decoder internals.  Crucially, also
    freezes ``wm_cross_attn_proj`` so the WM decoder sees the same key-value
    space as during pretraining and only adjusts query/action processing.

    Batch composition: ``pretrain_ratio`` from pretrain, rest from online data
    (defaults to all trajectories including failures for OOD state coverage).

    Args:
        policy_path: Path to pretrained_model directory.
        buffer: :class:`TrajectoryBuffer` with online episodes.
        commit_hash: Git commit hash for checkpoint namespacing.
        n_steps: Number of gradient update steps.
        lr: Learning rate (default 1e-6, conservative for WM-only).
        batch_size: Total batch size (split between pretrain and online).
        pretrain_ratio: Fraction of each batch from pretrain data (default 0.6).
        online_mode: ``"all"``, ``"success_only"``, or ``"failure_only"``
            for the online portion of each batch.
        grad_clip_norm: Max gradient norm for clipping.
        device: Torch device.
        log_interval: Log metrics to wandb every N steps.
        val_interval: Compute validation metrics every N steps.
        health_interval: Compute representation health every N steps.
        wandb_run: Active wandb run object (or None to skip logging).
        global_step: Starting step counter for wandb logging.

    Returns:
        (checkpoint_path, new_global_step)
    """
    from lerobot.datasets.utils import cycle

    policy, preprocessor, postprocessor, policy_cfg = _load_model_and_preprocessors(
        policy_path, device, detach_encoder_from_wm=True,
    )
    chunk_size = policy_cfg.chunk_size

    # ── freeze: everything except WM strict params ───────────────
    for name, param in policy.named_parameters():
        if not any(name.startswith(p) for p in _WM_STRICT_PREFIXES):
            param.requires_grad = False

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total_params = sum(p.numel() for p in policy.parameters())
    logger.info(
        "finetune_wm: %d / %d params trainable (%.1f%%), wm_cross_attn_proj FROZEN",
        n_trainable, n_total_params, 100 * n_trainable / max(n_total_params, 1),
    )
    policy.train()

    # ── online dataset ───────────────────────────────────────────
    online_ds = buffer.as_dataset(mode=online_mode, chunk_size=chunk_size)
    pretrain_bs = max(1, round(batch_size * pretrain_ratio))
    online_bs = max(1, batch_size - pretrain_bs)

    online_loader = DataLoader(
        online_ds, batch_size=online_bs, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"), drop_last=True,
    )

    # ── pretrain replay + validation ─────────────────────────────
    pretrain_train_ds, pretrain_val_ds = load_pretrain_datasets(chunk_size=chunk_size)
    pretrain_loader = DataLoader(
        pretrain_train_ds, batch_size=pretrain_bs, shuffle=True, num_workers=2,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        pretrain_val_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"), drop_last=True,
    )

    online_iter = cycle(online_loader)
    pretrain_iter = cycle(pretrain_loader)
    val_iter = cycle(val_loader)

    logger.info(
        "finetune_wm: %d online %s samples, %d pretrain samples, "
        "batch=%d (pretrain=%d + online=%d), lr=%.1e, steps=%d",
        len(online_ds), online_mode, len(pretrain_train_ds),
        batch_size, pretrain_bs, online_bs, lr, n_steps,
    )

    # ── optimizer ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # ── training loop ────────────────────────────────────────────
    for step in range(1, n_steps + 1):
        pretrain_batch = next(pretrain_iter)
        online_batch = next(online_iter)
        batch = _cat_batches(pretrain_batch, online_batch)
        batch = preprocessor(batch)

        loss, info = _wm_forward_with_diagnostics(policy, batch)

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()

        if hasattr(policy, "update"):
            policy.update()

        # ── logging ──────────────────────────────────────────────
        gs = global_step + step
        if wandb_run is not None and step % log_interval == 0:
            log_dict = {f"finetune_wm/{k}": v for k, v in info.items()}
            log_dict["finetune_wm/lr"] = lr
            wandb_run.log(log_dict, step=gs)

        if wandb_run is not None and step % val_interval == 0:
            val_metrics = _compute_validation_metrics(policy, val_iter, preprocessor, device)
            if val_metrics:
                wandb_run.log({f"finetune_wm/{k}": v for k, v in val_metrics.items()}, step=gs)
            policy.train()

        if wandb_run is not None and step % health_interval == 0:
            with torch.no_grad():
                health_batch = next(online_iter)
                health_batch = preprocessor(health_batch)
                _, h_info = _wm_forward_with_diagnostics(policy, health_batch)
            # Compute effective rank
            eff_rank = -1.0
            try:
                # Quick forward to get z_pred for SVD
                from lerobot.policies.act_simple_with_awm_head.modeling_act_simple_with_awm_head import (
                    _slice_obs_batch,
                )
                from lerobot.utils.constants import OBS_IMAGES

                config = policy.config
                model = policy.model
                cb = _slice_obs_batch(health_batch, 0)
                if config.image_features:
                    cb = dict(cb)
                    cb[OBS_IMAGES] = [cb[key] for key in config.image_features]
                _, _, _, enc_in = model._encode(cb)
                actions = health_batch["action"]
                T = actions.shape[1]
                ae = model.wm_action_proj(actions).transpose(0, 1)
                ap = model.wm_action_pos_embed.weight[:T].unsqueeze(1)
                S = model.n_encoder_tokens
                B = enc_in.shape[1]
                qp = model.wm_query_pos_embed.weight.unsqueeze(1)
                q = (model.wm_query_tokens + qp).expand(-1, B, -1)
                wi = torch.cat([q, ae + ap], dim=0)
                ckv = model.wm_cross_attn_proj(enc_in.detach())
                from lerobot.policies.act_simple_with_awm_head.modeling_act_simple_with_awm_head import (
                    _slice_obs_batch as _sob,
                )
                # encoder_pos from encode
                _, _, epos, _ = model._encode(cb)
                wo = model.wm_decoder(wi, ckv, epos)
                zp = model.wm_proj_head(wo[:S])
                eff_rank = _compute_effective_rank(zp)
            except Exception:
                pass

            wandb_run.log({
                "finetune_wm/health/effective_rank": eff_rank,
                **{f"finetune_wm/health/{k}": v for k, v in h_info.items()},
            }, step=gs)

        if step % (log_interval * 5) == 0 or step == 1:
            parts = [f"[finetune_wm] step {step}/{n_steps}"]
            for k in ("loss", "wm_loss", "wm_cosine_sim", "z_pred_target_norm_ratio"):
                if k in info:
                    parts.append(f"{k}={info[k]:.4f}")
            logger.info("  ".join(parts))

    # ── save checkpoint ──────────────────────────────────────────
    new_global_step = global_step + n_steps
    save_dir = (
        Path(policy_path).parent / "self_improvement" / commit_hash / f"step_{new_global_step}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)
    logger.info("Saved WM-finetuned model → %s", save_dir)

    return str(save_dir), new_global_step
