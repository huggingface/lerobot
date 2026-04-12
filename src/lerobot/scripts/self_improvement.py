"""Self-improvement pipeline v2 — thin orchestrator over lerobot-train / lerobot-eval.

Pipeline per iteration:
    1. Evaluate current policy & collect on-policy trajectories.
    2. Package trajectories into a LeRobotDataset on disk.
    3. Call ``train()`` in-process with a ``_FinetuneDataset`` that
       concatenates pretrain + online data.

Usage (CLI):
    lerobot-self-improve --policy_path=/path/to/pretrained_model --n_iters=3
    # or
    python -m lerobot.scripts.self_improvement --policy_path=/path/to/pretrained_model
"""

import datetime as dt
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lerobot.configs import parser
from lerobot.policies.act_simple_with_awm_head.planning import PlanningConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SelfImprovementConfig:
    """Configuration for the self-improvement pipeline."""

    # ── Pretrain checkpoint (the starting point) ──────────────────
    policy_path: str = ""
    pretrain_dataset_repo_id: str = "lerobot/pusht"
    pretrain_dataset_root: str | None = None
    task_description: str = "Push the T-shaped block onto the target."

    # ── Self-improvement loop ─────────────────────────────────────
    n_iters: int = 0
    n_collect_episodes: int = 50
    finetune_steps: int = 100
    finetune_lr: float | None = 5e-6
    batch_size: int = 8
    bc_mask_mode: str = "none"  # "none" or "failure"
    trainable_param_keywords: list[str] | None = None  # e.g. ["wm_"] to only train WM head
    eval_seed: int = 42
    log_freq: int = 50
    save_freq: int | None = None

    # ── Collection (and default eval) planning ─────────────────────
    use_planning: bool = True
    planner: PlanningConfig = field(default_factory=PlanningConfig)

    # ── Final eval ────────────────────────────────────────────────
    eval_n_episodes: int = 250
    # Override planning config for final eval only.
    # When None, final eval uses the same use_planning/planner as collection.
    eval_use_planning: bool | None = None
    eval_planner: PlanningConfig | None = None

    # ── WandB ─────────────────────────────────────────────────────
    wandb_project: str = "awm"
    wandb_entity: str = ""

    # ── General ───────────────────────────────────────────────────
    experiment_name: str = ""  # e.g. "bc-finetune-lr5e6"
    seed: int = 1000
    cudnn_deterministic: bool = True
    device: str = "cuda"


# ═════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════

def eval_and_collect(
    policy_path: str,
    n_episodes: int,
    seed: int = 42,
    device: str = "cuda",
    use_planning: bool = False,
    planner: PlanningConfig | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate the policy and collect trajectory data.

    Returns ``(metrics, episodes)`` where *episodes* is a list of dicts
    ready for :func:`episodes_to_lerobot_dataset`.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.envs.goal_provider import make_goal_provider
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.scripts.self_improvement_data import episodes_from_eval_info

    # ── Environment ──────────────────────────────────────────
    env_cfg = make_env_config("pusht")
    envs_dict = make_env(env_cfg, n_envs=n_episodes)
    vec_env = next(iter(next(iter(envs_dict.values())).values()))

    # ── Policy ───────────────────────────────────────────────
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = device

    if use_planning and planner is not None:
        policy_cfg.use_planning = True
        policy_cfg.planning = planner

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    # ── Processors ───────────────────────────────────────────
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg,
    )

    # ── Goal provider (needed for GBP/MPPI planning) ──────────
    goal_provider = make_goal_provider("pusht") if use_planning else None

    # ── Run evaluation ───────────────────────────────────────
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

    metrics = {
        **info["aggregated"],
        "n_episodes": n_episodes,
        "per_episode": info["per_episode"],
    }
    episodes = episodes_from_eval_info(info)
    return metrics, episodes


def _build_finetune_dataset(
    episodes: list[dict],
    pretrain_cfg,
    fps: int,
    features: dict,
    task_description: str,
    bc_mask_mode: str = "none",
):
    """Build a _FinetuneDataset (pretrain + online) for in-process training.

    Writes the online episodes to a temp directory (LeRobotDataset requires
    disk backing), loads the pretrain dataset, resolves delta_timestamps,
    reloads the online dataset with the correct timestamps, and wraps both
    in a _FinetuneDataset.

    Returns ``(finetune_dataset, tmp_dir)`` — caller must clean up tmp_dir.
    """
    from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.lerobot_train import _FinetuneDataset
    from lerobot.scripts.self_improvement_data import episodes_to_lerobot_dataset

    # 1. Write online episodes to a temp dir
    tmp_dir = tempfile.mkdtemp(prefix="self_improve_online_")
    online_root = Path(tmp_dir) / "dataset"
    episodes_to_lerobot_dataset(
        episodes=episodes,
        repo_id="self_improve/online_accumulated",
        root=online_root,
        fps=fps,
        features=features,
        task_description=task_description,
        bc_mask_mode=bc_mask_mode,
    )

    # 2. Load pretrain dataset
    pretrain_ds = make_dataset(pretrain_cfg)

    # 3. Reload online dataset with correct delta_timestamps
    delta_timestamps = resolve_delta_timestamps(pretrain_cfg.policy, pretrain_ds.meta)
    online_ds = LeRobotDataset(
        repo_id="self_improve/online_accumulated",
        root=online_root,
        delta_timestamps=delta_timestamps,
        tolerance_s=pretrain_cfg.tolerance_s,
    )

    logger.info(
        "Built finetune dataset: pretrain=%d frames + online=%d frames (%d episodes)",
        pretrain_ds.num_frames, online_ds.num_frames, online_ds.num_episodes,
    )

    return _FinetuneDataset(pretrain_ds, online_ds), tmp_dir


def _build_train_config(
    config_path: str,
    checkpoint_path: Path,
    total_steps: int,
    output_dir: str,
    job_name: str,
    finetune_lr: float | None = None,
    batch_size: int = 8,
    log_freq: int = 50,
    save_freq: int | None = None,
    wandb_project: str = "awm",
    wandb_entity: str = "",
    trainable_param_keywords: list[str] | None = None,
):
    """Load a TrainPipelineConfig from a checkpoint and apply overrides."""
    from lerobot.configs.train import TrainPipelineConfig

    cfg = TrainPipelineConfig.from_pretrained(config_path)

    # Pre-set fields that validate() normally resolves from CLI args
    cfg.resume = True
    cfg.checkpoint_path = checkpoint_path
    if cfg.policy is not None:
        cfg.policy.pretrained_path = checkpoint_path / "pretrained_model"

    # Apply overrides
    cfg.steps = total_steps
    cfg.output_dir = Path(output_dir)
    cfg.job_name = job_name
    cfg.batch_size = batch_size
    cfg.log_freq = log_freq
    cfg.save_freq = save_freq if save_freq is not None else total_steps
    cfg.eval_freq = 0
    cfg.cudnn_deterministic = True
    cfg.override_lr = finetune_lr

    cfg.wandb.enable = True
    cfg.wandb.project = wandb_project
    cfg.wandb.entity = wandb_entity

    cfg.trainable_param_keywords = trainable_param_keywords

    return cfg


def _find_last_checkpoint(output_dir: str) -> str:
    """Find the last checkpoint's pretrained_model directory."""
    ckpt_dir = Path(output_dir) / "checkpoints" / "last" / "pretrained_model"
    if not ckpt_dir.exists():
        checkpoints_dir = Path(output_dir) / "checkpoints"
        step_dirs = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name != "last"],
            key=lambda d: int(d.name),
        )
        if not step_dirs:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        ckpt_dir = step_dirs[-1] / "pretrained_model"
    return str(ckpt_dir)


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

@parser.wrap()
def self_improve(cfg: SelfImprovementConfig):
    from lerobot.scripts.self_improvement_data import (
        get_pretrain_info,
        read_training_step,
    )
    from lerobot.utils.random_utils import set_seed

    # ── Determinism ──────────────────────────────────────────
    if cfg.cudnn_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

    set_seed(cfg.seed)

    # ── Resolve paths ────────────────────────────────────────
    policy_dir = Path(cfg.policy_path)
    checkpoint_dir = policy_dir.parent  # e.g. .../checkpoints/last/
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = cfg.experiment_name or f"run_pid{os.getpid()}"
    run_id = f"{timestamp}_{slug}"
    base_output = str(checkpoint_dir / "self_improvement" / run_id)

    # ── Pretrain info ────────────────────────────────────────
    pretrain_info = get_pretrain_info(cfg.pretrain_dataset_repo_id, cfg.pretrain_dataset_root)
    pretrain_step = read_training_step(checkpoint_dir)
    logger.info(
        "Pretrain: %s — %d episodes, %d frames, step %d",
        cfg.pretrain_dataset_repo_id, pretrain_info["num_episodes"],
        pretrain_info["num_frames"], pretrain_step,
    )

    ckpt = cfg.policy_path
    current_step = pretrain_step

    all_collected_episodes = []  # Accumulate across iterations

    for iteration in range(cfg.n_iters):
        logger.info("=" * 60)
        logger.info("ITERATION %d / %d  (step %d)", iteration, cfg.n_iters, current_step)
        logger.info("=" * 60)

        # ── 1. Evaluate and collect ──────────────────────────
        logger.info("Collecting %d episodes...", cfg.n_collect_episodes)
        metrics, episodes = eval_and_collect(
            ckpt,
            n_episodes=cfg.n_collect_episodes,
            seed=cfg.eval_seed,
            device=cfg.device,
            use_planning=cfg.use_planning,
            planner=cfg.planner,
        )
        n_success = sum(1 for e in episodes if e["success"])
        n_fail = len(episodes) - n_success
        logger.info(
            "Collection: %.1f%% success (%d success, %d fail)",
            metrics["pc_success"], n_success, n_fail,
        )

        # ── 2. Accumulate and package online data ────────────
        all_collected_episodes.extend(episodes)
        logger.info(
            "Accumulated dataset: %d total episodes (%d new this iteration)",
            len(all_collected_episodes), len(episodes),
        )

        # ── 3. Build dataset and train in-process ────────────
        total_steps = current_step + cfg.finetune_steps
        iter_dir = Path(base_output) / f"iter_{iteration}"
        ft_output_dir = str(iter_dir / "train")
        ft_config_path = str(Path(ckpt) / "train_config.json")

        train_cfg = _build_train_config(
            config_path=ft_config_path,
            checkpoint_path=Path(ckpt).parent,
            total_steps=total_steps,
            output_dir=ft_output_dir,
            job_name=f"self-improve-{slug}",
            finetune_lr=cfg.finetune_lr,
            batch_size=cfg.batch_size,
            log_freq=cfg.log_freq,
            save_freq=cfg.save_freq,
            wandb_project=cfg.wandb_project,
            wandb_entity=cfg.wandb_entity,
            trainable_param_keywords=cfg.trainable_param_keywords,
        )

        dataset, tmp_dir = _build_finetune_dataset(
            episodes=all_collected_episodes,
            pretrain_cfg=train_cfg,
            fps=pretrain_info["fps"],
            features=pretrain_info["features"],
            task_description=cfg.task_description,
            bc_mask_mode=cfg.bc_mask_mode,
        )

        try:
            from lerobot.scripts.lerobot_train import train
            train(train_cfg, dataset=dataset)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Cleaned up temp online dataset: %s", tmp_dir)

        ckpt = _find_last_checkpoint(ft_output_dir)
        current_step = total_steps
        logger.info("Finetune done -> %s (step %d)", ckpt, current_step)

    # ═════════════════════════════════════════════════════════
    # Final evaluation
    # ═════════════════════════════════════════════════════════
    final_use_planning = cfg.eval_use_planning if cfg.eval_use_planning is not None else cfg.use_planning
    final_planner = cfg.eval_planner if cfg.eval_planner is not None else cfg.planner

    # Always run BC baseline eval first (no planning).
    logger.info("Running BC baseline evaluation (%d episodes)...", cfg.eval_n_episodes)
    bc_metrics, _ = eval_and_collect(
        ckpt,
        n_episodes=cfg.eval_n_episodes,
        seed=cfg.eval_seed,
        device=cfg.device,
        use_planning=False,
    )
    print(f"BC_EVAL_RESULTS: {bc_metrics.get('pc_success', 0):.1f}% success")
    print(f"BC_EVAL_AVG_MAX_REWARD: {bc_metrics.get('avg_max_reward', 0):.4f}")
    print(f"BC_EVAL_EP_S: {bc_metrics.get('eval_ep_s', 0):.3f}")

    # If final eval uses planning, run it as a second eval.
    # Otherwise BC baseline IS the final eval — no need to duplicate.
    if final_use_planning:
        logger.info(
            "Running planning evaluation (%d episodes, planner=%s)...",
            cfg.eval_n_episodes, final_planner.algorithm if final_planner else "default",
        )
        plan_metrics, _ = eval_and_collect(
            ckpt,
            n_episodes=cfg.eval_n_episodes,
            seed=cfg.eval_seed,
            device=cfg.device,
            use_planning=True,
            planner=final_planner,
        )
        print(f"PLAN_EVAL_RESULTS: {plan_metrics.get('pc_success', 0):.1f}% success")
        print(f"PLAN_EVAL_AVG_MAX_REWARD: {plan_metrics.get('avg_max_reward', 0):.4f}")
        print(f"PLAN_EVAL_EP_S: {plan_metrics.get('eval_ep_s', 0):.3f}")

    print(f"CHECKPOINT: {ckpt}")


def main():
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.utils import init_logging

    init_logging()
    register_third_party_plugins()
    self_improve()


if __name__ == "__main__":
    main()
