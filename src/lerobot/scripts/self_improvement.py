"""Self-improvement pipeline v2 — thin orchestrator over lerobot-train / lerobot-eval.

Pipeline per iteration:
    1. Evaluate current policy & collect on-policy trajectories.
    2. Package trajectories into a LeRobotDataset on disk.
    3. Call ``lerobot-train --resume --online_dataset_root=<path>`` to
       continue training on the pretrain + online data (concatenated at
       load time — no data copying).

Usage:
    python -u src/lerobot/scripts/self_improvement.py <commit_hash>
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import torch

# ═════════════════════════════════════════════════════════════════
# Determinism — must be set before any CUDA operations
# ═════════════════════════════════════════════════════════════════
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)

from lerobot.utils.random_utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════
# Config — edit these for each experiment
# ═════════════════════════════════════════════════════════════════
COMMIT = sys.argv[1] if len(sys.argv) > 1 else "unknown"

# ── Pretrain checkpoint (the starting point) ──────────────────
POLICY = (
    "/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/outputs/"
    "act_simple_awm_pusht_wm1.0_l2norm_improved_decoder/checkpoints/last/pretrained_model"
)
PRETRAIN_DATASET_REPO_ID = "lerobot/pusht"
PRETRAIN_DATASET_ROOT = None  # None = use HF cache
TASK_DESCRIPTION = "Push the T-shaped block onto the target."

# ── Self-improvement loop ─────────────────────────────────────
N_ITERS = 1                   # Number of collect→finetune cycles
N_COLLECT_EPISODES = 50       # Episodes per eval_and_collect
FINETUNE_STEPS = 100          # Training steps per iteration
FINETUNE_LR = 5e-6            # LR for finetuning (None = keep pretrain LR)
BATCH_SIZE = 8
BC_MASK_MODE = "none"         # "none" or "failure"
EVAL_SEED = 42
LOG_FREQ = 50
SAVE_FREQ = None              # None = save only at end of finetune

# ── Final eval ────────────────────────────────────────────────
EVAL_N_EPISODES = 250         # Final evaluation episodes
EVAL_USE_PLANNING = False
EVAL_PLANNING_ALGORITHM = "gcp"
EVAL_PLANNING_OVERRIDES = {}

# ── WandB ─────────────────────────────────────────────────────
WANDB_PROJECT = "awm"
WANDB_ENTITY = "pair-diffusion"


# ═════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════

def eval_and_collect(
    policy_path: str,
    n_episodes: int,
    seed: int = 42,
    device: str = "cuda",
    use_planning: bool = False,
    planning_algorithm: str = "gcp",
    planning_overrides: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate the policy and collect trajectory data.

    Returns ``(metrics, episodes)`` where *episodes* is a list of dicts
    ready for :func:`episodes_to_lerobot_dataset`.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
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

    if use_planning:
        policy_cfg.use_planning = True
        policy_cfg.planning.algorithm = planning_algorithm
        if planning_overrides:
            for key, value in planning_overrides.items():
                if hasattr(policy_cfg.planning, key):
                    setattr(policy_cfg.planning, key, value)

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
        )

    vec_env.close()

    metrics = {
        **info["aggregated"],
        "n_episodes": n_episodes,
        "per_episode": info["per_episode"],
    }
    episodes = episodes_from_eval_info(info)
    return metrics, episodes


def run_finetune(
    config_path: str,
    online_dataset_root: str,
    total_steps: int,
    output_dir: str,
    commit: str,
    finetune_lr: float | None = None,
    batch_size: int = 8,
    log_freq: int = 50,
    save_freq: int | None = None,
) -> str:
    """Call ``lerobot-train --resume`` as a subprocess.

    The pretrain dataset is loaded from the checkpoint's saved config.
    The online dataset is concatenated via ``--online_dataset_root``.

    Returns the path to the last checkpoint's pretrained_model directory.
    """
    actual_save_freq = save_freq if save_freq is not None else total_steps

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--config_path={config_path}",
        "--resume=true",
        f"--online_dataset_root={online_dataset_root}",
        f"--steps={total_steps}",
        f"--output_dir={output_dir}",
        f"--job_name=self-improve-{commit[:7]}",
        f"--batch_size={batch_size}",
        f"--log_freq={log_freq}",
        f"--save_freq={actual_save_freq}",
        f"--eval_freq=0",
        f"--wandb.enable=true",
        f"--wandb.project={WANDB_PROJECT}",
        f"--wandb.entity={WANDB_ENTITY}",
        "--cudnn_deterministic=true",
    ]
    if finetune_lr is not None:
        cmd.append(f"--override_lr={finetune_lr}")

    logger.info("Running lerobot-train:\n  %s", " \\\n    ".join(cmd))

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"lerobot-train failed with return code {result.returncode}")

    # Find the last checkpoint
    ckpt_dir = Path(output_dir) / "checkpoints" / "last" / "pretrained_model"
    if not ckpt_dir.exists():
        # Fall back to step-based directory
        checkpoints_dir = Path(output_dir) / "checkpoints"
        step_dirs = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name != "last"],
            key=lambda d: int(d.name),
        )
        if not step_dirs:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        ckpt_dir = step_dirs[-1] / "pretrained_model"

    logger.info("Finetune checkpoint: %s", ckpt_dir)
    return str(ckpt_dir)


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    from lerobot.scripts.self_improvement_data import (
        episodes_to_lerobot_dataset,
        get_pretrain_info,
        read_training_step,
    )

    set_seed(1000)

    # ── Resolve paths ────────────────────────────────────────
    policy_dir = Path(POLICY)
    checkpoint_dir = policy_dir.parent  # e.g. .../checkpoints/last/
    base_output = str(checkpoint_dir / "self_improvement" / COMMIT[:8])

    # ── Pretrain info ────────────────────────────────────────
    pretrain_info = get_pretrain_info(PRETRAIN_DATASET_REPO_ID, PRETRAIN_DATASET_ROOT)
    pretrain_step = read_training_step(checkpoint_dir)
    logger.info(
        "Pretrain: %s — %d episodes, %d frames, step %d",
        PRETRAIN_DATASET_REPO_ID, pretrain_info["num_episodes"],
        pretrain_info["num_frames"], pretrain_step,
    )

    ckpt = POLICY
    current_step = pretrain_step

    for iteration in range(N_ITERS):
        logger.info("=" * 60)
        logger.info("ITERATION %d / %d  (step %d)", iteration, N_ITERS, current_step)
        logger.info("=" * 60)

        # ── 1. Evaluate and collect ──────────────────────────
        logger.info("Collecting %d episodes...", N_COLLECT_EPISODES)
        metrics, episodes = eval_and_collect(
            ckpt,
            n_episodes=N_COLLECT_EPISODES,
            seed=EVAL_SEED,
            use_planning=EVAL_USE_PLANNING,
            planning_algorithm=EVAL_PLANNING_ALGORITHM,
            planning_overrides=EVAL_PLANNING_OVERRIDES or None,
        )
        n_success = sum(1 for e in episodes if e["success"])
        n_fail = len(episodes) - n_success
        logger.info(
            "Collection: %.1f%% success (%d success, %d fail)",
            metrics["pc_success"], n_success, n_fail,
        )

        # ── 2. Package online data as LeRobotDataset ─────────
        iter_dir = Path(base_output) / f"iter_{iteration}"
        online_root = iter_dir / "online_dataset"
        online_ds = episodes_to_lerobot_dataset(
            episodes=episodes,
            repo_id=f"self_improve/online_iter{iteration}",
            root=online_root,
            fps=pretrain_info["fps"],
            features=pretrain_info["features"],
            task_description=TASK_DESCRIPTION,
            bc_mask_mode=BC_MASK_MODE,
        )

        # ── 3. Finetune via lerobot-train ────────────────────
        # No merge needed — lerobot-train concatenates the pretrain
        # dataset (from the checkpoint config) with the online dataset
        # via --online_dataset_root.
        total_steps = current_step + FINETUNE_STEPS
        ft_output_dir = str(iter_dir / "train")
        ft_config_path = str(Path(ckpt) / "train_config.json")

        ckpt = run_finetune(
            config_path=ft_config_path,
            online_dataset_root=str(online_ds.root),
            total_steps=total_steps,
            output_dir=ft_output_dir,
            commit=COMMIT,
            finetune_lr=FINETUNE_LR,
            batch_size=BATCH_SIZE,
            log_freq=LOG_FREQ,
            save_freq=SAVE_FREQ,
        )
        current_step = total_steps
        logger.info("Finetune done → %s (step %d)", ckpt, current_step)

    # ═════════════════════════════════════════════════════════
    # Final evaluation
    # ═════════════════════════════════════════════════════════
    logger.info("Running final evaluation (%d episodes)...", EVAL_N_EPISODES)
    final_metrics, _ = eval_and_collect(
        ckpt,
        n_episodes=EVAL_N_EPISODES,
        seed=EVAL_SEED,
        use_planning=EVAL_USE_PLANNING,
        planning_algorithm=EVAL_PLANNING_ALGORITHM,
        planning_overrides=EVAL_PLANNING_OVERRIDES or None,
    )
    print(f"EVAL_RESULTS: {final_metrics.get('pc_success', 0):.1f}% success")
    print(f"EVAL_AVG_MAX_REWARD: {final_metrics.get('avg_max_reward', 0):.4f}")
    print(f"EVAL_EP_S: {final_metrics.get('eval_ep_s', 0):.3f}")
    print(f"CHECKPOINT: {ckpt}")


if __name__ == "__main__":
    main()
