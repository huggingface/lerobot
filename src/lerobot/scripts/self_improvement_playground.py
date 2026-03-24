"""Self-improvement playground — manual experimentation pipeline.

Pipeline: eval_and_collect → finetune (end-to-end on successes)
          → finetune_wm (WM-only on all/failure data) → repeat.

Usage:
    python -u src/lerobot/scripts/self_improvement_playground.py <commit_hash>
"""

import logging
import sys

import wandb

from lerobot.scripts.self_improvement_utils import (
    TrajectoryBuffer,
    eval_and_collect,
    evaluate_final,
    finetune,
    finetune_wm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COMMIT = sys.argv[1]
POLICY = (
    "/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/outputs/"
    "act_simple_awm_pusht_wm1.0_l2norm_improved_decoder/checkpoints/last/pretrained_model"
)

# ═════════════════════════════════════════════════════════════════
# Config — edit these for each experiment
# ═════════════════════════════════════════════════════════════════
N_ITERS = 1                 # number of eval→finetune→finetune_wm cycles
N_COLLECT_EPISODES = 50     # episodes per eval_and_collect
FINETUNE_STEPS = 100        # steps for end-to-end finetune (on successes)
FINETUNE_LR = 5e-6          # LR for end-to-end finetune
FINETUNE_WM_STEPS = 100     # steps for WM-only finetune
FINETUNE_WM_LR = 1e-6       # LR for WM-only finetune
BATCH_SIZE = 8
PRETRAIN_RATIO_FT = 0.5     # pretrain fraction for finetune
PRETRAIN_RATIO_WM = 0.6     # pretrain fraction for finetune_wm
WM_ONLINE_MODE = "all"      # "all", "success_only", or "failure_only"

# ═════════════════════════════════════════════════════════════════
# WandB init
# ═════════════════════════════════════════════════════════════════
run = wandb.init(
    project="awm",
    entity="pair-diffusion",
    name=f"self-improve-{COMMIT[:7]}",
    config={
        "commit": COMMIT,
        "n_iters": N_ITERS,
        "n_collect_episodes": N_COLLECT_EPISODES,
        "finetune_steps": FINETUNE_STEPS,
        "finetune_lr": FINETUNE_LR,
        "finetune_wm_steps": FINETUNE_WM_STEPS,
        "finetune_wm_lr": FINETUNE_WM_LR,
        "batch_size": BATCH_SIZE,
        "pretrain_ratio_ft": PRETRAIN_RATIO_FT,
        "pretrain_ratio_wm": PRETRAIN_RATIO_WM,
        "wm_online_mode": WM_ONLINE_MODE,
    },
)

# ═════════════════════════════════════════════════════════════════
# Self-improvement loop
# ═════════════════════════════════════════════════════════════════
buf = TrajectoryBuffer()
ckpt = POLICY
global_step = 0

for iteration in range(N_ITERS):
    logger.info("=" * 60)
    logger.info("ITERATION %d / %d", iteration, N_ITERS)
    logger.info("=" * 60)

    # ── 1. Evaluate and collect trajectories ─────────────────────
    logger.info("Collecting %d episodes...", N_COLLECT_EPISODES)
    metrics, episodes = eval_and_collect(ckpt, n_episodes=N_COLLECT_EPISODES)
    buf.add(episodes)
    logger.info(
        "Collection: %.1f%% success | Buffer: %s",
        metrics["pc_success"], buf,
    )
    run.log({
        "eval/pc_success": metrics["pc_success"],
        "eval/avg_sum_reward": metrics.get("avg_sum_reward", 0),
        "eval/n_episodes": N_COLLECT_EPISODES,
        "eval/buffer_n_total": buf.n_total,
        "eval/buffer_n_success": buf.n_success,
        "eval/buffer_n_fail": buf.n_fail,
    }, step=global_step)

    # ── 2. End-to-end finetune on successes ──────────────────────
    if buf.n_success >= BATCH_SIZE:
        logger.info("Finetuning end-to-end on %d successes...", buf.n_success)
        ckpt, global_step = finetune(
            ckpt, buf, commit_hash=COMMIT,
            n_steps=FINETUNE_STEPS,
            lr=FINETUNE_LR,
            batch_size=BATCH_SIZE,
            pretrain_ratio=PRETRAIN_RATIO_FT,
            wandb_run=run,
            global_step=global_step,
        )
        logger.info("Finetune done → %s (global_step=%d)", ckpt, global_step)
    else:
        logger.warning("Too few successes (%d) for finetune, skipping", buf.n_success)

    # ── 3. WM-only finetune on suboptimal data ──────────────────
    n_online = buf.n_total if WM_ONLINE_MODE == "all" else (
        buf.n_fail if WM_ONLINE_MODE == "failure_only" else buf.n_success
    )
    if n_online >= BATCH_SIZE:
        logger.info("Finetuning WM-only on %d %s episodes...", n_online, WM_ONLINE_MODE)
        ckpt, global_step = finetune_wm(
            ckpt, buf, commit_hash=COMMIT,
            n_steps=FINETUNE_WM_STEPS,
            lr=FINETUNE_WM_LR,
            batch_size=BATCH_SIZE,
            pretrain_ratio=PRETRAIN_RATIO_WM,
            online_mode=WM_ONLINE_MODE,
            wandb_run=run,
            global_step=global_step,
        )
        logger.info("Finetune_wm done → %s (global_step=%d)", ckpt, global_step)
    else:
        logger.warning("Too few %s episodes (%d) for finetune_wm, skipping", WM_ONLINE_MODE, n_online)

# ═════════════════════════════════════════════════════════════════
# Final evaluation (non-blocking SLURM jobs)
# ═════════════════════════════════════════════════════════════════
logger.info("Submitting final multi-seed evaluation...")
job_ids, eval_dir = evaluate_final(ckpt, policy_overrides=[
    "--policy.use_planning=true",
    "--policy.planning.algorithm=gcp",
])
print(f"EVAL_JOBS: {job_ids}")
print(f"EVAL_DIR: {eval_dir}")

run.log({"final/checkpoint": ckpt, "final/global_step": global_step}, step=global_step)
wandb.finish()
