"""Self-improvement playground — manual experimentation pipeline.

Pipeline: eval_and_collect → finetune (end-to-end on successes)
          → finetune_wm (WM-only on all/failure data) → repeat.

Usage:
    python -u src/lerobot/scripts/self_improvement_playground.py <commit_hash>
"""

import logging
import os
import sys

import torch
import wandb

# ═════════════════════════════════════════════════════════════════
# Determinism — must be set before any CUDA operations
# ═════════════════════════════════════════════════════════════════
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)

from lerobot.utils.random_utils import set_seed

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
from pathlib import Path
OUTPUT_DIR = str(Path(POLICY).parent / "self_improvement")

# ═════════════════════════════════════════════════════════════════
# Config — edit these for each experiment
# ═════════════════════════════════════════════════════════════════
# ── Self-improvement loop ──────────────────────────────────────
N_ITERS = 1                 # 0 = eval-only (skip loop), 1+ = collect→finetune cycles
N_COLLECT_EPISODES = 50     # episodes per eval_and_collect
FINETUNE_STEPS = 100        # 0 = skip end-to-end finetune
FINETUNE_LR = 5e-6          # LR for end-to-end finetune
FINETUNE_WM_STEPS = 0       # 0 = skip WM-only finetune
FINETUNE_WM_LR = 1e-6       # LR for WM-only finetune
BATCH_SIZE = 8
MIXING = "naive"            # "naive" = concat online+pretrain, uniform sample
                            # "ratio" = fixed pretrain/online split per batch
PRETRAIN_RATIO_FT = 0.5     # pretrain fraction for finetune    (only if MIXING="ratio")
PRETRAIN_RATIO_WM = 0.6     # pretrain fraction for finetune_wm (only if MIXING="ratio")
LOAD_OPTIMIZER = True       # load Adam state from checkpoint (False = fresh optimizer)
FINETUNE_ONLINE_MODE = "success_only"  # "success_only", "all", or "failure_only" for e2e finetune
BC_MASK_MODE = "none"       # "none"    = full BC+WM loss on all episodes (default)
                            # "failure" = zero BC loss on failure episodes, full on successes
                            # "all"     = zero BC loss on ALL episodes (WM-only loss, all params trainable)
WM_ONLINE_MODE = "all"      # "all", "success_only", or "failure_only"

# ── Final eval ─────────────────────────────────────────────────
EVAL_N_EPISODES = 250       # episodes for final evaluation
EVAL_USE_PLANNING = False   # True to enable GBP/MPPI planning at eval time
EVAL_PLANNING_ALGORITHM = "gcp"  # "gcp" or "mppi"
EVAL_PLANNING_OVERRIDES = {}     # e.g. {"lr": 0.3, "lr_decay": 1.0, "n_iters": 20}

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
        "mixing": MIXING,
        "pretrain_ratio_ft": PRETRAIN_RATIO_FT,
        "pretrain_ratio_wm": PRETRAIN_RATIO_WM,
        "load_optimizer": LOAD_OPTIMIZER,
        "finetune_online_mode": FINETUNE_ONLINE_MODE,
        "bc_mask_mode": BC_MASK_MODE,
        "wm_online_mode": WM_ONLINE_MODE,
        "eval_n_episodes": EVAL_N_EPISODES,
        "eval_use_planning": EVAL_USE_PLANNING,
        "eval_planning_algorithm": EVAL_PLANNING_ALGORITHM,
        "eval_planning_overrides": EVAL_PLANNING_OVERRIDES,
    },
)

# ═════════════════════════════════════════════════════════════════
# Self-improvement loop
# ═════════════════════════════════════════════════════════════════
set_seed(1000)
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

    # ── 2. End-to-end finetune ─────────────────────────────────────
    n_ft_online = (
        buf.n_total if FINETUNE_ONLINE_MODE == "all"
        else buf.n_fail if FINETUNE_ONLINE_MODE == "failure_only"
        else buf.n_success
    )
    if FINETUNE_STEPS > 0 and n_ft_online >= BATCH_SIZE:
        logger.info("Finetuning e2e on %d %s episodes (bc_mask_mode=%s)...",
                     n_ft_online, FINETUNE_ONLINE_MODE, BC_MASK_MODE)
        ckpt, global_step = finetune(
            ckpt, buf, commit_hash=COMMIT,
            output_dir=OUTPUT_DIR,
            n_steps=FINETUNE_STEPS,
            lr=FINETUNE_LR,
            batch_size=BATCH_SIZE,
            mixing=MIXING,
            pretrain_ratio=PRETRAIN_RATIO_FT,
            online_mode=FINETUNE_ONLINE_MODE,
            bc_mask_mode=BC_MASK_MODE,
            load_optimizer=LOAD_OPTIMIZER,
            wandb_run=run,
            global_step=global_step,
        )
        logger.info("Finetune done → %s (global_step=%d)", ckpt, global_step)
    elif FINETUNE_STEPS > 0:
        logger.warning("Too few %s episodes (%d) for finetune, skipping",
                        FINETUNE_ONLINE_MODE, n_ft_online)

    # ── 3. WM-only finetune on suboptimal data ──────────────────
    n_online = buf.n_total if WM_ONLINE_MODE == "all" else (
        buf.n_fail if WM_ONLINE_MODE == "failure_only" else buf.n_success
    )
    if FINETUNE_WM_STEPS > 0 and n_online >= BATCH_SIZE:
        logger.info("Finetuning WM-only on %d %s episodes...", n_online, WM_ONLINE_MODE)
        ckpt, global_step = finetune_wm(
            ckpt, buf, commit_hash=COMMIT,
            output_dir=OUTPUT_DIR,
            n_steps=FINETUNE_WM_STEPS,
            lr=FINETUNE_WM_LR,
            batch_size=BATCH_SIZE,
            mixing=MIXING,
            pretrain_ratio=PRETRAIN_RATIO_WM,
            online_mode=WM_ONLINE_MODE,
            load_optimizer=LOAD_OPTIMIZER,
            wandb_run=run,
            global_step=global_step,
        )
        logger.info("Finetune_wm done → %s (global_step=%d)", ckpt, global_step)
    elif FINETUNE_WM_STEPS > 0:
        logger.warning("Too few %s episodes (%d) for finetune_wm, skipping", WM_ONLINE_MODE, n_online)

# ═════════════════════════════════════════════════════════════════
# Final evaluation (non-blocking SLURM jobs)
# ═════════════════════════════════════════════════════════════════
logger.info("Running final evaluation (%d episodes)...", EVAL_N_EPISODES)
final_metrics, eval_dir = evaluate_final(
    ckpt,
    n_episodes=EVAL_N_EPISODES,
    use_planning=EVAL_USE_PLANNING,
    planning_algorithm=EVAL_PLANNING_ALGORITHM,
    planning_overrides=EVAL_PLANNING_OVERRIDES or None,
)
print(f"EVAL_RESULTS: {final_metrics.get('pc_success', 0):.1f}% success")
print(f"EVAL_DIR: {eval_dir}")

run.log({
    "final/checkpoint": ckpt,
    "final/global_step": global_step,
    "final/pc_success": final_metrics.get("pc_success", 0),
    "final/avg_sum_reward": final_metrics.get("avg_sum_reward", 0),
}, step=global_step)
wandb.finish()
