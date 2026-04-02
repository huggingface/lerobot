"""Quick end-to-end test of the self-improvement v2 pipeline.

Run via SLURM:
    sbatch compute_inference.sh python -u test_self_improvement_v2.py
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────
POLICY = (
    "/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/outputs/"
    "act_simple_awm_pusht_wm1.0_l2norm_improved_decoder/checkpoints/last/pretrained_model"
)
PRETRAIN_REPO_ID = "lerobot/pusht"
TEST_DIR = Path("/storage/home/hcoda1/6/vgiridhar6/forks/lerobot/outputs/test_si_v2")
N_COLLECT = 5   # Very small for quick test
FINETUNE_STEPS = 10
FINETUNE_LR = 5e-6
BATCH_SIZE = 4


def main():
    from lerobot.utils.random_utils import set_seed
    set_seed(1000)

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True)

    # ── Step 1: Import and check ─────────────────────────────
    from lerobot.scripts.self_improvement_data import (
        episodes_to_lerobot_dataset,
        get_pretrain_info,
        read_training_step,
    )
    from lerobot.scripts.self_improvement import eval_and_collect, run_finetune

    pretrain_info = get_pretrain_info(PRETRAIN_REPO_ID)
    pretrain_step = read_training_step(Path(POLICY).parent)
    logger.info("Pretrain: %d episodes, %d frames, step %d",
                pretrain_info["num_episodes"], pretrain_info["num_frames"], pretrain_step)

    # ── Step 2: Eval and collect ─────────────────────────────
    logger.info("Collecting %d episodes...", N_COLLECT)
    t0 = time.time()
    metrics, episodes = eval_and_collect(POLICY, n_episodes=N_COLLECT, seed=42)
    logger.info("Collection: %.1f%% success, took %.1fs",
                metrics["pc_success"], time.time() - t0)

    # ── Step 3: Create online dataset ────────────────────────
    logger.info("Creating online dataset...")
    online_ds = episodes_to_lerobot_dataset(
        episodes=episodes,
        repo_id="test/online",
        root=TEST_DIR / "online_dataset",
        fps=pretrain_info["fps"],
        features=pretrain_info["features"],
        task_description=pretrain_info["task_description"],
        bc_mask_mode="failure",
    )
    logger.info("Online: %d episodes, %d frames", online_ds.num_episodes, online_ds.num_frames)

    # ── Step 4: Finetune via lerobot-train --resume ────────────
    # No merge step — lerobot-train concatenates pretrain + online
    # via --online_dataset_root (instant, no data copying).
    logger.info("Starting finetune (%d steps)...", FINETUNE_STEPS)
    total_steps = pretrain_step + FINETUNE_STEPS
    ft_output_dir = str(TEST_DIR / "train")
    config_path = str(Path(POLICY) / "train_config.json")

    ckpt = run_finetune(
        config_path=config_path,
        online_dataset_root=str(online_ds.root),
        total_steps=total_steps,
        output_dir=ft_output_dir,
        commit="test1234",
        finetune_lr=FINETUNE_LR,
        batch_size=BATCH_SIZE,
        log_freq=5,
        save_freq=FINETUNE_STEPS,
    )
    logger.info("Finetune checkpoint: %s", ckpt)

    # Verify step continuity
    ft_step = read_training_step(Path(ckpt).parent)
    assert ft_step == total_steps, f"Step mismatch: {ft_step} != {total_steps}"
    logger.info("Step continuity verified: %d ✓", ft_step)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
