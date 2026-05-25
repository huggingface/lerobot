"""Offline ACT training on UR10 v2 demonstrations (absolute target poses).

This is the v2 training counterpart to ``act_train/train_ur10_act.py``. It
mirrors the working RC10 pipeline (``act_train/act_training_example.py``) as
closely as possible: minimal config, no overrides, default ACT hyperparameters.

Why this is so much shorter than v1
===================================
v1 carried two workarounds for delta-action pathologies — MIN_MAX normalization
to dodge the gripper-collapse failure (no longer relevant because absolute
target gripper states are naturally balanced), and a small chunk size with
heavy temporal ensembling to react around the dataset's bang-bang impulses
(no longer relevant because v2 actions are dense and meaningful at every
frame). Both can be dropped, returning to the published ACT defaults that
produced RC10's successful policy:

  - ``chunk_size = 100``  (3.33 s horizon at 30 Hz, matches RC10's setup 1:1)
  - ``temporal_ensemble_coeff = None``  (full open-loop chunk replay)
  - normalization defaults (MEAN_STD across visual/state/action)
  - vanilla L1 regression on chunked actions

Usage
=====
    python act_train/train_ur10_act_v2.py

Tune the constants below if you want to adjust training horizon, batch size,
or output directory. The dataset's action / observation shape is inferred from
its metadata — no need to hardcode (5, 11, …) anywhere.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def _log(msg: str) -> None:
    """Force-flush stderr so tqdm + logs interleave correctly when piped."""
    print(msg, flush=True, file=sys.stderr)


# -- user-tunable ---------------------------------------------------------------
DATASET_REPO_ID = "local/pcb_act_3cams_yaw_v2_1"
OUTPUT_DIR = Path("outputs/act/ur10/pcb_act_3cams_yaw_v2_1")
TRAINING_STEPS = 100_000
BATCH_SIZE = 32

# Optional fine-tuning: load weights from a previous run before training.
# Useful for HG-DAgger or iterating on a fresh demo set.
PRETRAINED_PATH = "outputs/act/ur10/pcb_act_3cams_yaw_v2_1/last"

# PRETRAINED_PATH: str | None = None

# Checkpointing & logging
LOG_FREQ = 100
SAVE_FREQ = 10_000
DEVICE = "cuda"
NUM_WORKERS = 4
# -------------------------------------------------------------------------------


def _make_delta_timestamps(delta_indices: list[int] | None, fps: float) -> list[float]:
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    _log("[train_v2] Reading dataset metadata...")
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID)
    features = dataset_to_policy_features(metadata.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}

    _log(
        f"[train_v2] Dataset: {metadata.total_episodes} episodes, "
        f"{metadata.total_frames} frames @ {metadata.fps} Hz"
    )
    _log(f"[train_v2] Inputs : { {k: tuple(v.shape) for k, v in input_features.items()} }")
    _log(f"[train_v2] Outputs: { {k: tuple(v.shape) for k, v in output_features.items()} }")

    # Default ACTConfig: chunk_size=100, n_action_steps=100, temporal_ensemble_coeff=None,
    # MEAN_STD normalization, ResNet-18 vision encoder, 4-layer transformer encoder/decoder.
    # These are the published defaults that RC10's working ACT pipeline uses unchanged.
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=DEVICE,
    )
    _log(
        f"[train_v2] ACTConfig: chunk_size={cfg.chunk_size}, "
        f"n_action_steps={cfg.n_action_steps}, "
        f"temporal_ensemble_coeff={cfg.temporal_ensemble_coeff}, "
        f"normalization_mapping={cfg.normalization_mapping}"
    )

    if PRETRAINED_PATH is not None:
        _log(f"[train_v2] Loading pretrained weights from {PRETRAINED_PATH} (fine-tuning mode)")
        policy = ACTPolicy.from_pretrained(PRETRAINED_PATH)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, pretrained_path=PRETRAINED_PATH, dataset_stats=metadata.stats,
        )
    else:
        _log("[train_v2] Building ACT policy from scratch (may download ResNet weights on first run)...")
        policy = ACTPolicy(cfg)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=metadata.stats)
    policy.train()
    policy.to(device)
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    _log(f"[train_v2] Policy on {device} — {n_params:.1f}M params")

    # Action chunking timestamps so the dataloader returns chunked targets.
    delta_timestamps = {
        "action": _make_delta_timestamps(cfg.action_delta_indices, metadata.fps),
    }
    # Observation delta-timestamps (image / state history) when configured.
    delta_timestamps |= {
        k: _make_delta_timestamps(cfg.observation_delta_indices, metadata.fps)
        for k in cfg.image_features
    }

    _log("[train_v2] Loading dataset (videos indexed on first pass)...")
    t0 = time.perf_counter()
    dataset = LeRobotDataset(DATASET_REPO_ID, delta_timestamps=delta_timestamps)
    _log(f"[train_v2] Dataset loaded: {len(dataset)} sample windows in {time.perf_counter() - t0:.1f}s")

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    _log(
        f"[train_v2] Starting training: {TRAINING_STEPS} steps, batch_size={BATCH_SIZE}, "
        f"num_workers={NUM_WORKERS}"
    )
    _log(f"[train_v2] Checkpoints every {SAVE_FREQ} steps → {OUTPUT_DIR}")

    pbar = tqdm(total=TRAINING_STEPS, desc="train_v2", unit="step", dynamic_ncols=True)
    step = 0
    done = False
    try:
        while not done:
            for batch in loader:
                batch = preprocessor(batch)
                loss, _ = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                loss_val = float(loss.item())
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_val:.4f}", refresh=False)

                if step % LOG_FREQ == 0:
                    pbar.write(f"[step {step:6d}]  loss={loss_val:.4f}")

                if step % SAVE_FREQ == 0:
                    ckpt = OUTPUT_DIR / f"step_{step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(ckpt)
                    preprocessor.save_pretrained(ckpt)
                    postprocessor.save_pretrained(ckpt)
                    pbar.write(f"[checkpoint] saved → {ckpt}")

                if step >= TRAINING_STEPS:
                    done = True
                    break
    finally:
        pbar.close()

    last = OUTPUT_DIR / "last"
    last.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(last)
    preprocessor.save_pretrained(last)
    postprocessor.save_pretrained(last)
    _log(f"[train_v2] Final checkpoint saved → {last}")


if __name__ == "__main__":
    main()
