"""Offline ACT training on UR10 demonstrations.

Mirrors ``act_train/act_training_example.py`` (RC10 reference) but reads the
UR10 dataset produced by ``record_ur10_act.py``. ACT is observation-agnostic
(see the planning notes); the 16-D mixed-frame state and 4-D delta action
work without any policy code changes.

Usage:
    python act_train/train_ur10_act.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def _log(msg: str) -> None:
    """Force flush so progress shows up immediately under pipe/redirect."""
    print(msg, flush=True, file=sys.stderr)


# -- user-tunable ---------------------------------------------------------------
DATASET_REPO_ID = "local/ur10_act_usb_insertion_state"
OUTPUT_DIR = Path("outputs/act/ur10/usb_insertion_state")
TRAINING_STEPS = 50_000
BATCH_SIZE = 32

# Chunk size = policy's prediction horizon in steps. See "Tuning CHUNK_SIZE" at
# the bottom of this file for a step-by-step procedure to pick the right value.
CHUNK_SIZE = 30          # 3 s @ 10 Hz; see procedure for tuning

# Temporal ensembling: at inference, query the policy every step and average each
# step's prediction with the still-relevant predictions from earlier (overlapping)
# chunks. Smooths actions and tolerates per-frame jitter; the ACT paper uses 0.01.
# Weights are wᵢ = exp(-coeff * i); smaller coeff = more averaging, smoother but
# laggier; larger coeff = recency-biased, more reactive. Setting this to None
# disables temporal ensembling (the policy then executes the predicted chunk in
# one shot before re-querying — equivalent to n_action_steps = CHUNK_SIZE).
#
# Hard constraint enforced by ACTConfig.__post_init__:
#   if TEMPORAL_ENSEMBLE_COEFF is not None: n_action_steps must be 1.
TEMPORAL_ENSEMBLE_COEFF: float | None = 0.01

LOG_FREQ = 100
SAVE_FREQ = 5_000
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

    _log("[train] Reading dataset metadata...")
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID)
    features = dataset_to_policy_features(metadata.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}
    _log(f"[train] Dataset: {metadata.total_episodes} episodes, {metadata.total_frames} frames @ {metadata.fps} Hz")
    _log(f"[train] Inputs:  { {k: tuple(v.shape) for k, v in input_features.items()} }")
    _log(f"[train] Outputs: { {k: tuple(v.shape) for k, v in output_features.items()} }")

    # When temporal ensembling is on the policy must be queried every step so it can
    # update the running average — `n_action_steps` is forced to 1 by ACTConfig in
    # that case. When ensembling is off we execute the full chunk before re-querying.
    n_action_steps = 1 if TEMPORAL_ENSEMBLE_COEFF is not None else CHUNK_SIZE

    # Action normalization: MIN_MAX (not the ACT default MEAN_STD). The action vector
    # mixes a continuous Cartesian delta (dims 0-2) with a discrete gripper command
    # in {0, 1, 2} (dim 3). With MEAN_STD, the gripper std (~0.19 in our dataset) +
    # mean (~1.0, dominated by the >96% STAY frames) maps CLOSE → −5.3, STAY → 0,
    # OPEN → +5.3 in normalized space — the regression network learns to predict 0
    # everywhere and the gripper collapses to STAY at inference. MIN_MAX over [0, 2]
    # maps CLOSE → −1, STAY → 0, OPEN → +1, which the network can actually reach.
    # The continuous dims also fit MIN_MAX cleanly (already in [−1, 1]).
    normalization_mapping = {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MIN_MAX,
    }

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=TEMPORAL_ENSEMBLE_COEFF,
        normalization_mapping=normalization_mapping,
        device=DEVICE,
    )
    _log(f"[train] ACT cfg: chunk_size={CHUNK_SIZE}, n_action_steps={n_action_steps}, temporal_ensemble_coeff={TEMPORAL_ENSEMBLE_COEFF}")

    _log("[train] Building ACT policy (may download resnet18 weights on first run)...")
    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=metadata.stats)
    policy.train()
    policy.to(device)
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    _log(f"[train] Policy on {device} — {n_params:.1f}M params")

    delta_timestamps = {
        "action": _make_delta_timestamps(cfg.action_delta_indices, metadata.fps),
    }
    # Observation delta-timestamps cover image/state history if cfg.observation_delta_indices is set.
    delta_timestamps |= {
        k: _make_delta_timestamps(cfg.observation_delta_indices, metadata.fps)
        for k in cfg.image_features
    }

    _log("[train] Loading dataset (this can take a while if videos must be indexed)...")
    t0 = time.perf_counter()
    dataset = LeRobotDataset(DATASET_REPO_ID, delta_timestamps=delta_timestamps)
    _log(f"[train] Dataset loaded: {len(dataset)} sample windows in {time.perf_counter() - t0:.1f}s")

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    _log(f"[train] Starting training: {TRAINING_STEPS} steps, batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}")
    _log(f"[train] Checkpoints every {SAVE_FREQ} steps → {OUTPUT_DIR}")

    pbar = tqdm(total=TRAINING_STEPS, desc="train", unit="step", dynamic_ncols=True)
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
    _log(f"[train] Final checkpoint saved → {last}")


if __name__ == "__main__":
    main()


# =============================================================================
# Tuning CHUNK_SIZE — procedure
# =============================================================================
#
# `chunk_size` is the only ACT hyperparameter that interacts heavily with the
# *physical* task (most others have safe defaults). It controls how far into the
# future the policy commits to a plan in one forward pass. Wrong choice degrades
# things in opposite ways:
#
#   * Too small  → policy reacts step-by-step; trajectory is jittery; can't
#                  commit to multi-step behaviour (e.g. a smooth insertion
#                  approach); training loss can plateau higher.
#   * Too large  → policy effectively predicts the entire future from one frame;
#                  overfits to the exact trajectories in the demos; fails when
#                  the scene differs even slightly from the demo distribution;
#                  temporal ensembling stops helping because there's nothing left
#                  to ensemble across.
#
# Procedure
# ---------
#
# Step 1 — Measure your typical episode length on disk.
#
#     python - <<'PY'
#     from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
#     m = LeRobotDatasetMetadata("local/ur10_act_usb_insertion")
#     lens = [m.episodes[i]["length"] for i in range(m.total_episodes)]
#     import statistics
#     print("episodes:", m.total_episodes,
#           "  median frames:", statistics.median(lens),
#           "  min/max:", min(lens), max(lens),
#           "  fps:", m.fps)
#     PY
#
# Call the median episode length T (in steps). For USB insertion T ≈ 80–120.
#
# Step 2 — Build a candidate sweep: 4 chunk sizes at {10 %, 30 %, 60 %, 100 %}
# of T, rounded to multiples of 10 for tidy logs. Examples for T = 100:
#
#     CHUNK_SIZE ∈ { 10, 30, 60, 100 }
#
#   * 10 % matches the ratio the ALOHA paper uses (chunk_size=100 at 50 Hz over
#     20 s episodes — 2 s prediction over 20 s episode).
#   * 30 % is a balanced default that's been our starting point.
#   * 60 % commits to most of an episode in one shot.
#   * 100 % effectively predicts the whole episode from t=0 — useful sanity
#     check; usually too rigid for contact-rich tasks.
#
# Step 3 — Train each variant for the SAME number of steps (≈30 k is enough to
# resolve which chunk size is best; 50 k for a final run). Keep
# TEMPORAL_ENSEMBLE_COEFF = 0.01 fixed. Use a separate OUTPUT_DIR per variant:
#
#     OUTPUT_DIR = Path(f"outputs/act/ur10/usb_insertion/chunk{CHUNK_SIZE}")
#
# Track training loss; expect it to drop faster with larger chunk_size in early
# training (the model fits the chunks more easily). Final loss is NOT the
# selection metric — overfit chunks have low loss but bad eval.
#
# Step 4 — Evaluate each checkpoint with `eval_ur10_act.py`. Point MODEL_DIR at
# the trained checkpoint and run NUM_EPISODES = 20 with fresh resets. Record:
#
#   * success rate (fraction of episodes where you press the success button)
#   * mean episode time (faster = more confident insertion)
#   * subjective smoothness (does the arm jitter? overshoot?)
#
# Optional quantitative smoothness metric — log the recorded action stream
# from eval and compute action jerk (third-derivative finite difference, taken
# over the 4-D action sequence). Lower jerk = smoother. Useful but the seat-
# of-the-pants test usually dominates.
#
# Step 5 — Pick the chunk size with the highest success rate; break ties by
# smoothness. Re-train at the chosen value for the full 50 k steps.
#
# Heuristic shortcuts (skip the full sweep when time-constrained)
# ---------------------------------------------------------------
#
# * Short, low-variance tasks (< 5 s, well-rehearsed): start at 50 % of T.
# * Contact-rich precision tasks (insertion, peg-in-hole): start at 30 % of T,
#   keep temporal ensembling on (averaging smooths the contact transients).
# * Long, multi-phase tasks (pick → carry → place): start at 20 % of T — the
#   policy needs to react when phases switch, so it shouldn't commit too far.
#
# When to also tune TEMPORAL_ENSEMBLE_COEFF
# -----------------------------------------
#
# * If actions are noticeably jittery with the best chunk_size, lower the coeff
#   (e.g. 0.005) for more averaging.
# * If the arm lags behind your intent during eval (e.g. doesn't react to a new
#   visual cue), raise the coeff (e.g. 0.05) so newer predictions dominate.
# * Setting to `None` disables ensembling entirely (the policy then executes the
#   full predicted chunk before re-querying). Faster at inference but loses the
#   smoothing benefit; use only if compute-constrained.
# =============================================================================
