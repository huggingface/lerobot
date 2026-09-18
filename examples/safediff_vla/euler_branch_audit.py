#!/usr/bin/env python
"""Offline audit: does Euler-angle branch-cut wraparound (±π discontinuity) in the training
dataset's rotation actions (`action[..., 3:6]` = rx, ry, rz) affect checkpoint B's
(`outputs/train/safediff_vla_temporal_decoder_v2_baseline_20k/checkpoints/020000`) rotation
predictions? Read-only analysis -- no model/architecture code is touched, no training happens.

Two independent measurements:

  1. Dataset-wide wrap-transition statistics, computed directly from the raw parquet action
     columns (no video decoding, no model) over the *entire* `lerobot/vlabench_unified` training
     set (3,114,872 frames / 10,977 episodes): consecutive-delta wrap-transition count/rate per
     axis, near-±π sample rate per axis, per-task wrap-transition counts, and what fraction of
     all action_horizon=50 chunks contain at least one wrap transition.
  2. Model-vs-GT comparison on two *fixed, reproducible* sample sets (index lists saved
     alongside the results so they can be reproduced exactly): a general random sample (for the
     raw-vs-unwrapped-Euler MSE / geodesic-error comparison) and a branch-cut-near sample (for
     the detailed per-sample branch dump).

Usage:
    uv run python examples/safediff_vla/euler_branch_audit.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.processor.rename_processor import rename_batch_keys
from lerobot.utils.constants import ACTION

sys.path.insert(0, str(Path(__file__).parent))
from eval_baseline_rollout import geodesic_angle  # noqa: E402  (reuse, don't reimplement)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_v2_baseline_20k/checkpoints/020000/pretrained_model"
DATASET_REPO_ID = "lerobot/vlabench_unified"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
AXES = ("rx", "ry", "rz")
ROTATION_DIMS = slice(3, 6)
ACTION_HORIZON = 50
EPS_NEAR_PI = 0.1  # radians (~5.7deg): how close to +-pi counts as "near the branch cut"
SAMPLE_SEED = 42
N_GENERAL_SAMPLES = 200
N_BRANCH_CUT_SAMPLES = 100


def load_full_action_table(root: Path) -> dict[str, np.ndarray]:
    """Read `action`/`episode_index`/`frame_index`/`index`/`task_index` directly from the
    dataset's data-chunk parquet files -- bypasses `LeRobotDataset`/video decoding entirely,
    since this pass only needs the 7-d action vectors, not images."""
    data_dir = root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    tables = [
        pq.read_table(f, columns=["action", "episode_index", "frame_index", "index", "task_index"])
        for f in parquet_files
    ]
    episode_index = np.concatenate([t.column("episode_index").to_numpy() for t in tables])
    frame_index = np.concatenate([t.column("frame_index").to_numpy() for t in tables])
    index = np.concatenate([t.column("index").to_numpy() for t in tables])
    task_index = np.concatenate([t.column("task_index").to_numpy() for t in tables])
    action = np.concatenate(
        [np.stack(t.column("action").to_numpy()).astype(np.float64) for t in tables], axis=0
    )
    order = np.argsort(index)
    return {
        "action": action[order],
        "episode_index": episode_index[order],
        "frame_index": frame_index[order],
        "index": index[order],
        "task_index": task_index[order],
    }


def dataset_wide_wrap_audit(table: dict[str, np.ndarray], task_index_to_name: dict[int, str]) -> dict:
    action = table["action"]
    episode_index = table["episode_index"]
    task_index = table["task_index"]
    n_frames = action.shape[0]

    # Split into per-episode contiguous blocks (data is sorted by global `index`, which is
    # contiguous per episode -- verified via `dataset_from_index`/`dataset_to_index`).
    boundaries = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n_frames]))

    n_transition_pairs = 0
    wrap_transition_counts = {axis: 0 for axis in AXES}
    near_pi_counts = {axis: 0 for axis in AXES}
    task_wrap_counts: dict[str, dict[str, int]] = {}
    n_chunks = 0
    chunk_wrap_counts = {axis: 0 for axis in AXES}
    chunk_wrap_any_axis = 0

    for start, end in zip(starts, ends, strict=True):
        ep_action = action[start:end]
        t = ep_action.shape[0]
        ep_task = task_index_to_name.get(int(task_index[start]), str(task_index[start]))
        task_bucket = task_wrap_counts.setdefault(ep_task, {axis: 0 for axis in AXES})

        rot = ep_action[:, ROTATION_DIMS]  # [T, 3] = rx, ry, rz
        near = (rot > (np.pi - EPS_NEAR_PI)) | (rot < (-np.pi + EPS_NEAR_PI))
        for i, axis in enumerate(AXES):
            near_pi_counts[axis] += int(near[:, i].sum())

        if t >= 2:
            delta = np.diff(rot, axis=0)  # [T-1, 3]
            wrap = np.abs(delta) > np.pi  # [T-1, 3]
            n_transition_pairs += t - 1
            for i, axis in enumerate(AXES):
                cnt = int(wrap[:, i].sum())
                wrap_transition_counts[axis] += cnt
                task_bucket[axis] += cnt

            # Chunk-level: for every possible chunk start position i in [0, t-1] (every frame is
            # a valid `action_horizon=50` training-chunk start; the chunk's *real* transitions
            # are wrap[i : i + ACTION_HORIZON - 1], clipped at the episode end -- anything past
            # that is padding, not a real transition), does at least one wrap occur?
            wrap_any = wrap.any(axis=1)  # [T-1]
            cum_any = np.concatenate(([0], np.cumsum(wrap_any)))
            cum_axis = {axis: np.concatenate(([0], np.cumsum(wrap[:, i]))) for i, axis in enumerate(AXES)}
            for i in range(t):
                window_end = min(i + ACTION_HORIZON - 1, t - 1)
                if cum_any[window_end] - cum_any[i] > 0:
                    chunk_wrap_any_axis += 1
                for axis in AXES:
                    if cum_axis[axis][window_end] - cum_axis[axis][i] > 0:
                        chunk_wrap_counts[axis] += 1
            n_chunks += t

    return {
        "eps_near_pi_rad": EPS_NEAR_PI,
        "n_frames": int(n_frames),
        "n_consecutive_transition_pairs": int(n_transition_pairs),
        "n_episodes": int(len(starts)),
        "n_chunks": int(n_chunks),
        "per_axis": {
            axis: {
                "wrap_transition_count": wrap_transition_counts[axis],
                "wrap_transition_rate": wrap_transition_counts[axis] / max(n_transition_pairs, 1),
                "near_pi_sample_count": near_pi_counts[axis],
                "near_pi_sample_rate": near_pi_counts[axis] / max(n_frames, 1),
                "chunk_with_wrap_count": chunk_wrap_counts[axis],
                "chunk_with_wrap_rate": chunk_wrap_counts[axis] / max(n_chunks, 1),
            }
            for axis in AXES
        },
        "any_axis": {
            "chunk_with_wrap_count": chunk_wrap_any_axis,
            "chunk_with_wrap_rate": chunk_wrap_any_axis / max(n_chunks, 1),
        },
        "per_task_wrap_transition_counts": task_wrap_counts,
    }


def classify_branch(angle: float, eps: float) -> str:
    if angle > np.pi - eps:
        return "+pi_branch"
    if angle < -np.pi + eps:
        return "-pi_branch"
    return "middle"


def pick_fixed_samples(
    table: dict[str, np.ndarray],
    episodes_df,
    task_index_to_name: dict[int, str],
    rng: np.random.Generator,
    n_general: int,
    n_branch_cut: int,
) -> tuple[list[dict], list[dict]]:
    """Reproducible (fixed `rng` seed) sample selection, restricted to frames with a *full*
    valid ACTION_HORIZON=50 forward window (no padding) so first-5/first-10 MSE is computed on
    real demonstrated actions only."""
    length_by_ep = dict(zip(episodes_df["episode_index"].tolist(), episodes_df["length"].tolist(), strict=True))
    frame_index = table["frame_index"]
    episode_index = table["episode_index"]
    global_index = table["index"]
    valid_mask = frame_index <= (np.array([length_by_ep[e] for e in episode_index]) - ACTION_HORIZON)
    valid_positions = np.flatnonzero(valid_mask)

    rot = table["action"][:, ROTATION_DIMS]
    near = ((rot > (np.pi - EPS_NEAR_PI)) | (rot < (-np.pi + EPS_NEAR_PI))).any(axis=1)
    branch_cut_positions = np.intersect1d(valid_positions, np.flatnonzero(near))

    def describe(positions: np.ndarray) -> list[dict]:
        return [
            {
                "sample_index": int(global_index[p]),
                "episode_index": int(episode_index[p]),
                "frame_index": int(frame_index[p]),
                "task": task_index_to_name.get(int(table["task_index"][p]), str(table["task_index"][p])),
            }
            for p in positions
        ]

    general_positions = rng.choice(valid_positions, size=min(n_general, len(valid_positions)), replace=False)
    branch_cut_sampled = rng.choice(
        branch_cut_positions, size=min(n_branch_cut, len(branch_cut_positions)), replace=False
    )
    logger.info(
        "sample pools: %d valid (full-horizon) frames, %d near-branch-cut frames",
        len(valid_positions),
        len(branch_cut_positions),
    )
    return describe(general_positions), describe(branch_cut_sampled)


def run_model_comparison(policy, device: str, dataset: LeRobotDataset, samples: list[dict]) -> tuple[dict, list[dict]]:
    """For each sample: GT raw/unwrapped Euler, B-checkpoint predicted raw Euler, and per-sample
    MSE/geodesic metrics -- all from a single `plan_action_chunk` forward pass per sample."""
    per_sample_records = []
    raw_mse5, raw_mse10, unwrap_mse5, unwrap_mse10, geo5, geo10 = [], [], [], [], [], []

    for s in samples:
        raw_frame = dataset[s["sample_index"]]
        batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in raw_frame.items()}
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = rename_batch_keys(batch, RENAME_MAP)
        batch["task"] = [raw_frame["task"]]
        gt_raw = batch[ACTION].clone()  # [1, 50, 7], dataset-native units

        pre_batch = PREPROCESSOR(batch)
        with torch.no_grad():
            pred_norm, _ = policy.plan_action_chunk(pre_batch)
        pred_raw = POSTPROCESSOR(pred_norm.cpu()).to(gt_raw.dtype)

        gt_rot = gt_raw[0, :10, ROTATION_DIMS].numpy()
        pred_rot = pred_raw[0, :10, ROTATION_DIMS].numpy()
        gt_unwrapped = np.unwrap(gt_rot, axis=0)
        pred_unwrapped = np.unwrap(pred_rot, axis=0)

        def mse(a, b, n):
            return float(np.mean((a[:n] - b[:n]) ** 2))

        raw_mse5.append(mse(pred_rot, gt_rot, 5))
        raw_mse10.append(mse(pred_rot, gt_rot, 10))
        unwrap_mse5.append(mse(pred_unwrapped, gt_unwrapped, 5))
        unwrap_mse10.append(mse(pred_unwrapped, gt_unwrapped, 10))
        geo = geodesic_angle(
            torch.from_numpy(pred_rot).unsqueeze(0), torch.from_numpy(gt_rot).unsqueeze(0)
        )[0]
        geo5.append(float(geo[:5].mean()))
        geo10.append(float(geo[:10].mean()))

        per_sample_records.append(
            {
                **s,
                "gt_raw_euler_first10": gt_rot.tolist(),
                "gt_unwrapped_euler_first10": gt_unwrapped.tolist(),
                "pred_raw_euler_first10": pred_rot.tolist(),
                "pred_branch_classification_first10": [
                    {axis: classify_branch(float(pred_rot[t, i]), EPS_NEAR_PI) for i, axis in enumerate(AXES)}
                    for t in range(10)
                ],
            }
        )

    summary = {
        "n_samples": len(samples),
        "raw_euler_mse_first5": float(np.mean(raw_mse5)) if raw_mse5 else None,
        "raw_euler_mse_first10": float(np.mean(raw_mse10)) if raw_mse10 else None,
        "unwrapped_euler_mse_first5": float(np.mean(unwrap_mse5)) if unwrap_mse5 else None,
        "unwrapped_euler_mse_first10": float(np.mean(unwrap_mse10)) if unwrap_mse10 else None,
        "geodesic_orientation_error_first5_rad": float(np.mean(geo5)) if geo5 else None,
        "geodesic_orientation_error_first10_rad": float(np.mean(geo10)) if geo10 else None,
    }
    return summary, per_sample_records


def main() -> None:
    global PREPROCESSOR, POSTPROCESSOR

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-general-samples", type=int, default=N_GENERAL_SAMPLES)
    parser.add_argument("--n-branch-cut-samples", type=int, default=N_BRANCH_CUT_SAMPLES)
    parser.add_argument("--audit-output", default="outputs/eval/euler_branch_audit/euler_branch_audit_B.json")
    parser.add_argument("--samples-output", default="outputs/eval/euler_branch_audit/euler_branch_samples_B.json")
    args = parser.parse_args()

    logger.info("=== loading dataset metadata ===")
    meta = LeRobotDatasetMetadata(DATASET_REPO_ID)
    task_index_to_name = {int(v): k for k, v in meta.tasks["task_index"].items()}
    episodes_df = meta.episodes.to_pandas()

    logger.info("=== 1. dataset-wide Euler wrap-transition audit (parquet, no video/model) ===")
    table = load_full_action_table(Path(meta.root))
    audit = dataset_wide_wrap_audit(table, task_index_to_name)
    logger.info("per_axis summary: %s", {a: audit["per_axis"][a]["wrap_transition_rate"] for a in AXES})
    logger.info("any_axis chunk_with_wrap_rate: %.4f", audit["any_axis"]["chunk_with_wrap_rate"])

    logger.info("=== 2. selecting fixed, reproducible sample sets (seed=%d) ===", SAMPLE_SEED)
    rng = np.random.default_rng(SAMPLE_SEED)
    general_samples, branch_cut_samples = pick_fixed_samples(
        table, episodes_df, task_index_to_name, rng, args.n_general_samples, args.n_branch_cut_samples
    )
    logger.info("general_samples=%d branch_cut_samples=%d", len(general_samples), len(branch_cut_samples))

    logger.info("=== loading policy from %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    PREPROCESSOR, POSTPROCESSOR = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    logger.info("=== building full dataset object with action_horizon=50 chunking (for direct global-index access) ===")
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DATASET_REPO_ID),
        policy=policy.config,
        rename_map=RENAME_MAP,
        batch_size=1,
    )
    dataset = make_dataset(train_cfg)

    logger.info("=== 3. model-vs-GT comparison on the general fixed sample set ===")
    general_summary, _ = run_model_comparison(policy, args.device, dataset, general_samples)
    logger.info("general_summary: %s", general_summary)

    logger.info("=== 4. branch-cut-near sample dump (with model predictions) ===")
    _, branch_cut_records = run_model_comparison(policy, args.device, dataset, branch_cut_samples)

    audit_report = {
        "checkpoint": "B",
        "checkpoint_path": str(Path(CHECKPOINT).resolve()),
        "dataset": DATASET_REPO_ID,
        "dataset_wide_wrap_audit": audit,
        "fixed_sample_set": {
            "seed": SAMPLE_SEED,
            "n_general_samples": len(general_samples),
            "general_sample_indices": [s["sample_index"] for s in general_samples],
            "n_branch_cut_samples": len(branch_cut_samples),
            "branch_cut_sample_indices": [s["sample_index"] for s in branch_cut_samples],
        },
        "model_vs_gt_general_sample_set": general_summary,
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_report, indent=2))
    logger.info("=== wrote %s ===", audit_path)

    samples_report = {
        "checkpoint": "B",
        "eps_near_pi_rad": EPS_NEAR_PI,
        "note": (
            "branch classification is per-axis, per-timestep (first 10 steps of the predicted "
            "chunk); '+pi_branch'/'-pi_branch' means the predicted raw Euler value for that axis "
            "at that timestep is within eps_near_pi_rad of +pi / -pi, otherwise 'middle'."
        ),
        "samples": branch_cut_records,
    }
    samples_path = Path(args.samples_output)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(json.dumps(samples_report, indent=2))
    logger.info("=== wrote %s ===", samples_path)


if __name__ == "__main__":
    main()
