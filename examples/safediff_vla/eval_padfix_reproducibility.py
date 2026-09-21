#!/usr/bin/env python
"""Reproducibility check for the padded-action-timestep loss-masking fix ("padfix",
`371c88c8`/upstream `585fc7db`), on `safediff_vla_temporal_decoder_padfix_20k` -- a *fresh* (no
resume) 20k-step run of canonical `libero-safety` code (sin/cos rotation + padfix, no
phase-conditioning/replan code at all) done on this machine/session.

The goal here is NOT to re-litigate whether padfix helps (that's `371c88c8`'s own job, already
argued for in its commit message and test additions) -- it's to check whether the *same canonical
code*, trained fresh from scratch on this machine, produces comparable training/eval behavior to
whatever reference run originally validated it. Same protocol as `eval_baseline_rollout.py`'s own
two-part structure (closed-loop rollout + open-loop calibration), extended with the specific
metrics padfix's own effect would show up in:

  - Closed-loop rollout: success, mean_abs_delta_action / mean_abs_delta2_action, video (exactly
    `eval_baseline_rollout.py`'s own `run_closed_loop_rollout`, reused unchanged).
  - Open-loop prediction error, but broken out per horizon position (0..49) instead of only
    first_5/first_10 -- a "tail error profile": padfix's whole point is that end-of-episode
    chunks were being padded by repeating the last real action past the episode boundary and
    trained against as if real, which (if it hurt anything) should show up specifically as worse
    error at the *tail* of the predicted chunk (high horizon-position indices) versus its head.
    Reports mean position L2 / geodesic rotation / gripper abs error per position, plus a
    head-vs-tail (first 10 vs last 10 positions) summary.
  - Outlier rate: fraction of (sample, horizon-position) position-error values that are robust
    statistical outliers (|error - median| > 3 * 1.4826 * MAD, i.e. a standard ~3-sigma-equivalent
    robust z-score cutoff on the position-error distribution) -- not a hand-picked physical
    threshold, since this repo has no established "acceptable error" unit for this quantity.

Usage:
    uv run python examples/safediff_vla/eval_padfix_reproducibility.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent))
import eval_baseline_rollout as ebr  # noqa: E402  (reuse run_closed_loop_rollout, don't reimplement)
from eval_baseline_rollout import geodesic_angle  # noqa: E402

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_dataset  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.processor.rename_processor import rename_batch_keys  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_padfix_20k/checkpoints/020000/pretrained_model"
DATASET_REPO_ID = "lerobot/vlabench_unified"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}


def robust_outlier_rate(errors: Tensor) -> dict:
    """Fraction of `errors` (any shape, flattened) that are robust statistical outliers:
    `|x - median| > 3 * 1.4826 * MAD`. `1.4826 * MAD` is a consistent estimator of the standard
    deviation for normally-distributed data, so this is the standard "~3-sigma-equivalent, but
    robust to the very outliers it's detecting" cutoff -- no task-specific physical threshold is
    assumed."""
    flat = errors.reshape(-1)
    median = flat.median()
    mad = (flat - median).abs().median()
    scale = 1.4826 * mad
    if scale.item() == 0:
        return {"outlier_rate": 0.0, "median": median.item(), "mad": mad.item(), "threshold": None}
    threshold = 3 * scale
    outlier_rate = (flat - median).abs().gt(threshold).float().mean().item()
    return {
        "outlier_rate": outlier_rate,
        "median": median.item(),
        "mad": mad.item(),
        "threshold_abs_deviation": threshold.item(),
    }


def run_open_loop_tail_profile(policy, device: str, n_episodes: int) -> dict:
    meta = LeRobotDatasetMetadata(DATASET_REPO_ID)
    eps_df = meta.episodes.to_pandas()
    eps_df["task0"] = eps_df["tasks"].apply(lambda t: t[0])
    poker_episodes = sorted(
        eps_df.loc[eps_df["task0"].str.contains("poker", case=False, na=False), "episode_index"].tolist()
    )
    sample_episodes = poker_episodes[:: max(1, len(poker_episodes) // n_episodes)][:n_episodes]
    logger.info("open-loop sample: %d select_poker episodes: %s", len(sample_episodes), sample_episodes)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DATASET_REPO_ID, episodes=sample_episodes),
        policy=policy.config,
        rename_map=RENAME_MAP,
        batch_size=len(sample_episodes),
    )
    dataset = make_dataset(cfg)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=len(sample_episodes), shuffle=False)
    raw_batch = next(iter(loader))
    for cam_key in dataset.meta.camera_keys:
        if cam_key in raw_batch and raw_batch[cam_key].dtype == torch.uint8:
            raw_batch[cam_key] = raw_batch[cam_key].to(dtype=torch.float32) / 255.0
    raw_batch = rename_batch_keys(raw_batch, RENAME_MAP)
    gt_actions_raw = raw_batch[ACTION].clone()  # [B, 50, 7], dataset-native (unnormalized) scale

    batch = preprocessor(raw_batch)
    with torch.no_grad():
        pred_actions_norm, _ = policy.plan_action_chunk(batch)
    pred_actions_raw = postprocessor(pred_actions_norm.cpu()).to(gt_actions_raw.dtype)

    horizon = pred_actions_raw.shape[1]
    pred_pos = pred_actions_raw[..., :3]
    gt_pos = gt_actions_raw[..., :3]
    position_error_per_bh = (pred_pos - gt_pos).norm(dim=-1)  # [B, H]

    pred_rot = pred_actions_raw[..., 3:6].reshape(-1, 3).numpy()
    gt_rot = gt_actions_raw[..., 3:6].reshape(-1, 3).numpy()
    r_pred = Rotation.from_euler("xyz", pred_rot)
    r_gt = Rotation.from_euler("xyz", gt_rot)
    geodesic_per_bh = torch.from_numpy((r_pred.inv() * r_gt).magnitude()).reshape(
        pred_actions_raw.shape[0], horizon
    )

    gripper_abs_error_per_bh = (pred_actions_raw[..., 6] - gt_actions_raw[..., 6]).abs()  # [B, H]

    per_position = []
    for h in range(horizon):
        per_position.append(
            {
                "horizon_position": h,
                "position_l2_raw_units": position_error_per_bh[:, h].mean().item(),
                "orientation_geodesic_rad": geodesic_per_bh[:, h].mean().item(),
                "gripper_abs_error_raw_units": gripper_abs_error_per_bh[:, h].mean().item(),
            }
        )

    head = slice(0, 10)
    tail = slice(horizon - 10, horizon)
    head_vs_tail = {
        "position_l2_head_first10": position_error_per_bh[:, head].mean().item(),
        "position_l2_tail_last10": position_error_per_bh[:, tail].mean().item(),
        "geodesic_head_first10": geodesic_per_bh[:, head].mean().item(),
        "geodesic_tail_last10": geodesic_per_bh[:, tail].mean().item(),
        "gripper_abs_error_head_first10": gripper_abs_error_per_bh[:, head].mean().item(),
        "gripper_abs_error_tail_last10": gripper_abs_error_per_bh[:, tail].mean().item(),
    }

    outliers = {
        "position_l2": robust_outlier_rate(position_error_per_bh),
        "orientation_geodesic": robust_outlier_rate(geodesic_per_bh),
        "gripper_abs_error": robust_outlier_rate(gripper_abs_error_per_bh),
    }

    logger.info("head_vs_tail: %s", head_vs_tail)
    logger.info(
        "outlier_rate: position_l2=%.4f geodesic=%.4f gripper=%.4f",
        outliers["position_l2"]["outlier_rate"],
        outliers["orientation_geodesic"]["outlier_rate"],
        outliers["gripper_abs_error"]["outlier_rate"],
    )

    # Also keep the original first_5/first_10 summary, matching eval_baseline_rollout.py's own
    # open-loop metric shape, for direct comparability with prior reports.
    first_n_summary = {}
    for n in (5, 10):
        pred_n = pred_actions_norm[:, :n].cpu()
        gt_n = batch[ACTION][:, :n].cpu()
        action_error = torch.nn.functional.mse_loss(pred_n, gt_n).item()
        first_n_summary[f"first_{n}"] = {
            "action_error_mse_normalized": action_error,
            "position_error_l2_raw_units": position_error_per_bh[:, :n].mean().item(),
            "orientation_geodesic_error_rad": geodesic_angle(
                pred_actions_raw[:, :n, 3:6], gt_actions_raw[:, :n, 3:6]
            )
            .mean()
            .item(),
        }

    return {
        "episodes": sample_episodes,
        "per_horizon_position": per_position,
        "head_vs_tail": head_vs_tail,
        "outliers": outliers,
        "first_n_summary": first_n_summary,
    }


def main() -> None:
    global CHECKPOINT

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--open-loop-episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", default="outputs/eval/safediff_vla_temporal_decoder_padfix_20k/report.json"
    )
    parser.add_argument(
        "--videos-dir", default="outputs/eval/safediff_vla_temporal_decoder_padfix_20k/videos"
    )
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint
    ebr.CHECKPOINT = args.checkpoint  # run_closed_loop_rollout reads its own module's CHECKPOINT

    set_seed(args.start_seed)

    logger.info("=== loading policy from %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    assert policy.config.architecture == "temporal_decoder", policy.config.architecture
    assert policy.config.action_horizon == 50, policy.config.action_horizon
    assert policy.config.execute_horizon == 50, policy.config.execute_horizon
    assert not policy.config.use_temporal_ensembling
    assert not hasattr(policy.config, "use_phase_conditioning")
    assert not hasattr(policy.config, "replan_on_gripper_close")
    logger.info(
        "condition confirmed: architecture=%s action_horizon=%d execute_horizon=%d "
        "use_temporal_ensembling=%s (no phase_conditioning/replan_on_gripper_close fields exist "
        "on this branch's config at all)",
        policy.config.architecture,
        policy.config.action_horizon,
        policy.config.execute_horizon,
        policy.config.use_temporal_ensembling,
    )

    videos_dir = Path(args.videos_dir)
    logger.info(
        "=== 1. closed-loop rollout (task=%s, n_episodes=%d, start_seed=%d) ===",
        TASK,
        args.n_episodes,
        args.start_seed,
    )
    rollout_results = ebr.run_closed_loop_rollout(
        policy, args.device, args.n_episodes, args.start_seed, videos_dir=videos_dir
    )
    logger.info("overall: %s", rollout_results["overall"])

    logger.info(
        "=== 2. open-loop tail-error profile (in-distribution, %d select_poker episodes) ===",
        args.open_loop_episodes,
    )
    tail_profile = run_open_loop_tail_profile(policy, args.device, args.open_loop_episodes)

    report = {
        "checkpoint": CHECKPOINT,
        "condition": {
            "architecture": policy.config.architecture,
            "action_horizon": policy.config.action_horizon,
            "execute_horizon": policy.config.execute_horizon,
            "use_temporal_ensembling": policy.config.use_temporal_ensembling,
            "freeze_backbone": policy.config.freeze_backbone,
            "lambda_smooth": policy.config.lambda_smooth,
        },
        "task": TASK,
        "closed_loop_rollout": rollout_results,
        "open_loop_tail_error_profile": tail_profile,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("=== wrote report to %s ===", out_path)


if __name__ == "__main__":
    main()
