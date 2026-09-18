#!/usr/bin/env python
"""Baseline rollout + open-loop action-error eval for the clean-v2 20k `temporal_decoder`
checkpoint (`outputs/train/safediff_vla_temporal_decoder_v2_baseline_20k/checkpoints/020000`).

Fixed execution-strategy condition for this run (the first of the planned execute_horizon /
temporal-ensembling ablation): execute_horizon=action_horizon=50 (full open-loop chunk, no
mid-chunk replanning), temporal ensembling OFF, subgoal OFF (architecture="temporal_decoder").

Two independent measurements, both against VLABench's `select_poker` primitive task (chosen
because it is well represented in the `lerobot/vlabench_unified` training set -- the originally
used `select_fruit` env task does not appear anywhere in that dataset, which would have made any
result here uninterpretable):

  1. Closed-loop rollout (the real eval): run `n_episodes` real VLABench episodes, record
     per-episode/overall success, wall-clock runtime, and smoothness of the *executed* action
     stream (mean_abs_delta_action / mean_abs_delta2_action, gripper open<->close transition
     steps).
  2. Open-loop prediction error: for a sample of `select_poker` episodes' first frame, compare
     `plan_action_chunk`'s predicted 50-step chunk against the dataset's real demonstrated chunk
     (same target the decoder was trained against) restricted to the first 5 / first 10 steps.
     NOTE: `eval_split=0.0` was used for training (see train_config.json), so there is no true
     held-out split -- these episodes were seen during training. This is an in-distribution
     calibration check, not a generalization test.

Usage:
    uv run python examples/safediff_vla/eval_baseline_rollout.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import VLABenchEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.processor.rename_processor import rename_batch_keys
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_v2_baseline_20k/checkpoints/020000/pretrained_model"
DATASET_REPO_ID = "lerobot/vlabench_unified"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
GRIPPER_INDEX = 6
GRIPPER_THRESHOLD = 0.5  # VLABench gripper action range is [0, 1] (envs/vlabench.py ACTION_LOW/HIGH)


def mean_abs_delta(actions: Tensor) -> tuple[float, float]:
    """First/second differences of an executed [T, A] action stream, mirroring the
    `|da/dt|` / `|d2a/dt2|` convention already used by `sanity_check_temporal_decoder.py`."""
    if actions.shape[0] < 2:
        return float("nan"), float("nan")
    velocity = actions[1:] - actions[:-1]
    mean_abs_delta_action = velocity.abs().mean().item()
    if actions.shape[0] < 3:
        return mean_abs_delta_action, float("nan")
    acceleration = velocity[1:] - velocity[:-1]
    mean_abs_delta2_action = acceleration.abs().mean().item()
    return mean_abs_delta_action, mean_abs_delta2_action


def gripper_transition_steps(actions: Tensor) -> list[int]:
    """Step indices where the executed gripper command crosses `GRIPPER_THRESHOLD`. Per
    `envs/vlabench.py`'s `_build_ctrl_from_action` (`finger_qpos = gripper * FINGER_OPEN`), higher
    gripper values are *more open* -- so `side=1` (gripper > threshold) is "open", `side=0` is
    "closed"."""
    gripper = actions[:, GRIPPER_INDEX]
    side = (gripper > GRIPPER_THRESHOLD).to(torch.int8)
    changes = (side[1:] != side[:-1]).nonzero(as_tuple=True)[0]
    return [int(i) + 1 for i in changes.tolist()]


def failure_onset_step(actions: Tensor) -> int | None:
    """Step index of the first open->closed gripper transition (the episode's first grasp
    attempt). VLABench's env exposes no intermediate failure signal (only terminal
    `is_success`) -- used as an operational proxy for "when did this episode's outcome become
    effectively decided" on episodes that end in failure. `None` if the gripper never closes."""
    gripper = actions[:, GRIPPER_INDEX]
    side = (gripper > GRIPPER_THRESHOLD).to(torch.int8)
    closing = ((side[:-1] == 1) & (side[1:] == 0)).nonzero(as_tuple=True)[0]
    return int(closing[0].item()) + 1 if len(closing) > 0 else None


def probe_video(video_path: str) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None, int]:
    """Decode a written mp4's first and last frame (verification only -- not part of the write
    path) to confirm the file is real and non-empty, reusing the `av` dependency `write_video`
    already requires. Returns (first_frame_shape, last_frame_shape, n_frames)."""
    import av

    with av.open(video_path) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if not frames:
        return None, None, 0
    return frames[0].shape, frames[-1].shape, len(frames)


def geodesic_angle(euler_pred: Tensor, euler_gt: Tensor) -> Tensor:
    """Per-step geodesic (SO(3)) angle in radians between two batches of `[..., 3]` Euler-angle
    rotation actions, treating them as intrinsic XYZ Euler angles per VLABench's own action
    convention (`ACTION_DIM = 7  # pos(3) + euler(3) + gripper(1)`, `envs/vlabench.py`). These are
    raw (unnormalized) dataset/action-space units, NOT converted to any further physical scale --
    no per-step angular-delta-limit constant for VLABench's controller is available in this repo
    (unlike LIBERO's `osc_output_scale`, which does not apply to VLABench actions)."""
    shape = euler_pred.shape[:-1]
    r_pred = Rotation.from_euler("xyz", euler_pred.reshape(-1, 3).cpu().numpy())
    r_gt = Rotation.from_euler("xyz", euler_gt.reshape(-1, 3).cpu().numpy())
    r_rel = r_pred.inv() * r_gt
    angles = r_rel.magnitude()  # radians, shape [-1]
    return torch.from_numpy(angles).reshape(shape).to(euler_pred.dtype)


def run_closed_loop_rollout(
    policy, device: str, n_episodes: int, start_seed: int, videos_dir: Path | None = None
) -> dict:
    """Thin wrapper around the standard `lerobot.scripts.lerobot_eval.eval_policy()` -- env
    construction, seed propagation (`start_seed` -> `eval_policy`'s own per-batch `range(...)`),
    `policy.reset()`, `env.reset(seed=...)`, the step loop, done/truncated masking, the
    render_callback/write_video pipeline, episode indexing, and the returned dict shape are all
    exactly `eval_policy`'s own -- nothing here reimplements them. The only extension is
    `episode_callback`, a hook `eval_policy` invokes once per finished episode with the
    already-computed `rollout_data`, used here to derive the baseline-specific smoothness metrics
    without an extra rollout() call or the much heavier `return_episode_data=True` path."""
    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    extra_metrics: dict[int, dict] = {}

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        actions_ep = rollout_data[ACTION][env_idx, :ep_len]
        mad, mad2 = mean_abs_delta(actions_ep)
        extra_metrics[episode_ix] = {
            "episode_len": ep_len,
            "mean_abs_delta_action": mad,
            "mean_abs_delta2_action": mad2,
            "gripper_transition_steps": gripper_transition_steps(actions_ep),
            "failure_onset_step": failure_onset_step(actions_ep),
        }
        logger.info(
            "episode %d len=%d |da/dt|=%.4f |d2a/dt2|=%.4f grip_transitions=%s failure_onset=%s",
            episode_ix,
            ep_len,
            mad,
            mad2,
            extra_metrics[episode_ix]["gripper_transition_steps"],
            extra_metrics[episode_ix]["failure_onset_step"],
        )

    try:
        info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=n_episodes if videos_dir is not None else 0,
            videos_dir=videos_dir,
            start_seed=start_seed,
            episode_callback=episode_callback,
        )
    finally:
        env.close()

    video_paths = info.get("video_paths", [])
    per_episode = []
    for ep in info["per_episode"]:
        ix = ep["episode_ix"]
        merged = {**ep, **extra_metrics.get(ix, {})}
        if ix < len(video_paths):
            merged["video_path"] = video_paths[ix]
            first_shape, last_shape, n_frames = probe_video(video_paths[ix])
            logger.info(
                "episode %d video: %d frames, first_frame.shape=%s last_frame.shape=%s -> %s",
                ix,
                n_frames,
                first_shape,
                last_shape,
                video_paths[ix],
            )
        per_episode.append(merged)

    agg = info["aggregated"]
    overall = {
        "task": TASK,
        "n_episodes": n_episodes,
        "pc_success": agg["pc_success"],
        "total_runtime_s": agg["eval_s"],
        "mean_runtime_per_episode_s": agg["eval_ep_s"],
    }
    return {"per_episode": per_episode, "per_task": {TASK: overall}, "overall": overall}


def run_open_loop_error(policy, device: str, n_episodes: int) -> dict:
    meta = LeRobotDatasetMetadata(DATASET_REPO_ID)
    eps_df = meta.episodes.to_pandas()
    eps_df["task0"] = eps_df["tasks"].apply(lambda t: t[0])
    poker_episodes = sorted(eps_df.loc[eps_df["task0"].str.contains("poker", case=False, na=False), "episode_index"].tolist())
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

    results = {}
    for horizon in (5, 10):
        pred_n = pred_actions_norm[:, :horizon].cpu()
        gt_n = batch[ACTION][:, :horizon].cpu()
        action_error = torch.nn.functional.mse_loss(pred_n, gt_n).item()

        pred_r = pred_actions_raw[:, :horizon]
        gt_r = gt_actions_raw[:, :horizon]
        position_error = (pred_r[..., :3] - gt_r[..., :3]).norm(dim=-1).mean().item()
        orientation_geodesic_error = geodesic_angle(pred_r[..., 3:6], gt_r[..., 3:6]).mean().item()

        results[f"first_{horizon}"] = {
            "action_error_mse_normalized": action_error,
            "position_error_l2_raw_units": position_error,
            "orientation_geodesic_error_rad": orientation_geodesic_error,
        }
    return {"episodes": sample_episodes, "metrics": results}


def main() -> None:
    global CHECKPOINT

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--open-loop-episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/safediff_vla_temporal_decoder_v2_baseline/execute_horizon_50_ensembling_off.json")
    parser.add_argument("--record-videos", action="store_true", help="Capture and save an mp4 per rollout episode.")
    parser.add_argument("--videos-dir", default="outputs/eval/safediff_vla_temporal_decoder_v2_baseline/videos")
    parser.add_argument("--skip-open-loop", action="store_true", help="Run only the closed-loop rollout (for quick repro tests).")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    set_seed(args.start_seed)

    logger.info("=== loading policy from %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    assert policy.config.architecture == "temporal_decoder", policy.config.architecture
    assert policy.config.action_horizon == 50, policy.config.action_horizon
    assert policy.config.execute_horizon == 50, policy.config.execute_horizon
    assert policy.config.use_temporal_ensembling is False, policy.config.use_temporal_ensembling
    logger.info(
        "condition confirmed: architecture=%s action_horizon=%d execute_horizon=%d use_temporal_ensembling=%s",
        policy.config.architecture,
        policy.config.action_horizon,
        policy.config.execute_horizon,
        policy.config.use_temporal_ensembling,
    )

    videos_dir = Path(args.videos_dir) if args.record_videos else None
    logger.info(
        "=== 1. closed-loop rollout (task=%s, n_episodes=%d, start_seed=%d, record_videos=%s) ===",
        TASK,
        args.n_episodes,
        args.start_seed,
        bool(videos_dir),
    )
    rollout_results = run_closed_loop_rollout(policy, args.device, args.n_episodes, args.start_seed, videos_dir=videos_dir)
    logger.info("overall: %s", rollout_results["overall"])

    report = {
        "checkpoint": CHECKPOINT,
        "condition": {
            "architecture": policy.config.architecture,
            "action_horizon": policy.config.action_horizon,
            "execute_horizon": policy.config.execute_horizon,
            "use_temporal_ensembling": policy.config.use_temporal_ensembling,
        },
        "task": TASK,
        "closed_loop_rollout": rollout_results,
    }

    if not args.skip_open_loop:
        logger.info("=== 2. open-loop prediction error (in-distribution, %d select_poker episodes) ===", args.open_loop_episodes)
        open_loop_results = run_open_loop_error(policy, args.device, args.open_loop_episodes)
        logger.info("metrics: %s", open_loop_results["metrics"])
        report["open_loop_prediction_error"] = open_loop_results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("=== wrote report to %s ===", out_path)


if __name__ == "__main__":
    main()
