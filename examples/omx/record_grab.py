#!/usr/bin/env python3
"""
Auto-record grab episodes for the OMX robot arm.

Each episode cycle:
  1. grab_and_place  — grab cube from workspace center and place at a random (pan, reach) position
  2. HOME            — return arm to home with gripper open
  3. record_grab     — execute a targeted grab to the stored position while recording
                       observations + actions to a LeRobotDataset

Usage (run from repo root):
    python -m examples.omx.record_grab \\
        --robot.type=omx_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=omx_follower \\
        --robot.cameras="{ wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \\
        --dataset.repo_id=<hf_username>/<dataset_name> \\
        --dataset.root=data/omx_grab \\
        --dataset.num_episodes=50 \\
        --dataset.single_task="Grab the cube" \\
        --dataset.streaming_encoding=true
"""

import logging
from dataclasses import dataclass
from pprint import pformat

import numpy as np

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.omx_follower import OmxFollower
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.robot_utils import precise_sleep

from .reset_environment import (
    APPROACH_SPEED,
    GRIPPER_CLOSE_POS,
    HOME_POSE,
    PUSH_END_ELBOW_FLEX,
    PUSH_END_SHOULDER_LIFT,
    PUSH_START_ELBOW_FLEX,
    PUSH_START_SHOULDER_LIFT,
    array_to_pose,
    grab_cube,
    horizontal_wrist_flex,
    move_to_pose,
    place_cube,
    pose_to_array,
)

# ── Grab-episode motion parameters ────────────────────────────────────────────

# Shoulder-lift offset for the raised approach phase (subtracted from the target sl, arm is higher).
GRAB_RAISE_SL_OFFSET = 20.0
GRAB_LOWER_SPEED = 20.0
RECORD_SPEED = 30.0

# Pose the arm travels to after closing the gripper (cube held).
GRAB_CARRY_POSE = {
    "shoulder_pan.pos": -23.0,
    "shoulder_lift.pos": 5.0,
    "elbow_flex.pos": 18.0,
    "wrist_flex.pos": -14.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": GRIPPER_CLOSE_POS,
}

# Per-joint jitter limits (degrees) applied to transit waypoints for human-like variation.
# Cube-approach and carry poses are never jittered to preserve precision.
_JITTER_LIMITS: dict[str, float] = {
    "shoulder_pan.pos": 5.0,
    "shoulder_lift.pos": 4.0,
    "elbow_flex.pos": 4.0,
    "wrist_flex.pos": 3.0,
    "wrist_roll.pos": 2.0,
    "gripper.pos": 0.0,
}


def _jitter_pose(pose: dict, rng: np.random.Generator) -> dict:
    """Return a copy of pose with independent per-joint random perturbations."""
    return {
        k: v + rng.uniform(-_JITTER_LIMITS.get(k, 0.0), _JITTER_LIMITS.get(k, 0.0)) for k, v in pose.items()
    }


def _random_stuck_pose(rng: np.random.Generator) -> dict:
    """Return a physically plausible stuck pose (failed grasp), gripper closed.

    ef bounds are piecewise-linear in sl so the arm stays in a reachable,
    table-safe envelope across the full sl range:
      sl=-50 → ef ∈ [  0,  50]   (arm raised, can be bent forward)
      sl=  0 → ef ∈ [-25,  25]   (mid reach)
      sl= 30 → ef ∈ [-20,   0]   (arm extended, little room to flex)
    wrist_flex is randomly offset from the horizontal value.
    """
    pan = float(rng.uniform(-5.0, 35.0))
    sl = float(rng.uniform(-50.0, 30.0))

    if sl <= 0.0:
        alpha = (sl + 50.0) / 50.0  # 0 at sl=-50, 1 at sl=0
        ef_lo = alpha * -25.0  # 0 → -25
        ef_hi = 50.0 + alpha * -25.0  # 50 → 25
    else:
        alpha = sl / 30.0  # 0 at sl=0, 1 at sl=30
        ef_lo = -25.0 + alpha * 5.0  # -25 → -20
        ef_hi = 25.0 + alpha * -25.0  # 25 → 0

    ef = float(rng.uniform(ef_lo, ef_hi))
    wf = horizontal_wrist_flex(sl, ef) + float(rng.uniform(-15.0, 15.0))
    return {
        "shoulder_pan.pos": pan,
        "shoulder_lift.pos": sl,
        "elbow_flex.pos": ef,
        "wrist_flex.pos": wf,
        "wrist_roll.pos": float(rng.uniform(-15.0, 15.0)),
        "gripper.pos": GRIPPER_CLOSE_POS,
    }


logger = logging.getLogger(__name__)


@dataclass
class OmxRecordGrabConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Resume recording on an existing dataset.
    resume: bool = False
    # Fraction of episodes that start from a random stuck pose (gripper closed) to
    # generate recovery data.  0.0 = disabled, 1.0 = all episodes are recovery starts.
    recovery_prob: float = 0.5


def record_episode_spline(
    robot: OmxFollower,
    waypoints: list[dict],
    speeds: list[float],
    dataset: LeRobotDataset,
    task: str,
) -> None:
    """Execute a Catmull-Rom-style spline through waypoints, recording each frame.

    Segment durations are parameterized from the maximum absolute joint delta
    between consecutive waypoints divided by the requested segment speed,
    producing non-uniform timing in joint space. Interior tangents are derived
    from the adjacent per-segment velocities, with clamped (zero-velocity)
    endpoints so the arm starts and stops smoothly. Each segment is cubic
    Hermite, giving C1 continuity at every waypoint.
    """
    pts = [pose_to_array(w) for w in waypoints]
    n = len(pts)

    # Steps and duration per segment
    n_steps_list = []
    timestamps = []
    for i in range(n - 1):
        max_dist = float(np.max(np.abs(pts[i + 1] - pts[i])))
        ns = max(1, int(max_dist / speeds[i] * dataset.fps)) if max_dist >= 0.5 else 0
        n_steps_list.append(ns)
        timestamps.append(ns / dataset.fps)

    # Velocity tangents (deg/sec) — clamped at endpoints, Catmull-Rom for interior
    vels = [np.zeros_like(pts[0])]
    for i in range(1, n - 1):
        v_prev = (pts[i] - pts[i - 1]) / timestamps[i - 1] if timestamps[i - 1] > 0 else np.zeros_like(pts[0])
        v_next = (pts[i + 1] - pts[i]) / timestamps[i] if timestamps[i] > 0 else np.zeros_like(pts[0])
        vels.append(0.5 * (v_prev + v_next))
    vels.append(np.zeros_like(pts[0]))

    dt = 1.0 / dataset.fps
    for seg in range(n - 1):
        ns = n_steps_list[seg]
        if ns == 0:
            continue
        p0, p1 = pts[seg], pts[seg + 1]
        # Scale velocity (deg/sec) to t-space tangent (deg/t-unit, where t: 0→1 over ns steps)
        m0 = vels[seg] * timestamps[seg]
        m1 = vels[seg + 1] * timestamps[seg]

        for step in range(1, ns + 1):
            t = step / ns
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            commanded = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

            action = array_to_pose(commanded)
            robot.send_action(action)
            obs = robot.get_observation()
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            dataset.add_frame({**obs_frame, **action_frame, "task": task})
            precise_sleep(dt)


def record_grab_episode(
    robot: OmxFollower,
    dataset: LeRobotDataset,
    pan: float,
    t: float,
    task: str,
    recovery_start: bool = False,
) -> None:
    """Execute a targeted grab to the stored (pan, t) position, recording every frame.

    Normal sequence (initial HOME move is NOT recorded):
      HOME → raised approach above cube → lower → close gripper
           → raise [jittered] → retract [jittered] → GRAB_CARRY_POSE → drop → HOME

    Recovery sequence (recovery_start=True): arm is moved to a random stuck pose
    (gripper closed) without recording, then recording begins from there:
      stuck_pose → raised approach above cube → [normal grab sequence from there]

    All segments are joined by a Catmull-Rom spline (C1-continuous velocities).
    """
    sl = PUSH_START_SHOULDER_LIFT + t * (PUSH_END_SHOULDER_LIFT - PUSH_START_SHOULDER_LIFT)
    ef = PUSH_START_ELBOW_FLEX + t * (PUSH_END_ELBOW_FLEX - PUSH_START_ELBOW_FLEX)
    sl_raised = sl - GRAB_RAISE_SL_OFFSET
    wf_horizontal = horizontal_wrist_flex(sl, ef)

    rng = np.random.default_rng()

    if recovery_start:
        stuck_pose = _random_stuck_pose(rng)
        logger.info(f"Recovery start: {stuck_pose}")
        move_to_pose(robot, stuck_pose, APPROACH_SPEED)
        first_waypoints = [stuck_pose]
        first_speeds = []
    else:
        jittery_start = _jitter_pose(HOME_POSE, rng)
        move_to_pose(robot, jittery_start, APPROACH_SPEED)
        first_waypoints = [jittery_start]
        first_speeds = []

    waypoints = first_waypoints + [
        {  # raised approach: arm above cube
            "shoulder_pan.pos": pan,
            "shoulder_lift.pos": sl_raised,
            "elbow_flex.pos": ef,
            "wrist_flex.pos": horizontal_wrist_flex(sl_raised, ef),
            "wrist_roll.pos": 0.0,
            "gripper.pos": 60.0,
        },
        {  # lower onto cube — no jitter: precision needed
            "shoulder_pan.pos": pan,
            "shoulder_lift.pos": sl,
            "elbow_flex.pos": ef,
            "wrist_flex.pos": wf_horizontal,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 60.0,
        },
        {  # close gripper — no jitter: precision needed
            "shoulder_pan.pos": pan,
            "shoulder_lift.pos": sl,
            "elbow_flex.pos": ef,
            "wrist_flex.pos": wf_horizontal,
            "wrist_roll.pos": 0.0,
            "gripper.pos": GRIPPER_CLOSE_POS,
        },
        _jitter_pose(
            {  # raise with cube
                "shoulder_pan.pos": pan,
                "shoulder_lift.pos": sl_raised,
                "elbow_flex.pos": ef,
                "wrist_flex.pos": horizontal_wrist_flex(sl_raised, ef),
                "wrist_roll.pos": 0.0,
                "gripper.pos": GRIPPER_CLOSE_POS,
            },
            rng,
        ),
        _jitter_pose(
            {  # retract: fold arm toward HOME before sweeping to carry zone
                "shoulder_pan.pos": pan * 0.25,
                "shoulder_lift.pos": HOME_POSE["shoulder_lift.pos"] + 5.0,
                "elbow_flex.pos": HOME_POSE["elbow_flex.pos"] - 5.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": GRIPPER_CLOSE_POS,
            },
            rng,
        ),
        GRAB_CARRY_POSE,  # no jitter: target drop zone
        {**GRAB_CARRY_POSE, "gripper.pos": 60.0},  # drop cube
        HOME_POSE,
    ]
    speeds = first_speeds + [
        RECORD_SPEED,  # (HOME →) raised approach
        GRAB_LOWER_SPEED,  # raised approach → lower
        GRAB_LOWER_SPEED,  # lower → close gripper
        RECORD_SPEED,  # close gripper → raise
        RECORD_SPEED,  # raise → retract
        RECORD_SPEED,  # retract → carry pose
        RECORD_SPEED,  # carry pose → drop
        RECORD_SPEED,  # drop → HOME
    ]

    record_episode_spline(robot, waypoints, speeds, dataset, task)

    # Dwell at HOME for ~0.5 s before next episode
    home_action = build_dataset_frame(dataset.features, HOME_POSE, prefix=ACTION)
    dt = 1.0 / dataset.fps
    for _ in range(int(dataset.fps * 0.5)):
        robot.send_action(HOME_POSE)
        obs = robot.get_observation()
        obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        dataset.add_frame({**obs_frame, **home_action, "task": task})
        precise_sleep(dt)


@parser.wrap()
def record_grab(cfg: OmxRecordGrabConfig) -> LeRobotDataset:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info(pformat(cfg))

    robot = make_robot_from_config(cfg.robot)
    use_videos = cfg.dataset.video

    teleop_action_processor, _, robot_obs_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_obs_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=use_videos,
        ),
    )

    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
    dataset = None

    try:
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                streaming_encoding=cfg.dataset.streaming_encoding,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
        else:
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=use_videos,
                streaming_encoding=cfg.dataset.streaming_encoding,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )

        robot.connect(calibrate=True)

        rng = np.random.default_rng()
        with VideoEncodingManager(dataset):
            for episode_idx in range(cfg.dataset.num_episodes):
                logger.info(f"=== Episode {episode_idx + 1}/{cfg.dataset.num_episodes} ===")

                logger.info("Step 1: grabbing and placing cube...")
                grab_cube(robot)
                pan, t = place_cube(robot)
                logger.info(f"Cube placed at pan={pan:.1f}, reach={t:.2f}")

                recovery_start = cfg.recovery_prob > 0 and float(rng.random()) < cfg.recovery_prob
                logger.info(f"Step 2: recording {'recovery ' if recovery_start else ''}grab episode...")
                record_grab_episode(
                    robot,
                    dataset,
                    pan,
                    t,
                    cfg.dataset.single_task,
                    recovery_start=recovery_start,
                )

                dataset.save_episode()
                logger.info(f"Episode {episode_idx + 1} saved.")

    finally:
        if dataset:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()

    if cfg.dataset.push_to_hub and dataset and dataset.num_episodes > 0:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


if __name__ == "__main__":
    record_grab()
