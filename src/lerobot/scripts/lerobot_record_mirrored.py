#!/usr/bin/env python

"""
Mirrored bimanual teleoperation recording:
- Right leader → Left follower
- Left leader → Right follower
- Invert twist joints: shoulder_pan (base) and wrist_roll (second-last before gripper)

Example:

python -m lerobot.scripts.lerobot_record_mirrored \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/ttyACM1 \
  --robot.right_arm_port=/dev/ttyACM0 \
  --robot.id=bimanual_follower \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/ttyACM2 \
  --teleop.right_arm_port=/dev/ttyACM3 \
  --teleop.id=bimanual_leader \
  --dataset.repo_id=<hf_username>/<dataset_name> \
  --dataset.num_episodes=5 \
  --dataset.single_task="Bimanual mirrored handover" \
  --display_data=true
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

import rerun as rr

# Register camera choice types so draccus can decode --robot.cameras
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import create_initial_features, aggregate_pipeline_dataset_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    make_teleoperator_from_config,
)
from lerobot.processor import make_default_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
# Add missing imports to mirror record.py behavior
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener

# Twist joints to invert sign on both arms
TWIST_JOINT_NAMES = ("shoulder_pan.pos", "wrist_roll.pos")


@dataclass
class MirroredDatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 30
    num_episodes: int = 10
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class MirroredRecordConfig:
    robot: RobotConfig
    dataset: MirroredDatasetRecordConfig
    teleop: TeleoperatorConfig
    display_data: bool = False
    # Use vocal synthesis to read events (parity with record.py)
    play_sounds: bool = True
    resume: bool = False


def mirror_and_invert(leader_action: dict[str, float]) -> dict[str, float]:
    """
    Swap left/right prefixes and invert twist joints (shoulder_pan, wrist_roll).
    Example input keys (from BiSO100Leader): 'left_shoulder_pan.pos', 'right_wrist_roll.pos', 'left_gripper.pos'
    Output keys (for BiSO100Follower):      'right_shoulder_pan.pos', 'left_wrist_roll.pos', 'right_gripper.pos'
    """
    mirrored: dict[str, float] = {}
    for key, val in leader_action.items():
        if key.startswith("left_"):
            suffix = key[len("left_"):]
            target_key = f"right_{suffix}"
        elif key.startswith("right_"):
            suffix = key[len("right_"):]
            target_key = f"left_{suffix}"
        else:
            suffix = key
            target_key = key

        if suffix in TWIST_JOINT_NAMES:
            val = -val

        mirrored[target_key] = float(val)
    return mirrored


def record_loop_mirrored(
    robot: Robot,
    teleop: Teleoperator,
    dataset: LeRobotDataset | None,
    fps: int,
    control_time_s: int | float,
    single_task: str | None,
    display_data: bool,
    # Add processors like record.py
    teleop_action_processor=None,
    robot_action_processor=None,
    robot_observation_processor=None,
):
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        # Get raw robot observation
        obs_raw = robot.get_observation()
        # Process observation via pipeline (like record.py)
        obs_processed = (
            robot_observation_processor(obs_raw) if robot_observation_processor is not None else obs_raw
        )
        observation_frame = (
            build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR) if dataset else None
        )

        # Get leader action, mirror+invert to follower action (raw)
        leader_action = teleop.get_action()
        follower_action_raw = mirror_and_invert(leader_action)

        # Process teleop action via pipeline (like record.py)
        action_values = (
            teleop_action_processor((follower_action_raw, obs_raw))
            if teleop_action_processor is not None
            else follower_action_raw
        )

        # Process robot action to send (like record.py)
        robot_action_to_send = (
            robot_action_processor((action_values, obs_raw))
            if robot_action_processor is not None
            else action_values
        )

        # Send to robot
        _ = robot.send_action(robot_action_to_send)

        # Write to dataset using processed obs and teleop action values
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1.0 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


class MirroredTeleop(Teleoperator):
    """Adapter that mirrors left/right and inverts twist joints.

    Subclasses Teleoperator so core recording loop recognizes it as a valid teleop.
    Delegates all behavior to the wrapped teleop except `get_action`, which mirrors/inverts.
    """

    # Minimal identifiers to satisfy Teleoperator expectations
    name = "mirrored_teleop"
    config_class = TeleoperatorConfig

    def __init__(self, inner: Teleoperator):
        # Do not call super().__init__ — this is a thin wrapper around an existing teleop
        self.inner = inner

    # ---- Delegated lifecycle
    def connect(self, calibrate: bool = True) -> None:
        self.inner.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self.inner.disconnect()

    @property
    def is_connected(self) -> bool:
        return self.inner.is_connected

    # ---- Calibration/config passthrough
    @property
    def is_calibrated(self) -> bool:
        return getattr(self.inner, "is_calibrated", True)

    def calibrate(self) -> None:
        if hasattr(self.inner, "calibrate"):
            self.inner.calibrate()

    def configure(self) -> None:
        if hasattr(self.inner, "configure"):
            self.inner.configure()

    # ---- Features passthrough
    @property
    def action_features(self) -> dict:
        return getattr(self.inner, "action_features", {})

    @property
    def feedback_features(self) -> dict:
        return getattr(self.inner, "feedback_features", {})

    # ---- Action/feedback
    def get_action(self) -> dict[str, float]:
        return mirror_and_invert(self.inner.get_action())

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Pass-through; if a mirrored feedback protocol is needed, add mapping here
        if hasattr(self.inner, "send_feedback"):
            self.inner.send_feedback(feedback)


@parser.wrap()
def record_mirrored(cfg: MirroredRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording_mirrored")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    mirrored_teleop = MirroredTeleop(teleop)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
        else:
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )

        robot.connect()
        mirrored_teleop.connect()

        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                # Announce episode start (parity with record.py)
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=mirrored_teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

                if not events["stop_recording"] and recorded < cfg.dataset.num_episodes - 1:
                    # Announce reset (parity with record.py)
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=mirrored_teleop,
                        dataset=None,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                dataset.save_episode()
                recorded += 1
    finally:
        # Announce stop (parity with record.py)
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if mirrored_teleop.is_connected:
            mirrored_teleop.disconnect()

        if listener and hasattr(listener, "stop"):
            listener.stop()

        if cfg.dataset.push_to_hub and dataset:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        if cfg.display_data:
            rr.rerun_shutdown()

        # Announce exit (parity with record.py)
        log_say("Exiting", cfg.play_sounds)

    return dataset


def main():
    register_third_party_plugins()
    record_mirrored()


if __name__ == "__main__":
    main()