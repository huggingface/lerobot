#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record a LeRobot dataset via NVIDIA Isaac Teleop -> SO-101.

Runs ``teleoperate.py``'s control loop while also saving each frame to a LeRobot dataset.
``--teleop.type`` selects the device (``xr_controller`` | ``so101_leader``) as in
``teleoperate.py``.

Usage::

    # XR (VR) controller: clutch + soft-orientation IK
    python -m examples.isaac_teleop_to_so101.record \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=so101_follower_arm \\
        --teleop.type=xr_controller \\
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --dataset.repo_id=<hf_user>/<dataset_name> \\
        --dataset.single_task="Pick up vial from rack on the left side" \\
        --dataset.num_episodes=3 \\
        --dataset.episode_time_s=20 \\
        --dataset.reset_time_s=5

    # SO-101 leader arm: 1:1 joint mirror (real leader on /dev/ttyACM1)
    python -m examples.isaac_teleop_to_so101.record \\
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_arm \\
        --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=so101_leader_arm \\
        --launch_plugin=/path/to/IsaacTeleop/install/plugins/so101_leader/so101_leader_plugin \\
        --dataset.repo_id=<hf_user>/<dataset_name> --dataset.single_task="Pick up the cube" \\
        --dataset.num_episodes=3 --dataset.episode_time_s=20 --dataset.reset_time_s=5

The loop/launch knobs mirror ``teleoperate.py`` (tagged ``[xr]`` / ``[leader]`` below).

Keyboard shortcuts: Right/n = end episode early and save, Left/r = discard + re-record,
Esc/q = stop after the current episode. All frames are recorded (including hold frames).
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import make_default_processors
from lerobot.robots import RobotConfig
from lerobot.robots.so_follower import SOFollowerConfig  # noqa: F401  (registers so101_follower)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .common import (
    ALIGN_DURATION_S,
    RESET_DURATION_S,
    Device,
    HoldLatch,
    build_device,
    init_keyboard_listener,
)
from .isaac_teleop import IsaacTeleopConfig


@dataclass
class RecordConfig:
    """CLI config for Isaac Teleop -> SO-101 dataset recording.

    ``--robot.*`` / ``--teleop.*`` / ``--dataset.*`` configure the follower, device, and
    recording; the loop/launch knobs below carry the same ``[xr]`` / ``[leader]`` tags as
    ``teleoperate.py``. Use ``--flag=false`` for booleans (draccus style).
    """

    robot: RobotConfig
    # --teleop.type=xr_controller|so101_leader, resolved against IsaacTeleopConfig's registry.
    teleop: IsaacTeleopConfig
    dataset: DatasetRecordConfig

    # [leader] Path to the so101_leader plugin binary to spawn after CloudXR is up (it then
    # inherits the runtime env). None (default) -> assume the plugin already runs externally.
    launch_plugin: str | None = None

    # [xr] Slew all joints to the reset pose before the first episode (--reset_to_origin=false to
    # keep the arm where it is). After the slew the clutch seeds its home from the measured pose.
    reset_to_origin: bool = True
    # [xr] Duration [s] of the reset-to-origin slew (passed through to setup_xr).
    reset_duration: float = RESET_DURATION_S

    # [leader] Slew the follower to the leader's first pose before mirroring (--align=false to
    # begin the 1:1 mirror immediately; the follower may snap).
    align: bool = True
    # [leader] Duration [s] of the startup alignment slew.
    align_duration: float = ALIGN_DURATION_S

    # Resume recording on an existing (previously interrupted) dataset.
    resume: bool = False


@safe_stop_image_writer
def _record_loop(
    robot,
    device: Device,
    motor_names: list[str],
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    control_time_s: float = 0.0,
    single_task: str | None = None,
) -> None:
    """Run one episode (or reset phase) of the control loop.

    When ``dataset`` is None the loop still controls the robot (so the operator
    can reposition the arm during the reset window) but does not record frames.
    """
    control_interval = 1.0 / fps
    timestamp = 0.0
    start_t = time.perf_counter()
    record_frames = dataset is not None
    hold = HoldLatch(motor_names)

    while timestamp < control_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()

        if record_frames:
            observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)

        # Device idle (XR clutch disengaged, or leader stream stale) -> hold the pose
        # latched on the active->idle edge.
        action = hold.resolve(device.compute(obs), obs)

        robot.send_action(action)

        if record_frames:
            action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            dataset.add_frame({**observation_frame, **action_frame, "task": single_task})

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(control_interval - dt_s, 0.0))
        timestamp = time.perf_counter() - start_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Connect the follower, build the selected Isaac device, and run its pre-loop startup
    # (reset slew / leader align) — shared with teleoperate.py.
    robot, device, motor_names = build_device(cfg)

    # Build dataset feature spec.  The IK pipeline lives inside device.compute(), so the
    # action features are exactly robot.action_features (joint positions in degrees).
    teleop_proc, _, obs_proc = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
    image_writer_threads = cfg.dataset.num_image_writer_threads_per_camera * num_cameras

    dataset: LeRobotDataset | None = None
    listener = None
    try:
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=image_writer_threads if num_cameras > 0 else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=image_writer_threads,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        listener, events = init_keyboard_listener()

        loop_kwargs = {
            "robot": robot,
            "device": device,
            "motor_names": motor_names,
            "events": events,
            "fps": cfg.dataset.fps,
            "single_task": cfg.dataset.single_task,
        }

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                logging.info(f"Recording episode {dataset.num_episodes}")
                _record_loop(
                    **loop_kwargs,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                )

                # Reset window: give the operator time to reposition the scene.
                # Skipped for the last episode (or if stop_recording was set).
                if not events["stop_recording"] and (
                    recorded_episodes < cfg.dataset.num_episodes - 1 or events["rerecord_episode"]
                ):
                    logging.info("Reset the environment")
                    _record_loop(
                        **loop_kwargs,
                        dataset=None,
                        control_time_s=cfg.dataset.reset_time_s,
                    )

                if events["rerecord_episode"]:
                    logging.info("Re-record episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        logging.info("Stop recording")

        # Hardware teardown FIRST, each step guarded: the arm must be freed promptly (not
        # after a potentially long finalize/encode), a cleanup failure must not skip the
        # follower disconnect (which is what disables torque), and neither must prevent
        # the dataset from being finalized below.
        try:
            device.cleanup()
        except Exception:
            logging.exception("Device cleanup failed")
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            logging.exception("Robot disconnect failed")

        # Restore the terminal before the (potentially long) finalize/encode.
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                logging.exception("Keyboard listener stop failed")

        if dataset is not None:
            dataset.finalize()

        if cfg.dataset.push_to_hub:
            if dataset is not None and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        logging.info("Exiting")

    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
