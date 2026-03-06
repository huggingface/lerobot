#!/usr/bin/env python
"""
Human-in-the-Loop (HIL) Data Collection with Policy Rollout.

Implements the RaC paradigm (Hu et al., 2025) for LeRobot with standard synchronous
inference. For large models with high inference latency, use hil_data_collection_rtc.py.

The workflow:
1. Policy runs autonomously
2. Press SPACE to pause - robot holds position
3. Press 'c' to take control - human provides RECOVERY + CORRECTION
4. Press 'p' to hand control back to policy and continue recording
5. Press → to end episode (save and continue to next)
6. Reset, then do next rollout

Keyboard Controls:
    SPACE  - Pause policy (robot holds position, no recording)
    c      - Take control (start correction, recording resumes)
    p      - Resume policy after pause/correction (recording continues)
    →      - End episode (save and continue to next)
    ←      - Re-record episode
    ESC    - Stop recording and push dataset to hub

Usage:
    # OpenArms (synchronous inference)
    # Cameras are configured at robot level via --robot.cameras
    python examples/rac/hil_data_collection.py \
        --robot.type=bi_openarm_follower \
        --robot.left_arm_config.port=can1 \
        --robot.left_arm_config.side=left \
        --robot.right_arm_config.port=can0 \
        --robot.right_arm_config.side=right \
        --robot.cameras='{left_wrist: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30}, right_wrist: {type: opencv, index_or_path: "/dev/video4", width: 1280, height: 720, fps: 30}, base: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30}}' \
        --teleop.type=openarm_mini \
        --teleop.port_left=/dev/ttyACM0 \
        --teleop.port_right=/dev/ttyACM1 \
        --policy.path=lerobot-data-collection/level2_final_quality2_rabc \
        --dataset.repo_id=lerobot-data-collection/hil_65 \
        --dataset.single_task="Fold the T-shirt properly" \
        --dataset.fps=30 \
        --dataset.episode_time_s=1000 \
        --dataset.num_episodes=1 \
        --interpolation_multiplier=2 \
        --dataset.push_to_hub=true
"""

import logging
import time
from dataclasses import dataclass
from pprint import pformat
from typing import Any

import torch
from hil_utils import (
    HILDatasetConfig,
    init_keyboard_listener,
    make_identity_processors,
    print_controls,
    reset_loop,
    teleop_disable_torque,
    teleop_smooth_move_to,
)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionInterpolator
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import BiOpenArmFollowerConfig  # noqa: F401
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.openarm_mini.config_openarm_mini import OpenArmMiniConfig  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import is_headless, predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


@dataclass
class HILConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: HILDatasetConfig
    policy: PreTrainedConfig | None = None
    interpolation_multiplier: int = 2  # Control rate multiplier (1=off, 2=2x, 3=3x)
    display_data: bool = True
    play_sounds: bool = True
    resume: bool = False
    device: str = "cuda"
    log_hz: bool = True
    hz_log_interval_s: float = 2.0

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("policy.path is required")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@safe_stop_image_writer
def rollout_loop(
    robot: Robot,
    teleop: Teleoperator,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset,
    events: dict,
    cfg: HILConfig,
):
    """Rollout loop with standard synchronous inference."""
    fps = cfg.dataset.fps
    device = get_safe_torch_device(cfg.device)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    frame_buffer = []
    teleop_disable_torque(teleop)

    was_paused = False
    waiting_for_takeover = False
    last_action: dict[str, Any] | None = None
    robot_action: dict[str, Any] = {}
    action_keys = list(dataset.features[ACTION]["names"])
    obs_state_names = list(dataset.features[f"{OBS_STR}.state"]["names"])
    obs_image_names = [
        key.removeprefix(f"{OBS_STR}.images.")
        for key in dataset.features
        if key.startswith(f"{OBS_STR}.images.")
    ]

    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    control_interval = interpolator.get_control_interval(fps)

    timestamp = 0
    start_t = time.perf_counter()
    stats_window_start = start_t
    policy_inference_count = 0
    robot_command_count = 0

    while timestamp < cfg.dataset.episode_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            events["policy_paused"] = False
            events["correction_active"] = False
            events["resume_policy"] = False
            break

        if events["resume_policy"] and (
            events["policy_paused"] or events["correction_active"] or waiting_for_takeover
        ):
            events["resume_policy"] = False
            events["start_next_episode"] = False
            events["policy_paused"] = False
            events["correction_active"] = False
            waiting_for_takeover = False
            was_paused = False
            last_action = None
            interpolator.reset()
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

        # Transition to paused state
        if events["policy_paused"] and not was_paused:
            obs = robot.get_observation()
            robot_pos = {
                k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features
            }
            teleop_smooth_move_to(teleop, robot_pos, duration_s=2.0, fps=50)
            events["start_next_episode"] = False
            waiting_for_takeover = True
            was_paused = True
            interpolator.reset()

        # Takeover
        if waiting_for_takeover and events["start_next_episode"]:
            teleop_disable_torque(teleop)
            events["start_next_episode"] = False
            events["correction_active"] = True
            waiting_for_takeover = False

        obs = robot.get_observation()
        obs_filtered = {k: obs[k] for k in obs_state_names if k in obs}
        obs_filtered.update({k: obs[k] for k in obs_image_names if k in obs})
        obs_frame = build_dataset_frame(dataset.features, obs_filtered, prefix=OBS_STR)

        if events["correction_active"]:
            robot_action = teleop.get_action()
            robot.send_action(robot_action)
            robot_command_count += 1
            action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
            frame_buffer.append({**obs_frame, **action_frame, "task": cfg.dataset.single_task})

        elif waiting_for_takeover or events["policy_paused"]:
            if last_action:
                robot.send_action(last_action)
                robot_command_count += 1

        else:
            # Policy execution with optional interpolation
            if interpolator.needs_new_action():
                action_values = predict_action(
                    observation=obs_frame,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=cfg.dataset.single_task,
                    robot_type=robot.robot_type,
                )
                policy_inference_count += 1
                robot_action = make_robot_action(action_values, dataset.features)
                action_tensor = torch.tensor([robot_action[k] for k in action_keys])
                interpolator.add(action_tensor)

            interp_action = interpolator.get()
            if interp_action is not None:
                robot_action = {k: interp_action[i].item() for i, k in enumerate(action_keys)}
                robot.send_action(robot_action)
                robot_command_count += 1
                last_action = robot_action
                action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
                frame_buffer.append({**obs_frame, **action_frame, "task": cfg.dataset.single_task})

        if cfg.display_data and robot_action:
            log_rerun_data(observation=obs_filtered, action=robot_action)

        dt = time.perf_counter() - loop_start
        if (sleep_time := control_interval - dt) > 0:
            precise_sleep(sleep_time)
        now = time.perf_counter()
        timestamp = now - start_t

        if cfg.log_hz and (window_elapsed := now - stats_window_start) >= cfg.hz_log_interval_s:
            policy_hz = policy_inference_count / window_elapsed
            robot_hz = robot_command_count / window_elapsed
            logger.info(
                "[HIL rates] policy=%.1f Hz (target=%.1f) | robot=%.1f Hz (target=%.1f)",
                policy_hz,
                fps,
                robot_hz,
                fps * cfg.interpolation_multiplier,
            )
            stats_window_start = now
            policy_inference_count = 0
            robot_command_count = 0

    teleop_disable_torque(teleop)

    for frame in frame_buffer:
        dataset.add_frame(frame)


@parser.wrap()
def hil_collect(cfg: HILConfig) -> LeRobotDataset:
    """Main HIL data collection function."""
    init_logging()
    logger.info(pformat(cfg.__dict__))

    if cfg.display_data:
        init_rerun(session_name="hil_collection")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    teleop_proc, obs_proc = make_identity_processors()

    action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith(".pos")}
    observation_features_hw = {}
    for k, v in robot.observation_features.items():
        if k.endswith(".pos") or isinstance(v, tuple):
            observation_features_hw[k] = v

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc,
            initial_features=create_initial_features(observation=observation_features_hw),
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
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )
            if hasattr(robot, "cameras") and robot.cameras:
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
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        policy = make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        robot.connect()
        teleop.connect()
        listener, events = init_keyboard_listener()

        print_controls(rtc=False)
        print(f"  Policy: {cfg.policy.pretrained_path}")
        print(f"  Task: {cfg.dataset.single_task}")
        print(f"  Interpolation: {cfg.interpolation_multiplier}x\n")

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Episode {dataset.num_episodes}", cfg.play_sounds)

                rollout_loop(
                    robot=robot,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    events=events,
                    cfg=cfg,
                )

                if events["rerecord_episode"]:
                    log_say("Re-recording", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded += 1

                if recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                    reset_loop(robot, teleop, events, cfg.dataset.fps)

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub and dataset is not None:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


def main():
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    hil_collect()


if __name__ == "__main__":
    main()
