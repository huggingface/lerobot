#!/usr/bin/env python
"""
Human-in-the-Loop (HIL) Data Collection with Real-Time Chunking (RTC).

Implements the RaC paradigm (Hu et al., 2025) with RTC for large flow-matching models
(Pi0, Pi0.5, SmolVLA) that have high inference latency. RTC generates action chunks
asynchronously in a background thread for smooth robot control.

For fast models (ACT, Diffusion), use hil_data_collection.py instead.

The workflow:
1. Policy runs autonomously with RTC
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
    # OpenArms (RTC)
    # Cameras are configured at robot level via --robot.cameras
    python examples/rac/hil_data_collection_rtc.py \
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
        --rtc.execution_horizon=20 \
        --rtc.max_guidance_weight=5.0 \
        --rtc.prefix_attention_schedule=LINEAR \
        --interpolation_multiplier=2 \
        --dataset.push_to_hub=true
"""

import logging
import math
import time
from dataclasses import dataclass, field
from pprint import pformat
from threading import Event, Lock, Thread
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
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionInterpolator, ActionQueue, LatencyTracker, RTCConfig
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import BiOpenArmFollowerConfig  # noqa: F401
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.openarm_mini.config_openarm_mini import OpenArmMiniConfig  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


@dataclass
class HILRTCConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: HILDatasetConfig
    policy: PreTrainedConfig | None = None
    rtc: RTCConfig = field(default_factory=lambda: RTCConfig(enabled=True, execution_horizon=20))
    interpolation_multiplier: int = 2  # Control rate multiplier (1=off, 2=2x, 3=3x)
    display_data: bool = True
    play_sounds: bool = True
    resume: bool = False
    device: str = "cuda"
    use_torch_compile: bool = False  # First compile takes minutes, disable for real-time
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
        self.rtc.enabled = True

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


class ThreadSafeRobot:
    """Thread-safe wrapper for robot operations."""

    def __init__(self, robot: Robot):
        self._robot = robot
        self._lock = Lock()

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return self._robot.get_observation()

    def send_action(self, action: dict) -> None:
        with self._lock:
            self._robot.send_action(action)

    @property
    def observation_features(self) -> dict:
        return self._robot.observation_features

    @property
    def action_features(self) -> dict:
        return self._robot.action_features

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def cameras(self):
        return getattr(self._robot, "cameras", {})


def rtc_inference_thread(
    policy: PreTrainedPolicy,
    obs_holder: dict,
    hw_features: dict,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    queue_holder: dict,
    shutdown_event: Event,
    policy_active: Event,
    cfg: HILRTCConfig,
):
    """Background thread for RTC action chunk generation."""
    latency_tracker = LatencyTracker()
    time_per_chunk = 1.0 / cfg.dataset.fps
    threshold = 30
    stats_window_start = time.perf_counter()
    policy_inference_count = 0
    latency_sum_s = 0.0

    while not shutdown_event.is_set():
        if not policy_active.is_set():
            time.sleep(0.01)
            continue

        queue = queue_holder.get("queue")
        obs = obs_holder.get("obs")
        if queue is None or obs is None:
            time.sleep(0.01)
            continue

        if queue.qsize() <= threshold:
            try:
                current_time = time.perf_counter()
                idx_before = queue.get_action_index()
                prev_actions = queue.get_left_over()

                latency = latency_tracker.max()
                delay = math.ceil(latency / time_per_chunk) if latency else 0

                obs_batch = build_dataset_frame(hw_features, obs, prefix="observation")
                for name in obs_batch:
                    obs_batch[name] = torch.from_numpy(obs_batch[name])
                    if "image" in name:
                        obs_batch[name] = obs_batch[name].float() / 255
                        obs_batch[name] = obs_batch[name].permute(2, 0, 1).contiguous()
                    obs_batch[name] = obs_batch[name].unsqueeze(0).to(cfg.device)

                obs_batch["task"] = [cfg.dataset.single_task]
                obs_batch["robot_type"] = obs_holder.get("robot_type", "unknown")

                preprocessed = preprocessor(obs_batch)
                actions = policy.predict_action_chunk(
                    preprocessed, inference_delay=delay, prev_chunk_left_over=prev_actions
                )

                original = actions.squeeze(0).clone()
                processed = postprocessor(actions).squeeze(0)
                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)
                queue.merge(original, processed, new_delay, idx_before)
                policy_inference_count += 1
                latency_sum_s += new_latency
                logger.debug(f"[RTC] Inference latency={new_latency:.2f}s, queue={queue.qsize()}")
            except Exception as e:
                logger.error(f"[RTC] Error: {e}")
                time.sleep(0.5)
        else:
            time.sleep(0.01)

        now = time.perf_counter()
        if cfg.log_hz and (window_elapsed := now - stats_window_start) >= cfg.hz_log_interval_s:
            policy_hz = policy_inference_count / window_elapsed
            avg_latency_ms = (latency_sum_s / policy_inference_count * 1000.0) if policy_inference_count else 0.0
            logger.info(
                "[HIL RTC rates] policy=%.1f Hz | avg_inference=%.1f ms | queue=%d",
                policy_hz,
                avg_latency_ms,
                queue.qsize(),
            )
            stats_window_start = now
            policy_inference_count = 0
            latency_sum_s = 0.0


@safe_stop_image_writer
def rollout_loop(
    robot: ThreadSafeRobot,
    teleop: Teleoperator,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset,
    events: dict,
    cfg: HILRTCConfig,
    queue_holder: dict,
    obs_holder: dict,
    policy_active: Event,
    hw_features: dict,
):
    """Rollout loop with RTC for asynchronous inference."""
    fps = cfg.dataset.fps

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    frame_buffer = []
    teleop_disable_torque(teleop)

    was_paused = False
    waiting_for_takeover = False
    last_action: dict[str, Any] | None = None
    action_keys = list(dataset.features[ACTION]["names"])
    obs_state_names = list(dataset.features[f"{OBS_STR}.state"]["names"])
    obs_image_names = [
        key.removeprefix(f"{OBS_STR}.images.")
        for key in dataset.features
        if key.startswith(f"{OBS_STR}.images.")
    ]

    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    control_interval = interpolator.get_control_interval(fps)

    robot_action: dict[str, Any] = {}
    timestamp = 0
    start_t = time.perf_counter()
    stats_window_start = start_t
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
            queue_holder["queue"] = ActionQueue(cfg.rtc)
            policy_active.clear()
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

        # Transition to paused state
        if events["policy_paused"] and not was_paused:
            policy_active.clear()
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

        obs_holder["obs"] = obs_filtered

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
            # Policy execution with RTC
            if not policy_active.is_set():
                policy_active.set()

            queue = queue_holder["queue"]

            if interpolator.needs_new_action():
                new_action = queue.get() if queue else None
                if new_action is not None:
                    interpolator.add(new_action.cpu())

            action_tensor = interpolator.get()
            if action_tensor is not None:
                robot_action = {
                    k: action_tensor[i].item() for i, k in enumerate(action_keys) if i < len(action_tensor)
                }
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
            robot_hz = robot_command_count / window_elapsed
            logger.info(
                "[HIL RTC rates] robot=%.1f Hz (target=%.1f)",
                robot_hz,
                fps * cfg.interpolation_multiplier,
            )
            stats_window_start = now
            robot_command_count = 0

    policy_active.clear()
    teleop_disable_torque(teleop)

    for frame in frame_buffer:
        dataset.add_frame(frame)


@parser.wrap()
def hil_rtc_collect(cfg: HILRTCConfig) -> LeRobotDataset:
    """Main HIL data collection function with RTC."""
    init_logging()
    logger.info(pformat(cfg.__dict__))

    if cfg.display_data:
        init_rerun(session_name="hil_rtc_collection")

    robot_raw = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    teleop_proc, obs_proc = make_identity_processors()

    action_features_hw = {k: v for k, v in robot_raw.action_features.items() if k.endswith(".pos")}
    observation_features_hw = {}
    for k, v in robot_raw.observation_features.items():
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
    shutdown_event = Event()
    policy_active = Event()
    rtc_thread = None

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
            if hasattr(robot_raw, "cameras") and robot_raw.cameras:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot_raw.cameras),
                )
        else:
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot_raw.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot_raw.cameras if hasattr(robot_raw, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        # Load policy with RTC
        policy_class = get_policy_class(cfg.policy.type)
        policy_config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
        if hasattr(policy_config, "compile_model"):
            policy_config.compile_model = cfg.use_torch_compile
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=policy_config)
        policy.config.rtc_config = cfg.rtc
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()
        policy = policy.to(cfg.device)
        policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        robot_raw.connect()
        robot = ThreadSafeRobot(robot_raw)
        teleop.connect()
        listener, events = init_keyboard_listener()

        queue_holder = {"queue": ActionQueue(cfg.rtc)}
        obs_holder = {"obs": None, "robot_type": robot.robot_type}
        hw_features = hw_to_dataset_features(observation_features_hw, "observation")

        rtc_thread = Thread(
            target=rtc_inference_thread,
            args=(
                policy,
                obs_holder,
                hw_features,
                preprocessor,
                postprocessor,
                queue_holder,
                shutdown_event,
                policy_active,
                cfg,
            ),
            daemon=True,
        )
        rtc_thread.start()

        print_controls(rtc=True)
        print(f"  Policy: {cfg.policy.pretrained_path}")
        print(f"  Task: {cfg.dataset.single_task}")
        print(f"  Interpolation: {cfg.interpolation_multiplier}x\n")

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Episode {dataset.num_episodes}", cfg.play_sounds)

                queue_holder["queue"] = ActionQueue(cfg.rtc)

                rollout_loop(
                    robot=robot,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    events=events,
                    cfg=cfg,
                    queue_holder=queue_holder,
                    obs_holder=obs_holder,
                    policy_active=policy_active,
                    hw_features=hw_features,
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

        shutdown_event.set()
        policy_active.clear()

        if rtc_thread and rtc_thread.is_alive():
            rtc_thread.join(timeout=2.0)

        if dataset:
            dataset.finalize()

        if robot_raw.is_connected:
            robot_raw.disconnect()
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
    hil_rtc_collect()


if __name__ == "__main__":
    main()
