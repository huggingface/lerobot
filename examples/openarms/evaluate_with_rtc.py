#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
OpenArms Policy Evaluation with Real-Time Chunking (RTC)

Evaluates a trained policy on the OpenArms robot using RTC for smooth, continuous motion.
RTC enables large flow-matching policies (Pi0, Pi0.5, SmolVLA) to produce reactive motion
despite high inference latency by asynchronously generating action chunks.

Features:
- Thread-based asynchronous action generation and execution
- RTC for smooth transitions between action chunks
- Dataset recording for evaluation episodes

Example usage:
    python examples/openarms/evaluate_with_rtc.py

    # With custom RTC parameters
    python examples/openarms/evaluate_with_rtc.py \
        --rtc.execution_horizon=12 \
        --rtc.max_guidance_weight=10.0

    # With action interpolation (policy at 30Hz, robot at 50Hz)
    python examples/openarms/evaluate_with_rtc.py \
        --action_interpolation_enabled=true \
        --control_hz=50
"""

import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread

import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor import make_default_processors
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging, log_say

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Default Configuration Constants
# ============================================================================

DEFAULT_HF_MODEL_ID = "lerobot-data-collection/three-folds-pi0"
DEFAULT_HF_EVAL_DATASET_ID = "lerobot-data-collection/three-folds-pi0_eval_rtc"
DEFAULT_TASK_DESCRIPTION = "three-folds-dataset"

DEFAULT_NUM_EPISODES = 1
DEFAULT_FPS = 30
DEFAULT_EPISODE_TIME_SEC = 300
DEFAULT_RESET_TIME_SEC = 60

DEFAULT_CONTROL_HZ = 50

DEFAULT_FOLLOWER_LEFT_PORT = "can0"
DEFAULT_FOLLOWER_RIGHT_PORT = "can1"

DEFAULT_CAMERA_CONFIG = {
    "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video5", width=640, height=480, fps=DEFAULT_FPS),
    "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video1", width=640, height=480, fps=DEFAULT_FPS),
    "base": OpenCVCameraConfig(index_or_path="/dev/video3", width=640, height=480, fps=DEFAULT_FPS),
}


# ============================================================================
# Thread-Safe Robot Wrapper
# ============================================================================


class RobotWrapper:
    """Thread-safe wrapper for robot operations."""

    def __init__(self, robot: OpenArmsFollower):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: dict) -> None:
        with self.lock:
            self.robot.send_action(action)

    @property
    def observation_features(self) -> dict:
        with self.lock:
            return self.robot.observation_features

    @property
    def action_features(self) -> dict:
        with self.lock:
            return self.robot.action_features

    @property
    def name(self) -> str:
        return self.robot.name


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class OpenArmsRTCEvalConfig(HubMixin):
    """Configuration for OpenArms evaluation with RTC."""

    policy: PreTrainedConfig | None = None

    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True,
            execution_horizon=10,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    model_id: str = DEFAULT_HF_MODEL_ID
    eval_dataset_id: str = DEFAULT_HF_EVAL_DATASET_ID
    task: str = DEFAULT_TASK_DESCRIPTION

    num_episodes: int = DEFAULT_NUM_EPISODES
    fps: float = DEFAULT_FPS
    episode_time_sec: float = DEFAULT_EPISODE_TIME_SEC
    reset_time_sec: float = DEFAULT_RESET_TIME_SEC

    follower_left_port: str = DEFAULT_FOLLOWER_LEFT_PORT
    follower_right_port: str = DEFAULT_FOLLOWER_RIGHT_PORT

    device: str = "cuda"

    # Should be higher than inference_delay + execution_horizon
    action_queue_size_to_get_new_actions: int = 30

    record_dataset: bool = True
    push_to_hub: bool = True

    action_interpolation_enabled: bool = False
    control_hz: float = DEFAULT_CONTROL_HZ

    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    torch_compile_disable_cudagraphs: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
            self.model_id = policy_path
        elif self.model_id:
            self.policy = PreTrainedConfig.from_pretrained(self.model_id)
            self.policy.pretrained_path = self.model_id

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ============================================================================
# Action Generation Thread
# ============================================================================


def get_actions_thread(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: OpenArmsRTCEvalConfig,
    episode_active: Event,
):
    """Thread function to asynchronously generate action chunks from the policy."""
    try:
        logger.info("[GET_ACTIONS] Starting action generation thread")

        latency_tracker = LatencyTracker()
        time_per_chunk = 1.0 / cfg.fps

        hw_features = hw_to_dataset_features(robot.observation_features, "observation")
        policy_device = policy.config.device

        logger.info(f"[GET_ACTIONS] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,
            preprocessor_overrides={
                "device_processor": {"device": cfg.device},
            },
        )

        logger.info("[GET_ACTIONS] Preprocessor/postprocessor loaded successfully")

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions
        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        while not shutdown_event.is_set():
            if not episode_active.is_set():
                time.sleep(0.01)
                continue

            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk) if inference_latency else 0

                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)

                obs_with_policy_features = build_dataset_frame(
                    hw_features, obs_processed, prefix="observation"
                )

                for name in obs_with_policy_features:
                    obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
                    if "image" in name:
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].type(torch.float32) / 255
                        )
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                        )
                    obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
                    obs_with_policy_features[name] = obs_with_policy_features[name].to(policy_device)

                obs_with_policy_features["task"] = [cfg.task]
                obs_with_policy_features["robot_type"] = robot.name

                preprocessed_obs = preprocessor(obs_with_policy_features)

                actions = policy.predict_action_chunk(
                    preprocessed_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                original_actions = actions.squeeze(0).clone()
                postprocessed_actions = postprocessor(actions).squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] action_queue_size_to_get_new_actions too small. "
                        "Should be higher than inference delay + execution horizon."
                    )

                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )

                logger.debug(
                    f"[GET_ACTIONS] Generated chunk, latency={new_latency:.3f}s, "
                    f"delay={new_delay}, queue_size={action_queue.qsize()}"
                )
            else:
                time.sleep(0.01)

        logger.info("[GET_ACTIONS] Action generation thread shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()
        sys.exit(1)


# ============================================================================
# Action Execution Thread
# ============================================================================


def _interpolate_actions(prev_action: Tensor, next_action: Tensor, alpha: float) -> Tensor:
    """Linear interpolation between two action tensors."""
    return prev_action + alpha * (next_action - prev_action)


def actor_thread(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: OpenArmsRTCEvalConfig,
    episode_active: Event,
    dataset: LeRobotDataset | None,
    dataset_lock: Lock,
    teleop_action_processor,
    robot_observation_processor,
):
    """Thread function to execute actions on the robot."""
    try:
        logger.info("[ACTOR] Starting actor thread")
        logger.info(f"[ACTOR] interpolation={cfg.action_interpolation_enabled}, control_hz={cfg.control_hz}")

        action_count = 0
        action_keys = [k for k in robot.action_features.keys() if k.endswith(".pos")]

        if cfg.action_interpolation_enabled:
            control_interval = 1.0 / cfg.control_hz
            interp_steps = int(cfg.control_hz / cfg.fps)
        else:
            control_interval = 1.0 / cfg.fps
            interp_steps = 1

        prev_action: Tensor | None = None
        current_action: Tensor | None = None
        interp_step = 0
        last_dataset_frame_time = 0.0

        while not shutdown_event.is_set():
            if not episode_active.is_set():
                prev_action = None
                current_action = None
                interp_step = 0
                time.sleep(0.01)
                continue

            start_time = time.perf_counter()

            if cfg.action_interpolation_enabled:
                if interp_step == 0 or current_action is None:
                    new_action = action_queue.get()
                    if new_action is not None:
                        prev_action = current_action if current_action is not None else new_action.cpu()
                        current_action = new_action.cpu()
                        interp_step = 0

                if current_action is not None:
                    if prev_action is not None and interp_steps > 1:
                        alpha = (interp_step + 1) / interp_steps
                        action_to_send = _interpolate_actions(prev_action, current_action, alpha)
                    else:
                        action_to_send = current_action

                    action_dict = {}
                    for i, key in enumerate(action_keys):
                        if i < len(action_to_send):
                            action_dict[key] = action_to_send[i].item()

                    action_processed = robot_action_processor((action_dict, None))
                    robot.send_action(action_processed)
                    action_count += 1

                    interp_step = (interp_step + 1) % interp_steps

                    if cfg.record_dataset and dataset is not None:
                        if time.perf_counter() - last_dataset_frame_time >= (1.0 / cfg.fps):
                            last_dataset_frame_time = time.perf_counter()
                            with dataset_lock:
                                obs = robot.get_observation()
                                obs_processed = robot_observation_processor(obs)
                                action_for_dataset = teleop_action_processor((action_dict, None))
                                frame = {}
                                for key, value in obs_processed.items():
                                    frame[f"observation.{key}"] = value
                                for key, value in action_for_dataset.items():
                                    frame[f"action.{key}"] = value
                                frame["task"] = cfg.task
                                dataset.add_frame(frame)
            else:
                action = action_queue.get()
                if action is not None:
                    action = action.cpu()
                    action_dict = {}
                    for i, key in enumerate(action_keys):
                        if i < len(action):
                            action_dict[key] = action[i].item()

                    action_processed = robot_action_processor((action_dict, None))
                    robot.send_action(action_processed)
                    action_count += 1

                    if cfg.record_dataset and dataset is not None:
                        with dataset_lock:
                            obs = robot.get_observation()
                            obs_processed = robot_observation_processor(obs)
                            action_for_dataset = teleop_action_processor((action_dict, None))
                            frame = {}
                            for key, value in obs_processed.items():
                                frame[f"observation.{key}"] = value
                            for key, value in action_for_dataset.items():
                                frame[f"action.{key}"] = value
                            frame["task"] = cfg.task
                            dataset.add_frame(frame)

            dt_s = time.perf_counter() - start_time
            sleep_time = max(0, control_interval - dt_s - 0.001)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()
        sys.exit(1)


# ============================================================================
# Main Evaluation Function
# ============================================================================


def _apply_torch_compile(policy, cfg: OpenArmsRTCEvalConfig):
    """Apply torch.compile to the policy's predict_action_chunk method."""
    if policy.name in ["pi05", "pi0"]:
        return policy

    try:
        if not hasattr(torch, "compile"):
            logger.warning(
                f"torch.compile not available. Requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}. Skipping compilation."
            )
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")

        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }

        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")
        logger.warning("Continuing without torch.compile")

    return policy


@parser.wrap()
def main(cfg: OpenArmsRTCEvalConfig):
    """Main evaluation function with RTC."""
    init_logging()

    print("=" * 60)
    print("OpenArms Policy Evaluation with RTC")
    print("=" * 60)
    print(f"\nModel: {cfg.model_id}")
    print(f"Evaluation Dataset: {cfg.eval_dataset_id}")
    print(f"Task: {cfg.task}")
    print(f"Episodes: {cfg.num_episodes}")
    print(f"Episode Duration: {cfg.episode_time_sec}s")
    print(f"RTC Enabled: {cfg.rtc.enabled}")
    print(f"RTC Execution Horizon: {cfg.rtc.execution_horizon}")
    print(f"RTC Max Guidance Weight: {cfg.rtc.max_guidance_weight}")
    print(f"Action Interpolation: {cfg.action_interpolation_enabled}")
    if cfg.action_interpolation_enabled:
        print(f"Control Hz: {cfg.control_hz}")
    print(f"Device: {cfg.device}")
    print("=" * 60)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event
    episode_active = Event()

    # Initialize Robot
    follower_config = OpenArmsFollowerConfig(
        port_left=cfg.follower_left_port,
        port_right=cfg.follower_right_port,
        can_interface="socketcan",
        id="openarms_follower",
        disable_torque_on_disconnect=True,
        max_relative_target=10.0,
        cameras=DEFAULT_CAMERA_CONFIG,
    )

    follower = OpenArmsFollower(follower_config)
    follower.connect(calibrate=False)

    if not follower.is_connected:
        raise RuntimeError("Follower robot failed to connect!")

    robot = RobotWrapper(follower)
    logger.info("Follower robot connected")

    # Build Processors and Dataset Features
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features_hw = {}
    for key, value in follower.action_features.items():
        if key.endswith(".pos"):
            action_features_hw[key] = value

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=True,
        ),
    )

    # Create or Load Dataset
    dataset = None
    dataset_lock = Lock()

    if cfg.record_dataset:
        dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / cfg.eval_dataset_id
        if dataset_path.exists():
            logger.info(f"Evaluation dataset exists at: {dataset_path}")
            logger.info("New episodes will be appended.")
            choice = input("Continue? (y/n): ").strip().lower()
            if choice != "y":
                logger.info("Aborting evaluation.")
                follower.disconnect()
                return

        dataset = LeRobotDataset.create(
            repo_id=cfg.eval_dataset_id,
            fps=int(cfg.fps),
            features=dataset_features,
            robot_type=follower.name,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=12,
        )
        logger.info(f"Dataset created: {cfg.eval_dataset_id}")

    # Load Policy
    logger.info(f"Loading policy from: {cfg.model_id}")

    policy_class = get_policy_class(cfg.policy.type)
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if cfg.policy.type in ["pi05", "pi0"]:
        config.compile_model = cfg.use_torch_compile

    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    policy.config.rtc_config = cfg.rtc
    policy.init_rtc_processor()

    assert policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 are supported for RTC"

    policy = policy.to(cfg.device)
    policy.eval()

    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    logger.info(f"Policy loaded: {policy.name}")

    # Create Action Queue and Start Threads
    action_queue = ActionQueue(cfg.rtc)

    get_actions_t = Thread(
        target=get_actions_thread,
        args=(
            policy,
            robot,
            robot_observation_processor,
            action_queue,
            shutdown_event,
            cfg,
            episode_active,
        ),
        daemon=True,
        name="GetActions",
    )
    get_actions_t.start()
    logger.info("Started action generation thread")

    actor_t = Thread(
        target=actor_thread,
        args=(
            robot,
            robot_action_processor,
            action_queue,
            shutdown_event,
            cfg,
            episode_active,
            dataset,
            dataset_lock,
            teleop_action_processor,
            robot_observation_processor,
        ),
        daemon=True,
        name="Actor",
    )
    actor_t.start()
    logger.info("Started actor thread")

    # Run Evaluation Episodes
    episode_idx = 0

    try:
        while episode_idx < cfg.num_episodes and not shutdown_event.is_set():
            log_say(f"Evaluating episode {episode_idx + 1} of {cfg.num_episodes}")
            logger.info(f"\n{'='*40}")
            logger.info(f"Episode {episode_idx + 1} / {cfg.num_episodes}")
            logger.info(f"{'='*40}")

            action_queue = ActionQueue(cfg.rtc)
            episode_active.set()
            episode_start_time = time.time()

            while (time.time() - episode_start_time) < cfg.episode_time_sec:
                if shutdown_event.is_set():
                    break

                elapsed = time.time() - episode_start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    logger.info(
                        f"[MAIN] Episode progress: {elapsed:.0f}/{cfg.episode_time_sec}s, "
                        f"queue_size={action_queue.qsize()}"
                    )

                time.sleep(0.5)

            episode_active.clear()
            logger.info(f"Episode {episode_idx + 1} completed")

            if cfg.record_dataset and dataset is not None:
                with dataset_lock:
                    if dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
                        logger.info(
                            f"Saving episode {episode_idx + 1} "
                            f"({dataset.episode_buffer['size']} frames)"
                        )
                        dataset.save_episode()

            episode_idx += 1

            # Manual reset between episodes
            if not shutdown_event.is_set() and episode_idx < cfg.num_episodes:
                log_say("Waiting for manual reset")
                logger.info("Manually reset the environment and press ENTER to continue")
                input("Press ENTER when ready...")

        logger.info(f"Evaluation complete! {episode_idx} episodes recorded")
        log_say("Evaluation complete", blocking=True)

    except KeyboardInterrupt:
        logger.info("\n\nEvaluation interrupted by user")

    finally:
        shutdown_event.set()
        episode_active.clear()

        if get_actions_t.is_alive():
            logger.info("Waiting for action generation thread to finish...")
            get_actions_t.join(timeout=5.0)

        if actor_t.is_alive():
            logger.info("Waiting for actor thread to finish...")
            actor_t.join(timeout=5.0)

        follower.disconnect()
        logger.info("Follower disconnected")

        if cfg.record_dataset and dataset is not None:
            dataset.finalize()
            if cfg.push_to_hub:
                logger.info("Uploading to Hugging Face Hub...")
                dataset.push_to_hub(private=True)

        logger.info("Cleanup completed")


if __name__ == "__main__":
    main()
