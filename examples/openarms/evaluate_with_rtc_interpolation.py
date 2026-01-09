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
OpenArms Policy Evaluation with RTC + Interpolation

Combines Real-Time Chunking (RTC) with smooth action interpolation:
- RTC for reactive motion despite high inference latency
- Action interpolation for smooth robot movements
- Speed multiplier to execute faster than training
- Velocity feedforward and PID tuning
- Decoupled inference (async) from robot control

Example usage:
    python examples/openarms/evaluate_with_rtc_interpolation.py

    # With custom RTC parameters
    python examples/openarms/evaluate_with_rtc_interpolation.py \
        --rtc.execution_horizon=12 \
        --rtc.max_guidance_weight=10.0
"""

import logging
import math
import sys
import time
import traceback
from collections import deque
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
DEFAULT_HF_EVAL_DATASET_ID = "lerobot-data-collection/three-folds-pi0_eval_rtc_interp"
DEFAULT_TASK_DESCRIPTION = "three-folds-dataset"

DEFAULT_NUM_EPISODES = 1
DEFAULT_CAMERA_FPS = 30           # Camera hardware limit
DEFAULT_POLICY_FPS = 30           # What the policy was trained with
DEFAULT_SPEED_MULTIPLIER = 1.0    # Execute actions faster (1.0 = normal, 1.2 = 20% faster)
DEFAULT_ROBOT_FPS = 50            # Robot command rate (higher = smoother)
DEFAULT_EPISODE_TIME_SEC = 300
DEFAULT_RESET_TIME_SEC = 60

DEFAULT_FOLLOWER_LEFT_PORT = "can0"
DEFAULT_FOLLOWER_RIGHT_PORT = "can1"

# PID tuning defaults
DEFAULT_KP_SCALE = 0.7            # Lower = smoother but slower
DEFAULT_KD_SCALE = 1.3            # Higher = less overshoot
DEFAULT_USE_VELOCITY_FF = True    # Velocity feedforward

DEFAULT_CAMERA_CONFIG = {
    "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video5", width=640, height=480, fps=DEFAULT_CAMERA_FPS),
    "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video1", width=640, height=480, fps=DEFAULT_CAMERA_FPS),
    "base": OpenCVCameraConfig(index_or_path="/dev/video3", width=640, height=480, fps=DEFAULT_CAMERA_FPS),
}


# ============================================================================
# Action Interpolator
# ============================================================================


class ActionInterpolator:
    """Interpolate between RTC actions for smoother robot control with velocity estimation."""
    
    def __init__(self, robot_fps: int):
        self.robot_fps = robot_fps
        self.prev_action: Tensor | None = None
        self.curr_action: Tensor | None = None
        self.prev_time: float = 0
        self.curr_time: float = 0
        self.last_interpolated: Tensor | None = None
        
    def update(self, new_action: Tensor) -> None:
        self.prev_action = self.curr_action
        self.prev_time = self.curr_time
        self.curr_action = new_action
        self.curr_time = time.perf_counter()
        
    def get_interpolated_action(self) -> tuple[Tensor | None, Tensor | None]:
        """Returns (interpolated_position, estimated_velocity)"""
        if self.curr_action is None:
            return None, None
        if self.prev_action is None:
            self.last_interpolated = self.curr_action.clone()
            return self.curr_action, torch.zeros_like(self.curr_action)
        
        # Time-based interpolation
        current_time = time.perf_counter()
        dt_actions = self.curr_time - self.prev_time
        if dt_actions <= 0:
            dt_actions = 1.0 / 30  # Fallback
            
        t = (current_time - self.prev_time) / dt_actions
        t = max(0.0, min(t, 1.5))  # Allow slight extrapolation
        
        interpolated = self.prev_action + t * (self.curr_action - self.prev_action)
        
        # Estimate velocity
        dt_robot = 1.0 / self.robot_fps
        if self.last_interpolated is not None:
            velocity = (interpolated - self.last_interpolated) / dt_robot
        else:
            velocity = (self.curr_action - self.prev_action) / dt_actions
        
        self.last_interpolated = interpolated.clone()
        return interpolated, velocity
    
    def reset(self):
        self.prev_action = None
        self.curr_action = None
        self.prev_time = 0
        self.curr_time = 0
        self.last_interpolated = None


class HzTracker:
    """Track and display actual loop frequency."""
    
    def __init__(self, name: str = "Loop", window_size: int = 100, print_interval: float = 2.0):
        self.name = name
        self.timestamps = deque(maxlen=window_size)
        self.last_print_time = 0
        self.print_interval = print_interval
        self.extra_info_fn = None  # Optional callback for extra info
        
    def tick(self) -> float | None:
        now = time.perf_counter()
        self.timestamps.append(now)
        
        if len(self.timestamps) < 2:
            return None
            
        hz = (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])
        
        if now - self.last_print_time >= self.print_interval:
            extra = ""
            if self.extra_info_fn:
                extra = self.extra_info_fn()
            print(f"[CONTROL] {self.name}: {hz:.1f} Hz{extra}", flush=True)
            self.last_print_time = now
            
        return hz
    
    def get_avg_hz(self) -> float | None:
        if len(self.timestamps) < 2:
            return None
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])
    
    def reset(self):
        self.timestamps.clear()
        self.last_print_time = 0


# ============================================================================
# Thread-Safe Robot Wrapper
# ============================================================================


class RobotWrapper:
    """Thread-safe wrapper for robot operations with custom PID gains."""

    def __init__(
        self, 
        robot: OpenArmsFollower,
        custom_kp: dict | None = None,
        custom_kd: dict | None = None,
        use_velocity_ff: bool = False,
    ):
        self.robot = robot
        self.lock = Lock()
        self.custom_kp = custom_kp
        self.custom_kd = custom_kd
        self.use_velocity_ff = use_velocity_ff

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: dict, velocity_ff: dict | None = None) -> None:
        with self.lock:
            vel_ff = velocity_ff if self.use_velocity_ff else None
            self.robot.send_action(
                action, 
                custom_kp=self.custom_kp, 
                custom_kd=self.custom_kd,
                velocity_feedforward=vel_ff,
            )

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
class OpenArmsRTCInterpEvalConfig(HubMixin):
    """Configuration for OpenArms evaluation with RTC + Interpolation."""

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
    camera_fps: float = DEFAULT_CAMERA_FPS
    policy_fps: float = DEFAULT_POLICY_FPS
    speed_multiplier: float = DEFAULT_SPEED_MULTIPLIER
    robot_fps: float = DEFAULT_ROBOT_FPS
    episode_time_sec: float = DEFAULT_EPISODE_TIME_SEC
    reset_time_sec: float = DEFAULT_RESET_TIME_SEC

    # PID tuning
    kp_scale: float | None = DEFAULT_KP_SCALE
    kd_scale: float | None = DEFAULT_KD_SCALE
    use_velocity_ff: bool = DEFAULT_USE_VELOCITY_FF

    follower_left_port: str = DEFAULT_FOLLOWER_LEFT_PORT
    follower_right_port: str = DEFAULT_FOLLOWER_RIGHT_PORT

    device: str = "cuda"

    # Should be higher than inference_delay + execution_horizon
    action_queue_size_to_get_new_actions: int = 30

    record_dataset: bool = True
    push_to_hub: bool = True

    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    torch_compile_disable_cudagraphs: bool = True

    @property
    def effective_policy_fps(self) -> int:
        return int(self.policy_fps * self.speed_multiplier)

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
# Action Generation Thread (RTC)
# ============================================================================


def get_actions_thread(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: OpenArmsRTCInterpEvalConfig,
    episode_active: Event,
):
    """Thread function to asynchronously generate action chunks from the policy using RTC."""
    try:
        logger.info("[GET_ACTIONS] Starting RTC action generation thread")

        latency_tracker = LatencyTracker()
        inference_hz_tracker = HzTracker(name="Inference", window_size=20, print_interval=5.0)
        time_per_chunk = 1.0 / cfg.effective_policy_fps  # Use effective FPS with speed multiplier

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
                
                # Set extra info to show latency
                inference_hz_tracker.extra_info_fn = lambda lat=new_latency, delay=new_delay: f" | Latency: {lat*1000:.0f}ms | Delay: {delay}"
                inference_hz_tracker.tick()

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
# Actor Thread with Interpolation
# ============================================================================


def actor_thread(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: OpenArmsRTCInterpEvalConfig,
    episode_active: Event,
    dataset: LeRobotDataset | None,
    dataset_lock: Lock,
    teleop_action_processor,
    robot_observation_processor,
    interpolator: ActionInterpolator,
    hz_tracker: HzTracker,
    dataset_features: dict,
):
    """Thread function to execute interpolated actions on the robot at high frequency."""
    try:
        logger.info(f"[ACTOR] Starting actor thread with interpolation at {cfg.robot_fps}Hz")

        action_count = 0
        robot_interval = 1.0 / cfg.robot_fps  # High frequency robot control
        effective_policy_interval = 1.0 / cfg.effective_policy_fps  # Action consume rate
        action_keys = [k for k in robot.action_features.keys() if k.endswith(".pos")]
        
        last_action_consume_time = 0
        interpolator.reset()
        hz_tracker.reset()
        
        # Set up extra info callback to show queue size
        hz_tracker.extra_info_fn = lambda: f" | Queue: {action_queue.qsize()}"

        while not shutdown_event.is_set():
            if not episode_active.is_set():
                time.sleep(0.01)
                interpolator.reset()
                hz_tracker.reset()
                last_action_consume_time = 0
                continue

            start_time = time.perf_counter()
            
            # Consume new action from RTC queue at effective_policy_fps rate
            current_time = time.perf_counter()
            if current_time - last_action_consume_time >= effective_policy_interval:
                action = action_queue.get()
                
                if action is not None:
                    action = action.cpu()
                    interpolator.update(action)
                    last_action_consume_time = current_time
                    
                    # Record to dataset at action consume rate
                    if cfg.record_dataset and dataset is not None:
                        with dataset_lock:
                            obs = robot.get_observation()
                            obs_processed = robot_observation_processor(obs)
                            
                            action_dict = {}
                            for i, key in enumerate(action_keys):
                                if i < len(action):
                                    action_dict[key] = action[i].item()
                            
                            action_for_dataset = teleop_action_processor((action_dict, None))

                            # Use build_dataset_frame to properly format keys
                            observation_frame = build_dataset_frame(
                                dataset_features, obs_processed, prefix="observation"
                            )
                            action_frame = build_dataset_frame(
                                dataset_features, action_for_dataset, prefix="action"
                            )
                            frame = {**observation_frame, **action_frame, "task": cfg.task}
                            dataset.add_frame(frame)
            
            # Get interpolated action and send to robot at robot_fps (highest rate)
            interp_action, velocity = interpolator.get_interpolated_action()
            
            if interp_action is not None:
                # Convert tensor to dict
                action_dict = {}
                velocity_dict = {}
                for i, key in enumerate(action_keys):
                    if i < len(interp_action):
                        action_dict[key] = interp_action[i].item()
                        if velocity is not None:
                            motor_name = key.removesuffix(".pos")
                            velocity_dict[motor_name] = velocity[i].item()
                
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed, velocity_ff=velocity_dict)
                action_count += 1

            hz_tracker.tick()

            # Maintain robot control rate
            dt_s = time.perf_counter() - start_time
            sleep_time = max(0, robot_interval - dt_s - 0.001)
            if sleep_time > 0:
                time.sleep(sleep_time)

        final_hz = hz_tracker.get_avg_hz()
        if final_hz:
            logger.info(f"[ACTOR] Final robot Hz: {final_hz:.1f}")
        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()
        sys.exit(1)


# ============================================================================
# Helper Functions
# ============================================================================


def build_custom_gains(robot: OpenArmsFollower, kp_scale: float | None, kd_scale: float | None) -> tuple[dict | None, dict | None]:
    """Build custom PID gains dict from robot config."""
    if kp_scale is None and kd_scale is None:
        return None, None
        
    custom_kp = {}
    custom_kd = {}
    for arm in ["right", "left"]:
        bus = robot.bus_right if arm == "right" else robot.bus_left
        for i, motor_name in enumerate(bus.motors):
            full_name = f"{arm}_{motor_name}"
            default_kp = robot.config.position_kp[i] if isinstance(robot.config.position_kp, list) else robot.config.position_kp
            default_kd = robot.config.position_kd[i] if isinstance(robot.config.position_kd, list) else robot.config.position_kd
            custom_kp[full_name] = default_kp * (kp_scale or 1.0)
            custom_kd[full_name] = default_kd * (kd_scale or 1.0)
    return custom_kp, custom_kd


def _apply_torch_compile(policy, cfg: OpenArmsRTCInterpEvalConfig):
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


# ============================================================================
# Main Evaluation Function
# ============================================================================


@parser.wrap()
def main(cfg: OpenArmsRTCInterpEvalConfig):
    """Main evaluation function with RTC + Interpolation."""
    init_logging()

    print("=" * 70)
    print("OpenArms Policy Evaluation with RTC + Interpolation")
    print("=" * 70)
    print(f"\nModel: {cfg.model_id}")
    print(f"Evaluation Dataset: {cfg.eval_dataset_id}")
    print(f"Task: {cfg.task}")
    print(f"\n--- Timing ---")
    print(f"Camera FPS: {cfg.camera_fps} (hardware limit)")
    print(f"Policy trained at: {cfg.policy_fps}Hz")
    print(f"Speed multiplier: {cfg.speed_multiplier}x")
    print(f"Effective policy FPS: {cfg.effective_policy_fps}Hz (action consume rate)")
    print(f"Robot FPS: {cfg.robot_fps}Hz (interpolated commands)")
    print(f"\n--- RTC ---")
    print(f"RTC Enabled: {cfg.rtc.enabled}")
    print(f"Execution Horizon: {cfg.rtc.execution_horizon}")
    print(f"Max Guidance Weight: {cfg.rtc.max_guidance_weight}")
    print(f"\n--- PID Tuning ---")
    print(f"KP scale: {cfg.kp_scale}")
    print(f"KD scale: {cfg.kd_scale}")
    print(f"Velocity feedforward: {cfg.use_velocity_ff}")
    print(f"\n--- Episodes ---")
    print(f"Episodes: {cfg.num_episodes}")
    print(f"Duration: {cfg.episode_time_sec}s per episode")
    print(f"Device: {cfg.device}")
    print("=" * 70)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event
    episode_active = Event()

    # Initialize Robot
    camera_config = {
        "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video5", width=640, height=480, fps=int(cfg.camera_fps)),
        "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video1", width=640, height=480, fps=int(cfg.camera_fps)),
        "base": OpenCVCameraConfig(index_or_path="/dev/video3", width=640, height=480, fps=int(cfg.camera_fps)),
    }

    follower_config = OpenArmsFollowerConfig(
        port_left=cfg.follower_left_port,
        port_right=cfg.follower_right_port,
        can_interface="socketcan",
        id="openarms_follower",
        disable_torque_on_disconnect=True,
        max_relative_target=15.0,
        cameras=camera_config,
    )

    follower = OpenArmsFollower(follower_config)
    follower.connect(calibrate=False)

    if not follower.is_connected:
        raise RuntimeError("Follower robot failed to connect!")

    # Build custom PID gains
    custom_kp, custom_kd = build_custom_gains(follower, cfg.kp_scale, cfg.kd_scale)
    if custom_kp:
        logger.info(f"Custom gains: kp_scale={cfg.kp_scale}, kd_scale={cfg.kd_scale}")
    if cfg.use_velocity_ff:
        logger.info("Velocity feedforward enabled")

    robot = RobotWrapper(
        follower, 
        custom_kp=custom_kp, 
        custom_kd=custom_kd,
        use_velocity_ff=cfg.use_velocity_ff,
    )
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

        # Dataset uses effective policy FPS
        dataset = LeRobotDataset.create(
            repo_id=cfg.eval_dataset_id,
            fps=cfg.effective_policy_fps,
            features=dataset_features,
            robot_type=follower.name,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=12,
        )
        logger.info(f"Dataset created: {cfg.eval_dataset_id} at {cfg.effective_policy_fps}Hz")

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

    # Create Action Queue, Interpolator, and Hz Tracker
    action_queue = ActionQueue(cfg.rtc)
    interpolator = ActionInterpolator(robot_fps=int(cfg.robot_fps))
    hz_tracker = HzTracker(name="Robot", window_size=100, print_interval=2.0)

    # Start Threads
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
    logger.info("Started RTC action generation thread")

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
            interpolator,
            hz_tracker,
            dataset_features,
        ),
        daemon=True,
        name="Actor",
    )
    actor_t.start()
    logger.info(f"Started actor thread with interpolation at {cfg.robot_fps}Hz")

    # Run Evaluation Episodes
    episode_idx = 0

    try:
        while episode_idx < cfg.num_episodes and not shutdown_event.is_set():
            log_say(f"Evaluating episode {episode_idx + 1} of {cfg.num_episodes}")
            logger.info(f"\n{'='*50}")
            logger.info(f"Episode {episode_idx + 1} / {cfg.num_episodes}")
            logger.info(f"{'='*50}")

            action_queue = ActionQueue(cfg.rtc)
            interpolator.reset()
            hz_tracker.reset()
            episode_active.set()
            episode_start_time = time.time()

            while (time.time() - episode_start_time) < cfg.episode_time_sec:
                if shutdown_event.is_set():
                    break

                elapsed = time.time() - episode_start_time
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
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

