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
from lerobot.policies.rtc.action_interpolator import ActionInterpolator
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor import (
    DeltaActionsProcessorStep,
    NormalizerProcessorStep,
    TransitionKey,
    create_transition,
    make_default_processors,
    to_delta_actions,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging, log_say

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Default Configuration Constants
# ============================================================================

DEFAULT_HF_MODEL_ID = "lerobot-data-collection/ablation2-5_0"
DEFAULT_HF_EVAL_DATASET_ID = "lerobot-data-collection/test"
DEFAULT_TASK_DESCRIPTION = "Fold the T-shirt properly"

DEFAULT_NUM_EPISODES = 1
DEFAULT_FPS = 30
DEFAULT_EPISODE_TIME_SEC = 1000
DEFAULT_RESET_TIME_SEC = 60

DEFAULT_FOLLOWER_LEFT_PORT = "can0"
DEFAULT_FOLLOWER_RIGHT_PORT = "can1"

DEFAULT_CAMERA_CONFIG = {
    "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video4", width=1280, height=720, fps=DEFAULT_FPS),
    "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video0", width=1280, height=720, fps=DEFAULT_FPS),
    "base": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=DEFAULT_FPS),
}


# ============================================================================
# Thread-Safe Robot Wrapper
# ============================================================================


class RobotWrapper:
    """Thread-safe wrapper for robot operations."""

    def __init__(self, robot: BiOpenArmFollower):
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
        execution_horizon=20,
        max_guidance_weight=5.0,
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
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

    interpolation: bool = True
    interpolation_multiplier: int = 3

    use_torch_compile: bool = False
    compile_warmup_inferences: int = 2
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


def _reanchor_delta_rtc_prefix(
    prev_actions_absolute: Tensor,
    current_state: Tensor,
    delta_step: DeltaActionsProcessorStep | None,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
) -> Tensor:
    """Convert absolute leftovers into model space for delta-action RTC policies."""
    if delta_step is None:
        return prev_actions_absolute.to(policy_device)

    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)

    action_cpu = prev_actions_absolute.detach().cpu()
    mask = delta_step._build_mask(action_cpu.shape[-1])
    delta_actions = to_delta_actions(action_cpu, state, mask)

    transition = create_transition(action=delta_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)


def _normalize_prev_actions_length(prev_actions: Tensor, target_steps: int) -> Tensor:
    """Pad/truncate RTC prefix actions to a fixed length for stable compiled inference."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected prev_actions to be 2D [T, A], got shape={tuple(prev_actions.shape)}")

    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]

    padded = torch.zeros(
        (target_steps, action_dim),
        dtype=prev_actions.dtype,
        device=prev_actions.device,
    )
    padded[:steps] = prev_actions
    return padded


def _resolve_state_joint_order(
    policy_action_names: list[str] | None,
    available_joint_names: list[str],
    log_prefix: str,
) -> list[str]:
    """Resolve joint-state ordering used to build observation.state."""
    if not policy_action_names:
        return available_joint_names

    policy_action_names = list(policy_action_names)
    available_set = set(available_joint_names)
    policy_set = set(policy_action_names)

    if len(policy_action_names) != len(available_joint_names) or policy_set != available_set:
        logger.warning(
            "%s policy.action_feature_names does not match available state joints; "
            "falling back to robot observation order",
            log_prefix,
        )
        return available_joint_names

    logger.info("%s Using policy.action_feature_names order for observation.state mapping", log_prefix)
    return policy_action_names


def _resolve_action_key_order(cfg: OpenArmsRTCEvalConfig, robot_action_keys: list[str]) -> list[str]:
    """Choose action name ordering used to map policy tensor outputs to robot action dict."""
    policy_action_names = getattr(cfg.policy, "action_feature_names", None)
    if not policy_action_names:
        return robot_action_keys

    policy_action_names = list(policy_action_names)
    if len(policy_action_names) != len(robot_action_keys):
        logger.warning(
            "[ACTOR] policy.action_feature_names length (%d) != robot action dim (%d); "
            "falling back to robot action order",
            len(policy_action_names),
            len(robot_action_keys),
        )
        return robot_action_keys

    robot_key_set = set(robot_action_keys)
    policy_key_set = set(policy_action_names)
    if robot_key_set != policy_key_set:
        logger.warning(
            "[ACTOR] policy.action_feature_names keys do not match robot action keys; "
            "falling back to robot action order"
        )
        return robot_action_keys

    return policy_action_names


def get_actions_thread(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: OpenArmsRTCEvalConfig,
    episode_active: Event,
    compile_warmup_done: Event | None = None,
):
    """Thread function to asynchronously generate action chunks from the policy."""
    try:
        logger.info("[GET_ACTIONS] Starting action generation thread")

        latency_tracker = LatencyTracker()
        time_per_chunk = 1.0 / cfg.fps

        # BiOpenArmFollower exposes pos/vel/torque for each joint (48D state).
        # PI0/PI05 checkpoints here expect only joint positions (16D state), so
        # keep only `.pos` joints plus camera streams for policy preprocessing.
        all_observation_features = robot.observation_features
        available_joint_names = [
            key for key, value in all_observation_features.items() if key.endswith(".pos") and value is float
        ]
        ordered_joint_names = _resolve_state_joint_order(
            getattr(cfg.policy, "action_feature_names", None),
            available_joint_names,
            "[GET_ACTIONS]",
        )
        observation_features_hw = {joint_name: all_observation_features[joint_name] for joint_name in ordered_joint_names}
        for key, value in all_observation_features.items():
            if isinstance(value, tuple):
                observation_features_hw[key] = value
        hw_features = hw_to_dataset_features(observation_features_hw, "observation")
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

        delta_step = next(
            (
                step
                for step in preprocessor.steps
                if isinstance(step, DeltaActionsProcessorStep) and step.enabled
            ),
            None,
        )
        normalizer_step = next(
            (step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
            None,
        )
        if delta_step is not None:
            if delta_step.action_names is None:
                cfg_action_names = getattr(cfg.policy, "action_feature_names", None)
                if cfg_action_names:
                    delta_step.action_names = list(cfg_action_names)
                else:
                    # Fallback to runtime robot joint ordering to preserve excluded joints
                    # (e.g. gripper) for checkpoints that do not store action names.
                    delta_step.action_names = [
                        key for key in robot.action_features.keys() if key.endswith(".pos")
                    ]
            logger.info("[GET_ACTIONS] Delta actions enabled: re-anchoring RTC prefix to current state")

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions
        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        inference_count = 0
        warmup_required = max(1, int(cfg.compile_warmup_inferences)) if cfg.use_torch_compile else 0

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

                if prev_actions is not None and delta_step is not None and OBS_STATE in obs_with_policy_features:
                    prev_actions_absolute = action_queue.get_processed_left_over()
                    if prev_actions_absolute is not None and prev_actions_absolute.numel() > 0:
                        prev_actions = _reanchor_delta_rtc_prefix(
                            prev_actions_absolute=prev_actions_absolute,
                            current_state=obs_with_policy_features[OBS_STATE],
                            delta_step=delta_step,
                            normalizer_step=normalizer_step,
                            policy_device=policy_device,
                        )

                if prev_actions is not None:
                    prev_actions = _normalize_prev_actions_length(
                        prev_actions, target_steps=cfg.rtc.execution_horizon
                    )

                actions = policy.predict_action_chunk(
                    preprocessed_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                original_actions = actions.squeeze(0).clone()
                postprocessed_actions = postprocessor(actions).squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)

                inference_count += 1
                is_warmup_inference = cfg.use_torch_compile and inference_count <= warmup_required
                if is_warmup_inference:
                    # Ignore compile warmup latency for RTC delay estimation.
                    latency_tracker.reset()
                else:
                    latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] action_queue_size_to_get_new_actions too small. "
                        "Should be higher than inference delay + execution horizon."
                    )

                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )

                logger.info(
                    "[GET_ACTIONS] inference=%.1f ms | delay=%d | queue_size=%d",
                    new_latency * 1000.0,
                    new_delay,
                    action_queue.qsize(),
                )
                if is_warmup_inference:
                    logger.info(
                        "[GET_ACTIONS] compile warmup inference %d/%d complete",
                        inference_count,
                        warmup_required,
                    )
                    if compile_warmup_done is not None and inference_count >= warmup_required:
                        compile_warmup_done.set()
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
        robot_action_keys = [k for k in robot.action_features.keys() if k.endswith(".pos")]
        action_keys = _resolve_action_key_order(cfg, robot_action_keys)
        if action_keys != robot_action_keys:
            logger.info("[ACTOR] Using policy.action_feature_names order for action tensor mapping")
        else:
            logger.info("[ACTOR] Using robot action feature order for action tensor mapping")

        if cfg.interpolation:
            interp_factor = max(1, int(cfg.interpolation_multiplier))
            logger.info(f"[ACTOR] Interpolation ON: policy={cfg.fps}Hz -> robot={cfg.fps * interp_factor}Hz")
        else:
            interp_factor = 1
            logger.info(f"[ACTOR] Interpolation OFF: policy={cfg.fps}Hz, robot={cfg.fps}Hz")

        interpolator = ActionInterpolator(multiplier=interp_factor)
        robot_interval = interpolator.get_control_interval(cfg.fps)

        robot_send_count = 0
        policy_consume_count = 0
        last_hz_print = time.perf_counter()
        last_dataset_time = 0.0

        while not shutdown_event.is_set():
            if not episode_active.is_set():
                interpolator.reset()
                robot_send_count = 0
                policy_consume_count = 0
                last_hz_print = time.perf_counter()
                time.sleep(0.01)
                continue

            start_time = time.perf_counter()

            if interpolator.needs_new_action():
                new_action = action_queue.get()
                if new_action is not None:
                    policy_consume_count += 1
                    interpolator.add(new_action.cpu())

            action_to_send = interpolator.get()
            if action_to_send is not None:

                action_dict = {}
                for i, key in enumerate(action_keys):
                    if i < len(action_to_send):
                        action_dict[key] = action_to_send[i].item()

                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)
                robot_send_count += 1

                if cfg.record_dataset and dataset is not None:
                    now = time.perf_counter()
                    if now - last_dataset_time >= (1.0 / cfg.fps):
                        last_dataset_time = now
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

            now = time.perf_counter()
            if now - last_hz_print >= 5.0:
                elapsed = now - last_hz_print
                actual_robot_hz = robot_send_count / elapsed if elapsed > 0 else 0
                actual_policy_hz = policy_consume_count / elapsed if elapsed > 0 else 0
                logger.info(f"[ACTOR] Actual Hz - Robot: {actual_robot_hz:.1f}, Policy: {actual_policy_hz:.1f}")
                robot_send_count = 0
                policy_consume_count = 0
                last_hz_print = now

            dt_s = time.perf_counter() - start_time
            sleep_time = max(0, robot_interval - dt_s)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("[ACTOR] Shutting down")
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
    print(f"Policy Hz: {cfg.fps}")
    print(f"Robot Hz: {cfg.fps * cfg.interpolation_multiplier if cfg.interpolation else cfg.fps}")
    print(f"Interpolation: {cfg.interpolation} (x{cfg.interpolation_multiplier})")
    print(f"Device: {cfg.device}")
    print("=" * 60)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event
    episode_active = Event()
    compile_warmup_done = Event()
    if not cfg.use_torch_compile:
        compile_warmup_done.set()

    # Initialize Robot
    follower_config = BiOpenArmFollowerConfig(
        id="bi_openarm_follower",
        left_arm_config=OpenArmFollowerConfigBase(
            port=cfg.follower_left_port,
            side="left",
            can_interface="socketcan",
            disable_torque_on_disconnect=True,
            max_relative_target=10.0,
        ),
        right_arm_config=OpenArmFollowerConfigBase(
            port=cfg.follower_right_port,
            side="right",
            can_interface="socketcan",
            disable_torque_on_disconnect=True,
            max_relative_target=10.0,
        ),
        cameras=DEFAULT_CAMERA_CONFIG,
    )

    follower = BiOpenArmFollower(follower_config)
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
            compile_warmup_done,
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
    actor_started = False
    if not cfg.use_torch_compile:
        actor_t.start()
        actor_started = True
        logger.info("Started actor thread")

    # Run Evaluation Episodes
    episode_idx = 0

    try:
        while episode_idx < cfg.num_episodes and not shutdown_event.is_set():
            log_say(f"Evaluating episode {episode_idx + 1} of {cfg.num_episodes}")
            logger.info(f"\n{'='*40}")
            logger.info(f"Episode {episode_idx + 1} / {cfg.num_episodes}")
            logger.info(f"{'='*40}")

            action_queue.clear()
            episode_active.set()

            if not actor_started:
                if cfg.use_torch_compile:
                    logger.info(
                        "[MAIN] Waiting for compile warmup (%d inferences) before starting actor thread...",
                        max(1, int(cfg.compile_warmup_inferences)),
                    )
                    while not compile_warmup_done.is_set() and not shutdown_event.is_set():
                        time.sleep(0.01)

                    if shutdown_event.is_set():
                        break

                logger.info("[MAIN] Waiting for first action chunk before starting actor thread...")
                while action_queue.qsize() == 0 and not shutdown_event.is_set():
                    time.sleep(0.01)

                if shutdown_event.is_set():
                    break

                actor_t.start()
                actor_started = True
                logger.info("Started actor thread")

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

        if actor_started and actor_t.is_alive():
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
