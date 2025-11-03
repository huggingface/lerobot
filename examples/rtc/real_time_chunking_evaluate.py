#!/usr/bin/env python

"""
Demo script showing how to use Real-Time Chunking (RTC) with action chunking policies.

This script demonstrates:
1. Creating a robot/environment and policy (SmolVLA, Pi0, etc.) with RTC
2. Consuming actions from the policy while the robot/environment executes
3. Periodically requesting new action chunks in the background using threads
4. Managing action buffers and timing for real-time operation

Usage:
    # With real robot
    python rtc_demo.py --policy.path=lerobot/smolvla_base --robot.type=so100

    # With simulation environment
    python rtc_demo.py --policy.path=lerobot/smolvla_base --env.type=pusht

    # With config file
    python rtc_demo.py --config_path=path/to/config.json

    # With policy compilation for faster inference (recommended for production)
    python rtc_demo.py --policy.path=lerobot/smolvla_base --robot.type=so100 --compile_policy=true

    # With aggressive compilation for maximum speed
    python rtc_demo.py --policy.path=lerobot/smolvla_base --robot.type=so100 --compile_policy=true --compile_mode=max-autotune

Performance Notes:
    - torch.compile() is NOT supported on MPS (Apple Silicon) due to attention operation limitations
    - For MPS optimization, reduce num_steps in the policy config (biggest speedup)
    - CUDA devices will see 2-5x speedup with compilation enabled
"""

import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

import numpy as np
import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.envs.configs import EnvConfig  # noqa: F401
from lerobot.envs.factory import make_env
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    so100_follower,
    so101_follower,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tensor_stats_str(tensor: Tensor | None, name: str = "tensor") -> str:
    """Generate readable statistics string for a tensor."""
    if tensor is None:
        return f"{name}: None"

    stats = (
        f"{name}:\n"
        f"  shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}\n"
        f"  min={tensor.min().item():.6f}, max={tensor.max().item():.6f}\n"
        f"  mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}"
    )
    return stats


def compare_tensors(tensor1: Tensor, tensor2: Tensor, name1: str = "tensor1", name2: str = "tensor2") -> str:
    """Compare two tensors and return detailed difference statistics."""
    if tensor1 is None or tensor2 is None:
        return f"Cannot compare: {name1}={tensor1 is not None}, {name2}={tensor2 is not None}"

    # Ensure same shape for comparison
    if tensor1.shape != tensor2.shape:
        return f"Shape mismatch: {name1}={tuple(tensor1.shape)} vs {name2}={tuple(tensor2.shape)}"

    diff = tensor1 - tensor2
    abs_diff = torch.abs(diff)

    # Per-timestep statistics
    if len(diff.shape) >= 2:
        # Shape is (batch, time, action_dim) or (time, action_dim)
        per_timestep_mean = abs_diff.mean(dim=-1)  # Average across action dimensions

        timestep_stats = "\n  Per-timestep abs diff (averaged across action dims):\n"
        if len(per_timestep_mean.shape) > 1:
            # Has batch dimension
            for batch_idx in range(per_timestep_mean.shape[0]):
                timestep_stats += f"    Batch {batch_idx}: ["
                for t in range(min(10, per_timestep_mean.shape[1])):  # Show first 10 timesteps
                    timestep_stats += f"{per_timestep_mean[batch_idx, t].item():.6f}, "
                if per_timestep_mean.shape[1] > 10:
                    timestep_stats += "..."
                timestep_stats += "]\n"
        else:
            timestep_stats += "    ["
            for t in range(min(10, len(per_timestep_mean))):
                timestep_stats += f"{per_timestep_mean[t].item():.6f}, "
            if len(per_timestep_mean) > 10:
                timestep_stats += "..."
            timestep_stats += "]\n"
    else:
        timestep_stats = ""

    result = (
        f"\nDifference: {name1} - {name2}:\n"
        f"  abs_diff: min={abs_diff.min().item():.6f}, max={abs_diff.max().item():.6f}\n"
        f"  abs_diff: mean={abs_diff.mean().item():.6f}, std={abs_diff.std().item():.6f}\n"
        f"  relative_diff: mean={abs_diff.mean().item() / (torch.abs(tensor2).mean().item() + 1e-8) * 100:.2f}%"
        f"{timestep_stats}"
    )

    return result


class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Tensor):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


class EnvWrapper:
    """Wrapper for gym environments to provide same interface as RobotWrapper."""

    def __init__(self, env, env_cfg: EnvConfig):
        self.env = env
        self.env_cfg = env_cfg
        self.lock = Lock()
        self._last_obs = None
        self._episode_count = 0
        self._step_count = 0

        # Initialize environment
        obs, _ = self.env.reset()
        self._last_obs = (
            obs[0]
            if isinstance(obs, tuple)
            or (hasattr(obs, "__getitem__") and len(obs) > 0 and not isinstance(obs, dict))
            else obs
        )

        # Cache feature names
        self._observation_features = None
        self._action_features = None

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation from environment.

        Returns observations in the same format as robot.get_observation():
        a dict mapping feature names to numpy arrays.
        """
        with self.lock:
            if self._last_obs is None:
                # Reset environment on first observation
                obs, _ = self.env.reset()
                self._last_obs = (
                    obs[0]
                    if isinstance(obs, tuple)
                    or (hasattr(obs, "__getitem__") and len(obs) > 0 and not isinstance(obs, dict))
                    else obs
                )

            # VectorEnv returns observations as numpy arrays in a batch
            # Extract first element if it's a vectorized observation
            obs = self._last_obs
            if isinstance(obs, dict):
                # Handle dict observations (extract first element from batch if needed)
                result = {}
                for key, value in obs.items():
                    if isinstance(value, np.ndarray) and len(value.shape) > 0 and value.shape[0] == 1:
                        # Remove batch dimension for single env
                        result[key] = value[0]
                    else:
                        result[key] = value
                return result
            else:
                # Handle array observations - shouldn't happen with our configs but handle it
                return {"observation": obs[0] if len(obs.shape) > 1 else obs}

    def send_action(self, action: dict):
        """Execute action in environment and update observation."""
        with self.lock:
            # Convert action dict to array based on action_features
            action_list = []
            for feature_name in self.action_features():
                if feature_name in action:
                    action_list.append(action[feature_name])

            action_array = np.array(action_list)

            # VectorEnv expects actions with batch dimension
            action_batch = action_array.reshape(1, -1)

            # Step environment
            obs, _reward, terminated, truncated, _info = self.env.step(action_batch)

            # Extract from batch
            self._last_obs = (
                obs[0]
                if isinstance(obs, tuple)
                or (hasattr(obs, "__getitem__") and len(obs) > 0 and not isinstance(obs, dict))
                else obs
            )
            self._step_count += 1

            # Check if episode is done (handle vectorized env format)
            is_done = terminated[0] if isinstance(terminated, (np.ndarray, list)) else terminated
            is_truncated = truncated[0] if isinstance(truncated, (np.ndarray, list)) else truncated

            # Reset if episode is done
            if is_done or is_truncated:
                logger.info(f"Episode {self._episode_count} finished after {self._step_count} steps")
                obs, _ = self.env.reset()
                self._last_obs = (
                    obs[0]
                    if isinstance(obs, tuple)
                    or (hasattr(obs, "__getitem__") and len(obs) > 0 and not isinstance(obs, dict))
                    else obs
                )
                self._episode_count += 1
                self._step_count = 0

    def observation_features(self) -> list[str]:
        """Get observation feature names from environment config."""
        if self._observation_features is not None:
            return self._observation_features

        with self.lock:
            features = []
            for feature_name in self.env_cfg.features:
                if feature_name != "action":
                    # Use the mapped name from features_map
                    mapped_name = self.env_cfg.features_map.get(feature_name, feature_name)
                    features.append(mapped_name)

            self._observation_features = features
            return features

    def action_features(self) -> list[str]:
        """Get action feature names from environment config."""
        if self._action_features is not None:
            return self._action_features

        with self.lock:
            # Return action dimension names
            action_dim = self.env_cfg.features["action"].shape[0]
            self._action_features = [f"action_{i}" for i in range(action_dim)]
            return self._action_features


class ActionQueue:
    def __init__(self, cfg: RTCConfig):
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def qsize(self) -> int:
        # with self.lock:
        if self.queue is None:
            return 0
        length = len(self.queue)

        return length - self.last_index

    def empty(self) -> bool:
        # with self.lock:
        if self.queue is None:
            return True

        length = len(self.queue)
        return length - self.last_index + 1 <= 0

    def get_action_index(self) -> int:
        # with self.lock:
        return self.last_index

    def get_left_over(self) -> Tensor:
        """Get left over ORIGINAL actions for RTC prev_chunk_left_over."""
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :]

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = 0,
    ):
        with self.lock:
            self._check_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, real_delay)
                return

            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, real_delay: int):
        self.original_queue = original_actions[real_delay:].clone()
        self.queue = processed_actions[real_delay:].clone()

        logger.info(f"original_actions shape: {self.original_queue.shape}")
        logger.info(f"processed_actions shape: {self.queue.shape}")
        logger.info(f"real_delay: {real_delay}")

        self.last_index = 0

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor):
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_delays(self, real_delay: int, action_index_before_inference: int | None = None):
        if action_index_before_inference is None:
            return

        indexes_diff = self.last_index - action_index_before_inference
        if indexes_diff != real_delay:
            # Let's check that action index difference (real delay calculated based on action queue)
            # is the same as dealy calculated based on inference latency
            logger.warning(
                f"[ACTION_QUEUE] Indexes diff is not equal to real delay. Indexes diff: {indexes_diff}, real delay: {real_delay}"
            )


@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Robot configuration (mutually exclusive with env)
    robot: RobotConfig | None = None

    # Environment configuration (mutually exclusive with robot)
    env: EnvConfig | None = None

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Demo parameters
    duration: float = 30.0  # Duration to run the demo (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)

    # Compute device
    device: str | None = None  # Device to run on (cuda, cpu, auto)

    # Compilation options
    compile_policy: bool = (
        False  # Compile policy with torch.compile() for faster inference (not supported on MPS)
    )
    compile_mode: str = "default"  # Compilation mode: default, reduce-overhead, max-autotune

    # Alternative optimization options (work on all devices including MPS)
    use_channels_last: bool = False  # Use channels_last memory format for images (faster on some devices)
    enable_cudnn_benchmark: bool = True  # Enable cuDNN benchmarking (CUDA only)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 30

    # Task to execute
    task: str = field(default="", metadata={"help": "Task to execute"})

    # Debug options
    verbose_rtc_comparison: bool = True  # Enable detailed RTC comparison output

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required")

        # Validate that either robot or env is provided, but not both
        if self.robot is None and self.env is None:
            raise ValueError("Either robot or env configuration must be provided")
        if self.robot is not None and self.env is not None:
            raise ValueError("Cannot specify both robot and env configuration. Choose one.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def get_actions(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to request action chunks from the policy.

    Args:
        policy: The policy instance (SmolVLA, Pi0, etc.)
        robot: The robot instance for getting observations
        robot_observation_processor: Processor for raw robot observations
        action_queue: Queue to put new action chunks
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[GET_ACTIONS] Starting get actions thread")

        latency_tracker = LatencyTracker()  # Track latency of action chunks
        fps = cfg.fps
        time_per_chunk = 1.0 / fps

        dataset_features = hw_to_dataset_features(robot.observation_features(), "observation")
        policy_device = policy.config.device

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
            },
        )

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                obs = robot.get_observation()

                # Apply robot observation processor
                obs_processed = robot_observation_processor(obs)

                obs_with_policy_features = build_dataset_frame(
                    dataset_features, obs_processed, prefix="observation"
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

                # for k, v in obs_with_policy_features.items():
                #     if isinstance(v, np.ndarray):
                #         obs_with_policy_features[k] = torch.from_numpy(v).to(policy_device)

                #     if is_image_key(k):
                #         obs_with_policy_features[k] = obs_with_policy_features[k].type(torch.float32) / 255
                #         obs_with_policy_features[k] = obs_with_policy_features[k].permute(2, 0, 1).unsqueeze(0)
                #     elif isinstance(obs_with_policy_features[k], torch.Tensor):
                #         obs_with_policy_features[k] = obs_with_policy_features[k].unsqueeze(0)

                obs_with_policy_features["task"] = cfg.task

                preproceseded_obs = preprocessor(obs_with_policy_features)

                noise_size = (1, policy.config.chunk_size, policy.config.max_action_dim)
                noise = policy.model.sample_noise(noise_size, policy_device)
                noise_clone = noise.clone()

                # Generate actions WITHOUT RTC for comparison (if verbose mode enabled)
                if cfg.verbose_rtc_comparison:
                    policy.config.rtc_config.enabled = False
                    not_rtc_actions = policy.predict_action_chunk(
                        preproceseded_obs,
                        noise=noise,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_actions,
                    )
                    policy.config.rtc_config.enabled = True

                # Generate actions WITH RTC
                actions = policy.predict_action_chunk(
                    preproceseded_obs,
                    noise=noise_clone if cfg.verbose_rtc_comparison else noise,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                # Store original actions (before postprocessing) for RTC
                original_actions = actions.squeeze(0).clone()

                # Detailed comparison output (if verbose mode enabled)
                if cfg.verbose_rtc_comparison:
                    logger.info("=" * 80)
                    logger.info("RTC ACTION COMPARISON")
                    logger.info("=" * 80)

                    # Print detailed statistics
                    logger.info("\n" + tensor_stats_str(not_rtc_actions, "not_rtc_actions (without RTC)"))
                    logger.info("\n" + tensor_stats_str(actions, "actions (with RTC)"))
                    logger.info(
                        "\n" + tensor_stats_str(prev_actions, "prev_actions (leftover from previous chunk)")
                    )

                    # Compare RTC vs non-RTC actions
                    logger.info(
                        compare_tensors(actions, not_rtc_actions, "actions (RTC)", "not_rtc_actions (no RTC)")
                    )

                    to_non_rtc_diff = actions - not_rtc_actions

                    print("to_non_rtc_diff", to_non_rtc_diff)
                    if prev_actions is not None:
                        prev_padded = torch.zeros_like(actions)
                        prev_padded[:, : prev_actions.shape[1], :] = prev_actions
                        to_prev_diff = actions - prev_padded
                        print("to_prev_diff", to_prev_diff)
                    print("=" * 80)

                postprocessed_actions = postprocessor(actions)

                postprocessed_actions = postprocessed_actions.squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                    )

                logger.debug(f"[GET_ACTIONS] new_delay: {new_delay}")
                logger.debug(f"[GET_ACTIONS] original_actions shape: {original_actions.shape}")
                logger.debug(f"[GET_ACTIONS] postprocessed_actions shape: {postprocessed_actions.shape}")
                logger.debug(f"[GET_ACTIONS] action_index_before_inference: {action_index_before_inference}")

                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )
            else:
                # Small sleep to prevent busy waiting
                time.sleep(0.1)

        logger.info("[GET_ACTIONS] get actions thread shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception in get_actions thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def actor_control(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to execute actions on the robot.

    Args:
        robot: The robot instance
        action_queue: Queue to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / cfg.fps

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            # Try to get an action from the queue with timeout
            action = action_queue.get()

            if action is not None:
                action = action.cpu()
                action = {key: action[i].item() for i, key in enumerate(robot.action_features())}
                action = robot_action_processor((action, None))
                robot.send_action(action)

                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep((action_interval - dt_s) - 0.001)

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception in actor_control thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def stop_by_duration(shutdown_event: Event, cfg: RTCDemoConfig):
    """Stop the demo by duration."""
    time.sleep(cfg.duration)
    shutdown_event.set()


@parser.wrap()
def demo_cli(cfg: RTCDemoConfig):
    """Main entry point for RTC demo with draccus configuration."""

    # Initialize logging
    init_logging()

    logger.info(f"Using device: {cfg.device}")

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    policy = None
    robot = None
    vec_env = None
    get_actions_thread = None
    actor_thread = None

    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path)

    # Turn on RTC
    policy.config.rtc_config = cfg.rtc

    # Init RTC processort, as by default if RTC disabled in the config
    # The processor won't be created
    policy.init_rtc_processor(verbose=cfg.verbose_rtc_comparison)

    assert policy.name in ["smolvla"], "Only smolvla are supported for RTC"

    policy = policy.to(cfg.device)
    policy.eval()

    # Apply memory format optimizations
    if cfg.use_channels_last:
        logger.info("Converting model to channels_last memory format")
        try:
            # Convert vision encoder to channels_last for better performance
            if hasattr(policy, "vision_encoder"):
                policy.vision_encoder = policy.vision_encoder.to(memory_format=torch.channels_last)
            logger.info("Successfully converted to channels_last format")
        except Exception as e:
            logger.warning(f"Failed to convert to channels_last: {e}")

    # Enable cuDNN benchmarking for CUDA
    if cfg.enable_cudnn_benchmark and cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmarking")

    # Compile policy if requested
    if cfg.compile_policy:
        # Check if device is MPS - torch.compile has issues with MPS backend
        if cfg.device == "mps":
            logger.warning("torch.compile() is not stable with MPS backend (Apple Silicon)")
            logger.warning("Skipping compilation. For better performance on MPS:")
            logger.warning("  1. Use torch.float32 instead of bfloat16")
            logger.warning("  2. Ensure model uses contiguous memory layouts")
            logger.warning("  3. Consider using CUDA if available")
        else:
            logger.info(f"Compiling policy with mode: {cfg.compile_mode}")
            logger.info("First inference will be slower due to compilation, subsequent calls will be faster")

            try:
                # Compile the predict_action_chunk method
                policy.predict_action_chunk = torch.compile(
                    policy.predict_action_chunk,
                    mode=cfg.compile_mode,
                    fullgraph=False,  # Allow graph breaks for flexibility
                    backend="inductor",  # Use inductor backend
                )
                logger.info("Policy compiled successfully")
            except Exception as e:
                logger.warning(f"Failed to compile policy: {e}")
                logger.warning("Continuing without compilation")

    # Create robot or environment
    if cfg.robot is not None:
        logger.info(f"Initializing robot: {cfg.robot.type}")
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        agent_wrapper = RobotWrapper(robot)
    else:
        logger.info(f"Initializing environment: {cfg.env.type}")
        # Create environment using make_env
        env_dict = make_env(cfg.env, n_envs=1, use_async_envs=False)

        # Validate environment structure: should have exactly one suite
        if len(env_dict) != 1:
            raise ValueError(
                f"Expected exactly one environment suite, but got {len(env_dict)}. "
                f"Suites: {list(env_dict.keys())}"
            )

        # Extract the actual env from the dict structure {suite: {task_id: vec_env}}
        suite_name = list(env_dict.keys())[0]
        task_dict = env_dict[suite_name]

        # Validate task structure: should have exactly one task
        if len(task_dict) != 1:
            raise ValueError(
                f"Expected exactly one task in suite '{suite_name}', but got {len(task_dict)}. "
                f"Tasks: {list(task_dict.keys())}"
            )

        vec_env = task_dict[0]
        logger.info(f"Created environment: suite='{suite_name}', task_id=0, num_envs={vec_env.num_envs}")

        # Validate that we have exactly 1 parallel environment
        if vec_env.num_envs != 1:
            raise ValueError(
                f"Expected exactly 1 parallel environment, but got {vec_env.num_envs}. "
                f"The EnvWrapper is designed for single environment instances."
            )

        agent_wrapper = EnvWrapper(vec_env, cfg.env)

    # Create robot observation processor
    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()

    # Create action queue for communication between threads
    action_queue = ActionQueue(cfg.rtc)

    # Start chunk requester thread
    get_actions_thread = Thread(
        target=get_actions,
        args=(policy, agent_wrapper, robot_observation_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()
    logger.info("Started get actions thread")

    # Start action executor thread
    actor_thread = Thread(
        target=actor_control,
        args=(agent_wrapper, robot_action_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="Actor",
    )
    actor_thread.start()
    logger.info("Started actor thread")

    logger.info("Started stop by duration thread")

    # Main thread monitors for duration or shutdown
    logger.info(f"Running demo for {cfg.duration} seconds...")
    start_time = time.time()

    while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
        time.sleep(10)

        # Log queue status periodically
        if int(time.time() - start_time) % 5 == 0:
            logger.info(f"[MAIN] Action queue size: {action_queue.qsize()}")

        if time.time() - start_time > cfg.duration:
            break

    logger.info("Demo duration reached or shutdown requested")

    # Signal shutdown
    shutdown_event.set()

    # Wait for threads to finish
    if get_actions_thread and get_actions_thread.is_alive():
        logger.info("Waiting for chunk requester thread to finish...")
        get_actions_thread.join()

    if actor_thread and actor_thread.is_alive():
        logger.info("Waiting for action executor thread to finish...")
        actor_thread.join()

    # Cleanup robot or environment
    if cfg.robot is not None:
        if robot:
            robot.disconnect()
            logger.info("Robot disconnected")
    else:
        # Close environment
        if vec_env:
            vec_env.close()
            logger.info("Environment closed")

    logger.info("Cleanup completed")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")
