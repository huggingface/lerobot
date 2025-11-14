#!/usr/bin/env python

"""
Demo script showing how to use Real-Time Chunking (RTC) with action chunking policies.

This script demonstrates:
1. Creating a robot/environment and policy (SmolVLA, Pi0, etc.) with RTC
2. Consuming actions from the policy while the robot/environment executes
3. Periodically requesting new action chunks in the background using threads
4. Managing action buffers and timing for real-time operation

Usage:
    # Run RTC with Real robot with RTC
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with Real robot without RTC
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=false \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120
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
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.envs.configs import EnvConfig  # noqa: F401
from lerobot.envs.factory import make_env
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
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

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 30

    # Task to execute
    task: str = field(default="", metadata={"help": "Task to execute"})

    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )

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

        # Load preprocessor and postprocessor from pretrained files
        # The stats are embedded in the processor .safetensors files
        logger.info(f"[GET_ACTIONS] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,  # Will load from pretrained processor files
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
            },
        )

        logger.info("[GET_ACTIONS] Preprocessor/postprocessor loaded successfully with embedded stats")

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

                obs_with_policy_features["task"] = [cfg.task]  # Task should be a list, not a string!
                obs_with_policy_features["robot_type"] = (
                    robot.robot.name if hasattr(robot.robot, "name") else ""
                )

                preproceseded_obs = preprocessor(obs_with_policy_features)

                # Generate actions WITH RTC
                actions = policy.predict_action_chunk(
                    preproceseded_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                # Store original actions (before postprocessing) for RTC
                original_actions = actions.squeeze(0).clone()

                postprocessed_actions = postprocessor(actions)

                postprocessed_actions = postprocessed_actions.squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                    )

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
                action_dict = {key: action[i].item() for i, key in enumerate(robot.action_features())}
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)

                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep((action_interval - dt_s) - 0.001)

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception in actor_control thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def _apply_torch_compile(policy, cfg: RTCDemoConfig):
    """Apply torch.compile to the policy's predict_action_chunk method.

    Args:
        policy: Policy instance to compile
        cfg: Configuration containing torch compile settings

    Returns:
        Policy with compiled predict_action_chunk method
    """

    # PI models handle their own compilation
    if policy.type == "pi05" or policy.type == "pi0":
        return policy

    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning(
                f"torch.compile is not available. Requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}. Skipping compilation."
            )
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")
        logger.info(f"  Backend: {cfg.torch_compile_backend}")
        logger.info(f"  Mode: {cfg.torch_compile_mode}")
        logger.info(f"  Disable CUDA graphs: {cfg.torch_compile_disable_cudagraphs}")

        # Compile the predict_action_chunk method
        # - CUDA graphs disabled to prevent tensor aliasing from in-place ops (x_t += dt * v_t)
        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }

        # Disable CUDA graphs if requested (prevents tensor aliasing issues)
        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("✓ Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")
        logger.warning("Continuing without torch.compile")

    return policy


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

    # Load config and set compile_model for pi0/pi05 models
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if cfg.policy.type == "pi05" or cfg.policy.type == "pi0":
        config.compile_model = cfg.use_torch_compile

    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    # Turn on RTC
    policy.config.rtc_config = cfg.rtc

    # Init RTC processort, as by default if RTC disabled in the config
    # The processor won't be created
    policy.init_rtc_processor()

    assert policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 are supported for RTC"

    policy = policy.to(cfg.device)
    policy.eval()

    # Apply torch.compile to predict_action_chunk method if enabled
    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

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
