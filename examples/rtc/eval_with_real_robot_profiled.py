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
Profiled version of eval_with_real_robot.py for performance analysis.

This version adds detailed timing measurements for:
- Policy inference
- Preprocessing
- Postprocessing
- Action queue operations
- Robot communication
- Thread execution times

Usage: Same as eval_with_real_robot.py but with profiling output.
"""

import logging
import math
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
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


class ProfileTimer:
    """Context manager and utility class for timing code sections."""

    def __init__(self, name: str, stats_dict: dict):
        self.name = name
        self.stats_dict = stats_dict
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        if self.name not in self.stats_dict:
            self.stats_dict[self.name] = []
        self.stats_dict[self.name].append(elapsed)


class ProfilingStats:
    """Global profiling statistics collector."""

    def __init__(self):
        self.stats = defaultdict(list)
        self.lock = Lock()

    def record(self, name: str, duration: float):
        with self.lock:
            self.stats[name].append(duration)

    def timer(self, name: str):
        """Return a context manager for timing."""
        return ProfileTimer(name, self.stats)

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all timings."""
        with self.lock:
            summary = {}
            for name, times in self.stats.items():
                if times:
                    summary[name] = {
                        "count": len(times),
                        "mean": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "total": sum(times),
                    }
            return summary

    def print_summary(self):
        """Print formatted summary of all timings."""
        summary = self.get_summary()
        
        logger.info("\n" + "=" * 80)
        logger.info("PROFILING SUMMARY")
        logger.info("=" * 80)
        
        # Sort by total time (descending)
        sorted_items = sorted(summary.items(), key=lambda x: x[1]["total"], reverse=True)
        
        for name, stats in sorted_items:
            logger.info(f"\n{name}:")
            logger.info(f"  Count:     {stats['count']}")
            logger.info(f"  Mean:      {stats['mean']*1000:.2f} ms")
            logger.info(f"  Min:       {stats['min']*1000:.2f} ms")
            logger.info(f"  Max:       {stats['max']*1000:.2f} ms")
            logger.info(f"  Total:     {stats['total']:.2f} s")
            logger.info(f"  Hz:        {stats['count']/stats['total']:.2f}")
        
        logger.info("\n" + "=" * 80)


# Global profiling stats
profiling_stats = ProfilingStats()


class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with profiling_stats.timer("robot.get_observation"):
            with self.lock:
                return self.robot.get_observation()

    def send_action(self, action: Tensor):
        with profiling_stats.timer("robot.send_action"):
            with self.lock:
                self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies and real robots."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Robot configuration
    robot: RobotConfig | None = None

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

        # Validate that robot configuration is provided
        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

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
    """Thread function to request action chunks from the policy with profiling.

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

        inference_count = 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                with profiling_stats.timer("get_actions.total_iteration"):
                    inference_count += 1
                    logger.info(f"[GET_ACTIONS] Starting inference #{inference_count}")

                    current_time = time.perf_counter()
                    action_index_before_inference = action_queue.get_action_index()
                    
                    with profiling_stats.timer("get_actions.get_prev_actions"):
                        prev_actions = action_queue.get_left_over()

                    inference_latency = latency_tracker.max()
                    inference_delay = math.ceil(inference_latency / time_per_chunk)

                    # Get observation
                    obs = robot.get_observation()

                    # Apply robot observation processor
                    with profiling_stats.timer("get_actions.robot_obs_processing"):
                        obs_processed = robot_observation_processor(obs)

                    # Build dataset frame
                    with profiling_stats.timer("get_actions.build_dataset_frame"):
                        obs_with_policy_features = build_dataset_frame(
                            dataset_features, obs_processed, prefix="observation"
                        )

                    # Convert to tensors and normalize
                    with profiling_stats.timer("get_actions.tensor_conversion"):
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
                        obs_with_policy_features["robot_type"] = (
                            robot.robot.name if hasattr(robot.robot, "name") else ""
                        )

                    # Preprocessing
                    with profiling_stats.timer("get_actions.preprocessing"):
                        preproceseded_obs = preprocessor(obs_with_policy_features)

                    # Policy inference
                    with profiling_stats.timer("get_actions.policy_inference"):
                        actions = policy.predict_action_chunk(
                            preproceseded_obs,
                            inference_delay=inference_delay,
                            prev_chunk_left_over=prev_actions,
                        )

                    # Clone for RTC
                    with profiling_stats.timer("get_actions.clone_actions"):
                        original_actions = actions.squeeze(0).clone()

                    # Postprocessing
                    with profiling_stats.timer("get_actions.postprocessing"):
                        postprocessed_actions = postprocessor(actions)
                        postprocessed_actions = postprocessed_actions.squeeze(0)

                    # Update latency tracker
                    new_latency = time.perf_counter() - current_time
                    new_delay = math.ceil(new_latency / time_per_chunk)
                    latency_tracker.add(new_latency)

                    logger.info(
                        f"[GET_ACTIONS] Inference #{inference_count} completed in {new_latency*1000:.2f}ms "
                        f"(delay={new_delay} chunks)"
                    )

                    if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                        logger.warning(
                            "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, "
                            "It should be higher than inference delay + execution horizon."
                        )

                    # Merge into action queue
                    with profiling_stats.timer("get_actions.action_queue_merge"):
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
    """Thread function to execute actions on the robot with profiling.

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

            with profiling_stats.timer("actor.total_iteration"):
                # Get action from queue
                with profiling_stats.timer("actor.queue_get"):
                    action = action_queue.get()

                if action is not None:
                    # Process action
                    with profiling_stats.timer("actor.action_processing"):
                        action = action.cpu()
                        action_dict = {key: action[i].item() for i, key in enumerate(robot.action_features())}
                        action_processed = robot_action_processor((action_dict, None))
                    
                    # Send to robot (includes robot.send_action timing)
                    robot.send_action(action_processed)
                    action_count += 1

            # Sleep to maintain target FPS
            dt_s = time.perf_counter() - start_time
            sleep_time = max(0, (action_interval - dt_s) - 0.001)
            if sleep_time > 0:
                time.sleep(sleep_time)

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
    """Main entry point for RTC demo with profiling."""

    # Initialize logging
    init_logging()

    logger.info(f"Using device: {cfg.device}")
    logger.info("=" * 80)
    logger.info("PROFILING MODE ENABLED")
    logger.info("=" * 80)

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    policy = None
    robot = None
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

    # Init RTC processor
    policy.init_rtc_processor()

    assert policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 are supported for RTC"

    policy = policy.to(cfg.device)
    policy.eval()

    # Apply torch.compile to predict_action_chunk method if enabled
    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    # Create robot
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_wrapper = RobotWrapper(robot)

    # Create robot observation processor
    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()

    # Create action queue for communication between threads
    action_queue = ActionQueue(cfg.rtc)

    # Start chunk requester thread
    get_actions_thread = Thread(
        target=get_actions,
        args=(policy, robot_wrapper, robot_observation_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()
    logger.info("Started get actions thread")

    # Start action executor thread
    actor_thread = Thread(
        target=actor_control,
        args=(robot_wrapper, robot_action_processor, action_queue, shutdown_event, cfg),
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

    # Cleanup robot
    if robot:
        robot.disconnect()
        logger.info("Robot disconnected")

    # Print profiling summary
    profiling_stats.print_summary()

    logger.info("Cleanup completed")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")

