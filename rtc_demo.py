#!/usr/bin/env python

"""
Demo script showing how to use Real-Time Chunking (RTC) with action chunking policies.

This script demonstrates:
1. Creating a robot and policy (SmolVLA, Pi0, etc.) with RTC
2. Consuming actions from the policy while the robot executes
3. Periodically requesting new action chunks in the background using threads
4. Managing action buffers and timing for real-time operation

Usage:
    # With SmolVLA
    python rtc_demo.py --policy.path=lerobot/smolvla_base --robot.type=so100

    # With Pi0 (note: Pi0 currently doesn't support predict_action_chunk)
    python rtc_demo.py --policy.type=pi0 --robot.type=so100

    # With config file
    python rtc_demo.py --config_path=path/to/config.json
"""

import logging
import time
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

from torch import Tensor

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.robots.robot import Robot
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.hub import HubMixin
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import busy_wait
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


class ActionQueue:
    def __init__(self):
        self.queue = None
        self.lock = Lock()
        self.last_index = 0

    def get(self, timeout: float = 0.1) -> Tensor | None:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def qsize(self) -> int:
        with self.lock:
            return self.queue.qsize()

    def empty(self) -> bool:
        with self.lock:
            return self.queue.empty()

    def get_action_index(self) -> int:
        with self.lock:
            return self.last_index

    def get_left_over(self) -> Tensor:
        with self.lock:
            return self.queue[: self.last_index]

    def merge(
        self, predicted_actions: Tensor, real_delay: float, action_index_before_inference: int | None = 0
    ):
        with self.lock:
            current_index = self.last_index

            if action_index_before_inference is not None:
                indexes_diff = action_index_before_inference - current_index
                if indexes_diff != real_delay:
                    # Let's check that action index difference (real delay calculated based on action queue)
                    # is the same as dealy calculated based on inference latency
                    logger.warning(
                        f"[ACTION_QUEUE] Indexes diff is not equal to real delay. Indexes diff: {indexes_diff}, real delay: {real_delay}"
                    )

            self.queue = predicted_actions.clone()

            # First real_delay actions are already executed
            self.queue = self.queue[real_delay:]
            self.last_index = 0


@dataclass
class RobotConfig:
    """Configuration for robot setup."""

    type: str = "so100"  # Robot type (so100, so101, koch, etc.)
    # Add any robot-specific config here if needed


@dataclass
class RTC:
    """Configuration for policy setup."""

    type: str = "smolvla"  # Policy type (smolvla, pi0, etc.)
    # Add any policy-specific config here if needed


@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Robot configuration
    robot: RobotConfig = field(default_factory=RobotConfig)

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        )
    )

    # Demo parameters
    duration: float = 30.0  # Duration to run the demo (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)

    # Compute device
    device: str | None = None  # Device to run on (cuda, cpu, auto)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 20


def get_actions(
    policy,
    robot: RobotWrapper,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to request action chunks from the policy.

    Args:
        policy: The policy instance (SmolVLA, Pi0, etc.)
        robot: The robot instance for getting observations
        rtc_processor: RTC processor for guidance
        action_queue: Queue to put new action chunks
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    logger.info("[GET_ACTIONS] Starting get actions thread")

    latency_tracker = LatencyTracker()  # Track latency of action chunks
    fps = cfg.fps

    while not shutdown_event.is_set():
        if action_queue.qsize() < cfg.action_queue_size_to_get_new_actions:
            current_time = time.perf_counter()
            action_index_before_inference = action_queue.get_action_index()
            prev_actions = action_queue.get_left_over()

            time_per_chunk = 1.0 / fps
            inference_latency = latency_tracker.max()
            inference_delay = (inference_latency / time_per_chunk).ceil()

            actions = policy.predict_action_chunk(
                robot.get_observation(),
                {"inference_delay": inference_delay, "prev_chunk_left_over": prev_actions},
            )

            new_latency = time.perf_counter() - current_time
            new_delay = new_latency / time_per_chunk
            latency_tracker.add(new_latency)

            if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                logger.warning(
                    "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                )

            action_queue.merge(actions, new_delay, action_index_before_inference)
        else:
            # Small sleep to prevent busy waiting
            time.sleep(0.01)

    logger.info("[GET_ACTIONS] get actions thread shutting down")


def actor_thread(
    robot: RobotWrapper,
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
    logger.info("[ACTOR] Starting actor thread")

    action_interval = 1.0 / cfg.fps
    action_count = 0

    while not shutdown_event.is_set():
        start_time = time.time()

        # Try to get an action from the queue with timeout
        action = action_queue.get()

        # Execute action on robot
        action_np = action.cpu().numpy()
        robot.send_action(action_np)

        action_count += 1
        logger.info(f"[ACTOR] Executed action {action_count}")

        # Wait for the next action time
        elapsed = time.time() - start_time
        if elapsed < action_interval:
            busy_wait(action_interval - elapsed)

    logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")


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

    cfg.validate()

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    policy = None
    robot = None
    get_actions_thread = None
    actor_thread = None

    logger.info(f"Loading policy from {cfg.policy.path}")
    # Use the factory to load the policy based on its config
    from lerobot.policies.pretrained import PreTrainedPolicy

    policy = PreTrainedPolicy.from_pretrained(cfg.policy.path)

    assert policy.name in ["smolvla", "pi0"], "Only smolvla and pi0 are supported for RTC"

    # Turn on RTC
    policy.rtc_config = cfg.rtc
    policy.rtc.enabled = True

    policy = policy.to(cfg.device)
    policy.eval()

    # Create robot
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # Create action queue for communication between threads
    action_queue = ActionQueue()

    # Start chunk requester thread
    get_actions_thread = Thread(
        target=get_actions,
        args=(policy, robot, action_queue, shutdown_event, cfg),
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()
    logger.info("Started get actions thread")

    # Start action executor thread
    actor_thread = Thread(
        target=actor_thread, args=(robot, action_queue, shutdown_event, cfg), daemon=True, name="Actor"
    )
    actor_thread.start()
    logger.info("Started actor thread")

    stop_thread = Thread(
        target=stop_by_duration, args=(shutdown_event, cfg), daemon=True, name="StopByDuration"
    )
    stop_thread.start()
    logger.info("Started stop by duration thread")

    # Main thread monitors for duration or shutdown
    logger.info(f"Running demo for {cfg.duration} seconds...")
    start_time = time.time()

    while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
        time.sleep(0.1)

        # Log queue status periodically
        if int(time.time() - start_time) % 5 == 0:
            logger.info(f"Action queue size: {action_queue.qsize()}")

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

    if stop_thread and stop_thread.is_alive():
        logger.info("Waiting for stop by duration thread to finish...")
        stop_thread.join()

    # Cleanup robot
    if robot:
        robot.disconnect()
        logger.info("Robot disconnected")

    logger.info("Cleanup completed")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")
