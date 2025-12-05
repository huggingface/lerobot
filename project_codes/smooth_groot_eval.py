#!/usr/bin/env python
"""
Smooth inference script for Groot policy with pre-emptive action chunk refresh.

This script requests new action chunks BEFORE the queue empties to avoid
blocking the control loop and eliminate choppy motion.

Usage:
    python smooth_groot_eval.py \
        --policy.path=YieumYoon/groot-bimanual-so100-cbasket-diffusion-003 \
        --robot.type=bi_so100_follower \
        --robot.left_arm_port=/dev/ttyACM1 \
        --robot.right_arm_port=/dev/ttyACM0 \
        --task="Grab the red cube and put it in a red basket"
"""

import argparse
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.robots.factory import make_robot_from_config
from lerobot.robots.config import RobotConfig


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmoothGrootController:
    """
    Controller that runs Groot inference in a background thread to avoid
    blocking the main control loop.
    """

    def __init__(
        self,
        policy_path: str,
        device: str = "cuda",
        # Request new chunk when queue has this many actions left
        refresh_threshold: int = 10,
    ):
        self.device = device
        self.refresh_threshold = refresh_threshold

        # Load policy
        logger.info(f"Loading policy from {policy_path}")
        self.policy = GrootPolicy.from_pretrained(
            pretrained_name_or_path=policy_path,
            strict=False,
        )
        self.policy.to(device)
        self.policy.config.device = device
        self.policy.eval()

        # Create processors
        self.preprocessor, self.postprocessor = make_groot_pre_post_processors(
            config=self.policy.config,
            dataset_stats=None,
        )

        # Action queue (thread-safe)
        self.action_queue = deque()
        self.queue_lock = threading.Lock()

        # Inference thread state
        self.latest_observation = None
        self.obs_lock = threading.Lock()
        self.inference_requested = threading.Event()
        self.running = True

        # Start background inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True)
        self.inference_thread.start()

        logger.info(
            f"Policy loaded. Chunk size: {self.policy.config.chunk_size}")

    def _inference_loop(self):
        """Background thread that runs inference when requested."""
        while self.running:
            # Wait for inference request
            self.inference_requested.wait(timeout=0.1)

            if not self.running:
                break

            if not self.inference_requested.is_set():
                continue

            self.inference_requested.clear()

            # Get latest observation
            with self.obs_lock:
                if self.latest_observation is None:
                    continue
                obs = {k: v.clone() if isinstance(v, torch.Tensor) else v
                       for k, v in self.latest_observation.items()}

            # Run inference
            try:
                start_time = time.perf_counter()

                with torch.no_grad():
                    # Preprocess
                    processed_obs = self.preprocessor(obs)

                    # Get action chunk (bypassing internal queue)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    actions = self.policy.predict_action_chunk(processed_obs)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Postprocess each action
                    # actions shape: (batch, chunk_size, action_dim)
                    # Remove batch dim -> (chunk_size, action_dim)
                    actions = actions.squeeze(0)

                inference_time = time.perf_counter() - start_time

                # Add actions to queue
                with self.queue_lock:
                    for i in range(actions.shape[0]):
                        action = self.postprocessor(actions[i:i+1])
                        self.action_queue.append(
                            action.squeeze(0).cpu().numpy())

                logger.debug(f"Inference completed in {inference_time*1000:.1f}ms, "
                             f"queue size: {len(self.action_queue)}")

            except Exception as e:
                logger.error(f"Inference error: {e}")

    def update_observation(self, observation: dict):
        """Update the latest observation for inference."""
        with self.obs_lock:
            self.latest_observation = observation

        # Check if we need to request new actions
        with self.queue_lock:
            queue_size = len(self.action_queue)

        if queue_size <= self.refresh_threshold:
            self.inference_requested.set()

    def get_action(self):
        """Get the next action from the queue (non-blocking if possible)."""
        with self.queue_lock:
            if len(self.action_queue) > 0:
                return self.action_queue.popleft()
            else:
                return None

    def get_action_blocking(self, timeout: float = 1.0):
        """Get action, waiting for inference if queue is empty."""
        start = time.time()
        while time.time() - start < timeout:
            action = self.get_action()
            if action is not None:
                return action
            time.sleep(0.01)
        return None

    @property
    def queue_size(self):
        with self.queue_lock:
            return len(self.action_queue)

    def stop(self):
        """Stop the background inference thread."""
        self.running = False
        self.inference_requested.set()
        self.inference_thread.join(timeout=2.0)


def prepare_observation(robot_obs: dict, device: str, task: str) -> dict:
    """Convert robot observation to policy input format."""
    obs = {}

    for key, value in robot_obs.items():
        if key.startswith("observation.images."):
            # Convert image: HWC uint8 -> CHW float32 [0,1]
            import numpy as np
            img = value.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            obs[key] = torch.from_numpy(img).unsqueeze(
                0).to(device)  # Add batch dim
        elif key == "observation.state":
            obs[key] = torch.from_numpy(value).unsqueeze(0).float().to(device)

    obs["task"] = task

    return obs


def main():
    parser = argparse.ArgumentParser(
        description="Smooth Groot evaluation with background inference")
    parser.add_argument("--policy.path", dest="policy_path",
                        type=str, required=True)
    parser.add_argument("--robot.type", dest="robot_type",
                        type=str, default="bi_so100_follower")
    parser.add_argument("--robot.left_arm_port",
                        dest="left_arm_port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--robot.right_arm_port",
                        dest="right_arm_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--robot.id", dest="robot_id",
                        type=str, default="bimanual_follower")
    parser.add_argument("--task", type=str,
                        default="Grab the red cube and put it in a red basket")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float,
                        default=60.0, help="Duration in seconds")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--refresh_threshold", type=int, default=15,
                        help="Request new actions when queue has this many left")

    args = parser.parse_args()

    # Create robot config
    robot_config = {
        "type": args.robot_type,
        "left_arm_port": args.left_arm_port,
        "right_arm_port": args.right_arm_port,
        "id": args.robot_id,
        "cameras": {
            "left_gripper": {
                "type": "opencv",
                "index_or_path": "/dev/video4",
                "width": 640,
                "height": 480,
                "fps": args.fps,
            },
            "top": {
                "type": "opencv",
                "index_or_path": "/dev/video0",
                "width": 640,
                "height": 480,
                "fps": args.fps,
            },
            "right_gripper": {
                "type": "opencv",
                "index_or_path": "/dev/video2",
                "width": 640,
                "height": 480,
                "fps": args.fps,
            },
        },
    }

    print("\n" + "=" * 60)
    print("🤖 Smooth Groot Evaluation")
    print("=" * 60)
    print(f"Policy: {args.policy_path}")
    print(f"FPS: {args.fps}")
    print(f"Refresh threshold: {args.refresh_threshold}")
    print("=" * 60 + "\n")

    # Initialize controller
    controller = SmoothGrootController(
        policy_path=args.policy_path,
        device=args.device,
        refresh_threshold=args.refresh_threshold,
    )

    # Initialize robot
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.factory import make_robot_from_config

    # Build proper config
    from lerobot.robots.so100_follower import SO100FollowerConfig, BiSO100FollowerConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig

    camera_configs = {
        "left_gripper": OpenCVCameraConfig(
            index_or_path="/dev/video4", width=640, height=480, fps=args.fps
        ),
        "top": OpenCVCameraConfig(
            index_or_path="/dev/video0", width=640, height=480, fps=args.fps
        ),
        "right_gripper": OpenCVCameraConfig(
            index_or_path="/dev/video2", width=640, height=480, fps=args.fps
        ),
    }

    robot_cfg = BiSO100FollowerConfig(
        left_arm_port=args.left_arm_port,
        right_arm_port=args.right_arm_port,
        id=args.robot_id,
        cameras=camera_configs,
    )

    robot = make_robot_from_config(robot_cfg)
    robot.connect()

    print("Robot connected. Starting control loop...")
    print("Press Ctrl+C to stop.\n")

    dt = 1.0 / args.fps
    start_time = time.time()
    step = 0

    try:
        while time.time() - start_time < args.duration:
            loop_start = time.perf_counter()

            # Get observation
            obs = robot.get_observation()

            # Prepare observation for policy
            policy_obs = prepare_observation(obs, args.device, args.task)

            # Update controller with latest observation
            controller.update_observation(policy_obs)

            # Get action (non-blocking if possible)
            action = controller.get_action()

            if action is None:
                # Queue empty, need to wait for inference
                logger.warning(
                    f"Step {step}: Queue empty, waiting for inference...")
                action = controller.get_action_blocking(timeout=0.5)

            if action is not None:
                # Send action to robot
                robot.send_action({"action": action})
            else:
                logger.warning(f"Step {step}: No action available!")

            # Log status periodically
            if step % 30 == 0:
                logger.info(
                    f"Step {step}: Queue size = {controller.queue_size}")

            step += 1

            # Maintain loop timing
            elapsed = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        controller.stop()
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
