"""
Latency-Adaptive Async Inference Robot Client Example

This example demonstrates how to use the improved robot client with:
- Jacobson-Karels latency estimation
- SPSC one-slot mailboxes
- Cool-down mechanism for inference triggering
- Frozen action invariant
- Freshest-observation-wins merging strategy

Usage:
    python examples/tutorial/async-inf/robot_client_improved.py

Environment variables:
    LEROBOT_DEBUG=1    Enable DEBUG-level logging
    LEROBOT_VERBOSE=1  Enable verbose output in threads
"""

import logging
import os
import threading
from pathlib import Path

from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client_improved import (
    RobotClientImproved,
    RobotClientImprovedConfig,
)
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig


def _enable_debug_logging_if_requested() -> None:
    """Enable DEBUG logs in the console.

    Note: async-inference uses `init_logging()` internally at import time, which sets the console
    handler to INFO by default. Setting the root logger level is not enough; we must also bump the
    handler level.
    """
    if os.getenv("LEROBOT_DEBUG", "0") != "1":
        return

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setLevel(logging.DEBUG)

    # Make sure the module logger itself does not filter DEBUG.
    logging.getLogger("robot_client_improved").setLevel(logging.DEBUG)


def main() -> None:
    _enable_debug_logging_if_requested()

    # -------------------------------------------------------------------------
    # 1. Configure cameras
    # -------------------------------------------------------------------------
    # These cameras must match the ones expected by the policy.
    # Find your cameras with: lerobot-find-cameras
    # Check the config.json on the Hub for the policy you are using.
    camera_cfg = {
        "camera1": OpenCVCameraConfig(
            index_or_path=Path("/dev/video1"),
            width=640,
            height=480,
            fps=15,
            fourcc="YUYV",
            use_threaded_async_read=True,
            # Prefer smooth control: don't block waiting for a fresh camera frame.
            allow_stale_frames=True,
        ),
        "camera2": OpenCVCameraConfig(
            index_or_path=Path("/dev/video6"),
            width=640,
            height=480,
            fps=15,
            fourcc="YUYV",
            use_threaded_async_read=True,
            allow_stale_frames=True,
        ),
    }

    # -------------------------------------------------------------------------
    # 2. Configure robot
    # -------------------------------------------------------------------------
    # Find ports using: lerobot-find-port
    follower_port = "/dev/ttyACM0"

    # The robot ID is used to load the right calibration files
    follower_id = "so101_white"

    robot_cfg = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        cameras=camera_cfg,
    )

    # -------------------------------------------------------------------------
    # 3. Configure client
    # -------------------------------------------------------------------------
    # Server address (use LAN IP if connecting over network)
    server_address = "192.168.4.37:8080"

    client_cfg = RobotClientImprovedConfig(
        robot=robot_cfg,
        server_address=server_address,
        policy_device="cuda",
        # Policy selection:
        # - `policy_type` must be one of the async-inference supported policies (includes "smolvla").
        # - `pretrained_name_or_path` is passed to `<Policy>.from_pretrained(...)` on the server.
        policy_type="smolvla",
        pretrained_name_or_path="david-12345/smolvla_so101_pen_pick_place_test",
        # Number of actions per chunk (should be <= policy's max action horizon)
        actions_per_chunk=50,
        # Control frequency
        fps=30,
        # Latency-adaptive parameters:
        # - epsilon: safety margin in action steps (triggers inference earlier)
        epsilon=5,
        # - Jacobson-Karels parameters (default values work well in most cases)
        latency_alpha=0.125,  # Smoothing factor for RTT mean
        latency_beta=0.25,  # Smoothing factor for RTT deviation
        latency_k=1.0,  # Scaling factor for deviation (K=1 for faster recovery)
        # Debug: visualize action queue size after stopping
        debug_visualize_queue_size=True,
        # Diagnostics (helpful to distinguish model stutter vs timing/latency jitter)
        diagnostics_enabled=True,
        diagnostics_interval_s=2.0,
        diagnostics_window_s=10.0,
        # Optional: use a deadline-based control clock for steadier action timing
        control_use_deadline_clock=True,
    )

    # -------------------------------------------------------------------------
    # 4. Create and start client
    # -------------------------------------------------------------------------
    client = RobotClientImproved(client_cfg)

    # Task description for VLA policies
    task = "Pickup the bright green pen and place it near the yellow duck"

    if client.start():
        # Start observation sender thread
        obs_sender_thread = threading.Thread(
            target=client.observation_sender_loop,
            name="observation_sender",
            daemon=True,
        )

        # Start action receiver thread
        action_receiver_thread = threading.Thread(
            target=client.action_receiver_loop,
            name="action_receiver",
            daemon=True,
        )

        obs_sender_thread.start()
        action_receiver_thread.start()

        try:
            # Main thread runs the control loop
            client.control_loop(task)

        except KeyboardInterrupt:
            print("\nStopping client...")

        finally:
            client.stop()
            obs_sender_thread.join(timeout=2.0)
            action_receiver_thread.join(timeout=2.0)

            # Visualize action queue size if enabled
            if client_cfg.debug_visualize_queue_size and client.action_queue_sizes:
                visualize_action_queue_size(client.action_queue_sizes)


if __name__ == "__main__":
    main()

