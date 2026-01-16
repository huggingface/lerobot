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
    LEROBOT_DEBUG=1        Enable DEBUG-level logging
    LEROBOT_VERBOSE=1      Enable verbose output in threads
    RTC_CONFIG_INDEX=N     RTC sweep config index (0-14), see table below
    RTC_BATCH=M            RTC sweep batch number

RTC Sweep Config Index Mapping (15 configs):
    Default sweep (sigma_d, full_traj):
        0: sigma_d=0.1, full_traj=False    1: sigma_d=0.1, full_traj=True
        2: sigma_d=0.2, full_traj=False    3: sigma_d=0.2, full_traj=True
        4: sigma_d=0.4, full_traj=False    5: sigma_d=0.4, full_traj=True
        6: sigma_d=0.6, full_traj=False    7: sigma_d=0.6, full_traj=True
        8: sigma_d=0.8, full_traj=False    9: sigma_d=0.8, full_traj=True
       10: sigma_d=1.0, full_traj=False   11: sigma_d=1.0, full_traj=True

    Alex Soare sweep (n, Beta) - sigma_d=0.2 fixed:
       12: n=5,  Beta=auto    (faster inference, less smooth)
       13: n=10, Beta=auto    (default, balanced)
       14: n=20, Beta=auto    (slower inference, smoother)

Reference: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html
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

# =============================================================================
# RTC Sweep Configuration
# =============================================================================
# (sigma_d, full_trajectory_alignment, num_flow_matching_steps, rtc_max_guidance_weight)
# num_flow_matching_steps: None = use policy default (10), int = override
# rtc_max_guidance_weight: None = auto (Beta = n), float = override
#
# Reference: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html
RTC_SWEEP_CONFIGS: list[tuple[float, bool, int | None, float | None]] = [
    # Default sweep: sigma_d and full_trajectory_alignment
    (0.1, False, None, None), (0.1, True, None, None),   # configs 0, 1
    (0.2, False, None, None), (0.2, True, None, None),   # configs 2, 3
    (0.4, False, None, None), (0.4, True, None, None),   # configs 4, 5
    (0.6, False, None, None), (0.6, True, None, None),   # configs 6, 7
    (0.8, False, None, None), (0.8, True, None, None),   # configs 8, 9
    (1.0, False, None, None), (1.0, True, None, None),   # configs 10, 11
    # Alex Soare sweep: denoising steps (n) with Beta = n (auto)
    # Fixed: sigma_d=0.2 (optimal), full_traj=False (use gradient guidance)
    (0.2, False, 5, None),    # config 12: n=5,  Beta=auto
    (0.2, False, 10, None),   # config 13: n=10, Beta=auto
    (0.2, False, 20, None),   # config 14: n=20, Beta=auto
]


def get_rtc_sweep_config() -> tuple[int | None, str | None, float, bool, int | None, float | None, str | None]:
    """Get RTC config from environment variables.

    Returns:
        Tuple of (config_index, batch, sigma_d, full_traj_alignment,
                  num_flow_matching_steps, rtc_max_guidance_weight, metrics_path).
        If not in sweep mode, returns defaults with None for sweep-specific values.
    """
    config_index_str = os.getenv("RTC_CONFIG_INDEX")
    batch = os.getenv("RTC_BATCH")

    # Default values (used when not in sweep mode)
    default_sigma_d = 0.2  # Alex Soare optimal
    default_full_traj = False
    default_num_steps = None  # Use policy default
    default_max_gw = None  # Auto (Beta = n)

    if config_index_str is None or batch is None:
        return None, None, default_sigma_d, default_full_traj, default_num_steps, default_max_gw, None

    config_index = int(config_index_str)
    if config_index < 0 or config_index >= len(RTC_SWEEP_CONFIGS):
        raise ValueError(f"RTC_CONFIG_INDEX must be 0-{len(RTC_SWEEP_CONFIGS)-1}, got {config_index}")

    sigma_d, full_traj, num_steps, max_gw = RTC_SWEEP_CONFIGS[config_index]

    # Create results directory and metrics path
    results_dir = Path("results/rtc_sweep")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(results_dir / f"batch{batch}_config{config_index}.csv")

    print(f"RTC Sweep Mode: config_index={config_index}, batch={batch}")
    print(f"  sigma_d={sigma_d}, full_trajectory_alignment={full_traj}")
    print(f"  num_flow_matching_steps={num_steps or 'default'}, rtc_max_guidance_weight={max_gw or 'auto'}")
    print(f"  Metrics will be saved to: {metrics_path}")

    return config_index, batch, sigma_d, full_traj, num_steps, max_gw, metrics_path


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
    # 0. Check for RTC sweep mode (via environment variables)
    # -------------------------------------------------------------------------
    (
        config_idx,
        batch,
        rtc_sigma_d,
        rtc_full_traj,
        num_flow_matching_steps,
        rtc_max_guidance_weight,
        experiment_metrics_path,
    ) = get_rtc_sweep_config()

    # -------------------------------------------------------------------------
    # 1. Configure cameras
    # -------------------------------------------------------------------------
    # These cameras must match the ones expected by the policy.
    # Find your cameras with: lerobot-find-cameras
    # Check the config.json on the Hub for the policy you are using.
    camera_cfg = {
        # "camera1": OpenCVCameraConfig(
        #     index_or_path=Path("/dev/video0"),
        #     width=640,
        #     height=480,
        #     fps=15,
        #     fourcc="YUYV",
        #     use_threaded_async_read=True,
        #     # Prefer smooth control: don't block waiting for a fresh camera frame.
        #     allow_stale_frames=True,
        # ),
        # "camera2": OpenCVCameraConfig(
        #     index_or_path=Path("/dev/video4"),
        #     width=640,
        #     height=480,
        #     fps=15,
        #     fourcc="YUYV",
        #     use_threaded_async_read=True,
        #     allow_stale_frames=True,
        # ),
         "camera2": OpenCVCameraConfig(index_or_path="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.0-video-index0", width=800, height=600, fps=30, fourcc="MJPG", use_threaded_async_read=True, allow_stale_frames=True),
        "camera1": OpenCVCameraConfig(index_or_path="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0", width=800, height=600, fps=30, fourcc="MJPG", use_threaded_async_read=True, allow_stale_frames=True),
    }

    # -------------------------------------------------------------------------
    # 2. Configure robot
    # -------------------------------------------------------------------------
    # Find ports using: lerobot-find-port
    follower_port = "/dev/ttyACM0"

    # The robot ID is used to load the right calibration files
    follower_id = "so101_follower_2026_01_03"

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
    # server_address = "127.0.0.1:8080"

    client_cfg = RobotClientImprovedConfig(
        robot=robot_cfg,
        server_address=server_address,
        policy_device="cuda",
        # Policy selection:
        # - `policy_type` must be one of the async-inference supported policies (includes "smolvla").
        # - `pretrained_name_or_path` is passed to `<Policy>.from_pretrained(...)` on the server.
        policy_type="smolvla",
        # pretrained_name_or_path="david-12345/smolvla_so101_pen_pick_place_test",
        # pretrained_name_or_path="jackvial/so101_smolvla_pickplaceorangecube_0_e50_10000",
        pretrained_name_or_path="/home/jack/code/self-driving-screwdriver-robot/wandb_downloads/so101_smolvla_pickplaceorangecube_e100_20260108_203916/100000/pretrained_model/",
        # Number of actions per chunk (should be <= policy's max action horizon).
        # For lower jitter over Wi‑Fi / variable server times, increasing this can help keep `sched` > 0.
        actions_per_chunk=50,
        # Control frequency
        fps=30,
        # Latency-adaptive parameters:
        # - epsilon: safety margin in action steps (triggers inference earlier)
        epsilon=15,
        # - Jacobson-Karels parameters (default values work well in most cases)
        latency_alpha=0.125,  # Smoothing factor for RTT mean
        latency_beta=0.25,  # Smoothing factor for RTT deviation
        latency_k=2.0,  # Scaling factor for deviation (K=1 for faster recovery)
        # Debug: visualize action queue size after stopping
        debug_visualize_queue_size=False,
        # Diagnostics (helpful to distinguish model stutter vs timing/latency jitter)
        diagnostics_enabled=True,
        diagnostics_interval_s=2.0,
        diagnostics_window_s=10.0,
        # Optional: use a deadline-based control clock for steadier action timing
        control_use_deadline_clock=True,
        # Robustness: if the robot state read occasionally fails, reuse the last good observation
        # to avoid stalling action production (reduces visible hitches).
        obs_fallback_on_failure=True,
        obs_fallback_max_age_s=2.0,
        # Trajectory visualization (sends data to policy server for real-time visualization)
        # Open http://localhost:8088 in your browser to view trajectories
        trajectory_viz_enabled=True,
        # RTC parameters (can be overridden by RTC_CONFIG_INDEX env var for sweep experiments)
        rtc_sigma_d=rtc_sigma_d,
        rtc_full_trajectory_alignment=rtc_full_traj,
        # Alex Soare parameters: denoising steps (n) and Beta
        # Reference: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html
        num_flow_matching_steps=num_flow_matching_steps,
        rtc_max_guidance_weight=rtc_max_guidance_weight,
        # Experiment metrics (set by RTC sweep mode, otherwise None)
        experiment_metrics_path=experiment_metrics_path,
    )

    # -------------------------------------------------------------------------
    # 4. Create and start client
    # -------------------------------------------------------------------------
    client = RobotClientImproved(client_cfg)

    # Task description for VLA policies
    task = "Pick up the orange cube and place it on the black X marker with the white background"

    if client.start():
        # Start observation sender thread
        obs_sender_thread = threading.Thread(
            target=client.observation_sender,
            name="observation_sender",
            daemon=True,
        )

        # Start action receiver thread
        action_receiver_thread = threading.Thread(
            target=client.action_receiver,
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

