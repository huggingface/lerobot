import logging
import os
import threading

from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client_drtc import (
    RobotClientDrtc,
    RobotClientDrtcConfig,
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
    logging.getLogger("robot_client_drtc").setLevel(logging.DEBUG)


def main() -> None:
    _enable_debug_logging_if_requested()

    # These cameras must match the ones expected by the policy.
    # Find your cameras with: lerobot-find-cameras
    # Check the config.json on the Hub for the policy you are using.
    camera_cfg = {
        "camera2": OpenCVCameraConfig(index_or_path="/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0", width=800, height=600, fps=30, fourcc="MJPG", use_threaded_async_read=True, allow_stale_frames=True),
        "camera1": OpenCVCameraConfig(index_or_path="/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:2:1.0-video-index0", width=800, height=600, fps=30, fourcc="MJPG", use_threaded_async_read=True, allow_stale_frames=True),
    }

    # Find ports using: lerobot-find-port
    follower_port = "/dev/ttyACM0"

    # The robot ID is used to load the right calibration files
    follower_id = "so101_follower_2026_01_03"

    robot_cfg = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        cameras=camera_cfg,
    )

    # Server address (use LAN IP if connecting over network)
    # Examples:
    #   - Local: 127.0.0.1:8080
    #   - LAN:   192.168.4.37:8080
    #   - Tunnel (see scripts/start_client.sh): 127.0.0.1:18080
    server_address = os.getenv("LEROBOT_SERVER_ADDRESS", "127.0.0.1:8080")

    client_cfg = RobotClientDrtcConfig(
        robot=robot_cfg,
        server_address=server_address,
        policy_device="cuda",
        # Policy selection:
        # - `policy_type` must be one of the async-inference supported policies (includes "smolvla").
        # - `pretrained_name_or_path` is passed to `<Policy>.from_pretrained(...)` on the server.
        policy_type="smolvla",
        pretrained_name_or_path="jackvial/so101_smolvla_pickplaceorangecube_e100",
        actions_per_chunk=50,
        # Control frequency
        fps=60,
        # RTC s_min (aka minimum execution horizon)
        s_min=15,
        # DRTC cooldown margin
        epsilon=2,
        # DRTC Jacobson-Karels parameters (default values work well in most cases)
        latency_alpha=0.125,  # Smoothing factor for RTT mean
        latency_beta=0.25,  # Smoothing factor for RTT deviation
        latency_k=2.0,  # Scaling factor for deviation (K=1 for faster recovery)
        
        # DRTC trajectory smoothing filter
        action_filter_mode="butterworth",
        action_filter_past_buffer_size=10,
        action_filter_butterworth_cutoff=3.0,  # Hz - passes motion, attenuates jitter
        action_filter_butterworth_order=2,       # Good balance of sharpness vs phase lag
        action_filter_gain=1.4,                  # Slight boost to compensate attenuation
        
        # Debug: visualize action queue size after stopping
        debug_visualize_queue_size=False,
        # Diagnostics (helpful to distinguish model stutter vs timing/latency jitter)
        metrics_diagnostic_enabled=True,
        metrics_diagnostic_interval_s=2.0,
        metrics_diagnostic_window_s=10.0,
        # Optional: use a deadline-based control clock for steadier action timing
        control_use_deadline_clock=True,
        # Robustness: if the robot state read occasionally fails, reuse the last good observation
        # to avoid stalling action production (reduces visible hitches).
        obs_fallback_on_failure=True,
        obs_fallback_max_age_s=2.0,
        # Trajectory visualization (sends data to policy server for real-time visualization)
        # Local:
        # - Open http://localhost:8088 in your browser to view trajectories
        # Tunnel (see scripts/start_client.sh):
        # - Open http://localhost:18088 in your browser to view trajectories
        trajectory_viz_enabled=True,
        trajectory_viz_ws_url=os.getenv("LEROBOT_TRAJECTORY_VIZ_WS_URL", "ws://localhost:8089"),
        # RTC parameters
        rtc_sigma_d=0.2,
        rtc_full_trajectory_alignment=False,
        num_flow_matching_steps=None,  # Use policy default
        rtc_max_guidance_weight=None,  # Auto (Beta = n)
        # Experiment metrics
        metrics_path="results/jitter_analysis.csv"
    )

    # -------------------------------------------------------------------------
    # 4. Create and start client
    # -------------------------------------------------------------------------
    client = RobotClientDrtc(client_cfg)

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

