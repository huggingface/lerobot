import logging
import os
import threading
from pathlib import Path

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
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
    logging.getLogger("robot_client").setLevel(logging.DEBUG)


def _verbose_if_requested() -> bool:
    return os.getenv("LEROBOT_VERBOSE", "0") == "1"


def main():
    _enable_debug_logging_if_requested()
    verbose = _verbose_if_requested()

    # these cameras must match the ones expected by the policy - find your cameras with lerobot-find-cameras
    # check the config.json on the Hub for the policy you are using to see the expected camera specs
    camera_cfg = {
        # OpenCV V4L2 camera devices (from `lerobot-find-cameras`)
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

    # # find ports using lerobot-find-port
    follower_port = "/dev/ttyACM0"

    # # the robot ids are used the load the right calibration files
    follower_id = "so101_white"

    robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_cfg)

    server_address = "192.168.4.37:8080"

    # 3. Create client configuration
    client_cfg = RobotClientConfig(
        robot=robot_cfg,
        server_address=server_address,
        policy_device="cuda",
        # Policy selection:
        # - `policy_type` must be one of the async-inference supported policies (includes "smolvla").
        # - `pretrained_name_or_path` is passed to `<Policy>.from_pretrained(...)` on the server.
        # SmolVLA requires extra deps, e.g. `pip install -e ".[smolvla]"` (plus `.[pi]` if on a Pi).
        policy_type="smolvla",
        pretrained_name_or_path="david-12345/smolvla_so101_pen_pick_place_test",
        chunk_size_threshold=0.3,
        actions_per_chunk=50,  # make sure this is less than the max actions of the policy
    )

    # 4. Create and start client
    client = RobotClient(client_cfg)

    # 5. Provide a textual description of the task
    task = "Pickup the bright green pen and place it near the yellow duck"

    if client.start():
        # Start action receiver thread
        action_receiver_thread = threading.Thread(
            target=client.receive_actions,
            kwargs={"verbose": verbose},
            daemon=True,
        )
        action_receiver_thread.start()

        try:
            # Run the control loop
            client.control_loop(task, verbose=verbose)
        except KeyboardInterrupt:
            client.stop()
            action_receiver_thread.join()
            # (Optionally) plot the action queue size
            visualize_action_queue_size(client.action_queue_size)


if __name__ == "__main__":
    main()
