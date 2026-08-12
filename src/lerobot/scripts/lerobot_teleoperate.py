# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Simple script to control a robot from teleoperation.

Requires: pip install 'lerobot[hardware]'

Example:
```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

To stream the data to Foxglove instead of Rerun, add ``--display_mode=foxglove``
(then connect the Foxglove app to ``ws://127.0.0.1:8765``; override the port with ``--display_port=<port>``):

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true \
    --display_mode=foxglove
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


@dataclass
class TeleoperateConfig:
    """Configuration for the `lerobot-teleoperate` CLI.

    Args:
        teleop (`TeleoperatorConfig`): Teleoperator device providing control actions.
        robot (`RobotConfig`): Robot being controlled.
        fps (`int`, *optional*, defaults to 60): Maximum control loop frequency, in Hz.
        teleop_time_s (`float | None`, *optional*): Duration, in seconds, to run the teleoperation
            loop for. Runs indefinitely when unset.
        display_data (`bool`, *optional*, defaults to `False`): Whether to display all cameras on screen.
        display_mode (`str`, *optional*, defaults to `"rerun"`): Visualization backend used when
            `display_data` is `True`: `"rerun"` or `"foxglove"`.
        display_ip (`str | None`, *optional*): For `"rerun"`, the IP of a remote server to send to.
            For `"foxglove"`, the interface to bind the WebSocket server to (`127.0.0.1` for local
            only, `0.0.0.0` for all interfaces).
        display_port (`int | None`, *optional*): For `"rerun"`, the port of the remote server. For
            `"foxglove"`, the port to bind the WebSocket server to.
        display_compressed_images (`bool`, *optional*, defaults to `False`): Whether to display
            compressed (JPEG) images instead of raw frames.
    """

    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False
    display_mode: str = "rerun"
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    display_mode: str = "rerun",
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """Continuously read actions from a teleoperation device and apply them to a robot.

    Reads actions from `teleop`, processes them through optional pipelines, sends them to `robot`, and
    optionally displays the robot's state. The loop runs at a specified frequency until a set duration
    is reached or it is manually interrupted.

    Args:
        teleop (`Teleoperator`): The teleoperator device instance providing control actions.
        robot (`Robot`): The robot instance being controlled.
        fps (`int`): The target frequency for the control loop in frames per second.
        teleop_action_processor (`DataProcessorPipeline`): Pipeline to process raw actions from the
            teleoperator.
        robot_action_processor (`DataProcessorPipeline`): Pipeline to process actions before they are
            sent to the robot.
        robot_observation_processor (`DataProcessorPipeline`): Pipeline to process raw observations
            from the robot.
        display_data (`bool`, *optional*, defaults to `False`): If `True`, fetches robot observations
            and displays them in the console and the visualization backend.
        display_mode (`str`, *optional*, defaults to `"rerun"`): Visualization backend to use when
            `display_data` is `True` (`"rerun"` or `"foxglove"`).
        duration (`float | None`, *optional*): The maximum duration of the teleoperation loop in
            seconds. If `None`, the loop runs indefinitely.
        display_compressed_images (`bool`, *optional*, defaults to `False`): If `True`, compresses
            images before sending them to the backend for display.
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        obs = robot.get_observation()

        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_visualization_data(
                display_mode,
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    """Connect `cfg.teleop` and `cfg.robot`, then run the teleoperation loop until stopped.

    Args:
        cfg (`TeleoperateConfig`): Parsed from the CLI.
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            display_mode=cfg.display_mode,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)
        teleop.disconnect()
        robot.disconnect()


def main():
    """CLI entry point for `lerobot-teleoperate`."""
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
