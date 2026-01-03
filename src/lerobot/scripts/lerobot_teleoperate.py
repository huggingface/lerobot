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

"""
Simple script to control a robot from teleoperation.

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

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    IdentityProcessorStep,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    so100_leader,
    so101_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Optional filtering to reduce "slow movement jitter" caused by small sensor noise / quantization
    # in the teleoperator position readings.
    #
    # - action_deadband: either a single float deadband applied to all action keys, or a per-key dict.
    #   If, for every key, abs(delta vs last *sent* action) <= its deadband, we skip sending a new
    #   command for that frame.
    # - action_smoothing_alpha: EMA smoothing factor in [0, 1]. Higher means less smoothing.
    action_deadband: float | dict[str, float] | None = None
    action_smoothing_alpha: float | None = None


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    action_deadband: float | dict[str, float] | None = None,
    action_smoothing_alpha: float | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    last_loop_report_s = start
    last_sent_action: dict[str, float] | None = None
    smoothed_action: dict[str, float] | None = None

    if action_smoothing_alpha is not None and not (0.0 <= action_smoothing_alpha <= 1.0):
        raise ValueError(f"action_smoothing_alpha must be in [0, 1], got {action_smoothing_alpha}")
    if action_deadband is not None and not isinstance(action_deadband, (float, dict)):
        raise ValueError(f"action_deadband must be a float, dict[str, float], or None, got {type(action_deadband)}")

    def _deadband_for_key(key: str) -> float | None:
        if action_deadband is None:
            return None
        if isinstance(action_deadband, float):
            return float(action_deadband)
        # Dict deadband: allow providing only a subset of keys.
        val = action_deadband.get(key)
        return None if val is None else float(val)

    def _is_identity_pipeline(pipeline: RobotProcessorPipeline) -> bool:
        # We only auto-skip observations when we're sure the pipeline won't use them.
        # Default teleoperate uses a single IdentityProcessorStep, but we accept multiple.
        return len(pipeline.steps) > 0 and all(isinstance(step, IdentityProcessorStep) for step in pipeline.steps)

    # Reading robot observations can be expensive (motor sync reads + camera async_read that waits for new frames),
    # and can unintentionally rate-limit the control loop to camera FPS. If we're not displaying data and both
    # action processors are pure identity, it's safe to skip observations entirely.
    should_read_observation = display_data or not (
        _is_identity_pipeline(teleop_action_processor) and _is_identity_pipeline(robot_action_processor)
    )
    if not should_read_observation:
        logging.info(
            "Teleop: skipping robot observations (including camera reads) to avoid camera-FPS rate limiting. "
            "Set --display_data=true to force observations/visualization."
        )

    while True:
        loop_start = time.perf_counter()

        # Get robot observation (optional).
        # Note: `robot_action_observation_to_transition` accepts `observation=None`.
        # obs = robot.get_observation() if should_read_observation else None
        obs = robot.get_observation()

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Optional smoothing (EMA) on the final robot action.
        if action_smoothing_alpha is not None:
            if smoothed_action is None:
                smoothed_action = {k: float(v) for k, v in robot_action_to_send.items()}
            else:
                alpha = float(action_smoothing_alpha)
                for key, value in robot_action_to_send.items():
                    prev = smoothed_action.get(key, float(value))
                    smoothed_action[key] = alpha * float(value) + (1.0 - alpha) * float(prev)
            robot_action_to_send = smoothed_action

        # Optional deadband vs last *sent* action: if nothing changed enough, skip sending.
        if action_deadband is not None and last_sent_action is not None:
            any_exceeds_deadband = False
            for key, value in robot_action_to_send.items():
                prev = last_sent_action.get(key, float(value))
                delta = abs(float(value) - float(prev))
                deadband = _deadband_for_key(key)
                # If no deadband specified for this key (dict mode), treat it as "always send if it changes".
                if deadband is None:
                    if delta > 0.0:
                        any_exceeds_deadband = True
                        break
                elif delta > deadband:
                    any_exceeds_deadband = True
                    break
            if not any_exceeds_deadband:
                dt_s = time.perf_counter() - loop_start
                precise_sleep(1 / fps - dt_s)
                continue

        # Send processed action to robot (robot_action_processor.to_output should return dict[str, Any])
        sent_action = robot.send_action(robot_action_to_send)
        last_sent_action = {k: float(v) for k, v in sent_action.items()}

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        # Printing every iteration can itself become a bottleneck at higher FPS.
        if time.perf_counter() - last_loop_report_s >= 1.0:
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)
            last_loop_report_s = time.perf_counter()

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    log_file_env = os.getenv("LEROBOT_LOG_FILE")
    log_file = Path(log_file_env) if log_file_env else None
    init_logging(log_file=log_file)
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation")

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
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            action_deadband=cfg.action_deadband,
            action_smoothing_alpha=cfg.action_smoothing_alpha,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
