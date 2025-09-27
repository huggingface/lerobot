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
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
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
    bi_koch_follower,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_koch_leader,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# FK and IK stuff
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)

from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)


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


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
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
    for _ in range(10):
        print('\n')



    while True:
        loop_start = time.perf_counter()
        # Retrieve normalized action and raw values if available
        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        try:
            raw_action, unnormalized_action = teleop.get_action_with_raw()
        except Exception:
            # Fallback to normalized-only in case a specific teleoperator does not support raw values
            raw_action = teleop.get_action()
            unnormalized_action = None
        # Retrieve normalized observation and raw values if available
        try:
            obs, raw_observation = robot.get_observation_with_raw()
        except Exception:
            obs = robot.get_observation()
            raw_observation = None
        if display_data:
            log_rerun_data(obs, raw_action)

        # Process teleop action through pipeline
        # teleop_action = teleop_action_processor((raw_action, obs))
        teleop_action = teleop_action_processor(raw_action)
        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return dict[str, Any])
        _ = robot.send_action(robot_action_to_send)

        output_lines: list[str] = []
        output_lines.append("-" * (display_len + 10))
        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            output_lines.append(f"{'TELEOP NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in teleop_action.items():
                output_lines.append(f"{motor:<{display_len}} | {value * 100:>7.2f}")

            output_lines.append(f"{'ROBOT NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                output_lines.append(f"{motor:<{display_len}} | {value:>7.2f}")

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start

        output_lines.append("-" * (display_len + 20))
        # Actions block
        output_lines.append("ACTIONS")
        if unnormalized_action is not None:
            output_lines.append(f"{'NAME':<{display_len}} | {'NORM':>7} | {'RAW':>7}")
            for motor, value in raw_action.items():
                raw_val = unnormalized_action.get(motor) if isinstance(unnormalized_action, dict) else None
                raw_display = f"{int(raw_val):>7}" if isinstance(raw_val, (int, float)) else f"{'N/A':>7}"
                output_lines.append(f"{motor:<{display_len}} | {value:>7.2f} | {raw_display}")
        else:
            output_lines.append(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in raw_action.items():
                output_lines.append(f"{motor:<{display_len}} | {value:>7.2f}")

        # Observations block
        output_lines.append("OBSERVATIONS")
        if raw_observation is not None:
            output_lines.append(f"{'NAME':<{display_len}} | {'NORM':>7} | {'RAW':>7}")
            for key, value in obs.items():
                if not key.endswith(".pos"):
                    continue
                raw_val = raw_observation.get(key) if isinstance(raw_observation, dict) else None
                raw_display = f"{int(raw_val):>7}" if isinstance(raw_val, (int, float)) else f"{'N/A':>7}"
                output_lines.append(f"{key:<{display_len}} | {value:>7.2f} | {raw_display}")
        else:
            output_lines.append(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for key, value in obs.items():
                if not key.endswith(".pos"):
                    continue
                output_lines.append(f"{key:<{display_len}} | {value:>7.2f}")

        # Timing line (keep a blank line before time for readability)
        output_lines.append("")
        output_lines.append(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        print("\n".join(output_lines))

        if duration is not None and time.perf_counter() - start >= duration:
            return

        # Move cursor up exactly the number of lines we printed
        move_cursor_up(len(output_lines) + 10)


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    # Build pipeline to convert teleop joints to EE action
    left_robot_kinematics_solver = RobotKinematics(
        urdf_path="assets/koch_follower.urdf",
        target_frame_name="link_6",
        entity_path_prefix="follower_left",
        display_data=cfg.display_data,
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
        offset=0.0,
    )
    right_robot_kinematics_solver = RobotKinematics(
        urdf_path="assets/koch_follower.urdf",
        target_frame_name="link_6",
        entity_path_prefix="follower_right",
        display_data=cfg.display_data,
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
        offset=0.2,
    )

    left_teleop_kinematics_solver = RobotKinematics(
        urdf_path="assets/koch_follower.urdf",
        target_frame_name="link_6",
        entity_path_prefix="leader_left",
        display_data=cfg.display_data,
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
        offset=0.4,
    )
    right_teleop_kinematics_solver = RobotKinematics(
        urdf_path="assets/koch_follower.urdf",
        target_frame_name="link_6",
        entity_path_prefix="leader_right",
        display_data=cfg.display_data,
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
        offset=0.6,
    )

    teleop_motor_names = list(teleop.left_arm.bus.motors.keys())
    robot_motor_names = list(robot.left_arm.bus.motors.keys())
    left_teleop_motor_names = ["left_" + motor for motor in teleop_motor_names]
    right_teleop_motor_names = ["right_" + motor for motor in teleop_motor_names]
    left_robot_motor_names = ["left_" + motor for motor in robot_motor_names]
    right_robot_motor_names = ["right_" + motor for motor in robot_motor_names]

    teleop_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=left_teleop_kinematics_solver, motor_names=left_teleop_motor_names, gripper_name="left_gripper"
            ),
            ForwardKinematicsJointsToEE(
                kinematics=right_teleop_kinematics_solver, motor_names=right_teleop_motor_names, gripper_name="right_gripper"
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # build pipeline to convert EE action to robot joints
    ee_to_robot_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
                max_ee_twist_step_rad=0.50,
                prefix="left_",
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
                max_ee_twist_step_rad=0.50,
                prefix="right_",
            ),
            InverseKinematicsEEToJoints(
                kinematics=left_robot_kinematics_solver,
                motor_names=left_robot_motor_names,
                initial_guess_current_joints=False,
                prefix="left_",
            ),
            InverseKinematicsEEToJoints(
                kinematics=right_robot_kinematics_solver,
                motor_names=right_robot_motor_names,
                initial_guess_current_joints=False,
                prefix="right_",
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    # Use the IK version
    teleop_action_processor = teleop_to_ee
    robot_action_processor = ee_to_robot_joints

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
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    teleoperate()


if __name__ == "__main__":
    main()
