import copy
from dataclasses import dataclass, field, replace
import numpy as np
import torch
from examples.real_robot_example.gym_real_world.robot import Robot
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsChain
from lerobot.common.robot_devices.motors.utils import MotorsChain


GRIPPER_INDEX = -1


@dataclass
class KochRobotConfig:
    """
    Example of usage:
    ```python
    KochRobotConfig()
    ```

    Example of only using left arm:
    ```python
    KochRobotConfig(
        activated_leaders=["left"],
        activated_followers=["left"],
    )
    ```
    """

    # Define all the components of the robot
    leader_motors: dict[str, MotorsChain] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsChain(
                #"/dev/tty.usbmodem575E0030111", {
                "/dev/tty.usbmodem575E0031751", {
                    1: "xl330-m077",
                    2: "xl330-m077",
                    3: "xl330-m077",
                    4: "xl330-m077",
                    5: "xl330-m077",
                    6: "xl330-m077",
                }
            ),
            #"left": DynamixelMotorsChain("/dev/ttyDXL_master_left", [1, 2, 3, 4, 5, 6]),
        }
    )
    follower_motors: dict[str, MotorsChain] = field(
        default_factory=lambda: {
            #"right": DynamixelMotorsChain("/dev/ttyDXL_puppet_right", [1, 2, 3, 4, 5, 6]),
            "left": DynamixelMotorsChain(
                # "/dev/tty.usbmodem575E0031691", {
                "/dev/tty.usbmodem575E0032081", {
                    1: "xl430-w250",
                    2: "xl430-w250",
                    3: "xl330-m288",
                    4: "xl330-m288",
                    5: "xl330-m288",
                    6: "xl330-m288",
                }
            ),
        }
    )
    cameras: dict[str, Camera] = field(
        default_factory=lambda: {}
    )


class KochRobot():
    """ Tau Robotics: https://tau-robotics.com

    Example of usage:
    ```python
    robot = KochRobot()
    ```
    """

    def __init__(self, config: KochRobotConfig | None = None, **kwargs):
        if config is None:
            config = KochRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.leader_motors = self.config.leader_motors
        self.follower_motors = self.config.follower_motors
        self.cameras = self.config.cameras

    def init_teleop(self):
        for name in self.follower_motors:
            self.follower_motors[name].enable_torque()
        for name in self.cameras:
            self.cameras[name].connect()

    def teleop_step(self, record_data=False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Prepare to assign the positions of the leader to the follower
        leader_pos = {}
        for name in self.leader_motors:
            leader_pos[name] = self.leader_motors[name].read_position()

        follower_goal_pos = {}
        for name in self.leader_motors:
            follower_goal_pos[name] = leader_pos[name]

        # Send action
        for name in self.follower_motors:
            self.follower_motors[name].write_goal_position(follower_goal_pos[name])

        # Early exit when recording data is not requested
        if not record_data:
            return

        # Read follower position
        follower_pos = {}
        for name in self.follower_motors:
            follower_pos[name] = self.follower_motors[name].read_position()

        # Create state by concatenating follower current position
        state = []
        for name in ["left", "right"]:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = np.concatenate(state)
        #state = pwm2pos(state)

        # Create action by concatenating follower goal position
        action = []
        for name in ["left", "right"]:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = np.concatenate(action)
        #action = pwm2pos(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            images[name] = self.cameras[name].read()

        # Populate output dictionnaries and format to pytorch
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = torch.from_numpy(state)
        action_dict["action"] = torch.from_numpy(action)
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] =  torch.from_numpy(images[name])

        return obs_dict, action_dict

    def send_action(self, action):
        from_idx = 0
        to_idx = 0
        follower_goal_pos = {}
        for name in ["left", "right"]:
            if name in self.follower_motors:
                to_idx += len(self.config.follower_motors[name].motor_ids)
                follower_goal_pos[name] = action[from_idx:to_idx].numpy()
                #follower_goal_pos[name] = pos2pwm(action[from_idx:to_idx].numpy())
                from_idx = to_idx

        for name in self.follower_motors:
            self.follower_motors[name].write_goal_position(follower_goal_pos[name].astype(np.int64))
