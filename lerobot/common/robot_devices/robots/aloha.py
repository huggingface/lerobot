import copy
from dataclasses import dataclass, field, replace
import numpy as np
import torch
from examples.real_robot_example.gym_real_world.robot import Robot
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.dynamixel_old import pos2pwm, pwm2pos

MAX_LEADER_GRIPPER_RAD = 0.7761942786701344
MAX_LEADER_GRIPPER_POS = 2567
MAX_FOLLOWER_GRIPPER_RAD = 1.6827769243105486
MAX_FOLLOWER_GRIPPER_POS = 3100

MIN_LEADER_GRIPPER_RAD = -0.12732040539450828
MIN_LEADER_GRIPPER_POS = 1984
MIN_FOLLOWER_GRIPPER_RAD = 0.6933593161243099
MIN_FOLLOWER_GRIPPER_POS = 2512

GRIPPER_INDEX = -1

def convert_gripper_range_from_leader_to_follower(leader_pos):
    follower_goal_pos = copy.copy(leader_pos)
    follower_goal_pos[GRIPPER_INDEX] = \
        (leader_pos[GRIPPER_INDEX] - MIN_LEADER_GRIPPER_POS) \
        / (MAX_LEADER_GRIPPER_POS - MIN_LEADER_GRIPPER_POS) \
        * (MAX_FOLLOWER_GRIPPER_POS - MIN_FOLLOWER_GRIPPER_POS) \
        + MIN_FOLLOWER_GRIPPER_POS
    return follower_goal_pos


import serial
import os
import time

def reset_usb_port(port):
    try:
        # Close the serial port if it's open
        ser = serial.Serial(port)
        ser.close()
    except serial.serialutil.SerialException as e:
        print(f"Exception while closing the port: {e}")

    # Find the USB device path
    usb_device_path = None
    for root, dirs, files in os.walk('/sys/bus/usb/drivers/usb'):
        for dir_name in dirs:
            if port in dir_name:
                usb_device_path = os.path.join(root, dir_name)
                break

    if usb_device_path:
        # Unbind and rebind the USB device
        try:
            unbind_path = os.path.join(usb_device_path, 'unbind')
            bind_path = os.path.join(usb_device_path, 'bind')
            usb_id = os.path.basename(usb_device_path)

            with open(unbind_path, 'w') as f:
                f.write(usb_id)
            time.sleep(1)  # Wait for a second
            with open(bind_path, 'w') as f:
                f.write(usb_id)
            print(f"USB port {port} has been reset.")
        except Exception as e:
            print(f"Exception during USB reset: {e}")
    else:
        print(f"Could not find USB device path for port: {port}")


@dataclass
class AlohaRobotConfig:
    """
    Example of usage:
    ```python
    AlohaRobotConfig()
    ```

    Example of only using left arm:
    ```python
    AlohaRobotConfig(
        activated_leaders=["left"],
        activated_followers=["left"],
    )
    ```
    """

    # Define all the components of the robot
    leader_devices: dict[str, str] = field(
        default_factory=lambda: {
            "right": {
                #"port": "/dev/ttyDXL_master_right",
                "port": "/dev/ttyDXL_master_left",
                "servos": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            },
            "left": {
                "port": "/dev/ttyDXL_master_left",
                "servos": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            },
        }
    )
    follower_devices: dict[str, str] = field(
        default_factory=lambda: {
            "right": {
                "port": "/dev/ttyDXL_puppet_right",
                "servos": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            },
            "left": {
                "port": "/dev/ttyDXL_puppet_left",
                "servos": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            },
        }
    )
    camera_devices: dict[str, Camera] = field(
        default_factory=lambda: {
            # "cam_high": OpenCVCamera(16),
            # "cam_low": OpenCVCamera(4),
            # "cam_left_wrist": OpenCVCamera(10),
            # "cam_right_wrist": OpenCVCamera(22),
        }
    )

    # Allows to easily pick a subset of all devices
    activated_leaders: list[str] | None = field(
        default_factory=lambda: ["left", "right"]
    )
    activated_followers: list[str] | None = field(
        default_factory=lambda: ["left", "right"]
    )
    activated_cameras: list[str] | None = field(
        default_factory=lambda: ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    )


class AlohaRobot():
    """ Trossen Robotics

    Example of usage:
    ```python
    robot = AlohaRobot()
    ```
    """

    def __init__(self, config: AlohaRobotConfig | None = None, **kwargs):
        if config is None:
            config = AlohaRobotConfig()
        # Overwrite config arguments using kwargs
        config = replace(config, **kwargs)
        self.config = config

        self.leaders = {}
        self.followers = {}
        self.cameras = {}

        if config.activated_leaders:
            for name in config.activated_leaders:
                info = config.leader_devices[name]
                self.leaders[name] = Robot(info["port"], servo_ids=info["servos"])

        if config.activated_followers:
            for name in config.activated_followers:
                info = config.follower_devices[name]
                self.followers[name] = Robot(info["port"], servo_ids=info["servos"])

        if config.activated_cameras:
            for name in config.activated_cameras:
                self.cameras[name] = config.camera_devices[name]

    def init_teleop(self):
        for name in self.followers:
            self.followers[name]._enable_torque()
        for name in self.cameras:
            self.cameras[name].connect()

    def teleop_step(self, record_data=False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Prepare to assign the positions of the leader to the follower
        leader_pos = {}
        for name in self.leaders:
            leader_pos[name] = self.leaders[name].read_position()

        # Update the position of the follower gripper to account for the different minimum and maximum range
        # position in range [0, 4096[ which corresponds to 4096 bins of 360 degrees
        # for all our dynamixel servors
        # gripper id=8 has a different range from leader to follower
        follower_goal_pos = {}
        for name in self.leaders:
            follower_goal_pos[name] = convert_gripper_range_from_leader_to_follower(leader_pos[name])

        # Send action
        for name in self.followers:
            self.followers[name].set_goal_pos(follower_goal_pos[name])

        # Early exit when recording data is not requested
        if not record_data:
            return

        # Read follower position
        follower_pos = {}
        for name in self.followers:
            follower_pos[name] = self.followers[name].read_position()

        # Create state by concatenating follower current position
        state = []
        for name in ["left", "right"]:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = np.concatenate(state)
        state = pwm2pos(state)

        # Create action by concatenating follower goal position
        action = []
        for name in ["left", "right"]:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = np.concatenate(action)
        action = pwm2pos(action)

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
            if name in self.followers:
                to_idx += len(self.config.follower_devices[name]["servos"])
                follower_goal_pos[name] = pos2pwm(action[from_idx:to_idx].numpy())
                from_idx = to_idx

        for name in self.followers:
            self.followers[name].set_goal_pos(follower_goal_pos[name])
