import time
from typing import Tuple
from dataclasses import replace

import numpy as np
import torch
import trossen_slate as slate

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.motors.utils import (
    MotorsBus,
    make_motors_buses_from_configs,
)
from lerobot.common.robot_devices.robots.configs import TrossenAIMobileRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ensure_safe_goal_position
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    SlateBaseSystemState,
)

class TrossenAIMobile():

    def __init__(self, config: TrossenAIMobileRobotConfig | None = None, **kwargs):
        if config is None:
            self.config = TrossenAIMobileRobotConfig(**kwargs)
        else:
            # Overwrite config arguments using kwargs
            self.config = replace(config, **kwargs)
        self.robot_type = self.config.type
        self.enable_motor_torque = self.config.enable_motor_torque
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs: dict[str, float] = {}
        self.base = slate.TrossenSlate()
        self.slate_base_data = slate.ChassisData()

    def get_motor_names(self, arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = ['linear_vel', 'angular_vel'] + self.get_motor_names(self.leader_arms)
        state_names = ['odom_x', 'odom_y', 'odom_theta', 'linear_vel', 'angular_vel'] + self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def teleop_safety_stop(self):
        for arms in self.leader_arms:
            self.leader_arms[arms].write("Reset", 1)
        for arms in self.follower_arms:
            self.follower_arms[arms].write("Reset", 1)
        time.sleep(2)
        for arms in self.leader_arms:
            self.leader_arms[arms].write("Torque_Enable", 0)
        for arms in self.follower_arms:
            self.follower_arms[arms].write("Torque_Enable", 1)

    def connect(self) -> None:
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "TrossenAIMobile is already connected. Do not run `robot.connect()` twice."
            )
        success, result = self.base.init_base()
        if not success:
            raise RobotDeviceNotConnectedError(
                f"{result}.\nMake sure the robot is powered on and connected to the computer.\n{self.check_base_state()[0]}"
            )
        try:
            if self.check_base_state()[1]==SlateBaseSystemState.SYS_ESTOP:
                raise RuntimeError(
                    f"Robot is in emergency stop state. Please release the emergency stop button and try again"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to check base state.\n{e}."
            )

        success, result = self.base.enable_motor_torque(self.enable_motor_torque)
        if not success:
            raise RuntimeError(
                f"Failed to enable motor torque.\n{result}.\n{self.check_base_state()[0]}"
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect."
            )

        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()
        time.sleep(2)

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].write("Torque_Enable", 1)

        # Check both arms can be read
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True


    def check_base_state(self) -> Tuple[str, SlateBaseSystemState]:
        success = self.base.update_state()
        if not success:
            return "Failed to get base state. Make sure the robot is powered on and connected to the computer."
            
        self.base.read(self.slate_base_data)
        state_code = self.slate_base_data.system_state
        try:
            state_name = SlateBaseSystemState(state_code).name
        except ValueError:
            state_name = f"UNKNOWN_STATE (code: {state_code})"

        return state_name, state_code


    def get_base_state(self) -> dict:
        success = self.base.update_state()
        if not success:
            raise RobotDeviceNotConnectedError(
                "Failed to get base state. Make sure the robot is powered on and connected to the computer."
            )
        self.base.read(self.slate_base_data)
        return {
            "odom_x": self.slate_base_data.odom_x,
            "odom_y": self.slate_base_data.odom_y,
            "odom_theta": self.slate_base_data.odom_z,
            "linear_vel": self.slate_base_data.vel_x,
            "angular_vel": self.slate_base_data.vel_z,
        }

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "TrossenAIMobile is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Send goal position to the follower
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Used when record_data=True
            follower_goal_pos[name] = goal_pos

            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        if not record_data:
            return

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        before_read_t = time.perf_counter()
        base_state = self.get_base_state()
        base_action = [base_state['linear_vel'], base_state['angular_vel']]
        self.logs["read_base_dt_s"] = time.perf_counter() - before_read_t

        # Create state by concatenating follower current position and base state
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state.append(torch.as_tensor(list(base_state.values())))
        state = torch.cat(state)

        # Create action by concatenating follower goal position and base action
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action.append(torch.as_tensor(list(base_action)))
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "TrossenAIMobile is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Read base state
        before_read_t = time.perf_counter()
        base_state = self.get_base_state()
        self.logs["read_base_dt_s"] = time.perf_counter() - before_read_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state.append(base_state.values())
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        obs_dict = {}
        obs_dict["observation.state"] = state

        return obs_dict

    def send_action(self, action):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "TrossenAIMobile is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms[name].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)

        linear_vel, angular_vel = action.tolist()[-2:]
        before_write_t = time.perf_counter()
        success = self.base.set_cmd_vel(linear_vel, angular_vel)
        if not success:
            raise RobotDeviceNotConnectedError(
                "Failed to send action to base slate. Make sure the robot is powered on and connected to the computer."
            )
        self.logs["write_base_dt_s"] = time.perf_counter() - before_write_t

        action_sent.append(action[-2:])

        return torch.cat(action_sent)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "TrossenAIMobile is not connected. You need to run `robot.connect()` before disconnecting."
            )
        success, result = self.base.enable_motor_torque(False)
        if not success:
            raise RuntimeError(
                f"Failed to disable motor torque.\n{result}."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()
        time.sleep(2)

        for name in self.cameras:
            self.cameras[name].disconnect()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
