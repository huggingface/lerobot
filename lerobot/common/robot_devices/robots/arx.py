from dataclasses import dataclass, field, replace
from enum import Enum
import time

import arx5_interface as arx5
# import torch
import numpy as np

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

class ARXArmModel(Enum):
    """
    Two models of ARX arms are supported: the X5 and L5.
    The main difference between the two is the type of motor used in the three base joints.
    Ensure you are using the right arm model before starting the robot, as choosing the wrong one can lead to dangerous movements.
    """
    X5 = "X5"
    L5 = "L5"

@dataclass
class ARXArmConfig:
    model: ARXArmModel
    interface_name: str # name of the communication interface, e.g. `can0` or `enx6c1ff70ac436`
    urdf_path: str  # link to the robot arm URDF file. Used for gravity compensation
    dof: int = 6    # degrees of freedom. Defaults to 6.

@dataclass
class ARXRobotConfig:
    """
    Example of usage:
    ```python
    ARXRobotConfig()
    ```
    """
    # Define all components of the robot
    leader_arms: dict[str, ARXArmConfig] = field(default_factory=lambda: {})
    follower_arms: dict[str, ARXArmConfig] = field(default_factory=lambda: {})
    # cameras: dict[str, Camera] = field(default_factory=lambda: {})

class ARXArm:
    """
    Class for controlling a single ARX arm."""
    
    def __init__(
        self,
        config: ARXArmConfig,
    ):
        self.config = config
        self.is_connected = False
        self.joint_controller = None

    def connect(self):
        print(self.config)
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ARXArm is already connected. Do not run `robot.connect()` twice."
            )
        self.joint_controller = arx5.Arx5JointController(self.config.model, self.config.interface_name)
        self.is_connected = True

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ARXArm is not connected. Do not run `robot.disconnect()` twice."
            )
        self.joint_controller.reset_to_home()
        self.joint_controller.set_to_damping()
        self.joint_controller = None
        self.is_connected = False

    def reset(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ARXArm is not connected. You need to run `robot.connect()`."
            )
        self.joint_controller.send_recv_once()
        self.joint_controller.reset_to_home()
        self.joint_controller.enable_gravity_compensation(self.config.urdf_path)
        self.joint_controller.send_recv_once()

    def get_state(self) -> arx5.JointState:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ARXArm is not connected. You need to run `robot.connect()`."
            )
        self.joint_controller.send_recv_once()
        return self.joint_controller.get_state()
    
    def send_command(self, cmd: arx5.JointState):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ARXArm is not connected. You need to run `robot.connect()`."
            )
        self.joint_controller.set_joint_cmd(cmd)
        self.joint_controller.send_recv_once()

class ARXRobot:
    """
    A class for controlling a robot consisting of one or more ARX arms.
    
    TODO(villekuosmanen): add support for multithreaded controls
    """

    def __init__(
        self,
        config: ARXRobotConfig | None = None,
        # calibration_path: Path = ".cache/calibration/koch.pkl",
        **kwargs,
    ):
        if config is None:
            config = ARXRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        # self.calibration_path = Path(calibration_path)

        self.leader_arms = {}
        self.follower_arms = {}
        for key, arm_config in self.config.leader_arms.items():
            self.leader_arms[key] = ARXArm(arm_config)
        for key, arm_config in self.config.follower_arms.items():
            self.follower_arms[key] = ARXArm(arm_config)

        self.cameras = {}
        self.is_connected = False
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ARXRobot is already connected. Do not run `robot.connect()` twice."
            )

        # if not self.leader_arms and not self.follower_arms and not self.cameras:
        #     raise ValueError(
        #         "ARXRobot doesn't have any device to connect. See example of usage in docstring of the class."
        #     )

        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()

        # Run calibration process which begins by resetting all arms
        self.run_calibration()

        # Connect the cameras
        # for name in self.cameras:
        #     self.cameras[name].connect()

        self.is_connected = True

    def run_calibration(self):
        # TODO: there's no "calibration" here - rather we just reset the arm back to its home position. Is that ok?
        for name in self.follower_arms:
            print(f"Calibrating {name} follower arm.")
            self.follower_arms[name].reset()
        # for name in self.leader_arms:
        #     print(f"Calibrating {name} leader arm.")
        #     self.leader_arms[name].reset()

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ARXRobot is not connected. You need to run `robot.connect()`."
            )
        
        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            joint_state = self.leader_arms[name].get_state()
            leader_pos[name] = np.concatenate([joint_state.pos().copy(), np.array([joint_state.gripper_pos])])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t
        follower_goal_pos = {}
        for name in self.leader_arms:
            follower_goal_pos[name] = leader_pos[name]

            # Send action
            if name in self.follower_arms:
                before_fwrite_t = time.perf_counter()

                action = leader_pos[name]
                dof = self.config.follower_arms[name].dof

                cmd = arx5.JointState(dof)
                cmd.pos()[0:dof] = action[0:dof]
                cmd.gripper_pos = action[dof]    # TODO: does the gripper need special calibration?
                self.follower_arms[name].send_command(cmd)

                self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        # Early exit when recording data is not requested
        if not record_data:
            return

        # TODO(rcadene): Add velocity and other info
        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            joint_state = self.follower_arms[name].get_state()
            follower_pos[name] = np.concatenate([joint_state.pos().copy(), np.array([joint_state.gripper_pos])])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = np.concatenate(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = np.concatenate(action)

        # Capture images from cameras
        # images = {}
        # for name in self.cameras:
        #     before_camread_t = time.perf_counter()
        #     images[name] = self.cameras[name].async_read()
        #     self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
        #     self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        # for name in self.cameras:
        #     obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])

        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ARXRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            joint_state = self.follower_arms[name].get_state()
            follower_pos[name] = np.concatenate([joint_state.pos().copy(), np.array([joint_state.gripper_pos])])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = np.concatenate(state)

        # Capture images from cameras
        # images = {}
        # for name in self.cameras:
        #     before_camread_t = time.perf_counter()
        #     images[name] = self.cameras[name].async_read()
        #     self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
        #     self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict = {}
        obs_dict["observation.state"] = state
        # for name in self.cameras:
        #     obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])
        return obs_dict

    def send_action(self, action: np.ndarray):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ARXRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        follower_goal_pos = {}
        for name in self.follower_arms:
            if name in self.follower_arms:
                to_idx += len(self.config.follower_arms[name].dof)
                cmd = arx5.JointState(self.config.follower_arms[name].dof)
                cmd.pos()[0:self.config.follower_arms[name].dof] = action[from_idx:to_idx]
                cmd.gripper_pos = action[to_idx]    # TODO: does the gripper need special calibration?
                follower_goal_pos[name] = cmd
                
                from_idx = to_idx + 1

        for name in self.follower_arms:
            self.follower_arms[name].send_command(follower_goal_pos[name])

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ARXRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        # for name in self.cameras:
        #     self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
