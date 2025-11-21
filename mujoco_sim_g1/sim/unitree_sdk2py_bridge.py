import sys
import threading
from typing import Dict, Tuple

import glfw
from loguru import logger
import mujoco
import numpy as np
import pygame
import scipy.spatial.transform
from termcolor import colored
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__WirelessController_,
    unitree_hg_msg_dds__HandCmd_ as HandCmd_default,
    unitree_hg_msg_dds__HandState_ as HandState_default,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, OdoState_
    HAS_ODOSTATE = True
except ImportError:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
    OdoState_ = None
    HAS_ODOSTATE = False
    print("Warning: OdoState_ not available in unitree_sdk2py")


class UnitreeSdk2Bridge:
    """
    This class is responsible for bridging the Unitree SDK2 with the Gr00t environment.
    It is responsible for sending and receiving messages to and from the Unitree SDK2.
    Both the body and hand are supported.
    """

    def __init__(self, config):
        # Note that we do not give the mjdata and mjmodel to the UnitreeSdk2Bridge.
        # It is unsafe and would be unflexible if we use a hand-plugged robot model

        robot_type = config["ROBOT_TYPE"]
        if "g1" in robot_type or "h1-2" in robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
                unitree_hg_msg_dds__LowCmd_,
                unitree_hg_msg_dds__LowState_ as LowState_default,
            )
            try:
                from unitree_sdk2py.idl.default import unitree_hg_msg_dds__OdoState_ as OdoState_default
            except ImportError:
                OdoState_default = None
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif "h1" == robot_type or "go2" == robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__LowCmd_,
                unitree_go_msg_dds__LowState_ as LowState_default,
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_, LowCmd_, LowState_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        else:
            raise ValueError(f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', or 'go2'.")

        self.num_body_motor = config["NUM_MOTORS"]
        self.num_hand_motor = config.get("NUM_HAND_MOTORS", 0)
        self.use_sensor = config["USE_SENSOR"]

        self.have_imu_ = False
        self.have_frame_sensor_ = False
        # if self.use_sensor:
        #     MOTOR_SENSOR_NUM = 3
        #     self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        #     # Check sensor
        #     for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
        #         name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
        #         if name == "imu_quat":
        #             self.have_imu_ = True
        #         if name == "frame_pos":
        #             self.have_frame_sensor_ = True

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        # Only create odo_state for supported robot types (if available)
        if ("g1" in robot_type or "h1-2" in robot_type) and HAS_ODOSTATE and OdoState_default:
            self.odo_state = OdoState_default()
            self.odo_state_puber = ChannelPublisher("rt/odostate", OdoState_)
            self.odo_state_puber.Init()
        else:
            self.odo_state = None
            self.odo_state_puber = None
        self.torso_imu_state = IMUState_default()
        self.torso_imu_puber = ChannelPublisher("rt/secondary_imu", IMUState_)
        self.torso_imu_puber.Init()

        self.left_hand_state = HandState_default()
        self.left_hand_state_puber = ChannelPublisher("rt/dex3/left/state", HandState_)
        self.left_hand_state_puber.Init()
        self.right_hand_state = HandState_default()
        self.right_hand_state_puber = ChannelPublisher("rt/dex3/right/state", HandState_)
        self.right_hand_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)

        self.left_hand_cmd = HandCmd_default()
        self.left_hand_cmd_suber = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
        self.left_hand_cmd_suber.Init(self.LeftHandCmdHandler, 1)
        self.right_hand_cmd = HandCmd_default()
        self.right_hand_cmd_suber = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
        self.right_hand_cmd_suber.Init(self.RightHandCmdHandler, 1)

        self.low_cmd_lock = threading.Lock()
        self.left_hand_cmd_lock = threading.Lock()
        self.right_hand_cmd_lock = threading.Lock()

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            "rt/wirelesscontroller", WirelessController_
        )
        self.wireless_controller_puber.Init()

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        self.joystick = None

        # Store config for initialization
        self.config = config
        
        self.reset()
        
        # Initialize motors with default KP/KD from config to make robot stiff at startup
        self._initialize_motor_defaults()

    def _initialize_motor_defaults(self):
        """Initialize motor commands with default KP/KD and joint positions"""
        if "MOTOR_KP" in self.config and "MOTOR_KD" in self.config:
            motor_kp = self.config["MOTOR_KP"]
            motor_kd = self.config["MOTOR_KD"]
            default_dof_angles = self.config.get("DEFAULT_DOF_ANGLES", [0.0] * self.num_body_motor)
            
            for i in range(min(self.num_body_motor, len(motor_kp))):
                self.low_cmd.motor_cmd[i].kp = motor_kp[i]
                self.low_cmd.motor_cmd[i].kd = motor_kd[i]
                self.low_cmd.motor_cmd[i].q = default_dof_angles[i] if i < len(default_dof_angles) else 0.0
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].tau = 0.0
            
            print(f"✓ Initialized {self.num_body_motor} motors with default KP/KD gains")
        else:
            print("⚠ Warning: MOTOR_KP/MOTOR_KD not found in config, robot will be limp at startup")

    def reset(self):
        with self.low_cmd_lock:
            self.low_cmd_received = False
            self.new_low_cmd = False
        with self.left_hand_cmd_lock:
            self.left_hand_cmd_received = False
            self.new_left_hand_cmd = False
        with self.right_hand_cmd_lock:
            self.right_hand_cmd_received = False
            self.new_right_hand_cmd = False

    def LowCmdHandler(self, msg):
        with self.low_cmd_lock:
            self.low_cmd = msg
            self.low_cmd_received = True
            self.new_low_cmd = True

    def LeftHandCmdHandler(self, msg):
        with self.left_hand_cmd_lock:
            self.left_hand_cmd = msg
            self.left_hand_cmd_received = True
            self.new_left_hand_cmd = True

    def RightHandCmdHandler(self, msg):
        with self.right_hand_cmd_lock:
            self.right_hand_cmd = msg
            self.right_hand_cmd_received = True
            self.new_right_hand_cmd = True

    def cmd_received(self):
        with self.low_cmd_lock:
            low_cmd_received = self.low_cmd_received
        with self.left_hand_cmd_lock:
            left_hand_cmd_received = self.left_hand_cmd_received
        with self.right_hand_cmd_lock:
            right_hand_cmd_received = self.right_hand_cmd_received
        return low_cmd_received or left_hand_cmd_received or right_hand_cmd_received

    def PublishLowState(self, obs: Dict[str, any]):
        # publish body state
        if self.use_sensor:
            raise NotImplementedError("Sensor data is not implemented yet.")
        else:
            for i in range(self.num_body_motor):
                self.low_state.motor_state[i].q = obs["body_q"][i]
                self.low_state.motor_state[i].dq = obs["body_dq"][i]
                self.low_state.motor_state[i].ddq = obs["body_ddq"][i]
                self.low_state.motor_state[i].tau_est = obs["body_tau_est"][i]

        if self.use_sensor and self.have_frame_sensor_:
            raise NotImplementedError("Frame sensor data is not implemented yet.")
        else:
            # Get data from ground truth
            # Publish odo state only if available
            if self.odo_state is not None:
                self.odo_state.position[:] = obs["floating_base_pose"][:3]
                self.odo_state.linear_velocity[:] = obs["floating_base_vel"][:3]
                self.odo_state.orientation[:] = obs["floating_base_pose"][3:7]
                self.odo_state.angular_velocity[:] = obs["floating_base_vel"][3:6]
            # quaternion: w, x, y, z
            self.low_state.imu_state.quaternion[:] = obs["floating_base_pose"][3:7]
            # angular velocity
            self.low_state.imu_state.gyroscope[:] = obs["floating_base_vel"][3:6]
            # linear acceleration
            self.low_state.imu_state.accelerometer[:] = obs["floating_base_acc"][:3]

            self.torso_imu_state.quaternion[:] = obs["secondary_imu_quat"]
            self.torso_imu_state.gyroscope[:] = obs["secondary_imu_vel"][3:6]

        # acceleration: x, y, z (only available when frame sensor is enabled)
        if self.have_frame_sensor_:
            raise NotImplementedError("Frame sensor data is not implemented yet.")
        self.low_state.tick = int(obs["time"] * 1e3)
        self.low_state_puber.Write(self.low_state)

        # Publish odo state only if available
        if self.odo_state is not None and self.odo_state_puber is not None:
            self.odo_state.tick = int(obs["time"] * 1e3)
            self.odo_state_puber.Write(self.odo_state)

        self.torso_imu_puber.Write(self.torso_imu_state)

        # publish hand state
        for i in range(self.num_hand_motor):
            self.left_hand_state.motor_state[i].q = obs["left_hand_q"][i]
            self.left_hand_state.motor_state[i].dq = obs["left_hand_dq"][i]
        self.left_hand_state_puber.Write(self.left_hand_state)

        for i in range(self.num_hand_motor):
            self.right_hand_state.motor_state[i].q = obs["right_hand_q"][i]
            self.right_hand_state.motor_state[i].dq = obs["right_hand_dq"][i]
        self.right_hand_state_puber.Write(self.right_hand_state)

    def GetAction(self) -> Tuple[np.ndarray, bool, bool]:
        with self.low_cmd_lock:
            body_q = [self.low_cmd.motor_cmd[i].q for i in range(self.num_body_motor)]
        with self.left_hand_cmd_lock:
            left_hand_q = [self.left_hand_cmd.motor_cmd[i].q for i in range(self.num_hand_motor)]
        with self.right_hand_cmd_lock:
            right_hand_q = [self.right_hand_cmd.motor_cmd[i].q for i in range(self.num_hand_motor)]
        with self.low_cmd_lock and self.left_hand_cmd_lock and self.right_hand_cmd_lock:
            is_new_action = self.new_low_cmd and self.new_left_hand_cmd and self.new_right_hand_cmd
            if is_new_action:
                self.new_low_cmd = False
                self.new_left_hand_cmd = False
                self.new_right_hand_cmd = False

        return (
            np.concatenate([body_q[:-7], left_hand_q, body_q[-7:], right_hand_q]),
            self.cmd_received(),
            is_new_action,
        )

    def PublishWirelessController(self):
        if self.joystick is not None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
            key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
            key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
            key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
            key_state[self.key_map["R2"]] = self.joystick.get_axis(self.axis_id["RT"]) > 0
            key_state[self.key_map["L2"]] = self.joystick.get_axis(self.axis_id["LT"]) > 0
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 3,  # Right stick axis x
                    "RY": 4,  # Right stick axis y
                    "LT": 2,  # Left trigger
                    "RT": 5,  # Right trigger
                    "DX": 6,  # Directional pad x
                    "DY": 7,  # Directional pad y
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 4,
                    "RB": 5,
                    "SELECT": 6,
                    "START": 7,
                    "XBOX": 8,
                    "LSB": 9,
                    "RSB": 10,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 2,  # Right stick axis x
                    "RY": 3,  # Right stick axis y
                    "LT": 4,  # Left trigger
                    "RT": 5,  # Right trigger
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 9,
                    "RB": 10,
                    "SELECT": 4,
                    "START": 6,
                    "XBOX": 5,
                    "LSB": 7,
                    "RSB": 8,
                    "DYU": 11,
                    "DYD": 12,
                    "DXL": 13,
                    "DXR": 14,
                }
            else:
                print("Unsupported OS. ")

        elif js_type == "switch":
            # Yuanhang: may differ for different OS, need to be checked
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def PrintSceneInformation(self):
        print(" ")
        logger.info(colored("<<------------- Link ------------->>", "green"))
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                logger.info(f"link_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Joint ------------->>", "green"))
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                logger.info(f"joint_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Actuator ------------->>", "green"))
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                logger.info(f"actuator_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Sensor ------------->>", "green"))
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name:
                logger.info(
                    f"sensor_index: {index}, name: {name}, dim: {self.mj_model.sensor_dim[i]}"
                )
            index = index + self.mj_model.sensor_dim[i]
        print(" ")


class ElasticBand:
    """
    ref: https://github.com/unitreerobotics/unitree_mujoco
    """

    def __init__(self):
        self.kp_pos = 10000
        self.kd_pos = 1000
        self.kp_ang = 1000
        self.kd_ang = 10
        self.point = np.array([0, 0, 1])
        self.length = 0
        self.enable = True

    def Advance(self, pose):
        """
        Args:
          pose: 13D array containing:
               - pose[0:3]: position in world frame
               - pose[3:7]: quaternion [w,x,y,z] in world frame
               - pose[7:10]: linear velocity in world frame
               - pose[10:13]: angular velocity in world frame
        Returns:
          np.ndarray: 6D vector [fx, fy, fz, tx, ty, tz]
        """
        pos = pose[0:3]
        quat = pose[3:7]
        lin_vel = pose[7:10]
        ang_vel = pose[10:13]

        δx = self.point - pos
        f = self.kp_pos * (δx + np.array([0, 0, self.length])) + self.kd_pos * (0 - lin_vel)

        # --- Orientation PD control for torque ---
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])  # reorder to [x,y,z,w] for scipy
        rot = scipy.spatial.transform.Rotation.from_quat(quat)
        rotvec = rot.as_rotvec()  # axis-angle error
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel

        return np.concatenate([f, torque])

    def MujuocoKeyCallback(self, key):
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable

    def handle_keyboard_button(self, key):
        if key == "9":
            self.enable = not self.enable
