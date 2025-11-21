import logging
import time
import struct
from functools import cached_property
from typing import Any
from pathlib import Path

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.calibration_gui import RangeFinderGUI
from lerobot.motors.feetech import (
    FeetechMotorsBus,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
import json
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_unitree_g1 import UnitreeG1Config

import numpy as np
import threading
import time
from enum import IntEnum
import sys
import select
import termios
import tty
from collections import deque

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState  # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as go_LowCmd, LowState_ as go_LowState  # idl for h1
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_


from typing import Union
import numpy as np
import time
import torch
import onnxruntime as ort

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)

from scipy.spatial.transform import Rotation as R

import struct

import yaml

from typing import Union

import logging_mp

from lerobot.robots.unitree_g1.robot_kinematic_processor import G1_29_ArmIK

import torch

logger_mp = logging_mp.get_logger(__name__)

kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35
H1_Num_Motors = 20


class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None
        self.tau_est = None  # Estimated torque
        self.temperature = None  # Motor temperature


class IMUState:
    def __init__(self):
        self.quaternion = None  # [w, x, y, z]
        self.gyroscope = None  # [x, y, z] angular velocity (rad/s)
        self.accelerometer = None  # [x, y, z] linear acceleration (m/s²)
        self.rpy = None  # [roll, pitch, yaw] (rad)
        self.temperature = None  # IMU temperature


class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]
        self.imu_state = IMUState()
        self.wireless_remote = None  # Raw wireless remote data

class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

#eventually observations should be everything: motor torques etc etc 
#motor class for unitree? 
#TODO: camera, sim 
class UnitreeG1(Robot):

    config_class = UnitreeG1Config
    name = "unitree_g1"

    def __init__(self, config: UnitreeG1Config):
        super().__init__(config)
        
        logger_mp.info("Initialize UnitreeG1...")

        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = config.motion_mode
        self.simulation_mode = config.simulation_mode
        self.kp_high = config.kp_high
        self.kd_high = config.kd_high
        self.kp_low = config.kp_low
        self.kd_low = config.kd_low
        self.kp_wrist = config.kp_wrist
        self.kd_wrist = config.kd_wrist

        self.all_motor_q = config.all_motor_q
        self.arm_velocity_limit = config.arm_velocity_limit
        self.control_dt = config.control_dt

        self._speed_gradual_max = config.speed_gradual_max
        self._gradual_start_time = config.gradual_start_time
        self._gradual_time = config.gradual_time

        self.freeze_body = config.freeze_body
        self.gravity_compensation = config.gravity_compensation


        self.calibrated = False

        self.calibrate()

        self.arm_ik = G1_29_ArmIK()


        # initialize lowcmd publisher and lowstate subscriber
        if self.simulation_mode:
            ChannelFactoryInitialize(0, "lo")
            
            # Launch MuJoCo simulation environment
            logger_mp.info("Launching MuJoCo simulation environment...")
            from lerobot.envs.factory import make_env
            self.mujoco_env = make_env("lerobot/unitree-g1-mujoco", trust_remote_code=True)
            logger_mp.info("MuJoCo environment launched successfully!")
        else:
            ChannelFactoryInitialize(0)


        if not self.config.simulation_mode:
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()

            status, result = self.msc.CheckMode()
            print(status, result)
            #check if result name first
            if result is not None and "name" in result:
                while result["name"]:
                    self.msc.ReleaseMode()
                    status, result = self.msc.CheckMode()
                    print(status, result)
                    time.sleep(1)

        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[UnitreeG1] Waiting to subscribe dds...")
        logger_mp.info("[UnitreeG1] Subscribe dds ok.")

        # initialize audio client for LED, TTS, and audio playback


        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()
        print(self.msg)

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...\n")

        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q = self.all_motor_q[id]
        #print current motor q, kp, kd



        if config.audio_client:
            self.audio_client = AudioClient()
            self.audio_client.SetTimeout(10.0)
            self.audio_client.Init()
            logger_mp.info("[UnitreeG1] Audio client initialized!")

        logger_mp.info("Lock OK!\n") #motors are not locked x
        # for i in range(10000):
        #     print(self.get_current_motor_q())
        #     time.sleep(0.05) 
        
        # Initialize control flags BEFORE starting threads
        self.keyboard_thread = None
        self.keyboard_running = False
        self.locomotion_thread = None
        self.locomotion_running = False
        self.motion_imitation_thread = None
        self.motion_imitation_running = False
        
        # Initialize publish thread ONLY if not using motion imitation or locomotion
        # (those modes handle their own motor commands and publishing)
        self.publish_thread = None
        self.ctrl_lock = threading.Lock()
        if not config.motion_imitation_control and not config.locomotion_control:
            self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
            self.publish_thread.daemon = True
            self.publish_thread.start()
            logger_mp.info("Arm control publish thread started")

        # Load locomotion policy if enabled
        self.policy = None
        self.policy_type = None  # 'torchscript', 'onnx', or 'motion_imitation'
        self.motion_loader = None
        
        if config.motion_imitation_control:
            # Motion imitation mode (dance, etc.)
            if config.motion_file_path is None:
                raise ValueError("motion_imitation_control is True but motion_file_path is not set")
            
            logger_mp.info(f"Loading motion reference from {config.motion_file_path}")
            
            # Load motion file
            self.motion_loader = self.MotionLoader(config.motion_file_path, config.motion_fps)
            
            # Load ONNX policy (optional for now - can run in direct playback mode)
            if config.motion_policy_path and Path(config.motion_policy_path).exists():
                logger_mp.info(f"Loading motion imitation policy from {config.motion_policy_path}")
                self.policy = ort.InferenceSession(config.motion_policy_path)
                self.policy_type = 'motion_imitation'
                logger_mp.info("Motion imitation ONNX policy loaded successfully")
                logger_mp.info(f"ONNX input: {self.policy.get_inputs()[0].name}, shape: {self.policy.get_inputs()[0].shape}")
                logger_mp.info(f"ONNX output: {self.policy.get_outputs()[0].name}, shape: {self.policy.get_outputs()[0].shape}")
            else:
                logger_mp.info("Running in DIRECT PLAYBACK mode (no policy - just reference motion)")
                self.policy = None
                self.policy_type = 'motion_playback'
            
            # Initialize motion imitation variables
            self.motion_counter = 0
            self.motion_qj_all = np.zeros(29, dtype=np.float32)  # All 29 joints from robot
            self.motion_dqj_all = np.zeros(29, dtype=np.float32)
            self.motion_action = np.zeros(29, dtype=np.float32)  # 29D action output
            self.motion_obs = np.zeros(154, dtype=np.float32)  # 154D observation
            self.motion_elapsed_time = 0.0
            
            # Initialize motion and start
            self.init_motion_imitation()
            
        elif config.locomotion_control:
            if config.policy_path is None:
                raise ValueError("locomotion_control is True but policy_path is not set")
            
            logger_mp.info(f"Loading locomotion policy from {config.policy_path}")
            
            # Check file extension and load accordingly
            if config.policy_path.endswith('.pt'):
                logger_mp.info("Detected TorchScript (.pt) policy")
                self.policy = torch.jit.load(config.policy_path)
                self.policy_type = 'torchscript'
                logger_mp.info("TorchScript policy loaded successfully")
            elif config.policy_path.endswith('.onnx'):
                logger_mp.info("Detected ONNX (.onnx) policy")
                self.policy = ort.InferenceSession(config.policy_path)
                self.policy_type = 'onnx'
                logger_mp.info("ONNX policy loaded successfully")
                logger_mp.info(f"ONNX input: {self.policy.get_inputs()[0].name}, shape: {self.policy.get_inputs()[0].shape}")
                logger_mp.info(f"ONNX output: {self.policy.get_outputs()[0].name}, shape: {self.policy.get_outputs()[0].shape}")
            else:
                raise ValueError(f"Unsupported policy format: {config.policy_path}. Only .pt (TorchScript) and .onnx (ONNX) are supported.")
            
            # Initialize locomotion variables
            self.remote_controller = self.RemoteController()
            self.locomotion_counter = 0
            self.qj = np.zeros(config.num_locomotion_actions, dtype=np.float32)
            self.dqj = np.zeros(config.num_locomotion_actions, dtype=np.float32)
            self.locomotion_action = np.zeros(config.num_locomotion_actions, dtype=np.float32)
            self.locomotion_obs = np.zeros(config.num_locomotion_obs, dtype=np.float32)
            self.locomotion_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            # GR00T-specific variables (for ONNX policies with 29 joints)
            if self.policy_type == 'onnx':
                self.groot_qj_all = np.zeros(29, dtype=np.float32)  # All 29 joints
                self.groot_dqj_all = np.zeros(29, dtype=np.float32)
                self.groot_action = np.zeros(15, dtype=np.float32)  # 15D action (legs + waist)
                self.groot_obs_single = np.zeros(86, dtype=np.float32)  # 86D single frame observation
                self.groot_obs_history = deque(maxlen=6)  # 6-frame history buffer
                self.groot_obs_stacked = np.zeros(516, dtype=np.float32)  # 86D × 6 = 516D stacked observation
                self.groot_height_cmd = 0.74  # Default base height
                self.groot_orientation_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # roll, pitch, yaw
                
                # Initialize history with zeros
                for _ in range(6):
                    self.groot_obs_history.append(np.zeros(86, dtype=np.float32))
            
            # Start keyboard controls if in simulation mode
            if self.simulation_mode:
                logger_mp.info("Starting keyboard controls for simulation...")
                self.start_keyboard_controls()
            
            # Use different init based on policy type
            if self.policy_type == 'onnx':
                self.init_groot_locomotion()
            else:
                self.init_locomotion()
        elif self.simulation_mode:
            # Even without locomotion, provide keyboard feedback in sim
            logger_mp.info("Simulation mode active (locomotion disabled)")


        logger_mp.info("Initialize G1 OK!\n")

    def _subscribe_motor_state(self):
        while True:
            start_time = time.time()
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()
                
                # Capture motor states
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                    lowstate.motor_state[id].tau_est = msg.motor_state[id].tau_est
                    lowstate.motor_state[id].temperature = msg.motor_state[id].temperature
                
                # Capture IMU state
                lowstate.imu_state.quaternion = list(msg.imu_state.quaternion)
                lowstate.imu_state.gyroscope = list(msg.imu_state.gyroscope)
                lowstate.imu_state.accelerometer = list(msg.imu_state.accelerometer)
                lowstate.imu_state.rpy = list(msg.imu_state.rpy)
                lowstate.imu_state.temperature = msg.imu_state.temperature
                
                # Capture wireless remote data
                lowstate.wireless_remote = msg.wireless_remote
                
                self.lowstate_buffer.SetData(lowstate)

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))#maintina constant control dt
            time.sleep(sleep_time)
            

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        """Arm control thread - publishes commands for arms only.
        NOTE: This thread is NOT started when motion_imitation_control or locomotion_control is True.
        Those modes handle their own publishing."""
        if self.motion_mode:
            self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = 1.0

        while True:
            start_time = time.time()
            
            with self.ctrl_lock:
                arm_q_target = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit=self.arm_velocity_limit)

            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]

            # Zero out specific joints when in simulation mode
            if self.simulation_mode:
                # Waist joints
                self.msg.motor_cmd[G1_29_JointIndex.kWaistYaw].q = 0.0
                self.msg.motor_cmd[G1_29_JointIndex.kWaistPitch].q = 0.0
                # Wrist joints
                self.msg.motor_cmd[G1_29_JointIndex.kLeftWristPitch].q = 0.0
                self.msg.motor_cmd[G1_29_JointIndex.kLeftWristyaw].q = 0.0
                self.msg.motor_cmd[G1_29_JointIndex.kRightWristPitch].q = 0.0
                self.msg.motor_cmd[G1_29_JointIndex.kRightWristYaw].q = 0.0

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        """Set control target values q & tau of the left and right arm motors."""
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        """Return current dds mode machine."""
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        """Return current state q of all body motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])

    def get_current_dual_arm_q(self):
        """Return current state q of the left and right arm motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])

    def get_current_dual_arm_dq(self):
        """Return current state dq of the left and right arm motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])

    def ctrl_dual_arm_go_home(self):
        """Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero."""
        logger_mp.info("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            #self.q_target[G1_29_JointIndex.kLeftElbow] = 0.5
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = weight
                        time.sleep(0.02)
                logger_mp.info("[G1_29_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        """Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0."""
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        """set arms velocity to the maximum value immediately, instead of gradually increasing."""
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{G1_29_JointArmIndex(motor).name}.pos": float for motor in G1_29_JointArmIndex}

    def calibrate(self) -> None:
        self.calibration = json.load(open('src/lerobot/robots/unitree_g1/arm_calibration.json'))
        self.calibrated = True

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()
        logger_mp.info(f"{self} connected with {len(self.cameras)} camera(s).")

    def disconnect(self):
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Close MuJoCo environment if in simulation mode
        if self.simulation_mode and hasattr(self, 'mujoco_env'):
            logger_mp.info("Closing MuJoCo environment...")
            self.mujoco_env.close()
        
        logger_mp.info(f"{self} disconnected.")

    def get_full_robot_state(self) -> dict[str, Any]:
        """
        Get full robot state including IMU and extended motor data.
        
        Returns:
            dict with keys:
                - 'imu': dict containing IMU data (quaternion, gyroscope, accelerometer, rpy, temperature)
                - 'motors': list of dicts, one per motor, containing q, dq, tau_est, temperature
        """
        lowstate = self.lowstate_buffer.GetData()
        if lowstate is None:
            raise RuntimeError("No robot state available. Is the robot connected?")
        
        # Extract IMU data
        imu_data = {
            'quaternion': lowstate.imu_state.quaternion,  # [w, x, y, z]
            'gyroscope': lowstate.imu_state.gyroscope,  # [x, y, z] rad/s
            'accelerometer': lowstate.imu_state.accelerometer,  # [x, y, z] m/s²
            'rpy': lowstate.imu_state.rpy,  # [roll, pitch, yaw] rad
            'temperature': lowstate.imu_state.temperature,  # °C
        }
        
        # Extract motor data
        motors_data = []
        for i in range(G1_29_Num_Motors):
            motor = lowstate.motor_state[i]
            motors_data.append({
                'id': i,
                'q': motor.q,  # position (rad)
                'dq': motor.dq,  # velocity (rad/s)
                'tau_est': motor.tau_est,  # estimated torque (Nm)
                'temperature': motor.temperature[0] if isinstance(motor.temperature, (list, tuple)) else motor.temperature,  # °C
            })
        
        return {
            'imu': imu_data,
            'motors': motors_data,
        }

    def audio_control(self, command, volume: int = 80):
        """
        Unified audio/LED control function for the G1 robot.
        
        Args:
            command: Can be one of:
                - str: Text to speak via TTS
                - tuple[int, int, int]: RGB values (0-255) for LED control
                - str (path): Path to WAV file to play
            volume: Volume level 0-100 (default: 80)
        
        Examples:
            robot.audio_control("Hello world")  # TTS
            robot.audio_control((255, 0, 0))     # Red LED
            robot.audio_control("audio.wav")     # Play WAV file
        """
        # Set volume
        self.audio_client.SetVolume(volume)
        
        # Detect command type and execute
        if isinstance(command, tuple) and len(command) == 3:
            # LED control - RGB tuple
            r, g, b = command
            logger_mp.info(f"Setting LED to RGB({r}, {g}, {b})")
            self.audio_client.LedControl(r, g, b)
            
        elif isinstance(command, str):
            # Check if it's a file path
            if Path(command).exists():
                # Play WAV file
                logger_mp.info(f"Playing audio file: {command}")
                self._play_wav_file(command)
            else:
                # Text-to-speech
                logger_mp.info(f"Speaking: {command}")
                self.audio_client.TtsMaker(command, 0)  # 0 for English
        else:
            raise ValueError(
                f"Invalid command type: {type(command)}. "
                "Expected str (text/path) or tuple[int, int, int] (RGB)"
            )

    def _read_wav_file(self, filename: str):
        """Read WAV file and return PCM data as bytes."""
        with open(filename, 'rb') as f:
            def read(fmt):
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

            # Read RIFF header
            chunk_id, = read('<I')
            if chunk_id != 0x46464952:  # "RIFF"
                raise ValueError("Not a valid WAV file (invalid RIFF header)")

            _chunk_size, = read('<I')
            format_tag, = read('<I')
            if format_tag != 0x45564157:  # "WAVE"
                raise ValueError("Not a valid WAV file (invalid WAVE format)")

            # Read fmt chunk
            subchunk1_id, = read('<I')
            subchunk1_size, = read('<I')

            # Skip JUNK chunk if present
            if subchunk1_id == 0x4B4E554A:  # "JUNK"
                f.seek(subchunk1_size, 1)
                subchunk1_id, = read('<I')
                subchunk1_size, = read('<I')

            if subchunk1_id != 0x20746D66:  # "fmt "
                raise ValueError("Invalid fmt chunk")

            if subchunk1_size not in [16, 18]:
                raise ValueError(f"Unsupported fmt chunk size: {subchunk1_size}")

            audio_format, = read('<H')
            if audio_format != 1:
                raise ValueError(f"Only PCM format supported, got format {audio_format}")

            num_channels, = read('<H')
            sample_rate, = read('<I')
            _byte_rate, = read('<I')
            _block_align, = read('<H')
            bits_per_sample, = read('<H')

            if bits_per_sample != 16:
                raise ValueError(f"Only 16-bit samples supported, got {bits_per_sample}-bit")

            if sample_rate != 16000:
                raise ValueError(f"Sample rate must be 16000 Hz, got {sample_rate} Hz")

            if num_channels != 1:
                raise ValueError(f"Must be mono (1 channel), got {num_channels} channels")

            if subchunk1_size == 18:
                extra_size, = read('<H')
                if extra_size != 0:
                    f.seek(extra_size, 1)

            # Find data chunk
            while True:
                subchunk2_id, subchunk2_size = read('<II')
                if subchunk2_id == 0x61746164:  # "data"
                    break
                f.seek(subchunk2_size, 1)

            # Read PCM data
            raw_pcm = f.read(subchunk2_size)
            if len(raw_pcm) != subchunk2_size:
                raise ValueError("Failed to read full PCM data")

            return raw_pcm

    def _play_wav_file(self, filename: str, chunk_size: int = 96000):
        """
        Play a WAV file through the robot's speaker.
        
        Args:
            filename: Path to WAV file (must be 16kHz, mono, 16-bit PCM)
            chunk_size: Bytes per chunk (default: 96000 = ~3 seconds at 16kHz)
        """
        # Read WAV file
        pcm_data = self._read_wav_file(filename)
        
        stream_id = str(int(time.time() * 1000))
        app_name = "lerobot"
        offset = 0
        chunk_index = 0
        total_size = len(pcm_data)

        logger_mp.info(f"Playing audio: {total_size} bytes in {(total_size // chunk_size) + 1} chunks")

        # Send audio in chunks
        while offset < total_size:
            remaining = total_size - offset
            current_chunk_size = min(chunk_size, remaining)
            chunk = pcm_data[offset:offset + current_chunk_size]

            # Send chunk
            ret_code, _ = self.audio_client.PlayStream(app_name, stream_id, list(chunk))
            if ret_code != 0:
                logger_mp.error(f"Failed to send chunk {chunk_index}, return code: {ret_code}")
                break
            else:
                logger_mp.debug(f"Sent chunk {chunk_index}/{(total_size // chunk_size)}")

            offset += current_chunk_size
            chunk_index += 1
            time.sleep(1.0)  # Wait between chunks

        # Calculate playback duration
        duration_seconds = len(pcm_data) / (16000 * 2)  # 16kHz, 16-bit (2 bytes)
        logger_mp.info(f"Audio playback will take ~{duration_seconds:.1f} seconds")

    def get_observation(self) -> dict[str, Any]:
        obs_array = self.get_current_dual_arm_q()
        obs_dict = {f"{G1_29_JointArmIndex(motor).name}.pos": val for motor, val in zip(G1_29_JointArmIndex, obs_array, strict=True)}
        
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger_mp.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return obs_dict

    @property
    def is_calibrated(self) -> bool:
        return self.calibrated

    @property
    def is_connected(self) -> bool:
        return all(cam.is_connected for cam in self.cameras.values())
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{G1_29_JointArmIndex(motor).name}.pos": float for motor in G1_29_JointArmIndex}
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        #need a any to any teleoperator solution. i wanna teleoperate a horse with a shoe. action
        #to action mapping, when you do teleoperate. the keys that are left empty are just set to 0 
        #also what would be fun is finding all sorts of robots and adding them to lerobot, see if people do the same.
        #then teleop them wiuth the glove hehe
        #then we get ALL THE DATA
        if self.is_calibrated:
            uncalibrated_action = action.copy()
            action = self.invert_calibration(action)
            #if an action was 0.5 write 0 in its place 
            for key, value in uncalibrated_action.items():
                if value == 0.5:
                    action[key] = 0.0
            #check if action is within bounds
            for key, value in action.items():
                if value < self.calibration[key]["range_min"] or value > self.calibration[key]["range_max"]:
                    raise ValueError(f"Action value {value} for {key} is out of bounds, actions are not normalized")
        if self.freeze_body:
            arm_joint_indices = set(range(15, 29))  # 15–28 are arms
            for jid in G1_29_JointIndex:
                if jid.value not in arm_joint_indices:
                    self.msg.motor_cmd[jid].mode = 1
                    self.msg.motor_cmd[jid].q = 0.0
                    self.msg.motor_cmd[jid].dq = 0.0
                    self.msg.motor_cmd[jid].tau = 0.0

        action_np = np.stack([v for v in action.values()])
        #action_np is just zeros    
        #action_np = np.zeros(14)
        #print(action_np)
        #exit()
        if self.gravity_compensation:
            tau = self.arm_ik.solve_tau(action_np)
        else:
            tau = np.zeros(14)
        
        self.ctrl_dual_arm(action_np, tau)

    def apply_calibration(self, action: dict[str, float]) -> dict[str, float]:
        """Map motor ranges to [0, 1]."""
        calibrated = {}
        for key, value in action.items():
            value = float(value.item())

            cal = self.calibration[key]
            mn, mx = cal["range_min"], cal["range_max"]

            if mx == mn:
                norm = 0.0
            else:
                norm = (value - mn) / (mx - mn)
                norm = max(0.0, min(1.0, norm))

            # Round to 5 decimal places to avoid floating point precision issues
            calibrated[key] = round(norm, 5)

        return calibrated


    def invert_calibration(self, action: dict[str, float]) -> dict[str, float]:
        """Map [0, 1] actions back to motor ranges."""
        calibrated = {}
        for key, value in action.items():
            value = float(value.item()) if hasattr(value, "item") else float(value)

            cal = self.calibration[key]
            mn, mx = cal["range_min"], cal["range_max"]

            # inverse mapping
            real_val = mn + value * (mx - mn)
            
            # Round to 5 decimal places to avoid floating point precision issues
            calibrated[key] = round(real_val, 5)

        return calibrated

    ###################LOCOMOTION CONTROL###################

    def locomotion_create_damping_cmd(self):
        """Set all motors to damping mode (kp=0, kd=8)."""
        size = len(self.msg.motor_cmd)
        for i in range(size):
            self.msg.motor_cmd[i].q = 0
            self.msg.motor_cmd[i].qd = 0
            self.msg.motor_cmd[i].kp = 0
            self.msg.motor_cmd[i].kd = 8
            self.msg.motor_cmd[i].tau = 0
        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)

    def locomotion_create_zero_cmd(self):
        """Set all motors to zero torque mode."""
        size = len(self.msg.motor_cmd)
        for i in range(size):
            self.msg.motor_cmd[i].q = 0
            self.msg.motor_cmd[i].qd = 0
            self.msg.motor_cmd[i].kp = 0
            self.msg.motor_cmd[i].kd = 0
            self.msg.motor_cmd[i].tau = 0
        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)

    def locomotion_zero_torque_state(self):
        """Enter zero torque state."""
        logger_mp.info("Enter zero torque state.")
        self.locomotion_create_zero_cmd()
        time.sleep(self.config.locomotion_control_dt)

    def locomotion_move_to_default_pos(self):
        """Move robot legs to default standing position over 2 seconds (arms are not moved)."""
        logger_mp.info("Moving legs to default locomotion pos.")
        total_time = 2.0
        num_step = int(total_time / self.config.locomotion_control_dt)
        
        # Only control legs, not arms
        dof_idx = self.config.leg_joint2motor_idx
        kps = self.config.locomotion_kps
        kds = self.config.locomotion_kds
        default_pos = np.array(self.config.default_leg_angles, dtype=np.float32)
        dof_size = len(dof_idx)
        
        # Get current lowstate
        lowstate = self.lowstate_buffer.GetData()
        if lowstate is None:
            logger_mp.error("Cannot get lowstate for locomotion")
            return
        
        # Record the current leg positions
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = lowstate.motor_state[dof_idx[i]].q
        
        # Move legs to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.msg.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.msg.motor_cmd[motor_idx].qd = 0
                self.msg.motor_cmd[motor_idx].kp = kps[j]
                self.msg.motor_cmd[motor_idx].kd = kds[j]
                self.msg.motor_cmd[motor_idx].tau = 0
            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
            time.sleep(self.config.locomotion_control_dt)
        logger_mp.info("Reached default locomotion position (legs only)")

    def locomotion_default_pos_state(self):
        """Hold default leg position for 2 seconds (arms are not controlled)."""
        logger_mp.info("Enter default pos state - holding legs for 2 seconds")
        
        # Only control legs, not arms
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.msg.motor_cmd[motor_idx].q = self.config.default_leg_angles[i]
            self.msg.motor_cmd[motor_idx].qd = 0
            self.msg.motor_cmd[motor_idx].kp = self.config.locomotion_kps[i]
            self.msg.motor_cmd[motor_idx].kd = self.config.locomotion_kds[i]
            self.msg.motor_cmd[motor_idx].tau = 0
            
        # Hold leg position for 2 seconds
        hold_time = 2.0
        num_steps = int(hold_time / self.config.locomotion_control_dt)
        for _ in range(num_steps):
            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
            time.sleep(self.config.locomotion_control_dt)
        logger_mp.info("Finished holding default leg position")
    

    class RemoteController:
        def __init__(self):
            self.lx = 0
            self.ly = 0
            self.rx = 0
            self.ry = 0
            self.button = [0] * 16

        def set(self, data):
            # wireless_remote
            keys = struct.unpack("H", data[2:4])[0]
            for i in range(16):
                self.button[i] = (keys & (1 << i)) >> i
            self.lx = struct.unpack("f", data[4:8])[0]
            self.rx = struct.unpack("f", data[8:12])[0]
            self.ry = struct.unpack("f", data[12:16])[0]
            self.ly = struct.unpack("f", data[20:24])[0]
    
    class MotionLoader:
        """Load and interpolate motion from CSV file for motion imitation."""
        def __init__(self, motion_file: str, fps: float = 60.0):
            """Load motion from CSV file.
            
            CSV format: [root_pos(3), root_quat_xyzw(4), joint_dof(29)] per row
            """
            self.dt = 1.0 / fps
            
            # Load CSV
            data = np.loadtxt(motion_file, delimiter=',')
            self.num_frames = data.shape[0]
            self.duration = self.num_frames * self.dt
            
            # Split data
            self.root_positions = data[:, 0:3]  # (N, 3)
            self.root_quaternions_xyzw = data[:, 3:7]  # (N, 4) [x, y, z, w]
            self.dof_positions = data[:, 7:]  # (N, 29)
            
            # Compute velocities (finite differences)
            self.dof_velocities = np.diff(self.dof_positions, axis=0, prepend=self.dof_positions[0:1]) / self.dt
            
            # Current playback state
            self.current_time = 0.0
            self.index_0 = 0
            self.index_1 = 0
            self.blend = 0.0
            
            logger_mp.info(f"MotionLoader: Loaded {self.num_frames} frames, duration={self.duration:.2f}s")
            
        def update(self, time: float):
            """Update motion to specific time (loops at duration)."""
            self.current_time = time % self.duration  # Loop
            phase = self.current_time / self.duration
            
            self.index_0 = int(phase * (self.num_frames - 1))
            self.index_1 = min(self.index_0 + 1, self.num_frames - 1)
            self.blend = (self.current_time - self.index_0 * self.dt) / self.dt
            
        def get_joint_pos(self) -> np.ndarray:
            """Get interpolated joint positions (29D)."""
            return self.dof_positions[self.index_0] * (1 - self.blend) + \
                   self.dof_positions[self.index_1] * self.blend
        
        def get_joint_vel(self) -> np.ndarray:
            """Get interpolated joint velocities (29D)."""
            return self.dof_velocities[self.index_0] * (1 - self.blend) + \
                   self.dof_velocities[self.index_1] * self.blend
        
        def get_root_quat_wxyz(self) -> np.ndarray:
            """Get interpolated root quaternion [w, x, y, z]."""
            # Spherical linear interpolation (SLERP)
            q0 = self.root_quaternions_xyzw[self.index_0]  # [x, y, z, w]
            q1 = self.root_quaternions_xyzw[self.index_1]
            
            # Convert to scipy format [x, y, z, w]
            r0 = R.from_quat(q0)
            r1 = R.from_quat(q1)
            
            # SLERP
            key_times = [0, 1]
            key_rots = R.from_quat([q0, q1])
            slerp = R.from_quat(key_rots.as_quat())  # Simplified - just use linear for now
            
            # Linear interpolation for simplicity
            quat_xyzw = q0 * (1 - self.blend) + q1 * self.blend
            # Normalize
            quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
            
            # Convert to [w, x, y, z]
            return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    
    def locomotion_get_gravity_orientation(self, quaternion):
        """Get gravity orientation from quaternion."""
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]
        
        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        
        return gravity_orientation
    
    def locomotion_transform_imu_data(self, waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
        """Transform IMU data from torso to pelvis frame."""
        RzWaist = R.from_euler("z", waist_yaw).as_matrix()
        R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
        R_pelvis = np.dot(R_torso, RzWaist.T)
        w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
        return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w
    
    def locomotion_run(self):
        """Main locomotion policy loop - runs policy and sends leg commands."""
        self.locomotion_counter += 1
        
        # Get current lowstate
        lowstate = self.lowstate_buffer.GetData()
        if lowstate is None:
            return
        
        # Update remote controller from lowstate
        if lowstate.wireless_remote is not None:
            self.remote_controller.set(lowstate.wireless_remote)
        else:
            # Default to zero commands if no remote data
            self.remote_controller.lx = 0.0
            self.remote_controller.ly = 0.0
            self.remote_controller.rx = 0.0
            self.remote_controller.ry = 0.0
        
        # Get the current joint position and velocity (LEGS ONLY)
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = lowstate.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = lowstate.motor_state[self.config.leg_joint2motor_idx[i]].dq
        
        # Get IMU data
        quat = lowstate.imu_state.quaternion
        ang_vel = np.array([lowstate.imu_state.gyroscope], dtype=np.float32)
        
        if self.config.locomotion_imu_type == "torso":
            # Transform IMU data from torso to pelvis frame
            waist_yaw = lowstate.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = lowstate.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = self.locomotion_transform_imu_data(waist_yaw, waist_yaw_omega, quat, ang_vel)
        
        # Create observation
        gravity_orientation = self.locomotion_get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - np.array(self.config.default_leg_angles)) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        
        # Calculate phase
        period = 0.8
        count = self.locomotion_counter * self.config.locomotion_control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        
        # Get velocity commands from remote controller (only if NOT in simulation mode)
        # In simulation mode, keyboard controls set self.locomotion_cmd directly
        if not self.simulation_mode:
            self.locomotion_cmd[0] = self.remote_controller.ly
            self.locomotion_cmd[1] = self.remote_controller.lx * -1
            self.locomotion_cmd[2] = self.remote_controller.rx * -1
            
            # Debug: print remote controller values every 50 iterations (~1 second at 50Hz)
            if self.locomotion_counter % 50 == 0:
                logger_mp.debug(f"Remote controller - lx:{self.remote_controller.lx:.2f}, ly:{self.remote_controller.ly:.2f}, rx:{self.remote_controller.rx:.2f}")
        
        # Build observation vector
        num_actions = self.config.num_locomotion_actions
        self.locomotion_obs[:3] = ang_vel
        self.locomotion_obs[3:6] = gravity_orientation
        self.locomotion_obs[6:9] = self.locomotion_cmd * np.array(self.config.cmd_scale) * np.array(self.config.max_cmd)
        self.locomotion_obs[9 : 9 + num_actions] = qj_obs
        self.locomotion_obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.locomotion_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.locomotion_action
        self.locomotion_obs[9 + num_actions * 3] = sin_phase
        self.locomotion_obs[9 + num_actions * 3 + 1] = cos_phase
        
        # Get action from policy network
        obs_tensor = torch.from_numpy(self.locomotion_obs).unsqueeze(0)
        
        if self.policy_type == 'torchscript':
            # TorchScript inference
            self.locomotion_action = self.policy(obs_tensor).detach().numpy().squeeze()
        elif self.policy_type == 'onnx':
            # ONNX inference
            ort_inputs = {self.policy.get_inputs()[0].name: obs_tensor.cpu().numpy()}
            ort_outs = self.policy.run(None, ort_inputs)
            self.locomotion_action = ort_outs[0].squeeze()
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
        
        # Transform action to target joint positions
        target_dof_pos = np.array(self.config.default_leg_angles) + self.locomotion_action * self.config.locomotion_action_scale
        
        # Send commands to LEG motors only
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.msg.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.msg.motor_cmd[motor_idx].qd = 0
            self.msg.motor_cmd[motor_idx].kp = self.config.locomotion_kps[i]
            self.msg.motor_cmd[motor_idx].kd = self.config.locomotion_kds[i]
            self.msg.motor_cmd[motor_idx].tau = 0
        
        # Hold WAIST motors at 0 (indices 12, 13, 14 = WaistYaw, WaistRoll, WaistPitch)
        waist_indices = self.config.arm_waist_joint2motor_idx[:3]  # First 3 are waist
        for i, motor_idx in enumerate(waist_indices):
            self.msg.motor_cmd[motor_idx].q = 0.0
            self.msg.motor_cmd[motor_idx].qd = 0
            self.msg.motor_cmd[motor_idx].kp = self.config.locomotion_arm_waist_kps[i]
            self.msg.motor_cmd[motor_idx].kd = self.config.locomotion_arm_waist_kds[i]
            self.msg.motor_cmd[motor_idx].tau = 0
        
        # Send command
        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)
    
    def groot_locomotion_run(self):
        """GR00T-style locomotion policy loop for ONNX policies - reads all 29 joints, outputs 15D action."""
        self.locomotion_counter += 1
        
        # Get current lowstate
        lowstate = self.lowstate_buffer.GetData()
        if lowstate is None:
            return
        
        # Update remote controller from lowstate
        if lowstate.wireless_remote is not None:
            self.remote_controller.set(lowstate.wireless_remote)
            
            # R1/R2 buttons for height control on real robot (button indices 4 and 5)
            if self.remote_controller.button[0]:  # R1 - raise height
                self.groot_height_cmd += 0.001  # Small increment per timestep (~0.05m per second at 50Hz)
                self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)
            if self.remote_controller.button[4]:  # R2 - lower height
                self.groot_height_cmd -= 0.001  # Small decrement per timestep
                self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)
        else:
            # Default to zero commands if no remote data
            self.remote_controller.lx = 0.0
            self.remote_controller.ly = 0.0
            self.remote_controller.rx = 0.0
            self.remote_controller.ry = 0.0
        
        # Get ALL 29 joint positions and velocities
        for i in range(29):
            self.groot_qj_all[i] = lowstate.motor_state[i].q
            self.groot_dqj_all[i] = lowstate.motor_state[i].dq
        
        # Get IMU data
        quat = lowstate.imu_state.quaternion
        ang_vel = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)
        
        # Transform IMU if using torso IMU
        if self.config.locomotion_imu_type == "torso":
            waist_yaw = lowstate.motor_state[12].q  # Waist yaw index
            waist_yaw_omega = lowstate.motor_state[12].dq
            quat, ang_vel_3d = self.locomotion_transform_imu_data(waist_yaw, waist_yaw_omega, quat, np.array([ang_vel]))
            ang_vel = ang_vel_3d.flatten()
        
        # Create observation
        gravity_orientation = self.locomotion_get_gravity_orientation(quat)
        joints_to_zero_obs = [12, 14, 20, 21, 27, 28]  # Note: NOT 13 (waist roll exists)
        for idx in joints_to_zero_obs:
            self.groot_qj_all[idx] = 0.0
            self.groot_dqj_all[idx] = 0.0
        # Scale joint positions and velocities
        qj_obs = self.groot_qj_all.copy()
        dqj_obs = self.groot_dqj_all.copy()
        
        # Subtract default angles for legs + waist (15 joints)
        # GR00T default_angles: [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0]
        groot_default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # left leg
                                         -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # right leg
                                         0.0, 0.0, 0.0,                     # waist
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left arm (zeroed)
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # right arm (zeroed)
        
        qj_obs = (qj_obs - groot_default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel_scaled = ang_vel * self.config.groot_ang_vel_scale  # Use GR00T-specific scaling!
        
        # Get velocity commands (keyboard or remote)
        if not self.simulation_mode:
            self.locomotion_cmd[0] = self.remote_controller.ly
            self.locomotion_cmd[1] = self.remote_controller.lx * -1
            self.locomotion_cmd[2] = self.remote_controller.rx * -1
        
        # Build 86D single frame observation (GR00T format)
        self.groot_obs_single[:3] = self.locomotion_cmd * np.array(self.config.groot_cmd_scale)  # cmd - use GR00T scaling!
        self.groot_obs_single[3] = self.groot_height_cmd  # height_cmd
        self.groot_obs_single[4:7] = self.groot_orientation_cmd  # roll, pitch, yaw cmd
        self.groot_obs_single[7:10] = ang_vel_scaled  # angular velocity
        self.groot_obs_single[10:13] = gravity_orientation  # gravity
        self.groot_obs_single[13:42] = qj_obs  # joint positions (29D)
        self.groot_obs_single[42:71] = dqj_obs  # joint velocities (29D)
        self.groot_obs_single[71:86] = self.groot_action  # previous actions (15D)
        
        # Add to history and stack observations (6 frames × 86D = 516D)
        self.groot_obs_history.append(self.groot_obs_single.copy())
        
        # Stack all 6 frames into 516D vector
        for i, obs_frame in enumerate(self.groot_obs_history):
            start_idx = i * 86
            end_idx = start_idx + 86
            self.groot_obs_stacked[start_idx:end_idx] = obs_frame
        
        # Run policy inference (ONNX) with 516D stacked observation
        obs_tensor = torch.from_numpy(self.groot_obs_stacked).unsqueeze(0)
        ort_inputs = {self.policy.get_inputs()[0].name: obs_tensor.cpu().numpy()}
        ort_outs = self.policy.run(None, ort_inputs)
        self.groot_action = ort_outs[0].squeeze()
        
        # Zero out waist actions (yaw=12, roll=13, pitch=14) - only use leg actions (0-11)
        # This ensures action history in observations matches what's actually executed
        self.groot_action[12] = 0.0  # Waist yaw
        self.groot_action[13] = 0.0  # Waist roll  
        self.groot_action[14] = 0.0  # Waist pitch
        
        # Transform action to target joint positions (15D: legs + waist, but waist actions are zeroed)
        target_dof_pos_15 = groot_default_angles[:15] + self.groot_action * self.config.locomotion_action_scale
        
        # Send commands to LEG motors (0-11)
        for i in range(12):
            motor_idx = i
            self.msg.motor_cmd[motor_idx].q = target_dof_pos_15[i]
            self.msg.motor_cmd[motor_idx].qd = 0
            self.msg.motor_cmd[motor_idx].kp = self.config.locomotion_kps[i]
            self.msg.motor_cmd[motor_idx].kd = self.config.locomotion_kds[i]
            self.msg.motor_cmd[motor_idx].tau = 0
        
        # Send WAIST commands - but SKIP waist yaw (12) and waist pitch (14)
        # Only send waist roll (13)
        waist_roll_idx = 13
        waist_roll_action_idx = 13  # In the 15D action
        self.msg.motor_cmd[waist_roll_idx].q = target_dof_pos_15[waist_roll_action_idx]
        self.msg.motor_cmd[waist_roll_idx].qd = 0
        self.msg.motor_cmd[waist_roll_idx].kp = self.config.locomotion_arm_waist_kps[1]  # index 1 is waist roll
        self.msg.motor_cmd[waist_roll_idx].kd = self.config.locomotion_arm_waist_kds[1]
        self.msg.motor_cmd[waist_roll_idx].tau = 0
        
        # Zero out the problematic joints (waist yaw, waist pitch, wrist pitch/yaw)
        problematic_joints = [12, 14, 20, 21, 27, 28]
        for joint_idx in problematic_joints:
            self.msg.motor_cmd[joint_idx].q = 0.0
            self.msg.motor_cmd[joint_idx].qd = 0
            if joint_idx in [12, 14]:  # waist
                kp_idx = 0 if joint_idx == 12 else 2  # yaw or pitch
                self.msg.motor_cmd[joint_idx].kp = self.config.locomotion_arm_waist_kps[kp_idx]
                self.msg.motor_cmd[joint_idx].kd = self.config.locomotion_arm_waist_kds[kp_idx]
            else:  # wrists (20, 21, 27, 28)
                self.msg.motor_cmd[joint_idx].kp = self.kp_wrist
                self.msg.motor_cmd[joint_idx].kd = self.kd_wrist
            self.msg.motor_cmd[joint_idx].tau = 0

        
        # Send command
        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)
    
    def _locomotion_thread_loop(self):
        """Background thread that runs the locomotion policy at specified rate."""
        logger_mp.info("Locomotion thread started")
        while self.locomotion_running:
            start_time = time.time()
            try:
                # Use different run function based on policy type
                if self.policy_type == 'onnx':
                    self.groot_locomotion_run()
                else:
                    self.locomotion_run()
            except Exception as e:
                logger_mp.error(f"Error in locomotion loop: {e}")
            
            # Sleep to maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.locomotion_control_dt - elapsed)
            time.sleep(sleep_time)
        logger_mp.info("Locomotion thread stopped")
    
    def start_locomotion_thread(self):
        """Start the background locomotion control thread."""
        if not self.config.locomotion_control:
            logger_mp.warning("locomotion_control is False, cannot start thread")
            return
        
        if self.locomotion_running:
            logger_mp.warning("Locomotion thread already running")
            return
        
        logger_mp.info("Starting locomotion control thread...")
        self.locomotion_running = True
        self.locomotion_thread = threading.Thread(target=self._locomotion_thread_loop, daemon=True)
        self.locomotion_thread.start()
        logger_mp.info("Locomotion control thread started!")
    
    def stop_locomotion_thread(self):
        """Stop the background locomotion control thread."""
        if not self.locomotion_running:
            return
        
        logger_mp.info("Stopping locomotion control thread...")
        self.locomotion_running = False
        if self.locomotion_thread:
            self.locomotion_thread.join(timeout=2.0)
        logger_mp.info("Locomotion control thread stopped")
        
        # Also stop keyboard thread if running
        if self.keyboard_running:
            self.stop_keyboard_controls()
    
    def _keyboard_listener_thread(self):
        """Background thread that listens for keyboard input (sim mode only)."""
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS ACTIVE!")
        print("  W/S: Forward/Backward")
        print("  A/D: Left/Right")
        print("  Q/E: Rotate Left/Right")
        print("  R/F: Raise/Lower Height (±5cm)")
        print("  Z: Stop (zero velocity commands)")
        print("="*60 + "\n")
        
        # Save terminal settings
        old_settings = None
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            
            while self.keyboard_running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    # Velocity commands
                    if key == 'w':
                        self.locomotion_cmd[0] += 0.4  # Forward
                    elif key == 's':
                        self.locomotion_cmd[0] -= 0.4  # Backward
                    elif key == 'a':
                        self.locomotion_cmd[1] += 0.25  # Left
                    elif key == 'd':
                        self.locomotion_cmd[1] -= 0.25  # Right
                    elif key == 'q':
                        self.locomotion_cmd[2] += 0.5  # Rotate left
                    elif key == 'e':
                        self.locomotion_cmd[2] -= 0.5  # Rotate right
                    elif key == 'z':
                        self.locomotion_cmd[:] = 0.0  # Stop
                    
                    # Height commands (only for GR00T ONNX policies)
                    elif key == 'r':
                        self.groot_height_cmd += 0.05  # Raise 5cm
                    elif key == 'f':
                        self.groot_height_cmd -= 0.05  # Lower 5cm
                    
                    # Clamp commands to reasonable limits
                    self.locomotion_cmd[0] = np.clip(self.locomotion_cmd[0], -0.8, 0.8)  # vx
                    self.locomotion_cmd[1] = np.clip(self.locomotion_cmd[1], -0.5, 0.5)  # vy
                    self.locomotion_cmd[2] = np.clip(self.locomotion_cmd[2], -1.0, 1.0)  # yaw_rate
                    
                    # Clamp height (reasonable range: 0.5m to 1.0m)
                    if hasattr(self, 'groot_height_cmd'):
                        self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)
                    
                    # Print current commands
                    print(f"[VEL CMD] vx={self.locomotion_cmd[0]:.2f}, vy={self.locomotion_cmd[1]:.2f}, yaw={self.locomotion_cmd[2]:.2f}", end="")
                    if hasattr(self, 'groot_height_cmd'):
                        print(f" | [HEIGHT] {self.groot_height_cmd:.3f}m", end="")
                    print()  # Newline
        
        finally:
            # Restore terminal settings
            if old_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("\nKeyboard controls stopped")
    
    def start_keyboard_controls(self):
        """Start the keyboard control thread (sim mode only)."""
        if not self.simulation_mode:
            logger_mp.warning("Keyboard controls only available in simulation mode")
            return
        
        if self.keyboard_running:
            logger_mp.warning("Keyboard controls already running")
            return
        
        self.keyboard_running = True
        self.keyboard_thread = threading.Thread(target=self._keyboard_listener_thread, daemon=True)
        self.keyboard_thread.start()
        logger_mp.info("Keyboard controls started!")
    
    def stop_keyboard_controls(self):
        """Stop the keyboard control thread."""
        if not self.keyboard_running:
            return
        
        logger_mp.info("Stopping keyboard controls...")
        self.keyboard_running = False
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=2.0)
        logger_mp.info("Keyboard controls stopped")


    def init_locomotion(self):
        """Test locomotion control sequence: home arms -> move legs to default -> start policy thread."""
        if not self.config.locomotion_control:
            logger_mp.warning("locomotion_control is False, cannot run test sequence")
            return
        
        logger_mp.info("Starting locomotion test sequence...")
        
        # 1. Home the arms first
        logger_mp.info("Homing arms to zero position...")
        #self.ctrl_dual_arm_go_home()

        # 2. Move legs to default position
        self.locomotion_move_to_default_pos()
        
        # 3. Wait 3 seconds
        time.sleep(3.0)
        
        # 4. Hold default leg position for 2 seconds
        self.locomotion_default_pos_state()
        
        # 5. Start locomotion policy thread (runs in background)
        logger_mp.info("Starting locomotion policy control...")
        self.start_locomotion_thread()
        
        logger_mp.info("Locomotion test sequence complete! Policy is now running in background.")
        logger_mp.info("Use robot.stop_locomotion_thread() to stop the policy.")

    def init_groot_locomotion(self):
        """Initialize GR00T-style locomotion for ONNX policies (29 DOF, 15D actions)."""
        if not self.config.locomotion_control:
            logger_mp.warning("locomotion_control is False, cannot run GR00T init")
            return
        
        logger_mp.info("Starting GR00T locomotion initialization...")
        
        # Move legs to default position (same as regular locomotion)
        self.locomotion_move_to_default_pos()
        
        # Wait 3 seconds
        time.sleep(3.0)
        
        # Hold default leg position for 2 seconds
        self.locomotion_default_pos_state()
        
        # Start locomotion policy thread (will use groot_locomotion_run)
        logger_mp.info("Starting GR00T locomotion policy control...")
        self.start_locomotion_thread()
        
        logger_mp.info("GR00T locomotion initialization complete! Policy is now running.")
        logger_mp.info("516D observations (86D × 6 frames), 15D actions (legs + waist)")

    def motion_imitation_run(self):
        """Motion imitation policy loop - tracks reference motion (dance_102, etc)."""
        self.motion_counter += 1
        self.motion_elapsed_time = self.motion_counter * self.config.motion_control_dt
        
        # Update motion loader to current time
        self.motion_loader.update(self.motion_elapsed_time)
        
        # Get current lowstate
        lowstate = self.lowstate_buffer.GetData()
        if lowstate is None:
            return
        
        # Get ALL 29 joint positions and velocities from robot
        # IMPORTANT: Convert from motor order to BFS order to match reference motion
        # The C++ code does: robot_bfs[i] = motor[joint_ids_map[i]]
        for i in range(29):
            motor_idx = self.config.motion_joint_ids_map[i]
            self.motion_qj_all[i] = lowstate.motor_state[motor_idx].q
            self.motion_dqj_all[i] = lowstate.motor_state[motor_idx].dq
        
        # ======== 23 DOF MODE CONFIGURATION ========
        # For real robot - zeros out joints not present in 23 DOF hardware
        # Waist: yaw(12), pitch(14) | Wrist: L_pitch/yaw(20,21), R_pitch/yaw(27,28)
        USE_23DOF = True  # Set to True for real robot without these joints
        JOINTS_TO_ZERO_23DOF = [12,14,20, 21, 27, 28]#12, 14, 20, 21, 27, 28]#
        
        # Apply 23 DOF zeroing to robot observations if enabled
        if USE_23DOF:
            for joint_idx in JOINTS_TO_ZERO_23DOF:
                self.motion_qj_all[joint_idx] = 0.0
                self.motion_dqj_all[joint_idx] = 0.0
            if self.motion_counter == 1:
                logger_mp.info("="*60)
                logger_mp.info("🤖 23 DOF MODE ENABLED")
                logger_mp.info(f"   Zeroing joints: {JOINTS_TO_ZERO_23DOF}")
                logger_mp.info("   Waist: yaw(12), pitch(14)")
                logger_mp.info("   Wrist L: pitch(20), yaw(21) | Wrist R: pitch(27), yaw(28)")
                logger_mp.info("   Applied to: robot obs, reference motion, policy actions")
                logger_mp.info("="*60)
        
        # Get IMU data
        robot_quat = lowstate.imu_state.quaternion  # [w, x, y, z]
        ang_vel = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)  # 3D
        
        if self.policy is None:
            # DIRECT PLAYBACK MODE (no policy)
            motion_joint_pos_dfs = self.motion_loader.get_joint_pos()
            
            # Zero out missing joints for 23 DOF mode
            if USE_23DOF:
                # Convert to BFS to zero out, then convert back
                motion_joint_pos_bfs_temp = np.zeros(29, dtype=np.float32)
                for i in range(29):
                    motion_joint_pos_bfs_temp[i] = motion_joint_pos_dfs[self.config.motion_joint_ids_map[i]]
                for joint_idx in JOINTS_TO_ZERO_23DOF:
                    motion_joint_pos_bfs_temp[joint_idx] = 0.0
                # Convert back to DFS for sending
                for i in range(29):
                    motion_joint_pos_dfs[self.config.motion_joint_ids_map[i]] = motion_joint_pos_bfs_temp[i]
            
            for i in range(29):
                motor_idx = self.config.motion_joint_ids_map[i]
                csv_idx = self.config.motion_joint_ids_map[i]
                self.msg.motor_cmd[motor_idx].q = motion_joint_pos_dfs[csv_idx]
                self.msg.motor_cmd[motor_idx].qd = 0
                self.msg.motor_cmd[motor_idx].kp = self.config.motion_stiffness[motor_idx]
                self.msg.motor_cmd[motor_idx].kd = self.config.motion_damping[motor_idx]
                self.msg.motor_cmd[motor_idx].tau = 0
        else:
            # POLICY MODE - Full observation construction and inference
            
            # ======== DEBUG TEST MODES ========
            # Mode 1: Direct playback (no policy) - set motion_policy_path = None in config instead
            # Mode 2: Send default pos (stand still) - TEST_SEND_DEFAULT_POS = True
            # Mode 3: Policy with zero reference - TEST_WITH_ZEROS = True, TEST_SEND_DEFAULT_POS = False
            # Mode 4: Policy with real reference - TEST_WITH_ZEROS = False, TEST_SEND_DEFAULT_POS = False
            TEST_WITH_ZEROS = False  # True = use zero reference motion in observation
            TEST_SEND_DEFAULT_POS = False  # True = bypass policy and send default pos (stand still)
            TEST_DIRECT_PLAYBACK = False  # True = bypass policy and send reference motion directly
            
            if TEST_DIRECT_PLAYBACK:
                # DEBUG: Play back reference motion without policy
                motion_joint_pos_dfs = self.motion_loader.get_joint_pos()  # 29D in DFS order
                
                # Zero out missing joints for 23 DOF mode
                if USE_23DOF:
                    # Convert to BFS to zero out, then convert back
                    motion_joint_pos_bfs_temp = np.zeros(29, dtype=np.float32)
                    for i in range(29):
                        motion_joint_pos_bfs_temp[i] = motion_joint_pos_dfs[self.config.motion_joint_ids_map[i]]
                    for joint_idx in JOINTS_TO_ZERO_23DOF:
                        motion_joint_pos_bfs_temp[joint_idx] = 0.0
                    # Convert back to DFS for sending
                    for i in range(29):
                        motion_joint_pos_dfs[self.config.motion_joint_ids_map[i]] = motion_joint_pos_bfs_temp[i]
                
                # Send directly to motors using joint_ids_map (same as direct playback mode)
                for i in range(29):
                    motor_idx = self.config.motion_joint_ids_map[i]
                    csv_idx = self.config.motion_joint_ids_map[i]
                    self.msg.motor_cmd[motor_idx].q = motion_joint_pos_dfs[csv_idx]
                    self.msg.motor_cmd[motor_idx].qd = 0
                    self.msg.motor_cmd[motor_idx].kp = self.config.motion_stiffness[motor_idx]
                    self.msg.motor_cmd[motor_idx].kd = self.config.motion_damping[motor_idx]
                    self.msg.motor_cmd[motor_idx].tau = 0
                
                if self.motion_counter == 1:
                    logger_mp.info("="*60)
                    logger_mp.info("⚠️  DEBUG MODE: DIRECT PLAYBACK (reference motion, no policy)")
                    logger_mp.info("="*60)
                
                target_joint_pos_bfs = None  # Not used in this mode
                
            else:
                # Run observation construction and policy
                if TEST_WITH_ZEROS:
                    # Send zeros for reference motion
                    motion_joint_pos_bfs = np.zeros(29, dtype=np.float32)
                    motion_joint_vel_bfs = np.zeros(29, dtype=np.float32)
                    if self.motion_counter == 1:
                        logger_mp.info("="*60)
                        logger_mp.info("⚠️  DEBUG MODE: Using ZERO reference motion + RUNNING POLICY")
                        logger_mp.info("="*60)
                else:
                    # Get reference motion (DFS order from CSV)
                    motion_joint_pos_dfs = self.motion_loader.get_joint_pos()  # 29D
                    motion_joint_vel_dfs = self.motion_loader.get_joint_vel()  # 29D
                    
                    # Convert from DFS to BFS order: bfs[i] = dfs[joint_ids_map[i]]
                    motion_joint_pos_bfs = np.zeros(29, dtype=np.float32)
                    motion_joint_vel_bfs = np.zeros(29, dtype=np.float32)
                    for i in range(29):
                        motion_joint_pos_bfs[i] = motion_joint_pos_dfs[self.config.motion_joint_ids_map[i]]
                        motion_joint_vel_bfs[i] = motion_joint_vel_dfs[self.config.motion_joint_ids_map[i]]
                    
                    # Zero out missing joints in reference motion for 23 DOF mode
                    if USE_23DOF:
                        for joint_idx in JOINTS_TO_ZERO_23DOF:
                            motion_joint_pos_bfs[joint_idx] = 0.0
                            motion_joint_vel_bfs[joint_idx] = 0.0
                
                # Compute motion_anchor_ori_b (6D rotation matrix representation)
                motion_quat_wxyz = self.motion_loader.get_root_quat_wxyz()
                robot_rot = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]).as_matrix()
                motion_rot = R.from_quat([motion_quat_wxyz[1], motion_quat_wxyz[2], motion_quat_wxyz[3], motion_quat_wxyz[0]]).as_matrix()
                relative_rot = robot_rot.T @ motion_rot
                motion_anchor_ori_b = np.array([relative_rot[0, 0], relative_rot[0, 1], 
                                                relative_rot[1, 0], relative_rot[1, 1],
                                                relative_rot[2, 0], relative_rot[2, 1]], dtype=np.float32)
                
                # Compute joint positions and velocities relative to default
                default_joint_pos = np.array(self.config.motion_default_joint_pos, dtype=np.float32)
                joint_pos_rel = self.motion_qj_all - default_joint_pos
                joint_vel_rel = self.motion_dqj_all.copy()
                
                # Build 154D observation:
                # motion_command (58D) = joint_pos (29D) + joint_vel (29D) from reference
                # motion_anchor_ori_b (6D)
                # base_ang_vel (3D)
                # joint_pos_rel (29D)
                # joint_vel_rel (29D)
                # last_action (29D)
                self.motion_obs[0:29] = motion_joint_pos_bfs
                self.motion_obs[29:58] = motion_joint_vel_bfs
                self.motion_obs[58:64] = motion_anchor_ori_b
                self.motion_obs[64:67] = ang_vel
                self.motion_obs[67:96] = joint_pos_rel
                self.motion_obs[96:125] = joint_vel_rel
                self.motion_obs[125:154] = self.motion_action
                
                if TEST_SEND_DEFAULT_POS:
                    # DEBUG: Just send default positions (should make robot stand still)
                    target_joint_pos_bfs = default_joint_pos.copy()
                    if self.motion_counter == 1:
                        logger_mp.info("="*60)
                        logger_mp.info("⚠️  DEBUG MODE: Sending DEFAULT positions (NO POLICY)")
                        logger_mp.info("="*60)
                        logger_mp.info(f"   Default pos BFS[0:5]: {target_joint_pos_bfs[0:5]}")
                    if self.motion_counter % 50 == 0:
                        logger_mp.info(f"   [DEFAULT MODE] Sending: [{target_joint_pos_bfs[0]:.4f}, {target_joint_pos_bfs[6]:.4f}, {target_joint_pos_bfs[12]:.4f}]")
                        logger_mp.info(f"   [DEFAULT MODE] Robot at: [{self.motion_qj_all[0]:.4f}, {self.motion_qj_all[6]:.4f}, {self.motion_qj_all[12]:.4f}]")
                else:
                    # Run ONNX policy inference
                    obs_tensor = torch.from_numpy(self.motion_obs).unsqueeze(0)
                    ort_inputs = {self.policy.get_inputs()[0].name: obs_tensor.cpu().numpy()}
                    ort_outs = self.policy.run(None, ort_inputs)
                    self.motion_action = ort_outs[0].squeeze()  # 29D action in BFS order
                    
                    # Zero out missing joints in policy actions for 23 DOF mode
                    if USE_23DOF:
                        for joint_idx in JOINTS_TO_ZERO_23DOF:
                            self.motion_action[joint_idx] = 0.0
                    
                    # Process actions: scale and add offset
                    action_scale = np.array(self.config.motion_action_scale, dtype=np.float32)
                    target_joint_pos_bfs = default_joint_pos + self.motion_action * action_scale
                
                # Send commands to motors: motor[joint_ids_map[i]] = action[i]
                for i in range(29):
                    motor_idx = self.config.motion_joint_ids_map[i]
                    self.msg.motor_cmd[motor_idx].q = target_joint_pos_bfs[i]
                    self.msg.motor_cmd[motor_idx].qd = 0
                    self.msg.motor_cmd[motor_idx].kp = self.config.motion_stiffness[motor_idx]
                    self.msg.motor_cmd[motor_idx].kd = self.config.motion_damping[motor_idx]
                    self.msg.motor_cmd[motor_idx].tau = 0
        
        # Debug print (only when running policy, not in TEST_SEND_DEFAULT_POS or TEST_DIRECT_PLAYBACK mode)
        if self.motion_counter == 1 and self.policy and not TEST_SEND_DEFAULT_POS and not TEST_DIRECT_PLAYBACK:
            logger_mp.info("="*60)
            logger_mp.info("POLICY MODE OBSERVATION CHECK (First iteration)")
            logger_mp.info("="*60)
            logger_mp.info(f"Reference motion (BFS) samples: [{motion_joint_pos_bfs[0]:.3f}, {motion_joint_pos_bfs[6]:.3f}, {motion_joint_pos_bfs[12]:.3f}]")
            logger_mp.info(f"Robot joints (BFS) samples:     [{self.motion_qj_all[0]:.3f}, {self.motion_qj_all[6]:.3f}, {self.motion_qj_all[12]:.3f}]")
            logger_mp.info(f"Default positions samples:      [{default_joint_pos[0]:.3f}, {default_joint_pos[6]:.3f}, {default_joint_pos[12]:.3f}]")
            logger_mp.info(f"Joint pos rel samples:          [{joint_pos_rel[0]:.3f}, {joint_pos_rel[6]:.3f}, {joint_pos_rel[12]:.3f}]")
            logger_mp.info(f"Joint vel rel samples:          [{joint_vel_rel[0]:.3f}, {joint_vel_rel[6]:.3f}, {joint_vel_rel[12]:.3f}]")
            logger_mp.info(f"Angular velocity:               [{ang_vel[0]:.3f}, {ang_vel[1]:.3f}, {ang_vel[2]:.3f}]")
            logger_mp.info(f"Motion anchor ori:              [{motion_anchor_ori_b[0]:.3f}, ..., {motion_anchor_ori_b[5]:.3f}]")
            logger_mp.info(f"Observation breakdown:")
            logger_mp.info(f"  [0:29]   motion_cmd_pos: range [{self.motion_obs[0:29].min():.3f}, {self.motion_obs[0:29].max():.3f}]")
            logger_mp.info(f"  [29:58]  motion_cmd_vel: range [{self.motion_obs[29:58].min():.3f}, {self.motion_obs[29:58].max():.3f}]")
            logger_mp.info(f"  [58:64]  anchor_ori:     range [{self.motion_obs[58:64].min():.3f}, {self.motion_obs[58:64].max():.3f}]")
            logger_mp.info(f"  [64:67]  ang_vel:        range [{self.motion_obs[64:67].min():.3f}, {self.motion_obs[64:67].max():.3f}]")
            logger_mp.info(f"  [67:96]  joint_pos_rel:  range [{self.motion_obs[67:96].min():.3f}, {self.motion_obs[67:96].max():.3f}]")
            logger_mp.info(f"  [96:125] joint_vel_rel:  range [{self.motion_obs[96:125].min():.3f}, {self.motion_obs[96:125].max():.3f}]")
            logger_mp.info(f"  [125:154] last_action:   range [{self.motion_obs[125:154].min():.3f}, {self.motion_obs[125:154].max():.3f}]")
            logger_mp.info(f"Full obs range: [{self.motion_obs.min():.3f}, {self.motion_obs.max():.3f}]")
            logger_mp.info(f"Action output (first): [{self.motion_action.min():.3f}, {self.motion_action.max():.3f}]")
            logger_mp.info(f"Action scale samples: [{action_scale[0]:.3f}, {action_scale[6]:.3f}, {action_scale[12]:.3f}]")
            logger_mp.info(f"Target positions samples: [{target_joint_pos_bfs[0]:.3f}, {target_joint_pos_bfs[6]:.3f}, {target_joint_pos_bfs[12]:.3f}]")
            logger_mp.info("="*60)
        
        if self.motion_counter % 50 == 0:
            if self.policy is None:
                mode = "DIRECT"
            elif TEST_DIRECT_PLAYBACK:
                mode = "DIRECT_DEBUG"
            elif TEST_SEND_DEFAULT_POS:
                mode = "DEFAULT_POS"
            elif TEST_WITH_ZEROS:
                mode = "POLICY_ZEROS"
            else:
                mode = "POLICY"
            logger_mp.info(f"Motion {mode}: t={self.motion_elapsed_time:.2f}s, frame={self.motion_loader.index_0}/{self.motion_loader.num_frames}")
            if self.policy and not TEST_SEND_DEFAULT_POS and not TEST_DIRECT_PLAYBACK:
                logger_mp.info(f"  Policy action range: [{self.motion_action.min():.3f}, {self.motion_action.max():.3f}]")
                logger_mp.info(f"  Sample actions[0,6,12]: [{self.motion_action[0]:.3f}, {self.motion_action[6]:.3f}, {self.motion_action[12]:.3f}]")
                logger_mp.info(f"  Target pos (after scale)[0,6,12]: [{target_joint_pos_bfs[0]:.3f}, {target_joint_pos_bfs[6]:.3f}, {target_joint_pos_bfs[12]:.3f}]")
                logger_mp.info(f"  Robot pos (BFS)[0,6,12]: [{self.motion_qj_all[0]:.3f}, {self.motion_qj_all[6]:.3f}, {self.motion_qj_all[12]:.3f}]")
        
        # Send command
        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)
    
    def _motion_imitation_thread_loop(self):
        """Background thread that runs the motion imitation policy at specified rate."""
        logger_mp.info("Motion imitation thread started")
        while self.motion_imitation_running:
            start_time = time.time()
            try:
                self.motion_imitation_run()
            except Exception as e:
                logger_mp.error(f"Error in motion imitation loop: {e}")
                import traceback
                traceback.print_exc()
            
            # Sleep to maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.motion_control_dt - elapsed)
            time.sleep(sleep_time)
        logger_mp.info("Motion imitation thread stopped")
    
    def start_motion_imitation_thread(self):
        """Start the background motion imitation control thread."""
        if not self.config.motion_imitation_control:
            logger_mp.warning("motion_imitation_control is False, cannot start thread")
            return
        
        if self.motion_imitation_running:
            logger_mp.warning("Motion imitation thread already running")
            return
        
        logger_mp.info("Starting motion imitation control thread...")
        self.motion_imitation_running = True
        self.motion_imitation_thread = threading.Thread(target=self._motion_imitation_thread_loop, daemon=True)
        self.motion_imitation_thread.start()
        logger_mp.info("Motion imitation control thread started!")
    
    def stop_motion_imitation_thread(self):
        """Stop the background motion imitation control thread."""
        if not self.motion_imitation_running:
            return
        
        logger_mp.info("Stopping motion imitation control thread...")
        self.motion_imitation_running = False
        if self.motion_imitation_thread:
            self.motion_imitation_thread.join(timeout=2.0)
        logger_mp.info("Motion imitation control thread stopped")
    
    def init_motion_imitation(self):
        """Initialize motion imitation - move to default standing pose and start policy."""
        if not self.config.motion_imitation_control:
            logger_mp.warning("motion_imitation_control is False, cannot run initialization")
            return
        
        logger_mp.info("Starting motion imitation initialization...")
        
        # Move to default standing position
        logger_mp.info("Moving to default standing position...")
        total_time = 3.0
        num_steps = int(total_time / self.config.motion_control_dt)
        
        # Get current positions (in motor order)
        current_q_motor = self.get_current_motor_q()
        
        # target_q is in BFS order from config, need to convert to motor order
        target_q_bfs = np.array(self.config.motion_default_joint_pos, dtype=np.float32)
        target_q_motor = np.zeros(29, dtype=np.float32)
        for i in range(29):
            motor_idx = self.config.motion_joint_ids_map[i]
            target_q_motor[motor_idx] = target_q_bfs[i]
        
        # Interpolate to target (both in motor order now)
        for i in range(num_steps):
            alpha = i / num_steps
            for motor_idx in range(29):
                self.msg.motor_cmd[motor_idx].q = current_q_motor[motor_idx] * (1 - alpha) + target_q_motor[motor_idx] * alpha
                self.msg.motor_cmd[motor_idx].qd = 0
                self.msg.motor_cmd[motor_idx].kp = self.config.motion_stiffness[motor_idx]
                self.msg.motor_cmd[motor_idx].kd = self.config.motion_damping[motor_idx]
                self.msg.motor_cmd[motor_idx].tau = 0
            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
            time.sleep(self.config.motion_control_dt)
        
        logger_mp.info("Reached default position")
        
        # Wait 2 seconds
        time.sleep(2.0)
        
        # Start motion imitation policy thread
        logger_mp.info("Starting motion imitation policy control...")
        self.start_motion_imitation_thread()
        
        logger_mp.info("Motion imitation initialization complete! Policy is now running.")
        logger_mp.info(f"154D observations, 29D actions. Motion duration: {self.motion_loader.duration:.2f}s")


class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12 #we're c
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34