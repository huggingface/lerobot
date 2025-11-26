import logging
import time
import struct
from functools import cached_property
from typing import Any
from pathlib import Path

from lerobot.cameras.utils import make_cameras_from_configs

import json
from ..robot import Robot
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

from typing import Union
import numpy as np
import time
import torch
import onnxruntime as ort

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState  # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)

from lerobot.envs.factory import make_env
from scipy.spatial.transform import Rotation as R

import struct


import torch

logger = logging.getLogger(__name__)

kTopicLowCommand_Debug = "rt/lowcmd"
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

class UnitreeG1(Robot):

    config_class = UnitreeG1Config
    name = "unitree_g1"

    def __init__(self, config: UnitreeG1Config):
        super().__init__(config)
        
        logger.info("Initialize UnitreeG1...")

        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
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

        self._gradual_start_time = config.gradual_start_time
        self._gradual_time = config.gradual_time

        # Teleop warmup: gradually move from current position to targets over 2 seconds
        self.teleop_warmup_duration = 2.0  # seconds
        self.teleop_warmup_start_time = None
        self.teleop_warmup_initial_q = None

        self.freeze_body = config.freeze_body
        self.gravity_compensation = config.gravity_compensation

        self.calibrated = False

        self.calibrate()

        if self.config.socket_host is not None:
            from lerobot.robots.unitree_g1.unitree_sdk2_socket import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
        else:
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds

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

        # initialize lowcmd nd lowstate subscriber
        if self.simulation_mode:
            ChannelFactoryInitialize(0, "lo")

            logger.info("Launching MuJoCo simulation environment...")
            self.mujoco_env = make_env("lerobot/unitree-g1-mujoco", trust_remote_code=True)
            logger.info("MuJoCo environment launched successfully!")
        else:
            ChannelFactoryInitialize(0)

        # Always use debug mode (direct motor control)
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
            logger.warning("[UnitreeG1] Waiting to subscribe dds...")
        logger.info("[UnitreeG1] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()
        print(self.msg)

        self.all_motor_q = self.get_current_motor_q()
        logger.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger.info("Lock all joints except two arms...\n")

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

        logger.info("Lock OK!\n") #motors are not locked x
        # for i in range(10000):
        #     print(self.get_current_motor_q())
        #     time.sleep(0.05) 
        
        # Initialize control flags BEFORE starting threads
        self.keyboard_thread = None
        self.keyboard_running = False
        self.locomotion_thread = None
        self.locomotion_running = False
        
        # Initialize publish thread for arm control
        # Note: This thread runs alongside locomotion thread
        # - Arm thread: controls arms (indices 15-28)
        # - Locomotion thread: controls legs (0-11), waist (12-14)
        # Both update different parts of self.msg, both call Write()
        self.publish_thread = None
        self.ctrl_lock = threading.Lock()
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        logger.info("Arm control publish thread started")

        # Load locomotion policy if enabled
        self.policy = None
        self.policy_type = None  # 'torchscript' or 'onnx'
        
        if config.locomotion_control:
            if config.policy_path is None:
                raise ValueError("locomotion_control is True but policy_path is not set")
            
            logger.info(f"Loading locomotion policy from {config.policy_path}")
            
            # Check file extension and load accordingly
            if config.policy_path.endswith('.pt'):
                logger.info("Detected TorchScript (.pt) policy")
                self.policy = torch.jit.load(config.policy_path)
                self.policy_type = 'torchscript'
                logger.info("TorchScript policy loaded successfully")
            elif config.policy_path.endswith('.onnx'):
                logger.info("Detected ONNX (.onnx) policy")
                
                # For GR00T-style policies, load both Balance and Walk policies
                # Balance policy for standing (low velocity commands)
                # Walk policy for locomotion (high velocity commands)
                balance_policy_path = config.policy_path.replace('Walk.onnx', 'Balance.onnx')
                walk_policy_path = config.policy_path
                
                if Path(balance_policy_path).exists() and Path(walk_policy_path).exists():
                    logger.info("Loading dual-policy system (Balance + Walk)")
                    self.policy_balance = ort.InferenceSession(balance_policy_path)
                    self.policy_walk = ort.InferenceSession(walk_policy_path)
                    self.policy = None  # Not used when dual policies are loaded
                    logger.info(f"Balance policy loaded from: {balance_policy_path}")
                    logger.info(f"Walk policy loaded from: {walk_policy_path}")
                    logger.info(f"ONNX input: {self.policy_balance.get_inputs()[0].name}, shape: {self.policy_balance.get_inputs()[0].shape}")
                    logger.info(f"ONNX output: {self.policy_balance.get_outputs()[0].name}, shape: {self.policy_balance.get_outputs()[0].shape}")
                else:
                    # Fallback to single policy
                    logger.info("Loading single ONNX policy")
                    self.policy = ort.InferenceSession(config.policy_path)
                    self.policy_balance = None
                    self.policy_walk = None
                    logger.info("ONNX policy loaded successfully")
                    logger.info(f"ONNX input: {self.policy.get_inputs()[0].name}, shape: {self.policy.get_inputs()[0].shape}")
                    logger.info(f"ONNX output: {self.policy.get_outputs()[0].name}, shape: {self.policy.get_outputs()[0].shape}")
                
                self.policy_type = 'onnx'
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
                logger.info("Starting keyboard controls for simulation...")
                self.start_keyboard_controls()
            
            # Use different init based on policy type
            if self.policy_type == 'onnx':
                self.init_groot_locomotion()
            else:
                self.init_locomotion()
        elif self.simulation_mode:
            # Even without locomotion, provide keyboard feedback in sim
            logger.info("Simulation mode active (locomotion disabled)")


        logger.info("Initialize G1 OK!\n")

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
        """Arm control thread - publishes commands for arms only."""
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

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger.debug(f"sleep_time:{sleep_time}")

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
        logger.info(f"{self} connected with {len(self.cameras)} camera(s).")

    def disconnect(self):
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Close MuJoCo environment if in simulation mode
        if self.simulation_mode and hasattr(self, 'mujoco_env'):
            logger.info("Closing MuJoCo environment...")
            print(self.mujoco_env)
            self.mujoco_env["hub_env"][0].envs[0].kill_sim()
        
        logger.info(f"{self} disconnected.")

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

    def get_observation(self) -> dict[str, Any]:
        obs_array = self.get_current_dual_arm_q()
        obs_dict = {f"{G1_29_JointArmIndex(motor).name}.pos": val for motor, val in zip(G1_29_JointArmIndex, obs_array, strict=True)}
        
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
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
        logger.info("Enter zero torque state.")
        self.locomotion_create_zero_cmd()
        time.sleep(self.config.locomotion_control_dt)

    def locomotion_move_to_default_pos(self):
        """Move robot legs to default standing position over 2 seconds (arms are not moved)."""
        logger.info("Moving legs to default locomotion pos.")
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
            logger.error("Cannot get lowstate for locomotion")
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
        logger.info("Reached default locomotion position (legs only)")

    def locomotion_default_pos_state(self):
        """Hold default leg position for 2 seconds (arms are not controlled)."""
        logger.info("Enter default pos state - holding legs for 2 seconds")
        
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
        logger.info("Finished holding default leg position")
    

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
                logger.debug(f"Remote controller - lx:{self.remote_controller.lx:.2f}, ly:{self.remote_controller.ly:.2f}, rx:{self.remote_controller.rx:.2f}")
        
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
        
        # Select appropriate policy based on command magnitude (dual-policy system)
        if self.policy_balance is not None and self.policy_walk is not None:
            # Dual-policy mode: switch between Balance and Walk
            cmd_magnitude = np.linalg.norm(self.locomotion_cmd)
            if cmd_magnitude < 0.05:
                # Use balance/standing policy for small commands
                selected_policy = self.policy_balance
            else:
                # Use walking policy for movement commands
                selected_policy = self.policy_walk
        else:
            # Single policy mode (fallback)
            selected_policy = self.policy
        
        ort_inputs = {selected_policy.get_inputs()[0].name: obs_tensor.cpu().numpy()}
        ort_outs = selected_policy.run(None, ort_inputs)
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
        logger.info("Locomotion thread started")
        while self.locomotion_running:
            start_time = time.time()
            try:
                # Use different run function based on policy type
                if self.policy_type == 'onnx':
                    self.groot_locomotion_run()
                else:
                    self.locomotion_run()
            except Exception as e:
                logger.error(f"Error in locomotion loop: {e}")
            
            # Sleep to maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.locomotion_control_dt - elapsed)
            time.sleep(sleep_time)
        logger.info("Locomotion thread stopped")
    
    def start_locomotion_thread(self):
        """Start the background locomotion control thread."""
        if not self.config.locomotion_control:
            logger.warning("locomotion_control is False, cannot start thread")
            return
        
        if self.locomotion_running:
            logger.warning("Locomotion thread already running")
            return
        
        logger.info("Starting locomotion control thread...")
        self.locomotion_running = True
        self.locomotion_thread = threading.Thread(target=self._locomotion_thread_loop, daemon=True)
        self.locomotion_thread.start()
        logger.info("Locomotion control thread started!")
    
    def stop_locomotion_thread(self):
        """Stop the background locomotion control thread."""
        if not self.locomotion_running:
            return
        
        logger.info("Stopping locomotion control thread...")
        self.locomotion_running = False
        if self.locomotion_thread:
            self.locomotion_thread.join(timeout=2.0)
        logger.info("Locomotion control thread stopped")
        
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
            logger.warning("Keyboard controls only available in simulation mode")
            return
        
        if self.keyboard_running:
            logger.warning("Keyboard controls already running")
            return
        
        self.keyboard_running = True
        self.keyboard_thread = threading.Thread(target=self._keyboard_listener_thread, daemon=True)
        self.keyboard_thread.start()
        logger.info("Keyboard controls started!")
    
    def stop_keyboard_controls(self):
        """Stop the keyboard control thread."""
        if not self.keyboard_running:
            return
        
        logger.info("Stopping keyboard controls...")
        self.keyboard_running = False
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=2.0)
        logger.info("Keyboard controls stopped")


    def init_locomotion(self):
        """Test locomotion control sequence: home arms -> move legs to default -> start policy thread."""
        if not self.config.locomotion_control:
            logger.warning("locomotion_control is False, cannot run test sequence")
            return
        
        logger.info("Starting locomotion test sequence...")


        # 2. Move legs to default position
        self.locomotion_move_to_default_pos()
        
        # 3. Wait 3 seconds
        time.sleep(3.0)
        
        # 4. Hold default leg position for 2 seconds
        self.locomotion_default_pos_state()
        
        # 5. Start locomotion policy thread (runs in background)
        logger.info("Starting locomotion policy control...")
        self.start_locomotion_thread()
        
        logger.info("Locomotion test sequence complete! Policy is now running in background.")
        logger.info("Use robot.stop_locomotion_thread() to stop the policy.")

    def init_groot_locomotion(self):
        """Initialize GR00T-style locomotion for ONNX policies (29 DOF, 15D actions)."""
        if not self.config.locomotion_control:
            logger.warning("locomotion_control is False, cannot run GR00T init")
            return
        
        logger.info("Starting GR00T locomotion initialization...")
        
        # Move legs to default position (same as regular locomotion)
        self.locomotion_move_to_default_pos()
        
        # Wait 3 seconds
        time.sleep(3.0)
        
        # Hold default leg position for 2 seconds
        self.locomotion_default_pos_state()
        
        # Start locomotion policy thread (will use groot_locomotion_run)
        logger.info("Starting GR00T locomotion policy control...")
        self.start_locomotion_thread()
        
        logger.info("GR00T locomotion initialization complete! Policy is now running.")
        logger.info("516D observations (86D × 6 frames), 15D actions (legs + waist)")


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