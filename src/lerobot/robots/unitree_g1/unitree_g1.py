import json
import logging
import struct
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowCmd_ as hg_LowCmd,
    LowState_ as hg_LowState,
)  # idl for g1, h1_2
from unitree_sdk2py.utils.crc import CRC

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex, G1_29_JointIndex
from lerobot.robots.unitree_g1.unitree_sdk2_socket import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config

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

    def __init__(self, config: UnitreeG1Config):
        super().__init__(config)

        logger.info("Initialize UnitreeG1...")

        self.config = config
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.simulation_mode = config.simulation_mode

        # Unified kp/kd arrays for all 35 motors
        self.kp = np.array(config.kp, dtype=np.float32)
        self.kd = np.array(config.kd, dtype=np.float32)

        self.arm_velocity_limit = config.arm_velocity_limit
        self.control_dt = config.control_dt

        # Initialize DDS
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
        logger.warning("[UnitreeG1] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        # Initialize all motors with unified kp/kd from config
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            self.msg.motor_cmd[id].kp = self.kp[id.value]
            self.msg.motor_cmd[id].kd = self.kd[id.value]
            self.msg.motor_cmd[id].q = self.get_current_motor_q()[id.value]

        # Both update different parts of self.msg, both call Write()
        self.publish_thread = None
        self.ctrl_lock = threading.Lock()
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        logger.warning("Arm control publish thread started")
        self.remote_controller = self.RemoteController()

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
            sleep_time = max(0, (self.control_dt - all_t_elapsed))  # maintina constant control dt
            time.sleep(sleep_time)

    def ctrl_dual_arm_go_home(self):
        """Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero."""
        logger.info("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            # self.q_target[G1_29_JointIndex.kLeftElbow] = 0.5
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = weight
                        time.sleep(0.02)
                logger.info("[G1_29_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

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
                cliped_arm_q_target = self.clip_arm_q_target(
                    arm_q_target, velocity_limit=self.arm_velocity_limit
                )

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

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{G1_29_JointArmIndex(motor).name}.pos": float for motor in G1_29_JointArmIndex}

    def calibrate(self) -> None:
        self.calibration = json.load(open("src/lerobot/robots/unitree_g1/arm_calibration.json"))
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
        if self.simulation_mode and hasattr(self, "mujoco_env"):
            logger.info("Closing MuJoCo environment...")
            print(self.mujoco_env)
            self.mujoco_env["hub_env"][0].envs[0].kill_sim()

        logger.info(f"{self} disconnected.")

    def get_observation(self) -> dict[str, Any]:
        return self.lowstate_buffer.GetData()

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
        self.msg.crc = self.crc.Crc(action)
        self.lowcmd_publisher.Write(action)

    def reset_legs(self):
        """Move robot legs to default standing position over 2 seconds (arms are not moved)."""
        total_time = 2.0
        num_step = int(total_time / self.control_dt)

        # Only control legs, not arms
        dof_idx = self.config.leg_joint2motor_idx
        default_pos = np.array(self.config.default_leg_angles, dtype=np.float32)
        dof_size = len(dof_idx)

        # Get current lowstate
        lowstate = self.lowstate_buffer.GetData()
        if lowstate is None:
            logger.error("Cannot get lowstate")
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
                self.msg.motor_cmd[motor_idx].kp = self.kp[motor_idx]
                self.msg.motor_cmd[motor_idx].kd = self.kd[motor_idx]
                self.msg.motor_cmd[motor_idx].tau = 0
            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
            time.sleep(self.control_dt)
        logger.info("Reached default position (legs only)")

    def get_gravity_orientation(self, quaternion):
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

    def transform_imu_data(self, waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
        """Transform IMU data from torso to pelvis frame."""
        RzWaist = R.from_euler("z", waist_yaw).as_matrix()
        R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
        R_pelvis = np.dot(R_torso, RzWaist.T)
        w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
        return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w
