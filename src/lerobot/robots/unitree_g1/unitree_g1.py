import json
import logging
import select
import struct
import sys
import termios
import threading
import time
import tty
from collections import deque
from functools import cached_property
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from scipy.spatial.transform import Rotation as R
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowCmd_ as hg_LowCmd,
    LowState_ as hg_LowState,
)  # idl for g1, h1_2
from unitree_sdk2py.utils.crc import CRC

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex, G1_29_JointIndex

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

        from lerobot.robots.unitree_g1.unitree_sdk2_socket import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )  # dds

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
        print(self.msg)

        self.all_motor_q = self.get_current_motor_q()
        print(self.all_motor_q)
        logger.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger.info("Lock all joints except two arms...\n")

        arm_indices = {member.value for member in G1_29_JointArmIndex}
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

        # Initialize control flags BEFORE starting threads
        self.keyboard_thread = None
        self.keyboard_running = False
        self.locomotion_thread = None
        self.locomotion_running = False

        # Both update different parts of self.msg, both call Write()
        self.publish_thread = None
        self.ctrl_lock = threading.Lock()
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        logger.warning("Arm control publish thread started")
        self.remote_controller = self.RemoteController()


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
            "quaternion": lowstate.imu_state.quaternion,  # [w, x, y, z]
            "gyroscope": lowstate.imu_state.gyroscope,  # [x, y, z] rad/s
            "accelerometer": lowstate.imu_state.accelerometer,  # [x, y, z] m/s²
            "rpy": lowstate.imu_state.rpy,  # [roll, pitch, yaw] rad
            "temperature": lowstate.imu_state.temperature,  # °C
        }

        # Extract motor data
        motors_data = []
        for i in range(G1_29_Num_Motors):
            motor = lowstate.motor_state[i]
            motors_data.append(
                {
                    "id": i,
                    "q": motor.q,  # position (rad)
                    "dq": motor.dq,  # velocity (rad/s)
                    "tau_est": motor.tau_est,  # estimated torque (Nm)
                    "temperature": motor.temperature[0]
                    if isinstance(motor.temperature, (list, tuple))
                    else motor.temperature,  # °C
                }
            )

        return {
            "imu": imu_data,
            "motors": motors_data,
        }

    def get_observation(self) -> dict[str, Any]:
        obs_array = self.get_current_dual_arm_q()
        obs_dict = {
            f"{G1_29_JointArmIndex(motor).name}.pos": val
            for motor, val in zip(G1_29_JointArmIndex, obs_array, strict=True)
        }

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
        # need a any to any teleoperator solution. i wanna teleoperate a horse with a shoe. action
        # to action mapping, when you do teleoperate. the keys that are left empty are just set to 0
        # also what would be fun is finding all sorts of robots and adding them to lerobot, see if people do the same.
        # then teleop them wiuth the glove hehe
        # then we get ALL THE DATA
        if self.is_calibrated:
            uncalibrated_action = action.copy()
            action = self.invert_calibration(action)
            # if an action was 0.5 write 0 in its place
            for key, value in uncalibrated_action.items():
                if value == 0.5:
                    action[key] = 0.0
            # check if action is within bounds
            for key, value in action.items():
                if value < self.calibration[key]["range_min"] or value > self.calibration[key]["range_max"]:
                    raise ValueError(
                        f"Action value {value} for {key} is out of bounds, actions are not normalized"
                    )
        if self.freeze_body:
            arm_joint_indices = set(range(15, 29))  # 15–28 are arms
            for jid in G1_29_JointIndex:
                if jid.value not in arm_joint_indices:
                    self.msg.motor_cmd[jid].mode = 1
                    self.msg.motor_cmd[jid].q = 0.0
                    self.msg.motor_cmd[jid].dq = 0.0
                    self.msg.motor_cmd[jid].tau = 0.0

        action_np = np.stack([v for v in action.values()])
        # action_np is just zeros
        # action_np = np.zeros(14)
        # print(action_np)
        # exit()

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
