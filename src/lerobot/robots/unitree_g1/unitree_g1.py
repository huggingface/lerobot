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

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState  # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as go_LowCmd, LowState_ as go_LowState  # idl for h1
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

import logging_mp

from lerobot.robots.unitree_g1.robot_kinematic_processor import G1_29_ArmIK


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
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

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
        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

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

