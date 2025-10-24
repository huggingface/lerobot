import numpy as np
import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState  # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as go_LowCmd, LowState_ as go_LowState  # idl for h1
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

import logging_mp

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


class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]

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


class G1_29_ArmController:
    def __init__(self, motion_mode=False, simulation_mode=False):
        logger_mp.info("Initialize G1_29_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 40.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 100.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

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
            logger_mp.warning("[G1_29_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[G1_29_ArmController] Subscribe dds ok.")

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

        logger_mp.info("Lock OK!\n") #motors are not locked x
        # for i in range(10000):
        #     print(self.get_current_motor_q())
        #     time.sleep(0.05) 
        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_29_ArmController OK!\n")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)

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

