# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_  # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

# for gripper
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import threading
from multiprocessing import Process, Value, Array

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


unitree_tip_indices = [4, 9, 14]  # [thumb, index, middle] in OpenXR
Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(
        self,
        left_hand_array_in,
        right_hand_array_in,
        dual_hand_data_lock=None,
        dual_hand_state_array_out=None,
        dual_hand_action_array_out=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process
        left_hand_array_in: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process
        right_hand_array_in: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process
        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array
        dual_hand_state_array_out: [output] Return left(7), right(7) hand motor state
        dual_hand_action_array_out: [output] Return left(7), right(7) hand motor action
        fps: Control frequency
        Unit_Test: Whether to enable unit testing
        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """
        logger_mp.info("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array = Array("d", Dex3_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Dex3_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array_in,
                right_hand_array_in,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array_out,
                dual_hand_action_array_out,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Dex3_1_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)

    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= self.id & 0x0F
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
        # logger_mp.debug("hand ctrl publish ok.")

    def control_process(
        self,
        left_hand_array_in,
        right_hand_array_in,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array_out=None,
        dual_hand_action_array_out=None,
    ):
        self.running = True

        # left_q_target = np.full(Dex3_Num_Motors, 0)
        # right_q_target = np.full(Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # initialize dex3-1's left hand cmd msg
        self.left_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q = q
            self.left_msg.motor_cmd[id].dq = dq
            self.left_msg.motor_cmd[id].tau = tau
            self.left_msg.motor_cmd[id].kp = kp
            self.left_msg.motor_cmd[id].kd = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode
            self.right_msg.motor_cmd[id].q = q
            self.right_msg.motor_cmd[id].dq = dq
            self.right_msg.motor_cmd[id].tau = tau
            self.right_msg.motor_cmd[id].kp = kp
            self.right_msg.motor_cmd[id].kd = kd

        try:
            while self.running:
                start_time = time.time()

                # get dual hand state
                with left_hand_array_in.get_lock():
                    left_hand_mat = np.array(left_hand_array_in[:]).copy()
                with right_hand_array_in.get_lock():
                    right_hand_mat = np.array(right_hand_array_in[:]).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                # get dual hand action
                action_data = np.concatenate((left_hand_mat, right_hand_mat))
                if dual_hand_state_array_out and dual_hand_action_array_out:
                    with dual_hand_data_lock:
                        dual_hand_state_array_out[:] = state_data
                        dual_hand_action_array_out[:] = action_data

                self.ctrl_dual_hand(left_hand_mat, right_hand_mat)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            print("Dex3_1_Controller has been closed.")


class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6


class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"


class Dex1_1_Gripper_Controller:
    def __init__(
        self,
        left_gripper_value_in,
        right_gripper_value_in,
        dual_gripper_data_lock=None,
        dual_gripper_state_out=None,
        dual_gripper_action_out=None,
        filter=True,
        fps=200.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process
        left_gripper_value_in: [input] Left ctrl data (required from XR device) to control_thread
        right_gripper_value_in: [input] Right ctrl data (required from XR device) to control_thread
        dual_gripper_data_lock: Data synchronization lock for dual_gripper_state_array and dual_gripper_action_array
        dual_gripper_state_out: [output] Return left(1), right(1) gripper motor state
        dual_gripper_action_out: [output] Return left(1), right(1) gripper motor action
        fps: Control frequency
        Unit_Test: Whether to enable unit testing
        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """

        logger_mp.info("Initialize Dex1_1_Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.gripper_sub_ready = False
        self.simulation_mode = simulation_mode

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftGripperCmb_publisher = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.LeftGripperCmb_publisher.Init()
        self.RightGripperCmb_publisher = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.RightGripperCmb_publisher.Init()

        self.LeftGripperState_subscriber = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.LeftGripperState_subscriber.Init()
        self.RightGripperState_subscriber = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.RightGripperState_subscriber.Init()

        # Shared Arrays for gripper states
        self.left_gripper_state_value = Value("d", 0.0, lock=True)
        self.right_gripper_state_value = Value("d", 0.0, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while not self.gripper_sub_ready:
            time.sleep(0.01)
            logger_mp.warning("[Dex1_1_Gripper_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex1_1_Gripper_Controller] Subscribe dds ok.")

        self.gripper_control_thread = threading.Thread(
            target=self.control_thread,
            args=(
                left_gripper_value_in,
                right_gripper_value_in,
                self.left_gripper_state_value,
                self.right_gripper_state_value,
                dual_gripper_data_lock,
                dual_gripper_state_out,
                dual_gripper_action_out,
            ),
        )
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        logger_mp.info("Initialize Dex1_1_Gripper_Controller OK!\n")

    def _subscribe_gripper_state(self):
        while True:
            left_gripper_msg = self.LeftGripperState_subscriber.Read()
            right_gripper_msg = self.RightGripperState_subscriber.Read()
            self.gripper_sub_ready = True
            if left_gripper_msg is not None and right_gripper_msg is not None:
                self.left_gripper_state_value.value = left_gripper_msg.states[0].q
                self.right_gripper_state_value.value = right_gripper_msg.states[0].q
            time.sleep(0.002)

    def ctrl_dual_gripper(self, dual_gripper_action):
        """set current left, right gripper motor cmd target q"""
        self.left_gripper_msg.cmds[0].q = dual_gripper_action[0]
        self.right_gripper_msg.cmds[0].q = dual_gripper_action[1]

        self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
        self.RightGripperCmb_publisher.Write(self.right_gripper_msg)
        # logger_mp.debug("gripper ctrl publish ok.")

    def control_thread(
        self,
        left_gripper_value_in,
        right_gripper_value_in,
        left_gripper_state_value,
        right_gripper_state_value,
        dual_hand_data_lock=None,
        dual_gripper_state_out=None,
        dual_gripper_action_out=None,
    ):
        self.running = True
        LEFT_MAPPED_MIN = 0.0  # The minimum initial motor position when the gripper closes at startup.
        RIGHT_MAPPED_MIN = 0.0  # The minimum initial motor position when the gripper closes at startup.
        # The maximum initial motor position when the gripper closes before calibration (with the rail stroke calculated as 0.6 cm/rad * 9 rad = 5.4 cm).

        dq = 0.0
        tau = 0.0
        kp = 5.00
        kd = 0.05
        # initialize gripper cmd msg
        self.left_gripper_msg = MotorCmds_()
        self.left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_gripper_msg = MotorCmds_()
        self.right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]

        self.left_gripper_msg.cmds[0].dq = dq
        self.left_gripper_msg.cmds[0].tau = tau
        self.left_gripper_msg.cmds[0].kp = kp
        self.left_gripper_msg.cmds[0].kd = kd

        self.right_gripper_msg.cmds[0].dq = dq
        self.right_gripper_msg.cmds[0].tau = tau
        self.right_gripper_msg.cmds[0].kp = kp
        self.right_gripper_msg.cmds[0].kd = kd
        try:
            while self.running:
                start_time = time.time()
                # get dual hand skeletal point state from XR device
                with left_gripper_value_in.get_lock():
                    left_gripper_value = left_gripper_value_in.value
                with right_gripper_value_in.get_lock():
                    right_gripper_value = right_gripper_value_in.value
                # get current dual gripper motor state
                dual_gripper_state = np.array([left_gripper_state_value.value, right_gripper_state_value.value])
                dual_gripper_action = np.array([left_gripper_value, right_gripper_value])

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])
                self.ctrl_dual_gripper(dual_gripper_action)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex1_1_Gripper_Controller has been closed.")


class Gripper_JointIndex(IntEnum):
    kGripper = 0
