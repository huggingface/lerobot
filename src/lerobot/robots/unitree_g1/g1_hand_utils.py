# Ported from prometheus/src/xr_teleoperate/teleop/robot_control/robot_hand_unitree.py

import numpy as np
import time
import threading
import logging
from multiprocessing import Process, Array, Value, Lock
from enum import IntEnum

# Requires unitree_sdk2py
try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
except ImportError:
    pass 

logger = logging.getLogger(__name__)

# Constants
Dex3_Num_Motors = 7
Dex3_Num_Sensors = (6*4 + 3*3) * 2
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

DEX3_LEFT_LOWER_LIMITS = np.array([-1.047, -0.724,  0.0, -1.57, -1.74, -1.57, -1.74], dtype=float)
DEX3_LEFT_UPPER_LIMITS = np.array([ 1.047,  0.920,  1.74,  0.0,  0.0,  0.0,  0.0], dtype=float)
DEX3_RIGHT_LOWER_LIMITS = np.array([-1.047, -0.920, -1.74,  0.0,  0.0,  0.0,  0.0], dtype=float)
DEX3_RIGHT_UPPER_LIMITS = np.array([ 1.047,  0.724,  0.0,  1.57,  1.74,  1.57,  1.74], dtype=float)

# Used for retargeting logic
DEX3_LEFT_MIN_JOINTs = DEX3_LEFT_LOWER_LIMITS
DEX3_LEFT_MAX_JOINTs = DEX3_LEFT_UPPER_LIMITS
# Original code had different arrays for retargeting but similar values. 
# Using simplified mapping for now.
DEX3_RIGHT_MIN_JOINTs = DEX3_RIGHT_LOWER_LIMITS
DEX3_RIGHT_MAX_JOINTs = DEX3_RIGHT_UPPER_LIMITS

# Special initial/final for right hand retargeting logic from original code
DEX3_RIGHT_INITIAL_JOINTs = np.array([ 1.047198, 0.724312, 0.0, 0.0, 0.0, 0.0, 0.0])
DEX3_RIGHT_FINAL_JOINTs = np.array([-1.047198, -0.920, -1.745, 1.57, 1.745, 1.57, 1.745])


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

class Dex3_1_Left_PressureTemperatureSensors(IntEnum):
    kLeftThumbBase = 0
    kLeftThumbTip = 1
    kLeftMiddleBase = 2
    kLeftMiddleTip = 3
    kLeftIndexBase = 4
    kLeftIndexTip = 5
    kLeftPalm0 = 6
    kLeftPalm1 = 7
    kLeftPalm2 = 8

class Dex3_1_Right_PressureTemperatureSensors(IntEnum):
    kRightThumbBase = 0
    kRightThumbTip = 1
    kRightMiddleBase = 2
    kRightMiddleTip = 3
    kRightIndexBase = 4
    kRightIndexTip = 5
    kRightPalm0 = 6
    kRightPalm1 = 7
    kRightPalm2 = 8

sensor_index = {
    Dex3_1_Left_PressureTemperatureSensors.kLeftThumbBase: [0, 2, 9, 11],
    Dex3_1_Left_PressureTemperatureSensors.kLeftThumbTip: [3, 6, 8],
    Dex3_1_Left_PressureTemperatureSensors.kLeftMiddleBase: [0, 2, 9, 11],
    Dex3_1_Left_PressureTemperatureSensors.kLeftMiddleTip: [3, 6, 8],
    Dex3_1_Left_PressureTemperatureSensors.kLeftIndexBase: [0, 2, 9, 11],
    Dex3_1_Left_PressureTemperatureSensors.kLeftIndexTip: [3, 6, 8],
    Dex3_1_Left_PressureTemperatureSensors.kLeftPalm0: [0, 2, 9, 11],
    Dex3_1_Left_PressureTemperatureSensors.kLeftPalm1: [0, 2, 9, 11],
    Dex3_1_Left_PressureTemperatureSensors.kLeftPalm2: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightThumbBase: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightThumbTip: [3, 6, 8],
    Dex3_1_Right_PressureTemperatureSensors.kRightMiddleBase: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightMiddleTip: [3, 6, 8],
    Dex3_1_Right_PressureTemperatureSensors.kRightIndexBase: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightIndexTip: [3, 6, 8],
    Dex3_1_Right_PressureTemperatureSensors.kRightPalm0: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightPalm1: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightPalm2: [0, 2, 9, 11],
}

def dex3_pinch_retarget(opening_left, thumb_left, opening_right, thumb_right):
    # Simplified retargeting logic from original code
    left_qpos = DEX3_LEFT_MIN_JOINTs + (DEX3_LEFT_MAX_JOINTs - DEX3_LEFT_MIN_JOINTs) * opening_left
    left_qpos[0] = DEX3_LEFT_MIN_JOINTs[0] + (DEX3_LEFT_MAX_JOINTs[0] - DEX3_LEFT_MIN_JOINTs[0]) * thumb_left

    # Note: Assumes specific joint order and mapping logic from original file
    right_qpos = DEX3_RIGHT_FINAL_JOINTs + (DEX3_RIGHT_INITIAL_JOINTs - DEX3_RIGHT_FINAL_JOINTs) * opening_right
    right_qpos[0] = DEX3_RIGHT_MIN_JOINTs[0] + (DEX3_RIGHT_MAX_JOINTs[0] - DEX3_RIGHT_MIN_JOINTs[0]) * thumb_right

    return left_qpos, right_qpos

class Dex3_1_Controller:
    def __init__(self, fps=100.0, simulation_mode=False):
        self.fps = fps
        self.simulation_mode = simulation_mode
        self.running = False
        
        # Shared Arrays
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors + Dex3_Num_Sensors, lock=True)  
        self.right_hand_state_array = Array('d', Dex3_Num_Motors + Dex3_Num_Sensors, lock=True)
        
        # Output arrays for robot class to read
        self.dual_hand_data_lock = Lock()
        self.dual_hand_state_array = Array('d', 14, lock=False)   # left(7) + right(7)
        self.dual_hand_action_array = Array('d', 14, lock=False)  # left(7) + right(7)
        
        # Tactile
        self.dual_hand_tactile_lock = Lock()
        self.dual_hand_tactile_state_array = Array('d', 132, lock=False)

        # Input values (to be written by robot class from action)
        # We can store target q here.
        # Original code computed q_target from retargeting logic inside the process.
        # Here we will allow passing q_target directly OR retargeting inputs.
        # For simplicity, let's assume we pass Q targets directly if 'hand' mode is externalized,
        # OR we pass the raw controller inputs if we want the process to handle retargeting.
        # Let's support passing Q targets for generality.
        self.left_q_target_shared = Array('d', Dex3_Num_Motors, lock=True)
        self.right_q_target_shared = Array('d', Dex3_Num_Motors, lock=True)

        self.subscribe_state_thread = None
        self.control_process_h = None

    def start(self):
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # Connect DDS
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state, daemon=True)
        self.subscribe_state_thread.start()
        
        self.control_process_h = Process(target=self.control_process, args=(
            self.left_hand_state_array, self.right_hand_state_array,
            self.left_q_target_shared, self.right_q_target_shared,
            self.dual_hand_data_lock, self.dual_hand_state_array, self.dual_hand_action_array,
            self.dual_hand_tactile_lock, self.dual_hand_tactile_state_array,
            self.fps
        ), daemon=True)
        self.control_process_h.start()
    
    def stop(self):
        if self.control_process_h and self.control_process_h.is_alive():
            self.control_process_h.terminate()
            self.control_process_h.join()
    
    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q

                # Update tactile (simplified copy from original)
                idx = 0
                for area_idx, id in enumerate(Dex3_1_Left_PressureTemperatureSensors):
                    for sensor_idx in sensor_index[id]:
                        self.left_hand_state_array[idx + Dex3_Num_Motors] = left_hand_msg.press_sensor_state[area_idx].pressure[sensor_idx]
                        idx += 1
                for area_idx, id in enumerate(Dex3_1_Left_PressureTemperatureSensors):
                    for sensor_idx in sensor_index[id]:
                        self.left_hand_state_array[idx + Dex3_Num_Motors] = left_hand_msg.press_sensor_state[area_idx].temperature[sensor_idx]
                        idx += 1
                
                idx = 0
                for area_idx, id in enumerate(Dex3_1_Right_PressureTemperatureSensors):
                    for sensor_idx in sensor_index[id]:
                        self.right_hand_state_array[idx + Dex3_Num_Motors] = right_hand_msg.press_sensor_state[area_idx].pressure[sensor_idx]
                        idx += 1
                for area_idx, id in enumerate(Dex3_1_Right_PressureTemperatureSensors):
                    for sensor_idx in sensor_index[id]:
                        self.right_hand_state_array[idx + Dex3_Num_Motors] = right_hand_msg.press_sensor_state[area_idx].temperature[sensor_idx]
                        idx += 1
            time.sleep(0.002)

    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def control_process(self, left_hand_state_array, right_hand_state_array,
                        left_q_target_shared, right_q_target_shared,
                        dual_hand_data_lock, dual_hand_state_array_out, dual_hand_action_array_out,
                        dual_hand_tactile_lock, dual_hand_tactile_state_array_out, fps):
        
        # Init publishers again inside process if needed
        left_msg  = unitree_hg_msg_dds__HandCmd_()
        right_msg = unitree_hg_msg_dds__HandCmd_()
        
        kp = 1.5
        kd = 0.2
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            left_msg.motor_cmd[id].mode = ris_mode._mode_to_uint8()
            left_msg.motor_cmd[id].kp   = kp
            left_msg.motor_cmd[id].kd   = kd
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            right_msg.motor_cmd[id].mode = ris_mode._mode_to_uint8()
            right_msg.motor_cmd[id].kp   = kp
            right_msg.motor_cmd[id].kd   = kd

        lh_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        lh_publisher.Init()
        rh_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        rh_publisher.Init()

        while True:
            start_time = time.time()
            
            # Read targets
            with left_q_target_shared.get_lock():
                left_cmd = np.array(left_q_target_shared[:])
            with right_q_target_shared.get_lock():
                right_cmd = np.array(right_q_target_shared[:])

            # Clamp
            left_cmd = np.clip(left_cmd, DEX3_LEFT_LOWER_LIMITS, DEX3_LEFT_UPPER_LIMITS)
            right_cmd = np.clip(right_cmd, DEX3_RIGHT_LOWER_LIMITS, DEX3_RIGHT_UPPER_LIMITS)

            # Update Msg
            for idx, id in enumerate(Dex3_1_Left_JointIndex):
                left_msg.motor_cmd[id].q = left_cmd[idx]
            for idx, id in enumerate(Dex3_1_Right_JointIndex):
                right_msg.motor_cmd[id].q = right_cmd[idx]

            # Publish
            lh_publisher.Write(left_msg)
            rh_publisher.Write(right_msg)

            # Update Output Arrays
            state_data = np.concatenate((np.array(left_hand_state_array[:Dex3_Num_Motors]), np.array(right_hand_state_array[:Dex3_Num_Motors])))
            action_data = np.concatenate((left_cmd, right_cmd))

            with dual_hand_data_lock:
                dual_hand_state_array_out[:] = state_data
                dual_hand_action_array_out[:] = action_data

            tactile = np.concatenate((np.array(left_hand_state_array[Dex3_Num_Motors:]), np.array(right_hand_state_array[Dex3_Num_Motors:])))
            with dual_hand_tactile_lock:
                dual_hand_tactile_state_array_out[:] = tactile

            # Sleep
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / fps) - elapsed)
            time.sleep(sleep_time)

