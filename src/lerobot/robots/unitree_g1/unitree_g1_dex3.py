#!/usr/bin/env python
"""
Unitree G1 Dex3 Robot - Unified DDS Architecture

This implementation uses the same DDS threading pattern as the body/arm code,
eliminating the multiprocessing complexity of the original Dex3_1_Controller.
"""

from dataclasses import dataclass, field
import logging
import threading
import time
import numpy as np
from functools import cached_property

from .unitree_g1 import UnitreeG1, UnitreeG1Config
from .g1_hand_utils import (
    Dex3_1_Left_JointIndex, 
    Dex3_1_Right_JointIndex,
    Dex3_Num_Motors,
    DEX3_LEFT_LOWER_LIMITS,
    DEX3_LEFT_UPPER_LIMITS,
    DEX3_RIGHT_LOWER_LIMITS,
    DEX3_RIGHT_UPPER_LIMITS,
    kTopicDex3LeftCommand,
    kTopicDex3RightCommand,
    kTopicDex3LeftState,
    kTopicDex3RightState,
)

from lerobot.processor import RobotAction, RobotObservation

logger = logging.getLogger(__name__)


@dataclass
class HandMotorState:
    """State of a single hand motor."""
    q: float = 0.0  # position


@dataclass
class HandState:
    """State of a single hand (7 motors for Dex3-1)."""
    motor_state: list[HandMotorState] = field(
        default_factory=lambda: [HandMotorState() for _ in range(Dex3_Num_Motors)]
    )


@dataclass
class UnitreeG1Dex3Config(UnitreeG1Config):
    """Configuration for Unitree G1 with Dex3-1 hands."""
    hand_kp: float = 1.5  # Position gain for hand motors
    hand_kd: float = 0.2  # Damping gain for hand motors
    hand_control_dt: float = 0.01  # 100 Hz control loop


class UnitreeG1Dex3(UnitreeG1):
    """
    Unitree G1 Robot with Dex3-1 Dexterous Hands.
    
    Uses the same DDS threading architecture as the body motors:
    - A background thread subscribes to hand state at 100Hz
    - Hand commands are published directly via DDS
    """
    config_class = UnitreeG1Dex3Config
    name = "unitree_g1_dex3"

    def __init__(self, config: UnitreeG1Dex3Config):
        super().__init__(config)
        
        # Hand state (similar to _lowstate for body)
        self._left_hand_state: HandState | None = None
        self._right_hand_state: HandState | None = None
        
        # Threading control
        self._hand_shutdown_event = threading.Event()
        self._hand_subscribe_thread: threading.Thread | None = None
        
        # DDS publishers/subscribers (initialized in connect)
        self._left_hand_cmd_pub = None
        self._right_hand_cmd_pub = None
        self._left_hand_state_sub = None
        self._right_hand_state_sub = None
        
        # Command messages (initialized in connect)
        self._left_hand_msg = None
        self._right_hand_msg = None
        
        # Joint name mapping (URDF-compatible names)
        self.left_hand_joint_names = [
             "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
             "left_hand_middle_0_joint", "left_hand_middle_1_joint",
             "left_hand_index_0_joint", "left_hand_index_1_joint"
        ]
        self.right_hand_joint_names = [
             "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
             "right_hand_middle_0_joint", "right_hand_middle_1_joint",
             "right_hand_index_0_joint", "right_hand_index_1_joint"
        ]

    def _subscribe_hand_state(self):
        """
        Background thread that polls hand state via DDS at ~100Hz.
        Similar to _subscribe_motor_state() in UnitreeG1.
        """
        while not self._hand_shutdown_event.is_set():
            start_time = time.time()
            
            # Read left hand state
            left_msg = self._left_hand_state_sub.Read()
            if left_msg is not None:
                left_state = HandState()
                for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
                    left_state.motor_state[idx].q = left_msg.motor_state[joint_id].q
                self._left_hand_state = left_state
            
            # Read right hand state
            right_msg = self._right_hand_state_sub.Read()
            if right_msg is not None:
                right_state = HandState()
                for idx, joint_id in enumerate(Dex3_1_Right_JointIndex):
                    right_state.motor_state[idx].q = right_msg.motor_state[joint_id].q
                self._right_hand_state = right_state
            
            # Maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.hand_control_dt - elapsed)
            time.sleep(sleep_time)

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot body and hands."""
        # Connect body first
        super().connect(calibrate=calibrate)
        
        # Skip hand connection in simulation mode
        if self.config.is_simulation:
            logger.info("Simulation mode: Skipping Dex3 hand connection.")
            return
        
        # Import hand-specific DDS types and REAL DDS channels (not ZMQ wrapper)
        # The ZMQ wrapper (unitree_sdk2_socket) only supports body lowcmd/lowstate
        # Hand topics require real DDS via unitree_sdk2py
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
        from unitree_sdk2py.core.channel import (
            ChannelSubscriber, 
            ChannelPublisher,
            ChannelFactoryInitialize as DDS_ChannelFactoryInitialize
        )
        
        # Initialize DDS factory for hand channels
        # Note: This is separate from the ZMQ wrapper used for body
        DDS_ChannelFactoryInitialize(0)
        
        # Initialize hand state subscribers (real DDS)
        self._left_hand_state_sub = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self._left_hand_state_sub.Init()
        self._right_hand_state_sub = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self._right_hand_state_sub.Init()
        
        # Initialize hand command publishers (real DDS)
        self._left_hand_cmd_pub = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self._left_hand_cmd_pub.Init()
        self._right_hand_cmd_pub = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self._right_hand_cmd_pub.Init()
        
        # Initialize command messages with default gains
        self._left_hand_msg = unitree_hg_msg_dds__HandCmd_()
        self._right_hand_msg = unitree_hg_msg_dds__HandCmd_()
        
        kp = self.config.hand_kp
        kd = self.config.hand_kd
        
        for joint_id in Dex3_1_Left_JointIndex:
            # Mode byte: bits 0-3 = id, bits 4-6 = status (0x01 = enabled), bit 7 = timeout
            mode = (joint_id & 0x0F) | (0x01 << 4)
            self._left_hand_msg.motor_cmd[joint_id].mode = mode
            self._left_hand_msg.motor_cmd[joint_id].kp = kp
            self._left_hand_msg.motor_cmd[joint_id].kd = kd
            
        for joint_id in Dex3_1_Right_JointIndex:
            mode = (joint_id & 0x0F) | (0x01 << 4)
            self._right_hand_msg.motor_cmd[joint_id].mode = mode
            self._right_hand_msg.motor_cmd[joint_id].kp = kp
            self._right_hand_msg.motor_cmd[joint_id].kd = kd
        
        # Start hand state subscription thread
        self._hand_subscribe_thread = threading.Thread(
            target=self._subscribe_hand_state, 
            daemon=True,
            name="Dex3HandStateSubscriber"
        )
        self._hand_subscribe_thread.start()
        
        # Wait for first hand state
        timeout = 3.0
        start = time.time()
        while self._left_hand_state is None or self._right_hand_state is None:
            if time.time() - start > timeout:
                logger.warning("Timeout waiting for Dex3 hand state. Hands may not be connected.")
                break
            time.sleep(0.01)
        
        if self._left_hand_state is not None and self._right_hand_state is not None:
            logger.info("Connected to Dex3 Hands.")
        else:
            logger.warning("Dex3 Hands not fully connected - hand state unavailable.")

    def disconnect(self):
        """Disconnect from robot body and hands."""
        # Signal hand thread to stop
        self._hand_shutdown_event.set()
        
        # Wait for hand thread to finish
        if self._hand_subscribe_thread is not None:
            self._hand_subscribe_thread.join(timeout=2.0)
            if self._hand_subscribe_thread.is_alive():
                logger.warning("Hand subscribe thread did not stop cleanly")
        
        # Disconnect body
        super().disconnect()

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action space including hand joints."""
        features = super().action_features
        for name in self.left_hand_joint_names:
            features[f"{name}.q"] = float
        for name in self.right_hand_joint_names:
            features[f"{name}.q"] = float
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation space including hand joints."""
        features = super().observation_features
        for name in self.left_hand_joint_names:
            features[f"{name}.q"] = float
        for name in self.right_hand_joint_names:
            features[f"{name}.q"] = float
        return features

    def get_observation(self) -> RobotObservation:
        """Get observation including hand joint positions."""
        obs = super().get_observation()
        
        # Add left hand state
        if self._left_hand_state is not None:
            for i, name in enumerate(self.left_hand_joint_names):
                obs[f"{name}.q"] = float(self._left_hand_state.motor_state[i].q)
        
        # Add right hand state
        if self._right_hand_state is not None:
            for i, name in enumerate(self.right_hand_joint_names):
                obs[f"{name}.q"] = float(self._right_hand_state.motor_state[i].q)
        
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send action to robot including hand commands."""
        # Send body action
        super().send_action(action)
        
        # Check if we have hand publishers
        if self._left_hand_cmd_pub is None or self._right_hand_cmd_pub is None:
            return action
        
        # Check if action contains hand commands
        first_joint = self.left_hand_joint_names[0]
        if f"{first_joint}.q" not in action:
            return action
        
        # Extract and clamp left hand targets
        left_q = np.zeros(Dex3_Num_Motors)
        for i, name in enumerate(self.left_hand_joint_names):
            left_q[i] = action.get(f"{name}.q", 0.0)
        left_q = np.clip(left_q, DEX3_LEFT_LOWER_LIMITS, DEX3_LEFT_UPPER_LIMITS)
        
        # Extract and clamp right hand targets
        right_q = np.zeros(Dex3_Num_Motors)
        for i, name in enumerate(self.right_hand_joint_names):
            right_q[i] = action.get(f"{name}.q", 0.0)
        right_q = np.clip(right_q, DEX3_RIGHT_LOWER_LIMITS, DEX3_RIGHT_UPPER_LIMITS)
        
        # Update command messages
        for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
            self._left_hand_msg.motor_cmd[joint_id].q = left_q[idx]
        for idx, joint_id in enumerate(Dex3_1_Right_JointIndex):
            self._right_hand_msg.motor_cmd[joint_id].q = right_q[idx]
        
        # Publish commands
        self._left_hand_cmd_pub.Write(self._left_hand_msg)
        self._right_hand_cmd_pub.Write(self._right_hand_msg)
        
        return action
