"""
TeleVuer to G1 Dex3 Processor.

Converts VR teleoperation output (wrist poses, hand keypoints) from TeleVuerTeleoperator
to G1 Dex3 robot joint positions using IK and hand retargeting.
"""

import logging
from typing import Any, Optional

import numpy as np

from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.robots.unitree_g1.g1_utils import (
    ARM_JOINT_NAMES,
    LEFT_HAND_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
)

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies (pinocchio, casadi, dex_retargeting)
_arm_ik = None
_hand_retargeting = None


def _get_arm_ik():
    """Lazy load arm IK solver."""
    global _arm_ik
    if _arm_ik is None:
        from lerobot.robots.unitree_g1.robot_control import G1_29_ArmIK
        _arm_ik = G1_29_ArmIK()
        logger.info("Loaded G1_29_ArmIK solver")
    return _arm_ik


def _get_hand_retargeting():
    """Lazy load hand retargeting."""
    global _hand_retargeting
    if _hand_retargeting is None:
        from lerobot.robots.unitree_g1.robot_control import HandRetargeting, HandType
        _hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        logger.info("Loaded Dex3 hand retargeting")
    return _hand_retargeting


class TeleVuerToG1Dex3Processor:
    """
    Processor that converts TeleVuer VR output to G1 Dex3 robot actions.
    
    Takes VR wrist poses and hand keypoints, runs them through:
    1. Arm IK solver (Pinocchio + CasADi) → 14 arm joint positions
    2. Hand retargeting (dex_retargeting) → 14 hand joint positions (7 per hand)
    
    Usage:
        processor = TeleVuerToG1Dex3Processor()
        robot_action = processor(vr_action, robot_obs)
    """
    
    # Use consolidated constants from g1_utils
    ARM_JOINT_NAMES = ARM_JOINT_NAMES
    LEFT_HAND_JOINT_NAMES = LEFT_HAND_JOINT_NAMES
    RIGHT_HAND_JOINT_NAMES = RIGHT_HAND_JOINT_NAMES
    
    def __init__(self, lazy_init: bool = True):
        """
        Initialize the processor.
        
        Args:
            lazy_init: If True, IK and retargeting are loaded on first use.
                       If False, they are loaded immediately (may be slow).
        """
        self._lazy_init = lazy_init
        if not lazy_init:
            _get_arm_ik()
            _get_hand_retargeting()
    
    def __call__(
        self, 
        vr_action: RobotAction, 
        obs: Optional[RobotObservation] = None
    ) -> RobotAction:
        """
        Process VR action to robot action.
        
        Args:
            vr_action: Output from TeleVuerTeleoperator.get_action() containing:
                - left_arm_pose: 4x4 left wrist transformation matrix
                - right_arm_pose: 4x4 right wrist transformation matrix  
                - left_hand_pos: (25, 3) left hand keypoints (we use first 21)
                - right_hand_pos: (25, 3) right hand keypoints
            obs: Current robot observation (optional, used for warm start)
            
        Returns:
            Dictionary mapping joint names to positions
        """
        robot_action = {}
        
        # Extract arm poses
        left_wrist = vr_action.get("left_arm_pose")
        right_wrist = vr_action.get("right_arm_pose")
        
        if left_wrist is not None and right_wrist is not None:
            # Get current arm positions for warm start
            current_arm_q = None
            if obs is not None:
                current_arm_q = self._extract_current_arm_q(obs)
            
            # Run IK
            arm_ik = _get_arm_ik()
            arm_q, _ = arm_ik.solve_ik(left_wrist, right_wrist, current_arm_q)
            
            # Map to joint names
            for i, name in enumerate(self.ARM_JOINT_NAMES):
                robot_action[name] = float(arm_q[i])
        
        # Extract hand keypoints
        left_hand_pos = vr_action.get("left_hand_pos")
        right_hand_pos = vr_action.get("right_hand_pos")
        
        if left_hand_pos is not None and right_hand_pos is not None:
            hand_retargeting = _get_hand_retargeting()
            
            # Use first 21 keypoints (standard hand skeleton)
            left_keypoints = np.array(left_hand_pos[:21])
            right_keypoints = np.array(right_hand_pos[:21])
            
            # Run retargeting
            left_hand_q = hand_retargeting.retarget_left(left_keypoints)
            right_hand_q = hand_retargeting.retarget_right(right_keypoints)
            
            # Map to joint names
            for i, name in enumerate(self.LEFT_HAND_JOINT_NAMES):
                robot_action[name] = float(left_hand_q[i])
            for i, name in enumerate(self.RIGHT_HAND_JOINT_NAMES):
                robot_action[name] = float(right_hand_q[i])
        
        # Copy through pinch values (useful for gripper-like actions)
        if "left_pinch_value" in vr_action:
            robot_action["left_pinch_value"] = vr_action["left_pinch_value"]
        if "right_pinch_value" in vr_action:
            robot_action["right_pinch_value"] = vr_action["right_pinch_value"]
        
        return robot_action
    
    def _extract_current_arm_q(self, obs: RobotObservation) -> Optional[np.ndarray]:
        """Extract current arm joint positions from observation for warm start."""
        try:
            arm_q = []
            for name in self.ARM_JOINT_NAMES:
                # Try different key formats
                key = f"{name}.q"
                if key in obs:
                    arm_q.append(obs[key])
                elif name in obs:
                    arm_q.append(obs[name])
            
            if len(arm_q) == len(self.ARM_JOINT_NAMES):
                return np.array(arm_q)
        except Exception:
            pass
        return None
    
    def reset(self) -> None:
        """Reset processor state."""
        global _arm_ik
        if _arm_ik is not None:
            _arm_ik.reset()


def make_televuer_g1_dex3_processor() -> TeleVuerToG1Dex3Processor:
    """Factory function to create a TeleVuer to G1 Dex3 processor."""
    return TeleVuerToG1Dex3Processor()
