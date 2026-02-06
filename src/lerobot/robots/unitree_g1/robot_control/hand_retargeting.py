"""
Hand Retargeting for Unitree Dex3 Hands using dex_retargeting library.
Ported from prometheus/src/xr_teleoperate/teleop/robot_control/hand_retargeting.py

Maps VR hand tracking keypoints to Dex3 finger joint positions.
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

try:
    from dex_retargeting import RetargetingConfig
    HAS_DEX_RETARGETING = True
except ImportError:
    HAS_DEX_RETARGETING = False
    RetargetingConfig = None

logger = logging.getLogger(__name__)

# Asset paths - check G1_ASSETS_DIR env var first, then fall back to local assets
# This allows assets to live in prometheus-vla while lerobot remains asset-free
_ENV_ASSETS_DIR = os.environ.get("G1_ASSETS_DIR")
if _ENV_ASSETS_DIR:
    ASSETS_DIR = Path(_ENV_ASSETS_DIR)
else:
    ASSETS_DIR = Path(__file__).parent.parent / "assets"


class HandType(Enum):
    """Supported hand types for retargeting."""
    UNITREE_DEX3 = "unitree_hand/unitree_dex3.yml"


class HandRetargeting:
    """
    Hand retargeting for Unitree Dex3 dexterous hands.
    
    Uses the dex_retargeting library to map VR hand keypoints (21 points × 3D)
    to robot hand joint positions (7 joints per hand).
    """
    
    # Dex3 joint names in API order (matching the DDS message structure)
    LEFT_DEX3_JOINT_NAMES = [
        'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
        'left_hand_middle_0_joint', 'left_hand_middle_1_joint',
        'left_hand_index_0_joint', 'left_hand_index_1_joint'
    ]
    
    # Right hand: thumb, INDEX, middle (matching Dex3_1_Right_JointIndex enum)
    RIGHT_DEX3_JOINT_NAMES = [
        'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
        'right_hand_index_0_joint', 'right_hand_index_1_joint',
        'right_hand_middle_0_joint', 'right_hand_middle_1_joint'
    ]
    
    def __init__(self, hand_type: HandType = HandType.UNITREE_DEX3):
        """
        Initialize hand retargeting.
        
        Args:
            hand_type: Type of hand to retarget to
        """
        if not HAS_DEX_RETARGETING:
            raise ImportError(
                "dex_retargeting is required for hand retargeting. "
                "Install with: pip install dex-retargeting"
            )
        
        # Set URDF base directory
        RetargetingConfig.set_default_urdf_dir(str(ASSETS_DIR))
        
        config_path = ASSETS_DIR / hand_type.value
        if not config_path.exists():
            raise FileNotFoundError(
                f"Hand retargeting config not found at {config_path}. "
                "Ensure assets are copied from prometheus/src/xr_teleoperate/assets/"
            )
        
        # Load config
        with config_path.open('r') as f:
            self.cfg = yaml.safe_load(f)
        
        if 'left' not in self.cfg or 'right' not in self.cfg:
            raise ValueError("Config must contain 'left' and 'right' keys")
        
        # Build retargeters
        left_config = RetargetingConfig.from_dict(self.cfg['left'])
        right_config = RetargetingConfig.from_dict(self.cfg['right'])
        
        self.left_retargeting = left_config.build()
        self.right_retargeting = right_config.build()
        
        # Get joint name mappings
        self.left_joint_names = self.left_retargeting.joint_names
        self.right_joint_names = self.right_retargeting.joint_names
        
        # Create mapping from retargeting output to hardware API order
        self.left_to_hardware = [
            self.left_joint_names.index(name) 
            for name in self.LEFT_DEX3_JOINT_NAMES
        ]
        self.right_to_hardware = [
            self.right_joint_names.index(name) 
            for name in self.RIGHT_DEX3_JOINT_NAMES
        ]
        
        logger.info(
            f"HandRetargeting initialized for {hand_type.name} "
            f"({len(self.left_joint_names)} left joints, "
            f"{len(self.right_joint_names)} right joints)"
        )
    
    def retarget_left(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Retarget left hand keypoints to joint positions.
        
        Args:
            keypoints: (21, 3) array of hand keypoints in wrist-relative frame
            
        Returns:
            Array of 7 joint positions in hardware API order
        """
        joint_pos = self.left_retargeting.retarget(keypoints)
        # Reorder to hardware API order
        return joint_pos[self.left_to_hardware]
    
    def retarget_right(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Retarget right hand keypoints to joint positions.
        
        Args:
            keypoints: (21, 3) array of hand keypoints in wrist-relative frame
            
        Returns:
            Array of 7 joint positions in hardware API order
        """
        joint_pos = self.right_retargeting.retarget(keypoints)
        # Reorder to hardware API order
        return joint_pos[self.right_to_hardware]
    
    def retarget(
        self, 
        left_keypoints: np.ndarray, 
        right_keypoints: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retarget both hands simultaneously.
        
        Args:
            left_keypoints: (21, 3) left hand keypoints
            right_keypoints: (21, 3) right hand keypoints
            
        Returns:
            Tuple of (left_joints, right_joints), each with 7 values
        """
        left_joints = self.retarget_left(left_keypoints)
        right_joints = self.retarget_right(right_keypoints)
        return left_joints, right_joints
