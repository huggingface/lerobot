from dataclasses import dataclass
import logging
import time
import numpy as np
from functools import cached_property

from .unitree_g1 import UnitreeG1, UnitreeG1Config
from .g1_hand_utils import (
    Dex3_1_Controller, 
    Dex3_1_Left_JointIndex, 
    Dex3_1_Right_JointIndex,
    Dex3_Num_Motors
)

from lerobot.processor import RobotAction, RobotObservation

logger = logging.getLogger(__name__)

@dataclass
class UnitreeG1Dex3Config(UnitreeG1Config):
    pass

class UnitreeG1Dex3(UnitreeG1):
    config_class = UnitreeG1Dex3Config
    name = "unitree_g1_dex3"

    def __init__(self, config: UnitreeG1Dex3Config):
        super().__init__(config)
        self.hand_ctrl: Dex3_1_Controller | None = None
        
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

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        
        if not self.config.is_simulation:
             self.hand_ctrl = Dex3_1_Controller(fps=100.0, simulation_mode=False)
             self.hand_ctrl.start()
             logger.info("Connected to Dex3 Hands.")

    def disconnect(self):
        if self.hand_ctrl:
            self.hand_ctrl.stop()
            self.hand_ctrl = None
        super().disconnect()

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = super().action_features
        for name in self.left_hand_joint_names:
            features[f"{name}.q"] = float
        for name in self.right_hand_joint_names:
            features[f"{name}.q"] = float
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = super().observation_features
        for name in self.left_hand_joint_names:
            features[f"{name}.q"] = float
        for name in self.right_hand_joint_names:
            features[f"{name}.q"] = float
        return features

    def get_observation(self) -> RobotObservation:
        obs = super().get_observation()
        
        if self.hand_ctrl:
             left_state = np.array(self.hand_ctrl.left_hand_state_array[:Dex3_Num_Motors])
             for i, name in enumerate(self.left_hand_joint_names):
                 obs[f"{name}.q"] = float(left_state[i])
             
             right_state = np.array(self.hand_ctrl.right_hand_state_array[:Dex3_Num_Motors])
             for i, name in enumerate(self.right_hand_joint_names):
                 obs[f"{name}.q"] = float(right_state[i])
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        super().send_action(action)
        
        if self.hand_ctrl:
             left_q_target = np.zeros(Dex3_Num_Motors)
             right_q_target = np.zeros(Dex3_Num_Motors)
             
             has_hand_action = False
             first_joint = self.left_hand_joint_names[0]
             if f"{first_joint}.q" in action:
                 has_hand_action = True
                 for i, name in enumerate(self.left_hand_joint_names):
                     left_q_target[i] = action.get(f"{name}.q", 0.0)
                 for i, name in enumerate(self.right_hand_joint_names):
                     right_q_target[i] = action.get(f"{name}.q", 0.0)
             
             if has_hand_action:
                 with self.hand_ctrl.left_q_target_shared.get_lock():
                     self.hand_ctrl.left_q_target_shared[:] = left_q_target
                 with self.hand_ctrl.right_q_target_shared.get_lock():
                     self.hand_ctrl.right_q_target_shared[:] = right_q_target
        
        return action
