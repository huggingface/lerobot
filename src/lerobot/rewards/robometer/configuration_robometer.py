# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons
Paper: https://arxiv.org/abs/2603.02115
Models: robometer/Robometer-4B
"""

from dataclasses import dataclass, field

from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig


@RewardModelConfig.register_subclass("robometer")
@dataclass
class RobometerConfig(RewardModelConfig):
    """Configuration for the Robometer vision-language reward model.

    Robometer is a pre-trained reward model (based on the Qwen-3-VL-4B-Instruct backbone) 
    that computes dense progress values from robot rollout videos and task descriptions. 
    Scores are scaled to float rewards in [0, 1].

    Args:
        name: model name
        task_key: Batch key containing the task language instruction string(s).
        task_instruction: If task_key is None, must provide task_instruction string: e.g., task_instruction="pick up the red cube"
        image_key: Batch key for the primary camera image(s). Accepts tensors of
            shape (B, T, C, H, W) for video.
        num_reasoning_frames: The video is evenly downsampled to this number of frames before computing rewards.
        view_type: Camera view types used for computing rewards: "external", "wrist", or "external and wrist".
    """

    name: str = "robometer"
    device: str = "cpu"
    task_key: str = "observation.language_instruction"
    task_instruction: str = ""  
    image_key: str = "observation.images.side"
    num_reasoning_frames: int = 10  
    view_type: str = "external"  

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=1e-5, weight_decay=1e-2)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

