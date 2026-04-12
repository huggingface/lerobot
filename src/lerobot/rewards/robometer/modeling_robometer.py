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

Inference only: generates dense progress values between 0 and 1 for robot rollout videos with a task instruction prompt.

Requirements:
    pip install "transformers>=4.51.0" qwen-vl-utils roboreason
"""

import logging
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.robometer.configuration_robometer import RobometerConfig

logger = logging.getLogger(__name__)

# Optional heavy imports — kept at module level so patch() can target them in tests.
try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoProcessor = None  # type: ignore[assignment,misc]
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment,misc]
    _TRANSFORMERS_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info

    _QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    process_vision_info = None  # type: ignore[assignment]
    _QWEN_VL_UTILS_AVAILABLE = False

try:
    import roboreason
    
    _ROBOREASON_AVAILABLE = True
except ImportError:
    roboreason = None 
    _ROBOREASON_AVAILABLE = False



class RobometerModel(PreTrainedRewardModel):
    """Robometer vision-language reward model.

    Wraps a pretrained Robometer-4B model (Qwen3-VL-4B-Instruct backbone) for
    inference-only reward computation. The VLM backbone is always frozen.

    Usage::
        config = RobometerConfig(model_name="robometer", device="cuda")
        model = RobometerModel(config)
        rewards = model.compute_reward(batch)  # Tensor of shape (B,)
    """
    
    name = "robometer"
    config_class = RobometerConfig

    def __init__(self, config: RobometerConfig):
        super().__init__(config)
        self.config = config

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "RobometerModel requires transformers>=4.51.0 with Qwen3-VL support.\n"
                "Install with: pip install 'transformers>=4.51.0'"
            )

        if not _QWEN_VL_UTILS_AVAILABLE:
            raise ImportError(
                "RobometerModel requires the qwen-vl-utils package (for compute_reward).\nInstall with: pip install qwen-vl-utils"
            )

        if not _ROBOREASON_AVAILABLE:
            raise ImportError(
                "RobometerModel requires the roboreason package.\nInstall with: pip install roboreason"
            )
        
        from roboreason.utils.model_utils import get_model_dir
        
        # pre-downloads the HF model and stores in local cache
        get_model_dir('robometer')

    # ------------------------------------------------------------------
    # Hub save / load
    # ------------------------------------------------------------------


    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: RobometerConfig | None = None,
        **kwargs,
    ) -> "RobometerModel":
        """Load config from a local directory or Hub repo and instantiate."""
        from lerobot.configs.rewards import RobometerConfig

        if config is None:
            config = RobometerConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        return cls(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tensor_to_frame(self, tensor: Tensor):
        """
        Convert a (C, H, W) float tensor in [0, 1]
        to a NumPy array (H, W, 3) uint8 (RGB),
        matching load_video_frames output.
        """
        import numpy as np
        frame = (
            tensor.detach()
            .cpu()
            .float()
            .permute(1, 2, 0)   # (C,H,W) → (H,W,C)
            .numpy()
            * 255.0
        ).clip(0, 255).astype(np.uint8)

        return frame

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute scalar rewards for a batch by generating progress scores.

        Args:
            batch: Must contain ``config.image_key`` (image tensor) and optionally
                ``config.task_key`` (task description strings). Image tensor shape:
                ``(B, C, H, W)`` (single frame) or ``(B, T, C, H, W)`` (video).

        Returns:
            Float tensor of shape ``(B,)`` with rewards in [0, 1].
        """
        if not _QWEN_VL_UTILS_AVAILABLE:
            raise ImportError(
                "compute_reward requires the qwen-vl-utils package.\nInstall with: pip install qwen-vl-utils"
            )
        
        if not _ROBOREASON_AVAILABLE:
            raise ImportError(
                "compute_reward requires the roboreason package.\nInstall with: pip install roboreason"
            )
                
        images = batch.get(self.config.image_key)
        if images is None:
            raise ValueError(f"Missing image key '{self.config.image_key}' in batch.")

        # Normalise to (B, T, C, H, W)
        if images.ndim == 4:
            images = images.unsqueeze(1)

        batch_size = images.shape[0]
        if self.config.task_key is not None:
            tasks = batch.get(self.config.task_key)
        else:
            # use self.config.task_instruction str for all samples if task_key is None
            tasks = [self.config.task_instruction] * batch_size
        task_list = _decode_tasks(tasks, batch_size)

        vlm_device = torch.device(self.config.device)
        
        from roboreason import generate
        
        rewards: list[float] = []
        for i in range(batch_size):
            frames = [self._tensor_to_frame(images[i, t]) for t in range(images.shape[1])]
            rewards_i, success_probs_i = generate(
                model=self.config.name,  
                task_description=task_list[i], 
                video_frames=frames, 
                view_type=self.config.view_type, 
                num_reasoning_frames=self.config.num_reasoning_frames,
            )
            # divide by 100 to convert from percentage to [0, 1] range
            rewards_i = [r / 100.0 for r in rewards_i]
            rewards.append(rewards_i)

        return torch.tensor(rewards, dtype=torch.float32, device=images.device)


    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        """Not implemented. Robometer is used inference-only via compute_reward()."""
        raise NotImplementedError(
            "RobometerModel does not support training via forward(). "
            "Use compute_reward() for inference. "
            "Fine-tuning support is planned as future work."
        )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def _decode_tasks(tasks: Any, batch_size: int) -> list[str]:
    """Decode task descriptions from various batch formats to a list of strings."""
    if tasks is None:
        return ["complete the task"] * batch_size
    if isinstance(tasks, (list, tuple)):
        return [str(t) for t in tasks]
    if isinstance(tasks, Tensor):
        # Byte-string tensor or similar edge case
        return [str(t.item()) if t.numel() == 1 else str(t.tolist()) for t in tasks]
    return [str(tasks)] * batch_size

