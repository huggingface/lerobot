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
RoboReward: General-Purpose Vision-Language Reward Models for Robotics.
Paper: https://arxiv.org/abs/2601.00675

Inference only: loads teetone/RoboReward-4B or teetone/RoboReward-8B (Qwen3-VL
fine-tunes) and scores robot rollout videos with a task description prompt.

The model generates a discrete score (1–5) via autoregressive decoding, which is
then mapped to a float reward in [0, 1] via the config's score_to_reward table.

Requirements:
    pip install "transformers>=4.51.0" qwen-vl-utils
"""

import logging
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.robo_reward.configuration_robo_reward import RoboRewardConfig

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

# Exact prompt from the teetone/RoboReward-8B model card.
_PROMPT_TEMPLATE = (
    "Given the task, assign a discrete progress score reward (1,2,3,4,5) "
    "for the robot in the video in the format: ANSWER: <score>\n"
    "Rubric for end-of-episode progress (judge only the final state without time limits):\n"
    "1 - No Success: Final state shows no goal-relevant change for the command.\n"
    "2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.\n"
    "3 - Partial Completion: The final state shows good progress toward the goal but violates "
    "more than one requirement or a major requirement.\n"
    "4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.\n"
    "5 - Perfect Completion: Final state satisfies all requirements.\n\n"
    "Task: {task}\n"
)


class RoboRewardModel(PreTrainedRewardModel):
    """RoboReward vision-language reward model.

    Wraps a pretrained Qwen3-VL model (teetone/RoboReward-4B or -8B) for
    inference-only reward computation. The VLM backbone is always frozen.

    Usage::

        config = RoboRewardConfig(model_name="teetone/RoboReward-8B", device="cuda")
        model = RoboRewardModel(config)
        rewards = model.compute_reward(batch)  # Tensor of shape (B,)
    """

    name = "robo_reward"
    config_class = RoboRewardConfig

    def __init__(self, config: RoboRewardConfig):
        super().__init__(config)
        self.config = config

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "RoboRewardModel requires transformers>=4.51.0 with Qwen3-VL support.\n"
                "Install with: pip install 'transformers>=4.51.0'"
            )

        logger.info(f"Loading RoboReward VLM from {config.model_name}")
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype="auto",
        )
        self.vlm_processor = AutoProcessor.from_pretrained(config.model_name)

        for param in self.vlm.parameters():
            param.requires_grad = False

        config.validate_features()

    # ------------------------------------------------------------------
    # Hub save / load
    # ------------------------------------------------------------------

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save config only. VLM weights are loaded from the Hub at init time."""
        self.config._save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: RoboRewardConfig | None = None,
        **kwargs,
    ) -> "RoboRewardModel":
        """Load config from a local directory or Hub repo and instantiate."""
        from lerobot.configs.rewards import RewardModelConfig

        if config is None:
            config = RewardModelConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        return cls(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tensor_to_pil(self, tensor: Tensor):
        """Convert a (C, H, W) float tensor in [0, 1] to a PIL Image."""
        import numpy as np
        from PIL import Image

        arr = (tensor.cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _build_messages(self, frames: list, task: str) -> list[dict]:
        """Build Qwen3-VL chat messages for a single sample."""
        prompt = _PROMPT_TEMPLATE.format(task=task)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _parse_score(self, text: str) -> int:
        """Extract integer score 1–5 from model output. Returns 1 on parse failure."""
        match = re.search(r"ANSWER:\s*([1-5])", text)
        if match:
            return int(match.group(1))
        # Fallback: any standalone digit 1–5
        match = re.search(r"\b([1-5])\b", text)
        if match:
            return int(match.group(1))
        logger.warning(f"Could not parse score from model output: '{text}'. Defaulting to 1.")
        return 1

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

        images = batch.get(self.config.image_key)
        if images is None:
            raise ValueError(f"Missing image key '{self.config.image_key}' in batch.")

        # Normalise to (B, T, C, H, W)
        if images.ndim == 4:
            images = images.unsqueeze(1)

        batch_size = images.shape[0]
        tasks = batch.get(self.config.task_key)
        task_list = _decode_tasks(tasks, batch_size)

        vlm_device = torch.device(self.config.device)
        rewards: list[float] = []

        for i in range(batch_size):
            frames = [self._tensor_to_pil(images[i, t]) for t in range(images.shape[1])]
            messages = self._build_messages(frames, task_list[i])

            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = {k: v.to(vlm_device) for k, v in inputs.items()}

            output_ids = self.vlm.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
            generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            decoded = self.vlm_processor.decode(generated_ids, skip_special_tokens=True)

            score = self._parse_score(decoded)
            rewards.append(self.config.score_to_reward.get(score, 0.0))

        return torch.tensor(rewards, dtype=torch.float32, device=images.device)

    def reset(self) -> None:
        pass

    def get_optim_params(self):
        """Returns trainable parameters. VLM is frozen; no params are returned by default."""
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        """Not implemented. RoboReward is used inference-only via compute_reward()."""
        raise NotImplementedError(
            "RoboRewardModel does not support training via forward(). "
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
