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

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.rewards import RewardModelConfig
from lerobot.utils.constants import OBS_IMAGES

# Default prompt scaffolding from the upstream TOPReward paper / reference
# implementation (``QwenClient.compute_instruction_reward``). The prompt
# scores the terminal ``True`` token in ``f"{instruction} ... True"``
# given the video.
DEFAULT_PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory that completes the following task: "
)
DEFAULT_PROMPT_SUFFIX_TEMPLATE = (
    "{instruction} Decide whether the above statement is True or not. The answer is: True"
)


@RewardModelConfig.register_subclass("topreward")
@dataclass
class TOPRewardConfig(RewardModelConfig):
    """Configuration for the TOPReward zero-shot reward model.

    TOPReward is **zero-shot**: it has no learnable parameters of its own.
    The "model" is a generic vision-language model (default
    ``Qwen/Qwen3-VL-8B-Instruct``) used with a fixed prompt to extract
    token log-probabilities as a reward signal. There is therefore no
    fine-tuned checkpoint to host: ``pretrained_path`` is unused at
    runtime — the model identity is :attr:`vlm_name` (an HF Hub id).

    Args:
        input_features (`dict[str, PolicyFeature]`, *optional*):
            A dictionary defining the `PolicyFeature` of the input data. The key represents the input
            data name, and the value is the `PolicyFeature`.
        output_features (`dict[str, PolicyFeature]`, *optional*):
            A dictionary defining the `PolicyFeature` of the output data, analogous to
            `input_features`.
        device (`str | None`, *optional*):
            Torch device to run the model on. Auto-resolved in `__post_init__` when unset or
            unavailable.
        pretrained_path (`str | None`, *optional*):
            Unused at runtime — TOPReward is zero-shot, so the model identity is `vlm_name`, not a
            fine-tuned checkpoint. Kept only for a local dir or Hub repo holding a `config.json`
            snapshot of this config.
        pretrained_revision (`str | None`, *optional*):
            Optional Hub revision (commit hash, branch, or tag) to pin the pretrained config version.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push this config to the Hugging Face Hub.
        repo_id (`str | None`, *optional*):
            Hub repository id to push to when `push_to_hub` is `True`.
        license (`str | None`, *optional*, defaults to `"mit"`):
            License tag for the Hub model card.
        tags (`list[str] | None`, *optional*):
            Hub model card tags.
        private (`bool | None`, *optional*):
            Whether the pushed Hub repository is private.
        vlm_name (`str`, *optional*, defaults to `"Qwen/Qwen3-VL-8B-Instruct"`):
            Hugging Face Hub id of the underlying VLM. Must be a Qwen3-VL family model (the only
            client implemented in this LeRobot port).
        torch_dtype (`str`, *optional*, defaults to `"auto"`):
            Torch dtype name passed to the VLM loader (`"auto"`, `"bfloat16"`, `"float16"`, ...).
        attn_implementation (`str | None`, *optional*):
            `transformers` attention implementation (e.g. `"flash_attention_2"`, `"sdpa"`). `None` so
            the upstream picks the best available.
        image_key (`str`, *optional*, defaults to `"observation.images.top"`):
            Observation key that holds the trajectory frames.
        task_key (`str`, *optional*, defaults to `"task"`):
            Complementary-data key that holds the task instruction.
        default_task (`str | None`, *optional*):
            Fallback instruction when `task_key` is absent.
        max_frames (`int | None`, *optional*, defaults to 16):
            Cap on the number of frames fed to the VLM per sample. `None` uses all frames.
        fps (`float`, *optional*, defaults to 2.0):
            Frames-per-second metadata for the Qwen video processor.
        prompt_prefix (`str`, *optional*, defaults to `"The above video shows a robot manipulation trajectory that completes the following task: "`):
            Text shown to the VLM right after the video and before the suffix template.
        prompt_suffix_template (`str`, *optional*, defaults to `"{instruction} Decide whether the above statement is True or not. The answer is: True"`):
            Suffix appended after `prompt_prefix`. Must contain `{instruction}`; the VLM scores the
            log-likelihood of the tokens that follow the prefix.
        add_chat_template (`bool`, *optional*, defaults to `False`):
            If `True`, wrap the full prompt with the tokenizer's chat template before tokenisation
            (matches upstream `add_chat_template=True`).
        success_threshold (`float`, *optional*, defaults to -inf):
            Optional log-prob threshold. If finite, `TOPRewardModel.compute_reward` returns
            `(reward > success_threshold).float()` instead of the raw log-prob.
        max_input_length (`int`, *optional*, defaults to 32768):
            Hard limit on the total tokenized input length; samples that exceed it raise a
            `ValueError`.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps feature types to their normalization mode.
    """

    # Path to a local LeRobot dir or HF repo that holds a ``config.json``
    # snapshot of this TOPRewardConfig. The VLM weights themselves are
    # always identified by ``vlm_name``.
    pretrained_path: str | None = None

    vlm_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "auto"
    attn_implementation: str | None = None

    image_key: str = OBS_IMAGES + ".top"
    task_key: str = "task"
    default_task: str | None = None
    max_frames: int | None = 16
    fps: float = 2.0

    prompt_prefix: str = DEFAULT_PROMPT_PREFIX
    prompt_suffix_template: str = DEFAULT_PROMPT_SUFFIX_TEMPLATE
    add_chat_template: bool = False

    success_threshold: float = float("-inf")
    max_input_length: int = 32768

    license: str | None = "mit"  # matches upstream TOPReward
    tags: list[str] | None = field(
        default_factory=lambda: ["reward-model", "vision-language", "qwen3-vl", "zero-shot"]
    )

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        """Validate field combinations and default `input_features`/`output_features`.

        Raises:
            ValueError: If `max_frames`, `fps`, `prompt_suffix_template`, or `max_input_length` hold
                an invalid value.
        """
        super().__post_init__()
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        if "{instruction}" not in self.prompt_suffix_template:
            raise ValueError(
                "prompt_suffix_template must contain `{instruction}` so the model "
                "scores the log-likelihood of the task suffix."
            )
        if self.max_input_length <= 0:
            raise ValueError(f"max_input_length must be > 0, got {self.max_input_length}")

        if self.image_key not in self.input_features:
            self.input_features[self.image_key] = PolicyFeature(shape=(3, 224, 224), type=FeatureType.VISUAL)
        self.output_features.setdefault("reward", PolicyFeature(shape=(1,), type=FeatureType.REWARD))

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """`None`: TOPReward samples its own frame window internally rather than via delta indices."""
        return None

    @property
    def action_delta_indices(self) -> None:
        """`None`: TOPReward does not consume actions."""
        return None

    @property
    def reward_delta_indices(self) -> None:
        """`None`: TOPReward does not consume past rewards."""
        return None

    def validate_features(self) -> None:
        """Validate that `image_key` is present in `input_features`.

        Raises:
            ValueError: If `image_key` is missing from `input_features`.
        """
        if self.image_key not in self.input_features:
            raise ValueError(f"TOPReward requires image input feature {self.image_key!r}")
