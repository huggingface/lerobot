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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.rewards import RewardModelConfig
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoTokenizer
else:
    AutoConfig = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


# Special tokens Robometer adds to the Qwen-VL tokenizer at construction time.
# The order is part of the data contract: upstream resized ``embed_tokens``
# after adding these tokens in this exact order, so changing the set or order
# would silently misalign the saved embedding rows with their token ids.
# ``<|reward_token|>`` and ``<|sim_token|>`` are leftover from earlier upstream
# heads (never read at inference) but still occupy rows the checkpoint expects.
ROBOMETER_SPECIAL_TOKENS = (
    "<|split_token|>",
    "<|reward_token|>",
    "<|pref_token|>",
    "<|sim_token|>",
    "<|prog_token|>",
)


@RewardModelConfig.register_subclass("robometer")
@dataclass
class RobometerConfig(RewardModelConfig):
    """Configuration for the Robometer reward model."""

    pretrained_path: str | None = "lerobot/Robometer-4B"
    image_key: str = OBS_IMAGES + ".top"
    task_key: str = "task"
    default_task: str | None = None

    max_frames: int | None = 8
    reward_output: str = "progress"  # "progress" or "success"
    success_threshold: float = 0.5

    license: str | None = "apache-2.0"
    tags: list[str] | None = field(
        default_factory=lambda: ["reward-model", "vision-language", "qwen3-vl", "zero-shot"]
    )

    base_model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    torch_dtype: str = "bfloat16"
    use_multi_image: bool = True
    use_per_frame_progress_token: bool = True
    average_temporal_patches: bool = True
    frame_pooling: str = "mean"  # "mean" | "boundary" | "attention"
    frame_pooling_attn_temperature: float = 1.0
    progress_loss_type: str = "discrete"  # "l1" | "l2" | "discrete"
    progress_discrete_bins: int = 10

    # Serialised Qwen backbone config (post-resize). Always populated by
    # ``__post_init__`` from ``base_model_id`` + ``len(tokenizer) + 5``, so it
    # is non-empty after construction. Saved into ``config.json`` automatically
    # by the base ``_save_pretrained``.
    vlm_config: dict[str, Any] = field(default_factory=dict)

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reward_output not in {"progress", "success"}:
            raise ValueError(f"reward_output must be 'progress' or 'success', got {self.reward_output!r}")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")
        if self.frame_pooling not in {"mean", "boundary", "attention"}:
            raise ValueError(f"frame_pooling must be mean/boundary/attention; got {self.frame_pooling!r}")
        if self.frame_pooling_attn_temperature <= 0:
            raise ValueError("frame_pooling_attn_temperature must be > 0")
        if self.progress_loss_type not in {"l1", "l2", "discrete"}:
            raise ValueError(f"progress_loss_type must be l1/l2/discrete; got {self.progress_loss_type!r}")
        if self.use_per_frame_progress_token and not self.use_multi_image:
            raise ValueError("use_per_frame_progress_token=True requires use_multi_image=True")

        if self.image_key not in self.input_features:
            self.input_features[self.image_key] = PolicyFeature(shape=(3, 224, 224), type=FeatureType.VISUAL)
        self.output_features.setdefault("progress", PolicyFeature(shape=(1,), type=FeatureType.REWARD))
        self.output_features.setdefault("success", PolicyFeature(shape=(1,), type=FeatureType.REWARD))

        # Deterministically populate ``vlm_config`` so it is non-empty after
        # construction. For ``Qwen/Qwen3-VL-4B-Instruct`` this gives
        # ``len(tokenizer) + 5 = 151,669 + 5 = 151,674`` — the exact post-resize
        # vocab the published ``Robometer-4B`` checkpoint was saved with.
        if not self.vlm_config:
            require_package("transformers", extra="robometer")
            vlm = AutoConfig.from_pretrained(self.base_model_id).to_dict()
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            text_config = vlm.get("text_config")
            if not isinstance(text_config, dict):
                raise ValueError(
                    f"Backbone config for {self.base_model_id!r} has no nested `text_config`; "
                    "Robometer expects a Qwen-VL-style config."
                )
            text_config["vocab_size"] = len(tokenizer) + len(ROBOMETER_SPECIAL_TOKENS)
            self.vlm_config = vlm

    @property
    def use_discrete_progress(self) -> bool:
        """Whether the progress head outputs distribution logits over bins."""
        return self.progress_loss_type.lower() == "discrete"

    @property
    def vlm_backbone_config(self):
        """Reconstruct the Qwen backbone config from :attr:`vlm_config`."""
        require_package("transformers", extra="robometer")
        config_dict = deepcopy(self.vlm_config)
        model_type = config_dict.pop("model_type", None)
        if model_type is None:
            raise ValueError("vlm_config must include `model_type` to reconstruct the backbone config")
        return AutoConfig.for_model(model_type, **config_dict)

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.image_key not in self.input_features:
            raise ValueError(f"Robometer requires image input feature {self.image_key!r}")
