# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode
from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim import AdamWConfig, LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import OBS_IMAGE


@RewardModelConfig.register_subclass(name="reward_classifier")
@dataclass
class RewardClassifierConfig(RewardModelConfig):
    """Configuration for the Reward Classifier model.

    Args:
        input_features (`dict`, *optional*):
            A dictionary defining the `PolicyFeature` of the input data. The key represents the input
            data name, and the value is the `PolicyFeature`.
        output_features (`dict`, *optional*):
            A dictionary defining the `PolicyFeature` of the output data, analogous to
            `input_features`.
        device (`str`, *optional*, defaults to `"cpu"`):
            Torch device to run the model on.
        pretrained_path (`str | None`, *optional*):
            Local directory or Hugging Face Hub repo id to load pretrained weights from.
        pretrained_revision (`str | None`, *optional*):
            Optional Hub revision (commit hash, branch, or tag) to pin the pretrained model version.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push this model to the Hugging Face Hub.
        repo_id (`str | None`, *optional*):
            Hub repository id to push to when `push_to_hub` is `True`.
        license (`str | None`, *optional*):
            License tag for the Hub model card.
        tags (`list[str] | None`, *optional*):
            Hub model card tags.
        private (`bool | None`, *optional*):
            Whether the pushed Hub repository is private.
        name (`str`, *optional*, defaults to `"reward_classifier"`):
            Registered name of the reward model.
        num_classes (`int`, *optional*, defaults to 2):
            Number of output classes for the success/failure classifier.
        hidden_dim (`int`, *optional*, defaults to 256):
            Hidden dimension of the classifier head.
        latent_dim (`int`, *optional*, defaults to 256):
            Dimension of the pooled image-embedding latent.
        image_embedding_pooling_dim (`int`, *optional*, defaults to 8):
            Output spatial size of the learned spatial-embedding pooling layer.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout probability applied in the classifier head.
        model_name (`str`, *optional*, defaults to `"lerobot/resnet10"`):
            Pretrained vision backbone to use.
        model_type (`str`, *optional*, defaults to `"cnn"`):
            Backbone architecture family: `"cnn"` or `"transformer"`.
        num_cameras (`int`, *optional*, defaults to 2):
            Number of camera views the classifier expects.
        learning_rate (`float`, *optional*, defaults to 0.0001):
            Learning rate for the AdamW optimizer preset.
        weight_decay (`float`, *optional*, defaults to 0.01):
            Weight decay for the AdamW optimizer preset.
        grad_clip_norm (`float`, *optional*, defaults to 1.0):
            Gradient-clipping norm for the AdamW optimizer preset.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps feature types to their normalization mode.
    """

    name: str = "reward_classifier"
    num_classes: int = 2
    hidden_dim: int = 256
    latent_dim: int = 256
    image_embedding_pooling_dim: int = 8
    dropout_rate: float = 0.1
    model_name: str = "lerobot/resnet10"
    device: str = "cpu"
    model_type: str = "cnn"
    num_cameras: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
        }
    )

    @property
    def observation_delta_indices(self) -> list | None:
        """`None`: the classifier only consumes the current-step observation."""
        return None

    @property
    def action_delta_indices(self) -> list | None:
        """`None`: the classifier does not consume actions."""
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        """`None`: the classifier does not consume past rewards."""
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        """AdamW preset using `learning_rate`/`weight_decay`/`grad_clip_norm`."""
        return AdamWConfig(
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """`None`: no learning-rate scheduler preset."""
        return None

    def validate_features(self) -> None:
        """Validate feature configurations."""
        has_image = any(key.startswith(OBS_IMAGE) for key in self.input_features)
        if not has_image:
            raise ValueError(
                "You must provide an image observation (key starting with 'observation.image') in the input features"
            )
