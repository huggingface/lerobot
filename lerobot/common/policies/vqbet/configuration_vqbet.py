#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
# and The HuggingFace Inc. team. All rights reserved.
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


@dataclass
class VQBeTConfig:
    """Configuration class for VQ-BeT.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - At least one key starting with "observation.image is required as an input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        n_action_pred_token: Total number of current token and future tokens that VQ-BeT predicts.
        action_chunk_size: Action chunk size of each action prediction token.
        input_shapes: A dictionary defining the shapes of the input data for the policy.
            The key represents the input data name, and the value is a list indicating the dimensions
            of the corresponding data. For example, "observation.image" refers to an input from
            a camera with dimensions [3, 96, 96], indicating it has three color channels and 96x96 resolution.
            Importantly, shapes doesnt include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
            The key represents the output data name, and the value is a list indicating the dimensions
            of the corresponding data. For example, "action" refers to an output shape of [14], indicating
            14-dimensional actions. Importantly, shapes doesnt include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
            within the image size. If None, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center crop in eval
            mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initalize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        n_vqvae_training_steps: Number of optimization steps for training Residual VQ.
        vqvae_n_embed: Number of embedding vectors in the RVQ dictionary (each layer).
        vqvae_embedding_dim: Dimension of each embedding vector in the RVQ dictionary.
        vqvae_enc_hidden_dim: Size of hidden dimensions of Encoder / Decoder part of Residaul VQ-VAE
        gpt_block_size: Max block size of minGPT (should be larger than the number of input tokens)
        gpt_input_dim: Size of output input of GPT. This is also used as the dimension of observation features.
        gpt_output_dim: Size of output dimension of GPT. This is also used as a input dimension of offset / bin prediction headers.
        gpt_n_layer: Number of layers of GPT
        gpt_n_head: Number of headers of GPT
        gpt_hidden_dim: Size of hidden dimensions of GPT
        dropout: Dropout rate for GPT
        mlp_hidden_dim: Size of hidden dimensions of offset header / bin prediction headers parts of VQ-BeT
        offset_loss_weight:  A constant that is multiplied to the offset loss
        primary_code_loss_weight: A constant that is multiplied to the primary code prediction loss
        secondary_code_loss_weight: A constant that is multiplied to the secondary code prediction loss
        bet_softmax_temperature: Sampling temperature of code for rollout with VQ-BeT
        sequentially_select: Whether select code of primary / secondary as sequentially (pick primary code,
            and then select secodnary code), or at the same time.
    """

    # Inputs / output structure.
    n_obs_steps: int = 5
    n_action_pred_token: int = 3
    action_chunk_size: int = 5

    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.image": [3, 96, 96],
            "observation.state": [2],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [2],
        }
    )

    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.image": "mean_std",
            "observation.state": "min_max",
        }
    )
    output_normalization_modes: dict[str, str] = field(default_factory=lambda: {"action": "min_max"})

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    # VQ-VAE
    n_vqvae_training_steps: int = 20000
    vqvae_n_embed: int = 16
    vqvae_embedding_dim: int = 256
    vqvae_enc_hidden_dim: int = 128
    # VQ-BeT
    gpt_block_size: int = 500
    gpt_input_dim: int = 512
    gpt_output_dim: int = 512
    gpt_n_layer: int = 8
    gpt_n_head: int = 8
    gpt_hidden_dim: int = 512
    dropout: float = 0.1
    mlp_hidden_dim: int = 1024
    offset_loss_weight: float = 10000.0
    primary_code_loss_weight: float = 5.0
    secondary_code_loss_weight: float = 0.5
    bet_softmax_temperature: float = 0.1
    sequentially_select: bool = False

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        image_keys = {k for k in self.input_shapes if k.startswith("observation.image")}
        if self.crop_shape is not None:
            for image_key in image_keys:
                if (
                    self.crop_shape[0] > self.input_shapes[image_key][1]
                    or self.crop_shape[1] > self.input_shapes[image_key][2]
                ):
                    raise ValueError(
                        f"`crop_shape` should fit within `input_shapes[{image_key}]`. Got {self.crop_shape} "
                        f"for `crop_shape` and {self.input_shapes[image_key]} for "
                        "`input_shapes[{image_key}]`."
                    )
        # Check that all input images have the same shape.
        first_image_key = next(iter(image_keys))
        for image_key in image_keys:
            if self.input_shapes[image_key] != self.input_shapes[first_image_key]:
                raise ValueError(
                    f"`input_shapes[{image_key}]` does not match `input_shapes[{first_image_key}]`, but we "
                    "expect all image shapes to match."
                )
