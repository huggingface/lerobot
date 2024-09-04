#!/usr/bin/env python

# Copyright 2024 Lirui Wang and The HuggingFace Inc. team. All rights reserved.
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
class HPTConfig:
    """Configuration class for the Heterogeneous Pre-trained Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        input_shapes (dict[str, list[int]]): A dictionary defining the shapes of the input data for the policy.
            The key represents the input data name, and the value is a list indicating the dimensions
            of the corresponding data. For example, "observation.images.top" refers to an input from
            a camera with dimensions [3, 480, 640], indicating it has three color channels and 480x640 resolution.
            Importantly, shapes don't include batch dimension or temporal dimension.
        output_shapes (dict[str, list[int]]): A dictionary defining the shapes of the output data for the policy.
            The key represents the output data name, and the value is a list indicating the dimensions
            of the corresponding data. For example, "action" refers to an output shape of [14], indicating
            14-dimensional actions. Importantly, shapes don't include batch dimension or temporal dimension.
        input_normalization_modes (dict[str, str]): A dictionary where the key represents the modality (e.g.,
            "observation.state"), and the value specifies the normalization mode to apply. The two available
            modes are "mean_std" (which subtracts the mean and divides by the standard deviation) and "min_max"
            (which rescales in a [-1, 1] range).
        output_normalization_modes (dict[str, str]): Similar dictionary to `input_normalization_modes`, but used to
            unnormalize to the original scale. This is also used for normalizing the training targets.
        domain_name (str): Name of the domain, e.g., 'robotics'.
        vision_backbone (str): Name of the torchvision ResNet backbone to use for encoding images, e.g., "resnet18".
        head_architecture (str): Architecture of the head network, e.g., "diffusion".
        embed_dim (int): Transformer model size.
        num_blocks (int): Number of blocks in the trunk transformer.
        num_heads (int): Number of heads in the trunk transformer.
        use_modality_embedding (bool): Whether to add modality-specific trainable parameters.
        use_domain_embedding (bool): Whether to add domain-specific trainable parameters.
        token_postprocessing (str): Method to pool the tokens, either "max" or "mean".
        weight_init_style (str): Weight initialization style, e.g., "pytorch".
        drop_path (float): Drop path rate in the trunk transformer.
        no_trunk (bool): Whether to disable the trunk transformer.
        load_pretrained (bool): Whether to load a pre-trained model for the trunk transformer.
        modalities (tuple): Modalities used in the model, e.g., ('image', 'state').
        modality_embed_dim (int): Embedding dimension for each modality.
        normalize_state (bool): Whether to normalize state vectors.
        state_embedding_dim (int): Dimension of positional encoding for state.
        image_encoder (str): Default image encoder, e.g., "resnet".
        crossattn_dim_head (int): Dimension of each head in cross-attention modules.
        crossattn_heads (int): Number of heads in cross-attention.
        crossattn_modality_dropout (float): Dropout ratio for cross-attention.
        n_obs_steps (int): Observation horizon.
        random_horizon_masking (bool): Whether to randomize observation input length.
        add_pos_embedding_to_state (bool): Whether to add positional embedding to the state.
        image_crossattn_latent (int): Latent dimension for cross-attention (image).
        state_crossattn_latent (int): Latent dimension for cross-attention (state).
        image_input_dim (int): Input dimension for the image encoder.
        image_output_dim (int): Output dimension for the image encoder.
        image_widths (tuple[int]): Widths of the layers for the image encoder.
        image_num_of_copy (int): Number of copies for the image encoder.
        state_input_dim (int): Input dimension for the state encoder, should be overwritten based on the environment state dimension.
        state_output_dim (int): Output dimension for the state encoder.
        state_widths (tuple[int]): Widths of the layers for the state encoder.
        state_num_of_copy (int): Number of copies for the state encoder.
        head_input_dim (int): Input dimension for the head network.
        head_tanh_end (bool): Whether to apply tanh to normalize action output.
        head_action_dim (int): Output dimension for the head network, should be overwritten based on the environment action dimension.
        action_chunk_size (int): Action horizon, should be overwritten based on the dataset.
        n_action_steps (int): Number of steps for action generation.
        head_dropout (bool): Whether to add dropout to the head network.
        head_widths (tuple[int]): Widths of the layers for the head network.
        down_dims (tuple[int]): Dimensions for down-sampling in the diffusion head network.
        kernel_size (int): Kernel size for convolutional layers in the diffusion head network.
        n_groups (int): Number of groups for normalization in the diffusion head network.
        diffusion_step_embed_dim (int): Embedding dimension for diffusion steps.
        use_film_scale_modulation (bool): Whether to use FiLM scale modulation in the diffusion head.
        noise_scheduler_type (str): Type of noise scheduler, e.g., "DDPM".
        num_train_timesteps (int): Number of training timesteps for the diffusion process.
        beta_schedule (str): Schedule type for beta, e.g., "squaredcos_cap_v2".
        beta_start (float): Starting value for beta in the diffusion process.
        beta_end (float): Ending value for beta in the diffusion process.
        prediction_type (str): Type of prediction in the diffusion process, e.g., "epsilon".
        clip_sample (bool): Whether to clip samples during the diffusion process.
        clip_sample_range (float): Range for clipping samples.
        num_inference_steps (int | None): Number of inference steps. If None, a default value is used.
        do_mask_loss_for_padding (bool): Whether to mask loss computation for padding tokens.
        dim_model (int): Dimension of the model for the ACT head.
        n_heads (int): Number of heads in the ACT head.
        dim_feedforward (int): Feedforward dimension in the ACT head.
        feedforward_activation (str): Activation function for the feedforward network, e.g., "relu".
        n_decoder_layers (int): Number of decoder layers in the ACT head.
        dropout (float): Dropout rate in the ACT head.
        pre_norm (bool): Whether to apply layer normalization before the main operations in the ACT head.
    """

    # Input / output structure.
    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.images.top": [3, 480, 640],
            "observation.state": [14],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [14],
        }
    )

    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.top": "mean_std",
            "observation.state": "min_max",
        }
    )
    output_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "action": "min_max",
        }
    )

    # Architecture.
    # Vision backbone.
    domain_name: str = "robotics"
    vision_backbone: str = "resnet18"
    head_architecture: str = "diffusion"
    action_chunk_size: int = 8
    n_action_steps: int = 4
    n_obs_steps: int = 2

    # Network configuration
    # Trunk Transformer
    embed_dim: int = 768
    num_blocks: int = 32
    num_heads: int = 16
    use_modality_embedding: bool = True
    use_domain_embedding: bool = False
    token_postprocessing: str = "mean"
    weight_init_style: str = "pytorch"
    drop_path: float = 0.1
    no_trunk: bool = False
    load_pretrained: str = "xlarge"
    freeze_trunk: bool = False

    # Stem network (projectors) for different modalities
    freeze_encoders: bool = False
    modalities: tuple = ("image", "state")
    modality_embed_dim: int = 256
    normalize_state: bool = True
    state_embedding_dim: int = 1
    image_encoder: str = "resnet"
    crossattn_dim_head: int = 64
    crossattn_heads: int = 8
    crossattn_modality_dropout: float = 0.1
    random_horizon_masking: bool = True
    add_pos_embedding_to_state: bool = False

    # Cross attention tokens
    image_crossattn_latent: int = 16
    state_crossattn_latent: int = 16

    # Modality: image
    image_input_dim: int = 512
    image_output_dim: int = 256
    image_widths: tuple = (128,)
    image_num_of_copy: int = 1

    # Modality: state
    state_input_dim: int = 14
    state_output_dim: int = 256
    state_widths: tuple = (128,)
    state_num_of_copy: int = 1

    # Modality: language (optional)
    language_input_dim: int = 768
    language_output_dim: int = 256
    language_widths: tuple = (128,)
    language_num_of_copy: int = 1

    # MLP Head
    head_input_dim: int = 256
    head_tanh_end: bool = True
    head_action_dim: int = 14
    head_dropout: bool = True
    head_widths: tuple = (256, 128)

    # Diffusion Head
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Transformer Head
    dim_model: int = 256
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_decoder_layers: int = 1
    dropout: float = 0.1
    pre_norm: bool = False

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if (
            not any(k.startswith("observation.image") for k in self.input_shapes)
            and "observation.environment_state" not in self.input_shapes
        ):
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
