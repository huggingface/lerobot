#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
        input_shapes: A dictionary defining the shapes of the input data for the policy.
            The key represents the input data name, and the value is a list indicating the dimensions
            of the corresponding data. For example, "observation.image" refers to an input from
            a camera with dimensions [3, 96, 96], indicating it has three color channels and 96x96 resolution.
            Importantly, shapes don't include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
            The key represents the output data name, and the value is a list indicating the dimensions
            of the corresponding data. For example, "action" refers to an output shape of [14], indicating
            14-dimensional actions. Importantly, shapes don't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
            within the image size. If None, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center crop in eval
            mode).
        embed_dim: Transformer model size.
        num_blocks: Number of blocks in the trunk transformer.
        num_heads: Number of heads in the trunk transformer.
        use_modality_embedding: Whether to add modality-specific trainable parameters.
        use_domain_embedding: Whether to add domain-specific trainable parameters.
        token_postprocessing: Method to pool the tokens, either "max" or "mean".
        weight_init_style: Weight initialization style.
        drop_path: Drop path in the trunk transformer.
        use_gpt_trunk: Load pre-trained trunk from GPT2.
        use_llama_trunk: Load pre-trained trunk from LLaMA2.
        hf_trunk: Load pre-trained transformer from Hugging Face.
        modalities: Modalities (e.g., 'image', 'language').
        modality_embed_dim: Embedding dimension for each modality.
        normalize_state: Normalize state vectors.
        state_embedding_dim: Dimension of positional encoding for state.
        image_encoder: Default image encoder.
        crossattn_dim_head: Dimension of each head in cross-attention modules.
        crossattn_heads: Number of heads in cross-attention.
        crossattn_modality_dropout: Dropout ratio for cross-attention.
        n_obs_steps: Observation horizon.
        random_horizon_masking: Randomize observation input length.
        add_pos_embedding_to_state: Positional embedding for the state.
        stem_num_blocks: Number of blocks for stem transformer's cross and self-attention.
        crossattn_latent_image: Latent dimension for cross-attention (image).
        crossattn_latent_state: Latent dimension for cross-attention (state).
        image_input_dim: Input dimension for the image encoder.
        image_output_dim: Output dimension for the image encoder.
        image_widths: Widths of the layers for the image encoder.
        image_num_of_copy: Number of copies for the image encoder.
        state_input_dim: Placeholder, should be overwritten based on the environment state dimension.
        state_output_dim: Output dimension for the state encoder.
        state_widths: Widths of the layers for the state encoder.
        head_input_dim: Input dimension for the head network.
        head_tanh_end: Whether to apply tanh to normalize action output.
        head_action_dim: Placeholder, should be overwritten based on the environment action dimension.
        action_horizon: Action horizon, should be overwritten based on the dataset.
        head_dropout: Add dropout to the head network.
        head_widths: Widths of the layers for the head network.
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

    # Network configuration
    # Trunk Transformer
    embed_dim: int = 256
    num_blocks: int = 16
    num_heads: int = 8
    use_modality_embedding: bool = True
    use_domain_embedding: bool = False
    token_postprocessing: str = "mean"
    weight_init_style: str = "pytorch"
    drop_path: float = 0.1
    no_trunk: bool = False
    load_pretrained: bool = False

    # Stem network (projectors) for different modalities
    modalities: tuple = ("image", "state")
    modality_embed_dim: int = 256
    normalize_state: bool = True
    state_embedding_dim: int = 1
    image_encoder: str = "resnet"
    crossattn_dim_head: int = 64
    crossattn_heads: int = 8
    crossattn_modality_dropout: float = 0.1
    n_obs_steps: int = 2
    random_horizon_masking: bool = True
    add_pos_embedding_to_state: bool = False

    # cross attention tokens
    image_crossattn_latent: int = 16
    state_crossattn_latent: int = 16

    # modality: image
    image_input_dim: int = 512
    image_output_dim: int = 256
    image_widths: tuple = (128,)
    image_num_of_copy: int = 1

    # modality: state
    state_input_dim: int = 14
    state_output_dim: int = 256
    state_widths: tuple = (128,)
    state_num_of_copy: int = 1

    # MLP Head network
    head_input_dim: int = 256
    head_tanh_end: bool = True
    head_action_dim: int = 14
    action_horizon: int = 8
    n_action_steps: int = 4
    head_dropout: bool = True
    head_widths: tuple = (256, 128)

    # Diffusion Head Network
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
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

    # ACT Head
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
