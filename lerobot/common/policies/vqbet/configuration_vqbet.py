from dataclasses import dataclass, field


@dataclass
class VQBeTConfig:
    """Configuration class for VQ-BeT.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        n_action_pred_token: TODO(jayLEE0301)
        n_action_pred_chunk: TODO(jayLEE0301)
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
        discretize_step: TODO(jayLEE0301)
        vqvae_groups: TODO(jayLEE0301)
        vqvae_n_embed: TODO(jayLEE0301)
        vqvae_embedding_dim: TODO(jayLEE0301)
        vqvae_enc_hidden_dim: TODO(jayLEE0301)
        gpt_block_size: TODO(jayLEE0301)
        gpt_input_dim: TODO(jayLEE0301)
        gpt_output_dim: TODO(jayLEE0301)
        gpt_n_layer: TODO(jayLEE0301)
        gpt_n_head: TODO(jayLEE0301)
        gpt_hidden_dim: TODO(jayLEE0301)
        dropout: TODO(jayLEE0301)
        mlp_hidden_dim: TODO(jayLEE0301)
        offset_loss_weight: TODO(jayLEE0301)
        secondary_code_loss_weight: TODO(jayLEE0301)
    """

    # Inputs / output structure.
    n_obs_steps: int = 5
    n_action_pred_token: int = 3
    n_action_pred_chunk: int = 5

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
    discretize_step: int = 3000
    vqvae_groups: int = 2
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
    offset_loss_weight: float = 10000.
    secondary_code_loss_weight: float = 0.5
    bet_softmax_temperature: float = 0.1

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if (
            self.crop_shape[0] > self.input_shapes["observation.image"][1]
            or self.crop_shape[1] > self.input_shapes["observation.image"][2]
        ):
            raise ValueError(
                f'`crop_shape` should fit within `input_shapes["observation.image"]`. Got {self.crop_shape} '
                f'for `crop_shape` and {self.input_shapes["observation.image"]} for '
                '`input_shapes["observation.image"]`.'
            )