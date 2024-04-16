from dataclasses import dataclass, field


@dataclass
class ActionChunkingTransformerConfig:
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `state_dim`, `action_dim` and `camera_names`.

    Args:
        state_dim: Dimensionality of the observation state space (excluding images).
        action_dim: Dimensionality of the action space.
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        camera_names: The (unique) set of names for the cameras.
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        image_normalization_mean: Value to subtract from the input image pixels (inputs are assumed to be in
            [0, 1]) for normalization.
        image_normalization_std: Value by which to divide the input image pixels (after the mean has been
            subtracted).
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        use_pretrained_backbone: Whether the backbone should be initialized with pretrained weights from
            torchvision.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        d_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        use_temporal_aggregation: Whether to blend the actions of multiple policy invocations for any given
            environment step.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Environment.
    state_dim: int = 14
    action_dim: int = 14

    # Inputs / output structure.
    n_obs_steps: int = 1
    camera_names: tuple[str] = ("top",)
    chunk_size: int = 100
    n_action_steps: int = 100

    # Vision preprocessing.
    image_normalization_mean: tuple[float, float, float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    image_normalization_std: tuple[float, float, float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    use_pretrained_backbone: bool = True
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    d_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    use_temporal_aggregation: bool = False

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # ---
    # TODO(alexander-soare): Remove these from the policy config.
    batch_size: int = 8
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10
    utd: int = 1

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError("`vision_backbone` must be one of the ResNet variants.")
        if self.use_temporal_aggregation:
            raise NotImplementedError("Temporal aggregation is not yet implemented.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "The chunk size is the upper bound for the number of action steps per model invocation."
            )
        if self.camera_names != ["top"]:
            raise ValueError("For now, `camera_names` can only be ['top']")
