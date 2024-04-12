from dataclasses import dataclass


@dataclass
class ActConfig:
    """
    TODO(now): Document all variables
    TODO(now): Pick sensible defaults for a use case?
    """

    # Environment.
    state_dim: int
    action_dim: int

    # Inputs / output structure.
    n_obs_steps: int
    camera_names: list[str]
    chunk_size: int
    n_action_steps: int

    # Vision preprocessing.
    image_normalization_mean: tuple[float, float, float]
    image_normalization_std: tuple[float, float, float]

    # Architecture.
    # Vision backbone.
    vision_backbone: str
    use_pretrained_backbone: bool
    replace_final_stride_with_dilation: int
    # Transformer layers.
    pre_norm: bool
    d_model: int
    n_heads: int
    dim_feedforward: int
    feedforward_activation: str
    n_encoder_layers: int
    n_decoder_layers: int
    # VAE.
    use_vae: bool
    latent_dim: int
    n_vae_encoder_layers: int

    # Inference.
    use_temporal_aggregation: bool

    # Training and loss computation.
    dropout: float
    kl_weight: float

    # ---
    # TODO(alexander-soare): Remove these from the policy config.
    batch_size: int
    lr: float
    lr_backbone: float
    weight_decay: float
    grad_clip_norm: float
    utd: int

    def __post_init__(self):
        """Input validation."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError("`vision_backbone` must be one of the ResNet variants.")
        if self.use_temporal_aggregation:
            raise NotImplementedError("Temporal aggregation is not yet implemented.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "The chunk size is the upper bound for the number of action steps per model invocation."
            )
