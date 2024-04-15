from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration class for Diffusion Policy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `state_dim`, `action_dim` and `image_size`.

    Args:
        state_dim: Dimensionality of the observation state space (excluding images).
        action_dim: Dimensionality of the action space.
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction horizon as detailed in the main policy documentation.
    """

    # Environment.
    # Inherit these from the environment config.
    state_dim: int = 2
    action_dim: int = 2
    image_size: tuple[int, int] = (96, 96)

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    # Vision preprocessing.
    image_normalization_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_normalization_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] = (84, 84)
    crop_is_random: bool = True
    use_pretrained_backbone: bool = False
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    film_scale_modulation: bool = True
    # Noise scheduler.
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    variance_type: str = "fixed_small"
    prediction_type: str = "epsilon"
    clip_sample: True

    # Inference
    num_inference_steps: int = 100

    # ---
    # TODO(alexander-soare): Remove these from the policy config.
    batch_size: int = 64
    grad_clip_norm: int = 10
    lr: float = 1.0e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    adam_betas: tuple[float, float] = (0.95, 0.999)
    adam_eps: float = 1.0e-8
    adam_weight_decay: float = 1.0e-6
    utd: int = 1
    use_ema: bool = True
    ema_update_after_step: int = 0
    ema_min_rate: float = 0.0
    ema_max_rate: float = 0.9999
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError("`vision_backbone` must be one of the ResNet variants.")
