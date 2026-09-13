from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("safediff_vla")
@dataclass
class SafeDiffVLAConfig(PreTrainedConfig):
    """Configuration for the external SafeDiff-VLA trajectory planner."""

    n_obs_steps: int = 1
    action_horizon: int = 16
    execute_horizon: int = 4
    # ACT-style temporal ensembling: replan every env step instead of every `execute_horizon`
    # steps, and blend each step's action across all still-relevant past chunk predictions
    # with exponential-decay weights (newest chunk weighted highest) instead of just executing
    # the freshest chunk's actions open-loop. Smooths chunk-boundary discontinuities at the cost
    # of calling `predict_action_chunk` on every step (vs. every `execute_horizon` steps), so it
    # meaningfully raises inference compute. Purely an inference-time behavior — a checkpoint
    # trained with this off can have it turned on at eval time with no retraining.
    use_temporal_ensembling: bool = False
    # Decay rate `m` in `weight_i = exp(-m * i)` for a prediction `i` steps old (0 = this step's
    # own chunk). Smaller = smoother blending across more history; larger = closer to using only
    # the freshest chunk. 0.01 is the value ACT (Zhao et al. 2023) reports working well.
    temporal_ensemble_coeff: float = 0.01
    backbone_name: str | None = None
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"
    freeze_backbone: bool = True
    # Adapt Safety-normalized observations/actions at the regular-LIBERO backbone boundary.
    use_backbone_domain_adapter: bool = False
    backbone_action_conversion_semantics: str = "per_step"
    freeze_vision_encoder: bool = True
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    num_diffusion_steps: int = 10
    beta_schedule: str = "cosine"
    prediction_type: str = "epsilon"
    use_vla_prior_init: bool = True

    latent_dim: int = 256
    planner_hidden_dim: int = 512
    # Hidden width of the self-supervised state-prediction head (see `state_predictor.py`). It
    # replaces the old task/risk critics, which needed `task_success`/`safety_violation` labels
    # that no dataset here actually provides and so never trained on anything but noise.
    state_head_hidden_dim: int = 256
    timestep_embedding_dim: int = 64

    lambda_diff: float = 1.0
    # Weight of the state-prediction regression loss (self-supervised against the state the
    # dataset actually observed `execute_horizon` steps later — always available, unlike
    # task/risk labels).
    lambda_state_pred: float = 1.0

    use_diffusion_refinement: bool = True
    # Per-sample squared-L2 gap (in normalized state units) between what the state predictor
    # expected `execute_horizon` steps after the previous chunk and the state actually observed
    # now, above which the previous chunk is considered "not complete": instead of committing to
    # a fresh full-length chunk, `select_action` replans one step at a time (closed-loop) until
    # the gap closes or `max_replan_retries` is hit.
    completion_threshold: float = 0.25
    max_replan_retries: int = 4
    enable_inference_metrics: bool = False

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # SafeDiff produces actions adapted for the LIBERO-Safety target domain when
    # `use_backbone_domain_adapter` is enabled; declare that explicitly here so
    # the runtime prefers the safety dataset contract for conversion boundaries.
    policy_action_contract: str | None = "libero_safety"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.execute_horizon <= self.action_horizon:
            raise ValueError("execute_horizon must be in [1, action_horizon]")
        if self.num_diffusion_steps < 1:
            raise ValueError("num_diffusion_steps must be positive")
        if self.completion_threshold < 0:
            raise ValueError("completion_threshold must be non-negative")
        if self.max_replan_retries < 0:
            raise ValueError("max_replan_retries must be non-negative")
        if self.temporal_ensemble_coeff < 0:
            raise ValueError("temporal_ensemble_coeff must be non-negative")
        if self.backbone_action_conversion_semantics not in {"per_step", "velocity"}:
            raise ValueError("backbone_action_conversion_semantics must be per_step or velocity")
        if self.use_backbone_domain_adapter and not self.backbone_name:
            raise ValueError("use_backbone_domain_adapter requires backbone_name")
        if self.use_lora and self.freeze_backbone:
            raise ValueError("use_lora requires freeze_backbone=False")
        if not self.freeze_vision_encoder and self.freeze_backbone:
            raise ValueError(
                "freeze_vision_encoder=False has no effect when freeze_backbone=True: the backbone-wide "
                "freeze is applied afterwards and overrides it. Set freeze_backbone=False (optionally "
                "with use_lora=True) to actually train the vision encoder."
            )
        if self.beta_schedule not in {"linear", "cosine"}:
            raise ValueError("beta_schedule must be 'linear' or 'cosine'")
        if self.prediction_type != "epsilon":
            raise ValueError("prediction_type must be 'epsilon'")

    def validate_features(self) -> None:
        if self.action_feature is None:
            raise ValueError("SafeDiff-VLA requires an action output feature")
        if self.robot_state_feature is None:
            raise ValueError(
                "SafeDiff-VLA requires an `observation.state` input feature: the diffusion "
                "planner conditions on it directly and the state-predictor head regresses "
                "against it."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr / 10,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def chunk_size(self) -> int:
        """Alias for `action_horizon`, the name most other policy configs (ACT, SmolVLA, ...) use for
        the same concept. Some call sites (e.g. the LIBERO-Safety dataset adapter) assume every policy
        config exposes `chunk_size`."""
        return self.action_horizon

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def state_observation_delta_indices(self) -> list[int]:
        """Overrides `observation_delta_indices` for `observation.state` only: in addition to the
        current frame (index 0), also load the state `execute_horizon` steps ahead so the
        state-prediction head has a real self-supervised regression target during training (see
        `state_predictor.py`). Images/language keep using `observation_delta_indices` (current
        frame only)."""
        return [0, self.execute_horizon]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
