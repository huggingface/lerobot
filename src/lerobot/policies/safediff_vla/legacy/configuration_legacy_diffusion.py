import logging
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig

logger = logging.getLogger(__name__)


@PreTrainedConfig.register_subclass("safediff_vla_legacy")
@dataclass
class LegacySafeDiffVLAConfig(PreTrainedConfig):
    """Configuration for the original SafeDiff-VLA design: a `ConditionalDiffusionPlanner`
    refining SmolVLA's own nominal action chunk.

    Not the main path -- see `..configuration_safediff_vla.SafeDiffVLAConfig` (policy type
    `"safediff_vla"`) for the current `temporal_decoder` architecture. This config/policy exists
    only to reproduce past experiments: every configuration of this design tried (critic-free,
    state-conditioned, subgoal-conditioned, temporal-conv-mixed across the horizon axis) matched
    or underperformed just executing the nominal chunk unmodified once the `action_horizon`/
    `execute_horizon` mismatch against the backbone's native `chunk_size` was fixed -- refining a
    nominal action chunk post hoc never once helped. Runnable via `lerobot-train
    --policy.type=safediff_vla_legacy ...` exactly like any other policy.
    """

    n_obs_steps: int = 1
    # Must match the backbone's own native chunk size (`backbone.config.chunk_size` /
    # `n_action_steps`) — `lerobot/smolvla_vlabench` uses 50. A mismatch here silently truncates
    # SmolVLA's own coherent 50-step plan to the first `action_horizon` steps and replans far more
    # often than the backbone was ever run at, which by itself measurably hurt these results;
    # `__post_init__` below only warns (not hard-fails) since deliberately mismatched experiments
    # are still sometimes useful, but treat any warning here as a red flag.
    action_horizon: int = 50
    execute_horizon: int = 50
    use_temporal_ensembling: bool = False
    temporal_ensemble_coeff: float = 0.01
    backbone_name: str | None = None
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"
    freeze_backbone: bool = True
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
    # 1D-conv kernel size / layer count `ConditionalDiffusionPlanner` uses to mix information
    # across the horizon axis (see `diffusion_planner.py`) — without this, each of the chunk's
    # timesteps is denoised in complete isolation from its neighbors, which per-step noise-MSE
    # doesn't penalize but produces trajectories with no coherence step-to-step.
    temporal_kernel_size: int = 5
    num_temporal_layers: int = 2
    # Hidden width of the self-supervised subgoal state-prediction head (see
    # `state_predictor.py`'s `StatePredictor`).
    state_head_hidden_dim: int = 256
    timestep_embedding_dim: int = 64

    lambda_diff: float = 1.0
    # Weight of the subgoal state-prediction regression loss. Only contributes when the batch
    # carries an `observation.subgoal_state` label (see `subgoal_labels_path` below) — otherwise
    # it's a no-op zero loss.
    lambda_subgoal: float = 1.0
    # Path to a local parquet produced by `examples/safediff_vla/compute_subgoal_labels.py`,
    # mapping each dataset frame's global `index` to the proprioceptive state at the next
    # demonstrated pick/place event (gripper open<->close transition). When set, `make_dataset()`
    # wraps the training dataset so every batch carries `observation.subgoal_state` — the
    # state-predictor head's regression target. Only affects training; at inference the head
    # predicts this from scratch (see `state_predictor.py`).
    subgoal_labels_path: str | None = None

    use_diffusion_refinement: bool = True
    # Per-sample squared-L2 gap (in normalized state units) between what the state predictor
    # expected `execute_horizon` steps after the previous chunk and the state actually observed
    # now, above which the previous chunk is considered "not complete": instead of committing to
    # a fresh full-length chunk, `select_action` replans one step at a time (closed-loop) until
    # the gap closes or `max_replan_retries` is hit.
    completion_threshold: float = 0.25
    max_replan_retries: int = 4
    use_completion_gate: bool = True
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

    policy_action_contract: str | None = "libero_safety"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.execute_horizon <= self.action_horizon:
            raise ValueError("execute_horizon must be in [1, action_horizon]")
        if self.num_diffusion_steps < 1:
            raise ValueError("num_diffusion_steps must be positive")
        if self.temporal_kernel_size < 1:
            raise ValueError("temporal_kernel_size must be positive")
        if self.num_temporal_layers < 1:
            raise ValueError("num_temporal_layers must be positive")
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
        backbone_chunk_size = self._backbone_native_chunk_size()
        if backbone_chunk_size is not None and backbone_chunk_size != self.action_horizon:
            logger.warning(
                "LegacySafeDiffVLAConfig.action_horizon=%d does not match backbone_name=%r's own "
                "native chunk_size=%d. This silently truncates/replans more often than the backbone "
                "was ever run at -- exactly the mismatch that measurably hurt earlier results. Set "
                "action_horizon=%d (and execute_horizon to match) unless this mismatch is deliberate.",
                self.action_horizon,
                self.backbone_name,
                backbone_chunk_size,
                backbone_chunk_size,
            )

    def _backbone_native_chunk_size(self) -> int | None:
        """Best-effort lookup of `backbone_name`'s own saved `chunk_size`, without downloading or
        constructing the full backbone (`__post_init__` runs on every config parse, including
        cheap CLI validation, so this must stay cheap and must never raise)."""
        if not self.backbone_name:
            return None
        try:
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

            return SmolVLAConfig.from_pretrained(self.backbone_name).chunk_size
        except Exception:  # noqa: BLE001 - best-effort; never block config construction on this
            return None

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
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
