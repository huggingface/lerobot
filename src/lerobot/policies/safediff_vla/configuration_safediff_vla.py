import logging
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig

logger = logging.getLogger(__name__)

ARCHITECTURES = {
    "smolvla_nominal",
    "smolvla_finetune",
    "temporal_decoder",
    "temporal_decoder_subgoal",
}


@PreTrainedConfig.register_subclass("safediff_vla")
@dataclass
class SafeDiffVLAConfig(PreTrainedConfig):
    """Configuration for SafeDiff-VLA's main path: a temporal action decoder generating full
    action chunks directly from a frozen (by default) SmolVLA multimodal encoder.

    The original nominal-refinement diffusion design (`legacy_diffusion`) has been moved out of
    this config entirely -- see `legacy/configuration_legacy_diffusion.py`'s `LegacySafeDiffVLAConfig`
    (policy type `"safediff_vla_legacy"`), kept only to reproduce past experiments.
    """

    # Which action-generation architecture to use (see `modeling_safediff_vla.py`'s module
    # docstring for the full data flow of each):
    #   "temporal_decoder" (default, recommended): VLM latent tokens + current state ->
    #       TemporalActionDecoder -> actions directly. Does not consult SmolVLA's own nominal
    #       action chunk at all.
    #   "temporal_decoder_subgoal": as above, plus a predicted subgoal state (from
    #       `SubgoalStatePredictor`) as additional decoder conditioning -- a single per-sample
    #       target broadcast to every horizon position inside the decoder, not a per-timestep
    #       future-state trajectory (see `temporal_decoder.py`).
    #   "smolvla_nominal": ablation baseline — returns SmolVLA's own nominal action chunk
    #       unmodified. No trainable parameters of its own; not meant to be trained, only eval'd.
    #   "smolvla_finetune": same module (SmolVLA's own flow-matching action expert, unmodified)
    #       as "smolvla_nominal", but trained: `forward()` delegates directly to the backbone's
    #       own flow-matching training loss (noise/time-sampled velocity regression -- see
    #       `modeling_smolvla.py`'s `VLAFlowMatching.forward`), so representation, architecture,
    #       and objective are all SmolVLA's original, unlike `temporal_decoder`'s from-scratch
    #       decoder + decomposed MSE. Requires `freeze_backbone=False` (there would be nothing to
    #       train otherwise). Exists to isolate whether `temporal_decoder`'s from-scratch design
    #       -- as opposed to e.g. dataset/task difficulty -- drives eval results.
    architecture: str = "temporal_decoder"
    n_obs_steps: int = 1
    # Must match the backbone's own native chunk size (`backbone.config.chunk_size` /
    # `n_action_steps`) — `lerobot/smolvla_vlabench` uses 50. A mismatch here silently truncates
    # SmolVLA's own coherent 50-step plan to the first `action_horizon` steps and replans far more
    # often than the backbone was ever run at; `__post_init__` below only warns (not hard-fails)
    # since deliberately mismatched experiments are still sometimes useful, but treat any warning
    # here as a red flag.
    action_horizon: int = 50
    execute_horizon: int = 50
    # ACT-style temporal ensembling: replan every env step instead of every `execute_horizon`
    # steps, and blend each step's action across all still-relevant past chunk predictions
    # with exponential-decay weights (newest chunk weighted highest) instead of just executing
    # the freshest chunk's actions open-loop. Smooths chunk-boundary discontinuities at the cost
    # of calling `predict_action_chunk` on every step (vs. every `execute_horizon` steps), so it
    # meaningfully raises inference compute. Purely an inference-time behavior, owned by
    # `execution.ActionExecutor` -- a checkpoint trained with this off can have it turned on at
    # eval time with no retraining.
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

    # Width of the pooled multimodal latent fed to `SubgoalStatePredictor` (`temporal_decoder_subgoal`
    # only) -- see `latent_pool_projection` in `modeling_safediff_vla.py`.
    latent_dim: int = 256
    # Hidden width of the self-supervised subgoal state-prediction head (see
    # `state_predictor.py`'s `SubgoalStatePredictor`). Only used by `temporal_decoder_subgoal`.
    state_head_hidden_dim: int = 256

    # -- `temporal_decoder` / `temporal_decoder_subgoal` (see `temporal_decoder.py`) --
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 1024
    decoder_dropout: float = 0.1
    # Weights of the primary supervised action-prediction loss, decomposed per action component
    # (see `validate_features()`: position = actions[..., :3], orientation = actions[..., 3:6],
    # gripper = actions[..., 6:7]) rather than one pooled MSE, so each can be reweighted or
    # monitored independently. `modeling_safediff_vla.py`'s `_forward_temporal_decoder` combines
    # them as `(lambda_pos*3*loss_pos + lambda_rot*3*loss_rot + lambda_grip*1*loss_grip) / 7`
    # (each per-component `F.mse_loss` pre-multiplied by its own slice width, divided by the
    # total 7) -- NOT a plain unweighted sum of the three MSEs, which would over-count the 1-wide
    # gripper slice relative to the 3-wide position/orientation slices. With this normalization,
    # the defaults (1.0 each) are numerically identical to the old single pooled
    # `F.mse_loss(pred_actions, clean)` over all 7 dims -- see
    # `test_decomposed_loss_equals_old_pooled_mse_at_default_weights`.
    lambda_pos: float = 1.0
    lambda_rot: float = 1.0
    lambda_grip: float = 1.0
    # Weight of an optional trajectory-smoothness regularizer (mean squared acceleration).
    # Defaults to off (0.0): the first experiment should characterize the plain supervised
    # decoder's own behavior before adding regularization on top of it.
    lambda_smooth: float = 0.0

    # Weight of the subgoal state-prediction regression loss. Only contributes when the batch
    # carries an `observation.subgoal_state` label (see `subgoal_labels_path` below) and the
    # architecture is `temporal_decoder_subgoal` -- otherwise it's a no-op zero loss.
    lambda_subgoal: float = 1.0
    # Path to a local parquet produced by `examples/safediff_vla/compute_subgoal_labels.py`,
    # mapping each dataset frame's global `index` to the proprioceptive state at the next
    # demonstrated pick/place event (gripper open<->close transition). When set, `make_dataset()`
    # wraps the training dataset so every batch carries `observation.subgoal_state` — the
    # `SubgoalStatePredictor` head's regression target. Only affects training; at inference the
    # head predicts this from scratch.
    subgoal_labels_path: str | None = None

    # Per-sample squared-L2 gap (in normalized state units) between what the subgoal predictor
    # expected `execute_horizon` steps after the previous chunk and the state actually observed
    # now, above which the previous chunk is considered "not complete": instead of committing to
    # a fresh full-length chunk, `execution.ActionExecutor.select_action` replans one step at a
    # time (closed-loop) until the gap closes or `max_replan_retries` is hit.
    completion_threshold: float = 0.25
    max_replan_retries: int = 4
    # Set False to always commit a fresh full-length chunk regardless of the subgoal gap —
    # useful to isolate an action-generation architecture's own quality from the gate's effect
    # (e.g. when comparing `temporal_decoder` against `temporal_decoder_subgoal`, put both
    # through with this off first). Has no effect on architectures with no subgoal signal
    # (`smolvla_nominal`, `smolvla_finetune`, `temporal_decoder`) — those never gate regardless.
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

    # SafeDiff produces actions adapted for the LIBERO-Safety target domain when
    # `use_backbone_domain_adapter` is enabled; declare that explicitly here so
    # the runtime prefers the safety dataset contract for conversion boundaries.
    policy_action_contract: str | None = "libero_safety"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.execute_horizon <= self.action_horizon:
            raise ValueError("execute_horizon must be in [1, action_horizon]")
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
        if self.architecture not in ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {sorted(ARCHITECTURES)}, got {self.architecture!r}"
            )
        if self.architecture == "smolvla_finetune" and self.freeze_backbone:
            raise ValueError(
                "architecture='smolvla_finetune' trains the backbone's own action expert -- it has no "
                "other trainable parameters, so freeze_backbone=True would train nothing. Set "
                "freeze_backbone=False."
            )
        if self.decoder_hidden_dim % self.decoder_num_heads != 0:
            raise ValueError("decoder_hidden_dim must be divisible by decoder_num_heads")
        if self.lambda_pos < 0:
            raise ValueError("lambda_pos must be non-negative")
        if self.lambda_rot < 0:
            raise ValueError("lambda_rot must be non-negative")
        if self.lambda_grip < 0:
            raise ValueError("lambda_grip must be non-negative")
        if self.lambda_smooth < 0:
            raise ValueError("lambda_smooth must be non-negative")
        backbone_chunk_size = self._backbone_native_chunk_size()
        if backbone_chunk_size is not None and backbone_chunk_size != self.action_horizon:
            logger.warning(
                "SafeDiffVLAConfig.action_horizon=%d does not match backbone_name=%r's own native "
                "chunk_size=%d. This silently truncates/replans more often than the backbone was "
                "ever run at. Set action_horizon=%d (and execute_horizon to match) unless this "
                "mismatch is deliberate.",
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
                "SafeDiff-VLA requires an `observation.state` input feature: the temporal decoder "
                "conditions on it directly and the subgoal state-predictor head regresses against it."
            )
        # The decomposed action loss (lambda_pos/lambda_rot/lambda_grip, see
        # `modeling_safediff_vla.py`) hard-codes the LIBERO 7-dim layout: [:3]=position,
        # [3:6]=orientation, [6:7]=gripper (matches `action_semantics`/`libero_contracts.py`'s
        # `gripper_index=6` and six-value `osc_output_scale`).
        if self.action_feature.shape[0] != 7:
            raise ValueError(
                "SafeDiff-VLA's action loss decomposition (lambda_pos/lambda_rot/lambda_grip) assumes "
                f"a 7-dim action (3 position + 3 orientation + 1 gripper), got shape "
                f"{self.action_feature.shape}."
            )
        if self.architecture in ("temporal_decoder", "temporal_decoder_subgoal"):
            # These architectures sin/cos-encode rotation (see `rotation_encoding.py`) for both
            # `observation.state` and `action`, assuming the *same* 7-dim [xyz, rx, ry, rz,
            # gripper] layout for both -- real state/action always share this layout (state is
            # the current pose, action the next commanded one), so this is a real constraint, not
            # an arbitrary one.
            if self.robot_state_feature.shape[0] != 7:
                raise ValueError(
                    "temporal_decoder/temporal_decoder_subgoal's sin/cos rotation encoding assumes a "
                    f"7-dim observation.state matching action's [xyz, rx, ry, rz, gripper] layout, got "
                    f"shape {self.robot_state_feature.shape}."
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
