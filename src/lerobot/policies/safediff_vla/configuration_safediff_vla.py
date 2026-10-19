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
    "temporal_decoder_grounded_grasp",
    "temporal_decoder_instruction",
    "temporal_decoder_text_crossattn",
    "temporal_decoder_grounded_grasp_v2",
    "temporal_decoder_grounded_grasp_v3",
    "temporal_decoder_grounded_grasp_v2_5",
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
    #   "temporal_decoder_grounded_grasp": experimental. `temporal_decoder` plus a `TargetPointHead`
    #       (see `target_point_head.py`) predicting one grasp-target xyz from the pooled VLM
    #       latent, fed into the *same* decoder as an added global-conditioning term (like
    #       `subgoal_state`, no change to the Transformer stack itself -- see
    #       `temporal_decoder.py`'s `use_target_xyz`). Exists to separate two causal bottlenecks
    #       found by diagnostic oracle experiments against the canonical `temporal_decoder`
    #       checkpoint (see `examples/safediff_vla/grasp_approach_oracle_correction.py` and
    #       `grasp_close_timing_oracle.py`): (1) target-xyz localization bias -- addressed by the
    #       explicit `TargetPointHead` + `lambda_target` regression below -- and (2) gripper-close
    #       *timing* not tracking actual proximity -- addressed at inference only, by ignoring the
    #       decoder's own regressed gripper channel as the close trigger and instead closing
    #       reactively once `current_state[:3]` comes within `grounded_grasp_close_threshold_m` of
    #       the predicted target (see `SafeDiffVLAPolicy._grounded_grasp_reactive_close`). No other
    #       architecture's behavior changes at all -- every new field/branch below is read only
    #       when `architecture == "temporal_decoder_grounded_grasp"`.
    #   "temporal_decoder_instruction": experimental. `temporal_decoder` plus the instruction
    #       itself added as one more global-conditioning term on every horizon query -- masked-mean
    #       TEXT_MODALITY tokens (see `utils.masked_mean_by_modality`), projected and added
    #       exactly like `subgoal_state`/`target_xyz` above (`temporal_decoder.py`'s
    #       `use_instruction`). No `TargetPointHead`, no auxiliary loss, no decoder-conditioning
    #       toggle, no reactive close -- the ONLY change from plain `temporal_decoder` is this one
    #       additive term on the query. Exists to test whether the canonical decoder's own action
    #       output can be made instruction-conditioned directly (bypassing the
    #       `temporal_decoder_grounded_grasp` `TargetPointHead` path entirely, which held-out
    #       xyz regression improved but never made the actual card selection track the
    #       instruction -- identity-flip rate stayed 0-25% across two independent pooling designs).
    #       Also did not move identity-flip/correct-card rates (see
    #       `temporal_decoder_text_crossattn` below).
    #   "temporal_decoder_text_crossattn": experimental. `temporal_decoder` plus an EXTRA
    #       cross-attention over the raw, un-pooled per-token text sequence, applied after the
    #       canonical multimodal cross-attention/self-attention stack (`temporal_decoder.py`'s
    #       `use_text_crossattn`): `queries -> canonical cross-attn(full multimodal memory) ->
    #       extra text-only cross-attn(text tokens, no mean/max/last-token pooling of any kind) ->
    #       action head`. Every other instruction-conditioning attempt so far (`temporal_decoder_
    #       instruction`'s additive pooled-instruction-embedding, and `temporal_decoder_grounded_
    #       grasp`'s two `TargetPointHead` pooling designs) collapsed the instruction into a single
    #       vector before it reached the query; this is the first to keep it as a full, un-pooled
    #       token sequence throughout. No `TargetPointHead`, no auxiliary loss, no reactive close.
    #       Also converged to the same identity-flip/correct-card rates as every other attempt.
    #   "temporal_decoder_grounded_grasp_v2": experimental. Forward-path tracing on the plain
    #       `temporal_decoder_grounded_grasp` checkpoint found the canonical model is NOT
    #       instruction-blind (text representation, multimodal memory, and the predicted
    #       trajectory all visibly change with the instruction; the nearest card to the predicted
    #       close position changed with the instruction in 5/5 traced scenes) -- so "the model
    #       ignores language entirely" was not the right diagnosis. What the same tracing found
    #       instead: overall attention mass on TEXT_MODALITY tokens is ~12-13%, but attention
    #       specifically on the tokens that name the target's identity (rank/suit) is only
    #       ~0.7-2.4% -- the identity-carrying tokens are present and attended to, at all, far less
    #       than boilerplate instruction tokens ("please", "pick", "the", "poker"). This
    #       architecture targets that specific gap with an EXPLICIT target-language-query ->
    #       visual-grounding-cross-attention pipeline (see `grounded_grasp_v2.py`):
    #         image / language / state -> latent_tokens (unchanged)
    #           -> `TargetQueryExtractor`: a small LEARNED query attention-pools over TEXT_MODALITY
    #              tokens only (NOT mean pooling, NOT hard-coded rank/suit token-position parsing)
    #              -> target_language_query [B, D], trained end-to-end against the target-xyz loss
    #              below so it can learn which tokens actually matter.
    #           -> `VisualGroundingCrossAttention`: that query attends over IMAGE_MODALITY tokens
    #              only -> grounded_target_feature [B, D].
    #           -> `TargetPointHead` (the "TargetXYZHead") -> predicted_target_xyz [B, 3].
    #       Target supervision (`utils.find_grasp_target`, same polarity/padding fix as
    #       `temporal_decoder_grounded_grasp`, unchanged): the first valid open->close transition's
    #       action xyz, masked to samples where such a transition exists in the future horizon.
    #       `TemporalActionDecoder` (`temporal_decoder.py`'s `use_grounded_grasp_v2`, canonical
    #       `nn.TransformerDecoder` core UNCHANGED) gets three new additive global-conditioning
    #       terms on top of `current_state` (already there for every architecture):
    #       `grounded_target_feature`, `predicted_target_xyz` (reusing the same `target_xyz_encoder`
    #       `temporal_decoder_grounded_grasp` already has), and a binary grasp-phase embedding
    #       (0=PRE_GRASP, 1=POST_GRASP -- during training, read directly off `observation.state`'s
    #       own gripper channel via `utils._state_gripper_is_open`; at inference, tracked as the
    #       policy's own persistent per-episode state, flipped by the reactive-close mechanism
    #       below). No target-conditioning ablation toggle this time (always on) -- there is no
    #       `grounded_grasp_condition_decoder_on_target`-equivalent field for this architecture.
    #
    #       Inference-only reactive close (`SafeDiffVLAPolicy._grounded_grasp_v2_reactive_close`)
    #       goes further than `temporal_decoder_grounded_grasp`'s: on proximity trigger it also
    #       flips the policy's own phase state to POST_GRASP and clears the executor's action
    #       queue, forcing a fresh replan (now conditioned on phase=POST_GRASP) on the very next
    #       step, instead of continuing to execute a chunk planned before the grasp closed. A
    #       one-shot latch keeps the gripper closed for the rest of the episode once triggered (no
    #       reopening) -- this behavior is EXPERIMENTAL and specific to `select_poker`'s
    #       grasp/lift-only validation; a place task would need a separate release phase this does
    #       not implement.
    #   "temporal_decoder_grounded_grasp_v3": experimental. v2's held-out eval (canonical baseline
    #       2/10 success vs. v2 1/10; baseline wrong_object_grounding 8/10 vs. v2 5/10; mean
    #       target-xyz error 0.205m vs. 0.145m) showed v2's single `TargetPointHead` MLP improved
    #       grounding/localization but never reached grasp-relevant precision, and did not move
    #       task success. This architecture splits v2's single "grounded feature -> direct xyz"
    #       regression into two stages (see `grounded_grasp_v3.py`):
    #         `TargetQueryExtractor` (v2's own module, reused unchanged) -> target_language_query
    #           -> `SharpenedVisualGrounding`: an EXPLICIT target-query -> image-token similarity,
    #              sharpened via temperature-scaled softmax (`grounded_grasp_v3_temperature`) and
    #              an optional hard top-k restriction (`grounded_grasp_v3_top_k`) -- replacing v2's
    #              4-head `nn.MultiheadAttention` (`VisualGroundingCrossAttention`), whose per-head
    #              weights are never exposed and whose head-averaged output can stay diffuse even
    #              when individual heads are peaked. Selection statistics (entropy, top-1 mass,
    #              top-k mass) are always computed and surfaced (see `_plan_temporal_decoder`'s
    #              metrics) so how concentrated the selection actually is can be inspected directly.
    #              -> coarse_grounded_feature [B, D]
    #         `CoarseTargetHead` (same small-MLP shape as `TargetPointHead`, separately
    #           parameterized -- v1/v2's `TargetPointHead` is untouched) -> coarse_xyz [B, 3]
    #         `FineResidualHead`, conditioned on [coarse_grounded_feature, coarse_xyz,
    #           current_state, target_language_query] -> a BOUNDED local correction:
    #           `delta_xyz = grounded_grasp_v3_max_refine_offset * tanh(raw_delta)`
    #         final_target_xyz = coarse_xyz + delta_xyz
    #       Supervision (`utils.find_grasp_target`, identical label/masking rule to v1/v2):
    #       `loss_target = lambda_target_coarse * MSE(coarse_xyz, gt) + lambda_target_refine *
    #       MSE(final_target_xyz, gt)`, both terms masked to samples with a valid transition,
    #       exactly like v1/v2's single `loss_target`. `TemporalActionDecoder` conditioning reuses
    #       `use_grounded_grasp_v2`'s existing wiring UNCHANGED (grounded_target_feature +
    #       final_target_xyz via the same `target_xyz_encoder` + binary grasp phase) -- the decoder
    #       cannot tell v2's and v3's `grounded_target_feature`/`target_xyz` apart, by design, since
    #       this experiment's only variable is the grounding/localization head. Reactive close
    #       (`SafeDiffVLAPolicy._grounded_grasp_v3_reactive_close`) is a straight copy of v2's
    #       proximity-trigger/phase-flip/one-shot-latch behavior, evaluated against
    #       `final_target_xyz` instead of v2's single predicted xyz -- threshold value and
    #       phase/close logic are deliberately UNCHANGED from v2 for this pilot (see
    #       `grounded_grasp_close_threshold_m`); this experiment isolates the grounding head only.
    #   "temporal_decoder_grounded_grasp_v2_5": experimental. v2's `TemporalActionDecoder` regresses
    #       one 50-step chunk end to end, including the final centimeters of the approach where
    #       diagnostic oracle experiments found close-TIMING (not just target-xyz accuracy) is a
    #       real bottleneck -- and v2's own held-out eval never moved task success even once
    #       target-xyz localization improved. This architecture changes NOTHING about v2's
    #       grounding/localization/global-decoder pipeline -- `TargetQueryExtractor`,
    #       `VisualGroundingCrossAttention`, `TargetPointHead`, `TemporalActionDecoder` are all
    #       reused exactly as v2 already computes them, in their own separate branch of
    #       `modeling_safediff_vla.py` (v2's own branch/tests are completely untouched). Instead it
    #       adds ONE new module, `LocalGraspRefiner` (see `local_grasp_refiner.py`), and a 3-stage
    #       runtime controller:
    #         PRE_GRASP: v2's own 50-step global approach, completely unchanged, until
    #           `||current_ee_xyz - predicted_target_xyz|| <= local_grasp_radius_m` (one-shot --
    #           this transition can only ever happen once per episode).
    #         LOCAL_GRASP: `LocalGraspRefiner` takes over -- genuinely closed-loop, re-observing and
    #           replanning EVERY single step (`local_grasp_execute_horizon=1`), unlike every other
    #           stage/architecture in this file, which all plan an open-loop chunk and execute it
    #           for `execute_horizon` steps before replanning. The refiner predicts a bounded
    #           RESIDUAL `(delta_xyz, delta_rotation, gripper)` relative to the CURRENT observed EE
    #           pose -- never an absolute target pose -- conditioned explicitly on
    #           `relative_xyz = predicted_target_xyz - current_ee_xyz` (v2's own predicted target,
    #           just consumed differently) alongside `grounded_target_feature`, a fresh
    #           masked-mean image-token pooling, and `current_state`. `delta_xyz`/`delta_rotation`
    #           are `tanh`-bounded by `local_grasp_max_delta_xyz_m`/`local_grasp_max_delta_rot_rad`
    #           so each step can only ever nudge the arm by a few cm/degrees. `gripper` is predicted
    #           directly by the refiner (NOT v1/v2/v3's proximity-based forced close, which rarely
    #           fired in v2's own held-out reproduction) -- once it first reads "closed", a one-shot
    #           latch holds it closed (preventing open/close chatter) and the SAME event ends
    #           LOCAL_GRASP.
    #         POST_GRASP: back to v2's own global decoder, now conditioned on phase=POST_GRASP,
    #           exactly like v2's own POST_GRASP stage -- reusing `use_grounded_grasp_v2`'s existing
    #           decoder-conditioning wiring UNCHANGED, since LOCAL_GRASP never calls the decoder at
    #           all and PRE_GRASP/POST_GRASP are the only two values ever fed to its existing binary
    #           `grasp_phase` embedding.
    #       Training (see `local_grasp_refiner.py`'s own docstring): `LocalGraspRefiner` is
    #       supervised only on samples within `local_grasp_window_k` steps of a demonstrated
    #       open->close transition (`utils.find_grasp_target`'s own transition-finding formula,
    #       duplicated -- not imported -- specifically so v1/v2's `find_grasp_target` itself is
    #       never touched), predicting a short `local_grasp_action_horizon`-step residual chunk
    #       against `action[t+i] - current_ee_xyz[t]` for each such sample -- never the full
    #       50-step target. `loss_total = loss_v2_global + lambda_local_grasp * loss_local`; v2's
    #       own loss terms (including `loss_target`) are computed identically to v2's own branch.
    #       The recommended first experiment (`local_grasp_freeze_global_modules`, default True)
    #       freezes every v2 module (backbone, `TargetQueryExtractor`,
    #       `VisualGroundingCrossAttention`, `TargetPointHead`, `TemporalActionDecoder`) so ONLY
    #       `LocalGraspRefiner` trains -- isolating "does the local closed-loop residual controller
    #       improve grasp precision" from any change to v2's own already-checkpointed behavior. A
    #       v2 checkpoint's weights may be loaded to initialize this architecture's shared modules;
    #       a fresh full-model run is a separate, later decision.
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

    # -- `temporal_decoder_grounded_grasp` only (see `target_point_head.py`, `utils.find_grasp_target`) --
    # Weight of the `TargetPointHead` grasp-xyz regression loss (`utils.find_grasp_target`'s
    # label, masked to pre-grasp samples with a real open->close transition in-horizon).
    lambda_target: float = 1.0
    # Hidden width of `TargetPointHead`'s small MLP (mirrors `state_head_hidden_dim` above).
    target_head_hidden_dim: int = 256
    # RAW-physical-scale gripper decision boundary used when extracting the GT grasp-target label
    # (see `utils.find_grasp_target`, `SafeDiffVLAPolicy._degrip_for_transition_detection`) --
    # same convention used throughout this codebase (e.g.
    # `examples/safediff_vla/place_phase_forensics.py`'s `GRIPPER_THRESHOLD`). Applied to
    # `action`'s gripper channel with `>threshold`==open, and separately to `observation.state`'s
    # gripper channel with `<threshold`==open -- these are NOT the same polarity, see
    # `utils._action_gripper_is_open` / `utils._state_gripper_is_open`. Always un-normalized to
    # raw physical units before this threshold is applied; never pass MEAN_STD-normalized values
    # in directly, or this threshold silently miscalibrates.
    grounded_grasp_gripper_open_threshold: float = 0.5
    # Inference-only: physical distance (meters, robot-base frame) between the current EE
    # position (`observation.state[:3]`, un-normalized) and the predicted grasp target
    # (`TargetPointHead`'s output, un-normalized) below which
    # `SafeDiffVLAPolicy._grounded_grasp_reactive_close` force-closes the gripper -- replacing the
    # decoder's own regressed gripper channel as the execution-time close trigger for this
    # architecture only. Untuned first-cut default; diagnostic oracle experiments
    # (`examples/safediff_vla/grasp_*_oracle*.py`) found actual contact typically within ~1-2cm
    # once xyz aim is corrected.
    grounded_grasp_close_threshold_m: float = 0.03
    # Ablation switch (see `examples/safediff_vla/train_ablation_5k.py`): when True (default,
    # unchanged existing behavior), `TargetPointHead`'s predicted xyz is fed into
    # `TemporalActionDecoder` as global conditioning (`use_target_xyz`, see `temporal_decoder.py`).
    # When False, `TargetPointHead` is still built and `lambda_target * loss_target` is still
    # trained exactly the same way -- only the decoder's own conditioning input changes: it never
    # sees the predicted target at all (`target_xyz=None` at both train and inference time,
    # decoder built with `use_target_xyz=False`). Isolates "does an auxiliary grasp-target
    # regression loss alone improve the pose trajectory" from "does conditioning the decoder on
    # that prediction help or hurt" -- two independently toggleable effects that were previously
    # coupled together under a single architecture. Has no effect on any other architecture.
    grounded_grasp_condition_decoder_on_target: bool = True

    # -- `temporal_decoder_grounded_grasp_v3` only (see `grounded_grasp_v3.py`) --
    # Weight of the coarse-stage (`CoarseTargetHead`) target-xyz regression loss. Left at parity
    # (1.0) with `lambda_target_refine` for this first pilot -- no sweep.
    lambda_target_coarse: float = 1.0
    # Weight of the final-stage (coarse + bounded residual) target-xyz regression loss. Left at
    # parity (1.0) with `lambda_target_coarse` for this first pilot -- no sweep.
    lambda_target_refine: float = 1.0
    # Softmax temperature for `SharpenedVisualGrounding`'s query-to-image-token similarity logits
    # (divides the scaled-dot-product logits before softmax -- smaller = sharper/more concentrated
    # selection). 0.5 is a moderate, untuned default -- sharper than a plain temperature=1
    # scaled-dot-product attention, but far short of near-one-hot selection -- left for a later,
    # data-audit-driven sweep this pilot deliberately does not run.
    grounded_grasp_v3_temperature: float = 0.5
    # Optional hard top-k restriction on `SharpenedVisualGrounding`'s image-token logits (keep only
    # the k highest-scoring per sample, `-inf` out the rest, before the softmax). `None` (default)
    # disables it, relying on `grounded_grasp_v3_temperature` alone -- left off rather than
    # hard-coding a token count until real per-scene image-token geometry has been audited.
    grounded_grasp_v3_top_k: int | None = None
    # Hard bound on `FineResidualHead`'s residual magnitude, in the same MEAN_STD-normalized
    # action-position space as `action[..., :3]` / `TargetPointHead`'s output -- NOT physical
    # meters: `delta_xyz = grounded_grasp_v3_max_refine_offset * tanh(raw_delta)`. Conservative
    # first-cut default so the fine head can only ever locally perturb the coarse estimate, never
    # re-regress the target from scratch; revisit once a data audit characterizes the actual
    # coarse-stage error distribution -- not optimized here.
    grounded_grasp_v3_max_refine_offset: float = 0.5

    # -- `temporal_decoder_grounded_grasp_v2_5` only (see `local_grasp_refiner.py`) --
    # Inference-only: physical distance (meters, robot-base frame) between the current EE position
    # and v2's own predicted grasp target below which `SafeDiffVLAPolicy.select_action` switches
    # from PRE_GRASP (v2's global 50-step decoder) to LOCAL_GRASP (`LocalGraspRefiner`, per-step
    # closed-loop). One-shot per episode -- see `_v25_stage`'s docstring in `local_grasp_refiner.py`.
    # Untuned first-cut default (same value used as v1/v2's own `grounded_grasp_close_threshold_m`
    # neighborhood-of-contact assumption); NOT swept in this pilot.
    local_grasp_radius_m: float = 0.10
    # Hard bound (physical meters) on `LocalGraspRefiner`'s per-step position residual:
    # `delta_xyz = local_grasp_max_delta_xyz_m * tanh(raw_delta_xyz)`. Conservative first-cut
    # default (a few cm) so each closed-loop step can only ever locally nudge the arm, never jump;
    # NOT swept in this pilot.
    local_grasp_max_delta_xyz_m: float = 0.03
    # Hard bound (radians) on `LocalGraspRefiner`'s per-step rotation residual, same `tanh`-bound
    # pattern as `local_grasp_max_delta_xyz_m`. Conservative first-cut default (~3 degrees); NOT
    # swept in this pilot.
    local_grasp_max_delta_rot_rad: float = 0.05
    # Width of the residual CHUNK `LocalGraspRefiner` predicts per call -- a receding-horizon
    # training target only (see `local_grasp_refiner.py`'s docstring); at inference only the first
    # `local_grasp_execute_horizon` step(s) are ever executed before a fresh re-observe/replan.
    local_grasp_action_horizon: int = 5
    # How many of `LocalGraspRefiner`'s predicted horizon steps are actually executed before
    # re-observing and replanning from scratch. Fixed at 1 for this pilot (genuine per-step
    # closed-loop control) -- see the architecture docstring above for why this is the one stage in
    # the whole policy that isn't open-loop-chunk-then-queue.
    local_grasp_execute_horizon: int = 1
    # `LocalGraspRefiner` is only ever supervised on training samples within this many steps of a
    # demonstrated open->close transition (`utils.find_grasp_target`'s own transition-finding
    # formula, duplicated -- not imported -- in `modeling_safediff_vla.py` so `find_grasp_target`
    # itself is never touched) -- never the full 50-step window a sample's own batch happens to
    # carry. NOT swept in this pilot.
    local_grasp_window_k: int = 10
    # Weight of `LocalGraspRefiner`'s own combined local loss
    # (`loss_local_pos`/`loss_local_rot`/`loss_local_grip`, dim-count-weighted like the global
    # action loss) added on top of v2's own (unmodified) global loss:
    # `loss_total = loss_v2_global + lambda_local_grasp * loss_local`. Initial value, no sweep.
    lambda_local_grasp: float = 1.0
    # Hidden width of `LocalGraspRefiner`'s small MLP (mirrors `target_head_hidden_dim`).
    local_grasp_hidden_dim: int = 256
    # First-experiment ablation (default True, see architecture docstring): freeze every v2 module
    # (backbone, `TargetQueryExtractor`, `VisualGroundingCrossAttention`, `TargetPointHead`,
    # `TemporalActionDecoder`) so ONLY `LocalGraspRefiner` is trainable -- isolating the local
    # controller's own effect from any drift in v2's already-checkpointed global behavior. Setting
    # this False (a full-model finetune) is a separate, later experiment, not run by this pilot.
    local_grasp_freeze_global_modules: bool = True

    # -- grasp-phase-aware `LocalGraspRefiner` loss (loss-ONLY ablation, no new module -- see
    # `modeling_safediff_vla.py`'s `has_target_head_v2_5` branch, guarded by this flag right after
    # `loss_local_grip`) -- opt-in (default False): existing v2_5 configs/checkpoints/loss values
    # are byte-for-byte unaffected unless this is explicitly turned on. Supersedes the
    # `LocalGraspCorrectionHead` direction below (discontinued after its first 5k-step experiment
    # showed the learned correction was ~10x too small to matter -- see that module's docstring):
    # instead of adding a new spatial-anchor head, this reshapes `LocalGraspRefiner`'s OWN existing
    # training objective to be aware of WHEN (not just where) the grasp happens.
    local_grasp_use_phase_aware_loss: bool = False
    # Weight of the terminal grasp-position term: `Huber(predicted_terminal_xyz, p_grasp_gt)`,
    # where `p_grasp_gt` is `find_grasp_target`'s own first-open->close-transition commanded xyz
    # (the SAME label `loss_target` regresses `TargetPointHead` against -- deliberately NOT object
    # center) and `predicted_terminal_xyz` is `LocalGraspRefiner`'s own predicted xyz at that
    # sample's close step WITHIN the predicted horizon (clamped to the last horizon step if the
    # real transition is farther out than `local_grasp_action_horizon` can see). First-experiment
    # value from the requested ablation design; NOT swept.
    lambda_local_terminal_xyz: float = 2.0
    # Weight of the grasp-window approach term: every predicted horizon step (not just the terminal
    # one) regressed toward the SAME `p_grasp_gt`, weighted by a fixed linear-in-horizon-position
    # schedule (`local_grasp_approach_weight_min` -> `_max`) so later (closer-to-close) steps count
    # more. First-experiment value; NOT swept.
    lambda_local_approach_xyz: float = 1.0
    # `local_grasp_approach_weight_min`/`_max`: the linear schedule above, evaluated across
    # `local_grasp_action_horizon` steps (position 0 -> `_min`, last position -> `_max`). First
    # experiment: simple linear 0.1 -> 1.0, NOT swept.
    local_grasp_approach_weight_min: float = 0.1
    local_grasp_approach_weight_max: float = 1.0
    # Weight of the gripper-timing-reweighted term: the SAME `gt_gripper` target `loss_local_grip`
    # already uses, but per-step-weighted by `local_grasp_grip_close_weight` within +/-1 horizon
    # step of the close step (else weight 1.0) -- `loss_local_grip` itself (uniform weight) stays
    # always computed too, unchanged, purely for metrics/comparability. First-experiment value; NOT
    # swept.
    lambda_local_grip_weighted: float = 2.0
    local_grasp_grip_close_weight: float = 3.0
    # `delta` for every `torch.nn.functional.huber_loss` call in this block (terminal + approach
    # terms). Not requested explicitly by the ablation design; exposed for completeness rather than
    # hard-coding PyTorch's own default inline. NOT swept.
    local_grasp_huber_delta: float = 1.0

    # -- `LocalGraspCorrectionHead` (see `grasp_correction_head.py`), `temporal_decoder_grounded_
    # grasp_v2_5` only -- opt-in (default False): existing v2_5 checkpoints/configs are unaffected
    # unless this is explicitly turned on. DISCONTINUED after its first 5k-step experiment (see
    # `train_grounded_grasp_v2_5_correction_head_5k.py`'s RESULT note): the learned correction was
    # ~10x too small to matter (mean 1.2cm vs. the ~13-16cm gap it needed to close) because its
    # supervision came from offline i.i.d. training frames where `target_point_head` is already
    # well-fit, not from the rollout-time distribution where the real bias lives. Kept in the
    # codebase (still opt-in, still False by default) as a documented negative result, not deleted
    # -- `local_grasp_use_phase_aware_loss` above is the follow-up direction instead. Adds ONE
    # bounded spatial correction on top of
    # `TargetPointHead`'s own `target_xyz`, LOCAL_GRASP-phase only -- see that module's docstring
    # and the manual 18-episode review (`examples/safediff_vla/local_grasp_correction_manual_review.py`)
    # motivating it. `TargetQueryExtractor`/`VisualGroundingCrossAttention`/`TargetPointHead`/
    # `TemporalActionDecoder`/`LocalGraspRefiner` are none of them modified by this flag; only what
    # `LocalGraspRefiner` is CALLED WITH changes (its own forward signature/weights do not).
    local_grasp_use_correction_head: bool = False
    # Hard bound (physical meters) on `LocalGraspCorrectionHead`'s single spatial correction:
    # `delta_grasp_xyz = local_grasp_max_grasp_correction_m * tanh(raw_delta)`. Set from the manual
    # review's A_FINE_GRASP_ERROR-class residual distribution's p90 (~15.6cm over n=4, 2 tasks --
    # SMALL SAMPLE, see the review's own caveats), rounded up slightly (not a round number picked
    # without that data). Deliberately NOT sized to cover the B_WRONG_OBJECT class's much larger
    # observed residuals (~20-22cm in that same review) -- see `grasp_correction_head.py`'s
    # docstring on why this module must not be able to attempt that class of correction.
    local_grasp_max_grasp_correction_m: float = 0.16
    # A training frame's own `||gt_target_xyz - target_xyz||` (i.e. what `LocalGraspCorrectionHead`
    # would need to predict as `delta_grasp_gt` for that frame) above this is excluded from
    # `loss_grasp_correction` entirely -- a proxy for "this frame's frozen `TargetPointHead`
    # grounding already looks B_WRONG_OBJECT-like (wrong object entirely), not a fine-grasp
    # correction this bounded head should be trained to attempt." Set just above the manual
    # review's A-class max (16.4cm) and below its smaller B-class observations (21.7cm/22.5cm) --
    # see that review's notes on why this proxy is imperfect (a B-class error can be SMALLER than
    # this in a tightly-packed scene, e.g. adjacent mahjong tiles/playing cards) and is only ever a
    # partial mitigation, not a guarantee.
    local_grasp_correction_outlier_clip_m: float = 0.20
    # Weight of `loss_grasp_correction` (masked MSE against `gt_target_xyz - target_xyz`, only over
    # `local_eligible & (frame's own residual <= local_grasp_correction_outlier_clip_m)`) added to
    # the total loss. Initial value, no sweep.
    lambda_grasp_correction: float = 1.0
    # Hidden width of `LocalGraspCorrectionHead`'s small MLP (mirrors `local_grasp_hidden_dim`).
    local_grasp_correction_hidden_dim: int = 256
    # First-experiment ablation (default True, requested explicitly for this pilot): ALSO freeze
    # `LocalGraspRefiner` (on top of `local_grasp_freeze_global_modules`'s existing global-module
    # freeze) so `LocalGraspCorrectionHead` is the ONLY trainable module -- isolating whether the
    # new head alone learns a real spatial-anchor correction, uncontaminated by any simultaneous
    # drift in `LocalGraspRefiner`'s own already-checkpointed control policy. Only meaningful when
    # `local_grasp_use_correction_head=True`; setting this False (joint finetune of both) is a
    # separate, later experiment.
    local_grasp_correction_freeze_refiner: bool = True

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
        if self.lambda_target < 0:
            raise ValueError("lambda_target must be non-negative")
        if not 0 < self.grounded_grasp_gripper_open_threshold < 1:
            raise ValueError("grounded_grasp_gripper_open_threshold must be in (0, 1)")
        if self.grounded_grasp_close_threshold_m <= 0:
            raise ValueError("grounded_grasp_close_threshold_m must be positive")
        if self.lambda_target_coarse < 0:
            raise ValueError("lambda_target_coarse must be non-negative")
        if self.lambda_target_refine < 0:
            raise ValueError("lambda_target_refine must be non-negative")
        if self.grounded_grasp_v3_temperature <= 0:
            raise ValueError("grounded_grasp_v3_temperature must be positive")
        if self.grounded_grasp_v3_top_k is not None and self.grounded_grasp_v3_top_k <= 0:
            raise ValueError("grounded_grasp_v3_top_k must be a positive integer or None")
        if self.grounded_grasp_v3_max_refine_offset <= 0:
            raise ValueError("grounded_grasp_v3_max_refine_offset must be positive")
        if self.local_grasp_radius_m <= 0:
            raise ValueError("local_grasp_radius_m must be positive")
        if self.local_grasp_max_delta_xyz_m <= 0:
            raise ValueError("local_grasp_max_delta_xyz_m must be positive")
        if self.local_grasp_max_delta_rot_rad <= 0:
            raise ValueError("local_grasp_max_delta_rot_rad must be positive")
        if self.local_grasp_action_horizon <= 0:
            raise ValueError("local_grasp_action_horizon must be positive")
        if not 0 < self.local_grasp_execute_horizon <= self.local_grasp_action_horizon:
            raise ValueError("local_grasp_execute_horizon must be in [1, local_grasp_action_horizon]")
        if self.local_grasp_window_k <= 0:
            raise ValueError("local_grasp_window_k must be positive")
        if self.lambda_local_grasp < 0:
            raise ValueError("lambda_local_grasp must be non-negative")
        if self.lambda_local_terminal_xyz < 0:
            raise ValueError("lambda_local_terminal_xyz must be non-negative")
        if self.lambda_local_approach_xyz < 0:
            raise ValueError("lambda_local_approach_xyz must be non-negative")
        if self.lambda_local_grip_weighted < 0:
            raise ValueError("lambda_local_grip_weighted must be non-negative")
        if self.local_grasp_grip_close_weight <= 0:
            raise ValueError("local_grasp_grip_close_weight must be positive")
        if self.local_grasp_approach_weight_min > self.local_grasp_approach_weight_max:
            raise ValueError("local_grasp_approach_weight_min must be <= local_grasp_approach_weight_max")
        if self.local_grasp_huber_delta <= 0:
            raise ValueError("local_grasp_huber_delta must be positive")
        if self.local_grasp_max_grasp_correction_m <= 0:
            raise ValueError("local_grasp_max_grasp_correction_m must be positive")
        if self.local_grasp_correction_outlier_clip_m <= 0:
            raise ValueError("local_grasp_correction_outlier_clip_m must be positive")
        if self.lambda_grasp_correction < 0:
            raise ValueError("lambda_grasp_correction must be non-negative")
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
        # These architectures sin/cos-encode rotation (see `rotation_encoding.py`) for both
        # `observation.state` and `action`, assuming the *same* 7-dim [xyz, rx, ry, rz,
        # gripper] layout for both -- real state/action always share this layout (state is
        # the current pose, action the next commanded one), so this is a real constraint, not
        # an arbitrary one.
        if (
            self.architecture
            in (
                "temporal_decoder",
                "temporal_decoder_subgoal",
                "temporal_decoder_grounded_grasp",
                "temporal_decoder_instruction",
                "temporal_decoder_text_crossattn",
                "temporal_decoder_grounded_grasp_v2",
                "temporal_decoder_grounded_grasp_v3",
                "temporal_decoder_grounded_grasp_v2_5",
            )
            and self.robot_state_feature.shape[0] != 7
        ):
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
