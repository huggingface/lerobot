"""SafeDiff-VLA: a temporal action decoder on top of a frozen (by default) SmolVLA multimodal
encoder.

## Data flow (`architecture="temporal_decoder"` / `"temporal_decoder_subgoal"`, default)

    image / language / state
        -> SmolVLA.embed_prefix() + SmolVLMWithExpertModel.forward(inputs_embeds=[prefix, None])
           -- a *direct* call neither `predict_action_chunk` nor `forward` on the backbone expose
           (both discard exactly this tensor: `sample_actions` assigns it to `_`) -- see
           `_encode_multimodal_latent`. Read-only wrt `modeling_smolvla.py`; no changes there.
        -> latent_tokens [B, N_tokens, hidden_size]: the VLM's own post-self-attention
           representation of image+language+state, never mean-pooled, never touching the
           flow-matching action-generation loop at all.
        -> TemporalActionDecoder(latent_tokens, current_state, [subgoal_state])
               learned per-position action-query tokens cross-attend to `latent_tokens` and
               self-attend across the horizon axis (see `temporal_decoder.py`)
        -> actions [B, action_horizon, action_dim] directly -- `nominal` (SmolVLA's own action
           head output) is never consulted; it exists elsewhere only as the `smolvla_nominal`
           ablation baseline.

Action generation (`plan_action_chunk`) and execution strategy (queueing, temporal ensembling,
completion-gated replanning) are deliberately separate: `select_action` below just calls into
`execution.ActionExecutor`, which holds no model weights, so the same checkpoint can be evaluated
under different execution strategies with no retraining.

The original nominal-refinement diffusion design (`legacy_diffusion`) has moved out of this module
entirely -- see `legacy/modeling_legacy_diffusion.py`'s `LegacySafeDiffVLAPolicy`
(`--policy.type=safediff_vla_legacy`), kept only to reproduce past experiments.
"""

from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_safediff_vla import SafeDiffVLAConfig
from .domain_adapter import LiberoBackboneDomainAdapter, load_processor_normalization_stats
from .execution import ActionExecutor
from .grasp_correction_head import LocalGraspCorrectionHead
from .grounded_grasp_v2 import TargetQueryExtractor, VisualGroundingCrossAttention
from .grounded_grasp_v3 import CoarseTargetHead, FineResidualHead, SharpenedVisualGrounding
from .language_grounded_target_pooling import LanguageGroundedTargetPooling
from .local_grasp_refiner import LOCAL_GRASP, POST_GRASP, PRE_GRASP, LocalGraspRefiner
from .rotation_encoding import (
    ENCODED_DIM,
    GRIPPER_INDEX_RAW,
    decode as decode_rotation,
    encode as encode_rotation,
)
from .state_predictor import SubgoalStatePredictor
from .target_point_head import TargetPointHead
from .temporal_decoder import TemporalActionDecoder
from .utils import (
    IMAGE_MODALITY,
    STATE_MODALITY,
    TEXT_MODALITY,
    _action_gripper_is_open,
    _state_gripper_is_open,
    compute_prefix_modality_ids,
    find_grasp_target,
    masked_mean_by_modality,
    masked_mse,
    pad_or_crop_horizon,
    pad_or_crop_mask,
)


def _first_grasp_transition_idx(
    current_state: Tensor, action: Tensor, gripper_index: int, gripper_open_threshold: float, action_is_pad: Tensor | None
) -> Tensor:
    """`temporal_decoder_grounded_grasp_v2_5` only: the SAME open->close transition-finding formula
    `utils.find_grasp_target` uses internally, duplicated here (not imported, not refactored out of
    that function) specifically so `find_grasp_target` itself -- shared by v1/v2/v3 -- is never
    touched by this architecture. Returns `[B]`, the index (within `action`'s own horizon) of each
    row's first open->close transition -- meaningless where `find_grasp_target`'s own returned
    `valid_mask` is False for that row (see its docstring); used by v2_5 to gate which samples fall
    within `local_grasp_window_k` steps of a real grasp event for `LocalGraspRefiner`'s own
    supervision (`local_eligible = target_valid_mask & (first_transition_idx <= window_k)`).
    """
    state_open_now = _state_gripper_is_open(current_state[..., gripper_index], gripper_open_threshold)
    action_open = _action_gripper_is_open(action[..., gripper_index], gripper_open_threshold)
    is_open_seq = torch.cat((state_open_now.unsqueeze(1), action_open), dim=1)
    transitions = is_open_seq[:, :-1] & ~is_open_seq[:, 1:]
    if action_is_pad is not None:
        transitions = transitions & ~action_is_pad
    return transitions.to(torch.float32).argmax(dim=1)


class SafeDiffVLAPolicy(PreTrainedPolicy):
    config_class = SafeDiffVLAConfig
    name = "safediff_vla"

    def __init__(
        self,
        config: SafeDiffVLAConfig,
        backbone: nn.Module | None = None,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
        backbone_stats: dict[str, dict[str, Any]] | None = None,
        **_: Any,
    ) -> None:
        super().__init__(config)
        config.validate_features()
        self.architecture = config.architecture
        self.backbone = backbone if backbone is not None else self._make_backbone()
        self.domain_adapter = self._make_domain_adapter(dataset_stats, backbone_stats)
        if not config.freeze_vision_encoder:
            self._unfreeze_backbone_vision_encoder()
        if config.use_lora:
            self.backbone = self.backbone.wrap_with_peft(
                peft_cli_overrides={
                    "method_type": "LORA",
                    "r": config.lora_rank,
                    "lora_alpha": config.lora_alpha,
                    "lora_dropout": config.lora_dropout,
                }
            )
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)

        if self.architecture in (
            "temporal_decoder",
            "temporal_decoder_subgoal",
            "temporal_decoder_grounded_grasp",
            "temporal_decoder_instruction",
            "temporal_decoder_text_crossattn",
            "temporal_decoder_grounded_grasp_v2",
            "temporal_decoder_grounded_grasp_v3",
            "temporal_decoder_grounded_grasp_v2_5",
        ):
            use_subgoal = self.architecture == "temporal_decoder_subgoal"
            has_target_head = self.architecture == "temporal_decoder_grounded_grasp"
            use_instruction = self.architecture == "temporal_decoder_instruction"
            use_text_crossattn = self.architecture == "temporal_decoder_text_crossattn"
            has_target_head_v2 = self.architecture == "temporal_decoder_grounded_grasp_v2"
            has_target_head_v3 = self.architecture == "temporal_decoder_grounded_grasp_v3"
            has_target_head_v2_5 = self.architecture == "temporal_decoder_grounded_grasp_v2_5"
            # Ablation switch (`grounded_grasp_condition_decoder_on_target`, default True):
            # `TargetPointHead` (below) is built whenever this architecture is used, regardless of
            # this flag -- the auxiliary target-regression loss is always trainable. Only the
            # DECODER's own conditioning input depends on it -- see
            # `SafeDiffVLAConfig.grounded_grasp_condition_decoder_on_target`'s docstring.
            # `temporal_decoder_grounded_grasp_v2`/`_v3`/`_v2_5` have no such ablation toggle --
            # target conditioning is always on for them.
            use_target_xyz = (
                (has_target_head and config.grounded_grasp_condition_decoder_on_target)
                or has_target_head_v2
                or has_target_head_v3
                or has_target_head_v2_5
            )
            self._register_rotation_stats(dataset_stats)
            # `ENCODED_DIM` (10) replaces the raw 7-D [xyz, rx, ry, rz, gripper] layout with
            # [xyz, sin(rx), cos(rx), sin(ry), cos(ry), sin(rz), cos(rz), gripper] everywhere the
            # decoder itself sees state/action -- see `rotation_encoding.py`. The *external* 7-D
            # contract (`config.action_feature`/`config.robot_state_feature`, the dataset/env/
            # postprocessor) is completely unaffected: `_encode_state` / `_encode_action_target` /
            # `_decode_action_prediction` convert at this policy's own boundary only.
            self.decoder = TemporalActionDecoder(
                ENCODED_DIM,
                ENCODED_DIM,
                self._multimodal_latent_dim(),
                config.decoder_hidden_dim,
                config.action_horizon,
                num_layers=config.decoder_num_layers,
                num_heads=config.decoder_num_heads,
                ffn_dim=config.decoder_ffn_dim,
                dropout=config.decoder_dropout,
                use_subgoal=use_subgoal,
                use_target_xyz=use_target_xyz,
                use_instruction=use_instruction,
                use_text_crossattn=use_text_crossattn,
                # v3/v2_5 both reuse this exact conditioning wiring (grounded_target_feature +
                # target_xyz + binary phase) UNCHANGED from v2 -- see `grounded_grasp_v3.py`'s and
                # `local_grasp_refiner.py`'s module docstrings; the decoder cannot tell v2's, v3's,
                # and v2_5's `grounded_target_feature`/`target_xyz` apart, by design. v2_5 only ever
                # calls this decoder for its PRE_GRASP/POST_GRASP stages -- LOCAL_GRASP uses
                # `LocalGraspRefiner` instead and never touches this decoder at all.
                use_grounded_grasp_v2=has_target_head_v2 or has_target_head_v3 or has_target_head_v2_5,
            )
            if use_subgoal:
                # `_pooled_latent`'s modality-aware pooling concatenates image/text-masked-mean +
                # the state token (each `_multimodal_latent_dim()`-wide) and projects back down to
                # `_multimodal_latent_dim()` -- used by `_predict_subgoal` only. The target head
                # below uses its own language-grounded cross-attention pooling instead of this.
                self.modality_pool_projection = nn.Linear(3 * self._multimodal_latent_dim(), self._multimodal_latent_dim())
                self.latent_pool_projection = nn.Linear(self._multimodal_latent_dim(), config.latent_dim)
                self.subgoal_state_predictor = SubgoalStatePredictor(
                    ENCODED_DIM, config.latent_dim, config.state_head_hidden_dim
                )
            if has_target_head:
                # Text-queried cross-attention over image tokens (see
                # `language_grounded_target_pooling.py`) replaces `_pooled_latent`'s
                # language-agnostic masked-mean as `TargetPointHead`'s input feature, so the
                # instruction itself picks which image tokens matter. Unlike the subgoal path
                # above, `TargetPointHead` reads this pooled feature directly (no intermediate
                # `config.latent_dim` projection) -- see its own docstring: it's deliberately
                # latent-only, no current-state input beyond the state token folded into pooling.
                self.language_grounded_target_pooling = LanguageGroundedTargetPooling(self._multimodal_latent_dim())
                self.target_point_head = TargetPointHead(self._multimodal_latent_dim(), config.target_head_hidden_dim)
                self._register_position_and_gripper_stats(dataset_stats)
            if has_target_head_v2:
                # Explicit target-language-query -> visual-grounding-cross-attention pipeline (see
                # `grounded_grasp_v2.py` and `configuration_safediff_vla.py`'s architecture
                # docstring for the forward-path-tracing findings motivating this) -- replaces both
                # `_pooled_latent` and `LanguageGroundedTargetPooling` (neither used here) as
                # `TargetPointHead`'s input feature.
                self.target_query_extractor = TargetQueryExtractor(self._multimodal_latent_dim())
                self.visual_grounding_cross_attn = VisualGroundingCrossAttention(self._multimodal_latent_dim())
                self.target_point_head = TargetPointHead(self._multimodal_latent_dim(), config.target_head_hidden_dim)
                self._register_position_and_gripper_stats(dataset_stats)
            if has_target_head_v3:
                # Coarse-then-refine target localization (see `grounded_grasp_v3.py`'s module
                # docstring): same `TargetQueryExtractor` class as v2 (a fresh, independently
                # trained instance -- weights are NOT shared with a v2 checkpoint), but the visual
                # grounding stage is `SharpenedVisualGrounding` (explicit, sharpened, loggable
                # attention over image tokens) instead of v2's 4-head `VisualGroundingCrossAttention`,
                # and target-xyz regression is split into `CoarseTargetHead` + `FineResidualHead`
                # instead of v1/v2's single `TargetPointHead` (untouched, not built here).
                self.target_query_extractor = TargetQueryExtractor(self._multimodal_latent_dim())
                self.sharpened_visual_grounding = SharpenedVisualGrounding(
                    self._multimodal_latent_dim(),
                    temperature=config.grounded_grasp_v3_temperature,
                    top_k=config.grounded_grasp_v3_top_k,
                )
                self.coarse_target_head = CoarseTargetHead(self._multimodal_latent_dim(), config.target_head_hidden_dim)
                self.fine_residual_head = FineResidualHead(
                    self._multimodal_latent_dim(),
                    ENCODED_DIM,
                    config.target_head_hidden_dim,
                    config.grounded_grasp_v3_max_refine_offset,
                )
                self._register_position_and_gripper_stats(dataset_stats)
            if has_target_head_v2_5:
                # v2's EXACT modules, own instances (fresh weights, or a v2 checkpoint's weights
                # loaded in afterwards -- see `local_grasp_freeze_global_modules`'s docstring). This
                # is a straight copy of `has_target_head_v2`'s own build block above -- deliberately
                # NOT shared code with it, so nothing added below (or in this architecture's own
                # forward/plan branches) can ever perturb v2's own build path.
                self.target_query_extractor = TargetQueryExtractor(self._multimodal_latent_dim())
                self.visual_grounding_cross_attn = VisualGroundingCrossAttention(self._multimodal_latent_dim())
                self.target_point_head = TargetPointHead(self._multimodal_latent_dim(), config.target_head_hidden_dim)
                self._register_position_and_gripper_stats(dataset_stats)
                self.local_grasp_refiner = LocalGraspRefiner(
                    latent_dim=self._multimodal_latent_dim(),
                    state_dim=ENCODED_DIM,
                    hidden_dim=config.local_grasp_hidden_dim,
                    horizon=config.local_grasp_action_horizon,
                    max_delta_xyz=config.local_grasp_max_delta_xyz_m,
                    max_delta_rot=config.local_grasp_max_delta_rot_rad,
                )
                if config.local_grasp_freeze_global_modules:
                    # First experiment (see architecture docstring): freeze every v2 module --
                    # `freeze_backbone` (checked in __post_init__/enforced above) already covers the
                    # backbone -- so ONLY `local_grasp_refiner` ends up in `get_optim_params()`.
                    self.target_query_extractor.requires_grad_(False)
                    self.visual_grounding_cross_attn.requires_grad_(False)
                    self.target_point_head.requires_grad_(False)
                    self.decoder.requires_grad_(False)
                if config.local_grasp_use_correction_head:
                    # See `grasp_correction_head.py`'s module docstring. Opt-in, `local_grasp_
                    # refiner` itself is never modified -- only what it's called WITH (below, in
                    # the `has_target_head_v2_5` forward branch and `_plan_local_grasp_step`).
                    self.local_grasp_correction_head = LocalGraspCorrectionHead(
                        latent_dim=self._multimodal_latent_dim(),
                        state_dim=ENCODED_DIM,
                        hidden_dim=config.local_grasp_correction_hidden_dim,
                        max_grasp_correction_m=config.local_grasp_max_grasp_correction_m,
                    )
                    if config.local_grasp_correction_freeze_refiner:
                        # First-experiment ablation (requested explicitly): isolate whether
                        # `local_grasp_correction_head` alone learns a real correction, with
                        # `local_grasp_refiner`'s own already-checkpointed control held fixed.
                        self.local_grasp_refiner.requires_grad_(False)
        elif self.architecture not in ("smolvla_nominal", "smolvla_finetune"):
            raise ValueError(f"Unknown architecture {self.architecture!r}")
        # "smolvla_nominal" / "smolvla_finetune": nothing to build -- both just call the backbone's
        # own flow-matching action expert directly (see `_nominal_actions`); the only difference is
        # whether the backbone is frozen (config.freeze_backbone, enforced in __post_init__ above).

        self.reset()

    def _make_domain_adapter(
        self,
        dataset_stats: dict[str, dict[str, Any]] | None,
        backbone_stats: dict[str, dict[str, Any]] | None,
    ) -> LiberoBackboneDomainAdapter | None:
        if not self.config.use_backbone_domain_adapter:
            return None
        source_stats = backbone_stats or load_processor_normalization_stats(self.config.backbone_name)
        target_stats = dataset_stats
        if target_stats is None and self.config.pretrained_path:
            target_stats = load_processor_normalization_stats(self.config.pretrained_path)
        if target_stats is None:
            raise ValueError(
                "Backbone domain adaptation needs target dataset_stats during training or "
                "a pretrained_path containing the SafeDiff processor during evaluation"
            )
        return LiberoBackboneDomainAdapter(
            source_stats,
            target_stats,
            state_dim=self.config.input_features["observation.state"].shape[0],
            action_dim=self.config.action_feature.shape[0],
            semantics=self.config.backbone_action_conversion_semantics,
        )

    def _register_rotation_stats(self, dataset_stats: dict[str, dict[str, Any]] | None) -> None:
        """Per-dimension mean/std for `observation.state[3:6]` / `action[3:6]` (rx, ry, rz),
        used by `rotation_encoding.encode`/`decode` to recover raw radians from the outer
        preprocessor's MEAN_STD-normalized values (and back). Same stats source/precedence as
        `_make_domain_adapter`: live `dataset_stats` during training, else the saved checkpoint's
        own normalizer stats when loading a pretrained model. Falls back to mean=0/std=1 (an
        effective no-op un-normalize) when neither is available -- only ever exercised by tests
        that build a policy from a bare config with synthetic data, where exact recovery of a
        "raw" angle from meaningless random values has no correct answer anyway.
        """
        stats = dataset_stats
        if stats is None and self.config.pretrained_path:
            try:
                stats = load_processor_normalization_stats(self.config.pretrained_path)
            except Exception:  # noqa: BLE001 - best-effort; fall back below
                stats = None

        def rot_stat(feature: str, statistic: str) -> Tensor:
            if stats is not None:
                try:
                    value = stats[feature][statistic]
                    value = value if isinstance(value, Tensor) else torch.as_tensor(value)
                    return value.reshape(-1)[3:6].float().clone()
                except (KeyError, TypeError, IndexError):
                    pass
            return torch.zeros(3) if statistic == "mean" else torch.ones(3)

        self.register_buffer("state_rot_mean", rot_stat(OBS_STATE, "mean"))
        self.register_buffer("state_rot_std", rot_stat(OBS_STATE, "std"))
        self.register_buffer("action_rot_mean", rot_stat(ACTION, "mean"))
        self.register_buffer("action_rot_std", rot_stat(ACTION, "std"))

    def _register_position_and_gripper_stats(self, dataset_stats: dict[str, dict[str, Any]] | None) -> None:
        """`temporal_decoder_grounded_grasp` only: per-dimension mean/std for `observation.state[:3]`
        / `action[:3]` (xyz), used by `_grounded_grasp_reactive_close` to un-normalize the current
        EE position and the predicted grasp target into physical meters (for
        `SafeDiffVLAConfig.grounded_grasp_close_threshold_m`); `action[6]` (gripper), used to
        compute the exact normalized value that decodes to a physically "closed" gripper command;
        and `observation.state[6]` (gripper), used by `_degrip_for_transition_detection` to
        un-normalize `observation.state`'s gripper channel back to raw physical units before
        `utils.find_grasp_target`'s threshold check (see that function's docstring on why
        `observation.state`'s and `action`'s gripper channels must never be conflated). Deliberately
        a separate, duplicated method rather than folding into `_register_rotation_stats` above --
        keeps that method, and every other architecture's behavior, completely untouched. Same
        stats source/precedence/fallback as `_register_rotation_stats`.
        """
        stats = dataset_stats
        if stats is None and self.config.pretrained_path:
            try:
                stats = load_processor_normalization_stats(self.config.pretrained_path)
            except Exception:  # noqa: BLE001 - best-effort; fall back below
                stats = None

        def stat(feature: str, statistic: str, index_slice: slice) -> Tensor:
            if stats is not None:
                try:
                    value = stats[feature][statistic]
                    value = value if isinstance(value, Tensor) else torch.as_tensor(value)
                    return value.reshape(-1)[index_slice].float().clone()
                except (KeyError, TypeError, IndexError):
                    pass
            width = index_slice.stop - index_slice.start
            return torch.zeros(width) if statistic == "mean" else torch.ones(width)

        pos_slice = slice(0, 3)
        grip_slice = slice(GRIPPER_INDEX_RAW, GRIPPER_INDEX_RAW + 1)
        self.register_buffer("state_pos_mean", stat(OBS_STATE, "mean", pos_slice))
        self.register_buffer("state_pos_std", stat(OBS_STATE, "std", pos_slice))
        self.register_buffer("action_pos_mean", stat(ACTION, "mean", pos_slice))
        self.register_buffer("action_pos_std", stat(ACTION, "std", pos_slice))
        self.register_buffer("action_grip_mean", stat(ACTION, "mean", grip_slice))
        self.register_buffer("action_grip_std", stat(ACTION, "std", grip_slice))
        self.register_buffer("state_grip_mean", stat(OBS_STATE, "mean", grip_slice))
        self.register_buffer("state_grip_std", stat(OBS_STATE, "std", grip_slice))

    def _degrip_for_transition_detection(self, current_state_raw: Tensor, clean_raw: Tensor) -> tuple[Tensor, Tensor]:
        """Return copies of `current_state_raw` / `clean_raw` with ONLY the gripper channel
        un-normalized back to raw physical scale -- xyz/rotation stay exactly as given (still
        MEAN_STD-normalized during training), since `find_grasp_target`'s returned `target_xyz`
        must stay in the same space `TargetPointHead` is trained against. Only the gripper index
        needs raw physical units, for `SafeDiffVLAConfig.grounded_grasp_gripper_open_threshold`'s
        raw-scale (0.5-ish) decision boundary to mean what it says -- passing normalized gripper
        values straight into `find_grasp_target` silently miscalibrates that threshold."""
        state_fixed = current_state_raw.clone()
        state_fixed[..., GRIPPER_INDEX_RAW] = (
            current_state_raw[..., GRIPPER_INDEX_RAW] * self.state_grip_std + self.state_grip_mean
        )
        action_fixed = clean_raw.clone()
        action_fixed[..., GRIPPER_INDEX_RAW] = (
            clean_raw[..., GRIPPER_INDEX_RAW] * self.action_grip_std + self.action_grip_mean
        )
        return state_fixed, action_fixed

    def _encode_state(self, state7: Tensor) -> Tensor:
        return encode_rotation(state7, self.state_rot_mean, self.state_rot_std)

    def _encode_action_target(self, action7: Tensor) -> Tensor:
        return encode_rotation(action7, self.action_rot_mean, self.action_rot_std)

    def _decode_action_prediction(self, action10: Tensor) -> Tensor:
        return decode_rotation(action10, self.action_rot_mean, self.action_rot_std)

    def _unfreeze_backbone_vision_encoder(self) -> None:
        """Undo the vision-encoder freeze baked into the loaded SmolVLA checkpoint.

        `_make_backbone()` reconstructs the backbone via `SmolVLAPolicy.from_pretrained`,
        which restores *that checkpoint's own* saved config — e.g. `HuggingFaceVLA/smolvla_libero`
        ships with `freeze_vision_encoder=True`, so `SmolVLMWithExpertModel.__init__` already set
        `requires_grad=False` on the vision tower before this class ever sees it. Setting
        `SafeDiffVLAConfig.freeze_backbone=False` does NOT undo that — it only skips the *additional*
        blanket `requires_grad_(False)` applied below. Note `SmolVLMWithExpertModel.set_requires_grad()`
        is one-directional (it only ever sets `requires_grad=False`, never back to `True`), so simply
        toggling the flag and re-calling it is a no-op here — we flip the params directly instead.
        """
        vlm_with_expert = getattr(getattr(self.backbone, "model", None), "vlm_with_expert", None)
        if vlm_with_expert is None or not hasattr(vlm_with_expert, "get_vlm_model"):
            raise RuntimeError(
                "freeze_vision_encoder=False requires a SmolVLA-style backbone exposing "
                "model.vlm_with_expert.get_vlm_model()."
            )
        vlm_with_expert.freeze_vision_encoder = False
        vlm_with_expert.get_vlm_model().vision_model.requires_grad_(True)

    def _multimodal_latent_dim(self) -> int:
        """Token width of `_encode_multimodal_latent`'s output, to size the decoder's own input
        projection up front. Resolved eagerly (instead of via `nn.LazyLinear`) because
        `lerobot-train` counts `policy.parameters()` before any forward pass, which raises on
        uninitialized lazy parameters."""
        if hasattr(self.backbone, "multimodal_latent_dim"):
            return self.backbone.multimodal_latent_dim
        return self.backbone.model.vlm_with_expert.config.text_config.hidden_size

    def _make_backbone(self) -> nn.Module:
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if self.config.backbone_name:
            return SmolVLAPolicy.from_pretrained(self.config.backbone_name)
        backbone_config = SmolVLAConfig(
            input_features=dict(self.config.input_features or {}),
            output_features=dict(self.config.output_features or {}),
            device=self.config.device,
            chunk_size=self.config.action_horizon,
            n_action_steps=self.config.action_horizon,
            vlm_model_name=self.config.vlm_model_name,
        )
        return SmolVLAPolicy(backbone_config)

    def get_optim_params(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def reset(self) -> None:
        self._executor = ActionExecutor(self.config)
        # Harmless no-ops for every other architecture -- only
        # `_grounded_grasp_reactive_close`/`_plan_temporal_decoder` (guarded on
        # `architecture == "temporal_decoder_grounded_grasp"`) ever read these.
        self._grounded_grasp_close_triggered = False
        self._grounded_grasp_last_target_xyz = None
        # Harmless no-ops for every other architecture -- only
        # `_grounded_grasp_v2_reactive_close`/`_plan_temporal_decoder` (guarded on
        # `architecture == "temporal_decoder_grounded_grasp_v2"`) ever read these.
        # `_v2_grasp_phase`: 0=PRE_GRASP, 1=POST_GRASP -- the policy's own persistent per-episode
        # phase, fed to the decoder as conditioning and flipped by the reactive-close mechanism
        # (not read from the environment at inference time).
        self._v2_grasp_phase = 0
        self._v2_close_triggered = False
        self._v2_last_target_xyz = None
        # Harmless no-ops for every other architecture -- only
        # `_grounded_grasp_v3_reactive_close`/`_plan_temporal_decoder` (guarded on
        # `architecture == "temporal_decoder_grounded_grasp_v3"`) ever read these. Same semantics
        # as `_v2_*` above, tracked independently since v3 is evaluated as its own checkpoint.
        self._v3_grasp_phase = 0
        self._v3_close_triggered = False
        self._v3_last_target_xyz = None
        # Harmless no-ops for every other architecture -- only v2_5's `select_action`/
        # `_plan_temporal_decoder` branches (guarded on
        # `architecture == "temporal_decoder_grounded_grasp_v2_5"`) ever read these.
        # `_v25_stage`: PRE_GRASP(0)/LOCAL_GRASP(1)/POST_GRASP(2) (see `local_grasp_refiner.py`) --
        # PRE_GRASP->LOCAL_GRASP can only happen once per episode (gated on
        # `self._v25_stage == PRE_GRASP`, never re-entered once it's moved past that), and
        # LOCAL_GRASP->POST_GRASP is likewise one-shot (gated on `_v25_gripper_closed_latch`).
        self._v25_stage = PRE_GRASP
        self._v25_gripper_closed_latch = False
        self._v25_last_target_xyz = None
        if hasattr(self.backbone, "reset"):
            self.backbone.reset()

    # ---- shared helpers -----------------------------------------------------------------

    def _prepare_backbone_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Never let anything but the *current* state reach the backbone (defensive: guards
        # against a caller passing a multi-frame `observation.state` in for some other reason).
        backbone_input = {**batch, OBS_STATE: self._current_state(batch)}
        return (
            self.domain_adapter.observation_for_backbone(backbone_input)
            if self.domain_adapter
            else backbone_input
        )

    @staticmethod
    def _current_state(batch: dict[str, Tensor]) -> Tensor:
        """The *current* (t=0) normalized state. Defensive `ndim` guard in case a caller passes
        a multi-frame `observation.state` in for some other reason — only index 0 is ever used."""
        state = batch[OBS_STATE]
        return state[:, 0] if state.ndim > 2 else state

    def _pooled_latent(
        self, latent_tokens: Tensor, latent_pad_mask: Tensor | None, latent_modality_ids: Tensor
    ) -> Tensor:
        """Modality-aware pooling for `TargetPointHead`/`SubgoalStatePredictor` only -- never the
        decoder's own full-token cross-attention (`TemporalActionDecoder` reads `latent_tokens`
        directly, untouched by this method).

        Replaces the old all-token unweighted mean (`latent_tokens.mean(dim=1)` /
        `(latent_tokens * pad_mask).sum(1) / pad_mask.sum(1)`), which drowns the handful of
        language tokens in the many more image patch tokens -- e.g. ~sub-10% of the pooled
        vector's weight coming from the instruction with a typical single-camera token count,
        silently diluting the one signal `TargetPointHead` actually needs to pick the right target
        among several visually similar candidates. Instead: masked-mean image tokens and text
        tokens SEPARATELY (each modality's own tokens average only against each other, never
        against the other modality's token count), keep the single state token as-is (no
        averaging), then project the concatenation back down to `_multimodal_latent_dim()` -- see
        `utils.compute_prefix_modality_ids` for how `latent_modality_ids` is derived.
        """
        if latent_pad_mask is None:
            latent_pad_mask = latent_tokens.new_ones(latent_tokens.shape[:2], dtype=torch.bool)
        valid = latent_pad_mask.to(torch.bool)

        def masked_mean(modality: int) -> Tensor:
            mask = ((latent_modality_ids == modality) & valid).unsqueeze(-1).to(latent_tokens.dtype)
            return (latent_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        image_pooled = masked_mean(IMAGE_MODALITY)
        text_pooled = masked_mean(TEXT_MODALITY)
        # State is exactly one token (see `compute_prefix_modality_ids`) -- summing rather than
        # indexing directly tolerates a test double that (unlike the real backbone) tags more than
        # one position as state, still degrading gracefully to their mean via the same clamp.
        state_mask = (latent_modality_ids == STATE_MODALITY).unsqueeze(-1).to(latent_tokens.dtype)
        state_token = (latent_tokens * state_mask).sum(dim=1) / state_mask.sum(dim=1).clamp_min(1)
        return self.modality_pool_projection(torch.cat([image_pooled, text_pooled, state_token], dim=-1))

    def _nominal_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """SmolVLA's own action-head output, unmodified. Used by `architecture="smolvla_nominal"`."""
        backbone_batch = self._prepare_backbone_batch(batch)
        with torch.set_grad_enabled(not self.config.freeze_backbone):
            nominal = self.backbone.predict_action_chunk(dict(backbone_batch))
        nominal = pad_or_crop_horizon(nominal, self.config.action_horizon)
        if self.domain_adapter is not None:
            nominal = self.domain_adapter.nominal_for_target(nominal)
        return nominal.detach() if self.config.freeze_backbone else nominal

    def _forward_smolvla_finetune(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """`architecture="smolvla_finetune"`: train the backbone's own flow-matching action expert
        directly -- same module, same noise/time-sampled velocity-regression objective, same
        multi-step Euler sampler at inference (`_nominal_actions` -> `backbone.predict_action_chunk`)
        as SmolVLA itself, unlike `temporal_decoder`'s from-scratch decoder + decomposed MSE. No new
        parameters of our own; requires `freeze_backbone=False` (enforced in config validation)."""
        backbone_batch = self._prepare_backbone_batch(batch)
        loss, backbone_metrics = self.backbone.forward(backbone_batch)
        metrics = {"loss": loss.item()}
        metrics.update(
            {f"backbone_{k}": (v.item() if torch.is_tensor(v) else v) for k, v in backbone_metrics.items()}
        )
        return loss, metrics

    def _encode_multimodal_latent(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None, Tensor]:
        """Direct multimodal-encoder path: SmolVLA's own `embed_prefix` plus one self-attention
        pass through its VLM transformer, *without* running the flow-matching action-generation
        loop at all. Neither `predict_action_chunk` nor `forward` on the backbone expose this —
        both discard exactly this tensor (`sample_actions` assigns it to `_`) — so this
        reimplements the first few lines of `sample_actions`. Entirely read-only wrt
        `modeling_smolvla.py` (no changes there, no new interface added to it): a test double can
        instead implement `encode_multimodal_latent(batch) -> (tokens, pad_mask_or_None,
        modality_ids)` and `multimodal_latent_dim` directly.

        The third return value, `latent_modality_ids` (see `utils.compute_prefix_modality_ids`),
        is only ever consumed by `_predict_subgoal`/`_predict_target_xyz` (via `_pooled_latent`)
        -- `TemporalActionDecoder`'s own full-token cross-attention (used by every architecture,
        including plain `temporal_decoder`) reads `latent_tokens`/`latent_pad_mask` exactly as
        before and never sees it.

        If `freeze_backbone=False` this is also how gradients would actually reach the VLM.
        """
        if hasattr(self.backbone, "encode_multimodal_latent"):
            return self.backbone.encode_multimodal_latent(self._prepare_backbone_batch(batch))

        backbone_batch = self._prepare_backbone_batch(batch)
        model = self.backbone.model
        with torch.set_grad_enabled(not self.config.freeze_backbone):
            images, img_masks = self.backbone.prepare_images(backbone_batch)
            state = self.backbone.prepare_state(backbone_batch)
            lang_tokens = backbone_batch[OBS_LANGUAGE_TOKENS]
            lang_masks = backbone_batch[OBS_LANGUAGE_ATTENTION_MASK]
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            # `use_cache=True` (matching `sample_actions`'s own prefix-only call) is required
            # here, not just an optimization: it's what makes `VLAFlowMatching.forward` take the
            # `forward_attn_layer` branch, the only one that tolerates `inputs_embeds[1]=None`.
            # `forward_cross_attn_layer` (the other branch) unconditionally dereferences it and
            # crashes if `use_cache=False` forces that path instead. We discard the returned KV
            # cache either way -- we only want `outputs_embeds[0]`.
            outputs_embeds, _ = model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        latent_tokens = outputs_embeds[0].float()
        # Cheap (no extra VLM/vision-tower forward pass): `prefix_att_masks` and `lang_tokens`'s
        # width were already computed above for `embed_prefix`/`prefix_att_2d_masks` -- see
        # `compute_prefix_modality_ids`'s docstring for exactly what they give us for free.
        latent_modality_ids = compute_prefix_modality_ids(prefix_att_masks, lang_tokens.shape[1])
        if self.config.freeze_backbone:
            latent_tokens = latent_tokens.detach()
        return latent_tokens, prefix_pad_masks, latent_modality_ids

    # ---- temporal_decoder / temporal_decoder_subgoal ---------------------------------------

    def _predict_subgoal(
        self, latent_tokens: Tensor, latent_pad_mask: Tensor | None, latent_modality_ids: Tensor, current_state: Tensor
    ) -> Tensor:
        """`temporal_decoder_subgoal` only: a single predicted subgoal state `[B, state_dim]` from
        the pooled scene latent + current state (see `state_predictor.py`'s `SubgoalStatePredictor`)."""
        pooled = self.latent_pool_projection(self._pooled_latent(latent_tokens, latent_pad_mask, latent_modality_ids))
        return self.subgoal_state_predictor(pooled, current_state)

    def _predict_target_xyz(self, latent_tokens: Tensor, latent_pad_mask: Tensor | None, latent_modality_ids: Tensor) -> Tensor:
        """`temporal_decoder_grounded_grasp` only: a single predicted grasp-target xyz `[B, 3]`
        from a language-grounded visual feature (see `language_grounded_target_pooling.py`) --
        NOT `_pooled_latent`'s language-agnostic pooling, which `_predict_subgoal` still uses."""
        grounded = self.language_grounded_target_pooling(latent_tokens, latent_pad_mask, latent_modality_ids)
        return self.target_point_head(grounded)

    def _forward_temporal_decoder(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        latent_tokens, latent_pad_mask, latent_modality_ids = self._encode_multimodal_latent(batch)
        current_state_raw = self._current_state(batch)
        current_state = self._encode_state(current_state_raw)
        clean_raw = pad_or_crop_horizon(batch[ACTION], self.config.action_horizon)
        clean = self._encode_action_target(clean_raw)

        # Episode-end chunks are padded by repeating the last valid frame's action past the
        # episode boundary (`DatasetReader._get_query_indices`), flagged by `action_is_pad` (`[B,
        # action_horizon]`, True = padded/repeated -- not a real supervision target). Exclude those
        # timesteps from the pose loss entirely rather than teaching the decoder to regress toward
        # a frozen "ghost" target it was never meant to predict, and (grounded_grasp only) from
        # ever being selected as a demonstrated grasp-transition step -- a padding repeat is not a
        # real event. Falls back to "everything valid" when the batch has no such key (e.g.
        # synthetic test batches), reproducing the old unmasked behavior exactly.
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad_h = pad_or_crop_mask(action_is_pad, self.config.action_horizon)
            valid_mask = ~action_is_pad_h
        else:
            action_is_pad_h = None
            valid_mask = clean.new_ones(clean.shape[:2], dtype=torch.bool)

        subgoal_state = None
        loss_subgoal = clean.new_zeros(())
        if self.architecture == "temporal_decoder_subgoal":
            predicted_subgoal = self._predict_subgoal(latent_tokens, latent_pad_mask, latent_modality_ids, current_state)
            subgoal_state = predicted_subgoal
            subgoal_target = batch.get("observation.subgoal_state")
            if subgoal_target is not None:
                loss_subgoal = F.mse_loss(predicted_subgoal, self._encode_state(subgoal_target))

        target_xyz = None
        loss_target = clean.new_zeros(())
        if self.architecture == "temporal_decoder_grounded_grasp":
            target_xyz = self._predict_target_xyz(latent_tokens, latent_pad_mask, latent_modality_ids)
            state_for_grasp, action_for_grasp = self._degrip_for_transition_detection(current_state_raw, clean_raw)
            gt_target_xyz, target_valid_mask = find_grasp_target(
                state_for_grasp,
                action_for_grasp,
                gripper_index=GRIPPER_INDEX_RAW,
                gripper_open_threshold=self.config.grounded_grasp_gripper_open_threshold,
                action_is_pad=action_is_pad_h,
            )
            loss_target = masked_mse(
                target_xyz.unsqueeze(1), gt_target_xyz.unsqueeze(1), target_valid_mask.unsqueeze(1)
            )

        grounded_target_feature = None
        grasp_phase = None
        if self.architecture == "temporal_decoder_grounded_grasp_v2":
            target_language_query = self.target_query_extractor(latent_tokens, latent_pad_mask, latent_modality_ids)
            grounded_target_feature = self.visual_grounding_cross_attn(
                target_language_query, latent_tokens, latent_pad_mask, latent_modality_ids
            )
            target_xyz = self.target_point_head(grounded_target_feature)
            state_for_grasp, action_for_grasp = self._degrip_for_transition_detection(current_state_raw, clean_raw)
            gt_target_xyz, target_valid_mask = find_grasp_target(
                state_for_grasp,
                action_for_grasp,
                gripper_index=GRIPPER_INDEX_RAW,
                gripper_open_threshold=self.config.grounded_grasp_gripper_open_threshold,
                action_is_pad=action_is_pad_h,
            )
            loss_target = masked_mse(
                target_xyz.unsqueeze(1), gt_target_xyz.unsqueeze(1), target_valid_mask.unsqueeze(1)
            )
            # PRE_GRASP (0) / POST_GRASP (1) read directly off the CURRENT observation's own
            # gripper channel (raw physical units, `_state_gripper_is_open`'s convention) -- a real
            # demonstrated state, well-defined for every sample regardless of `target_valid_mask`
            # (a post-grasp sample has no valid grasp-TARGET label, but still has a real phase).
            grasp_phase = (
                ~_state_gripper_is_open(state_for_grasp[..., GRIPPER_INDEX_RAW], self.config.grounded_grasp_gripper_open_threshold)
            ).long()

        loss_target_coarse = clean.new_zeros(())
        loss_target_final = clean.new_zeros(())
        residual_xyz = None
        visual_selection_stats = None
        if self.architecture == "temporal_decoder_grounded_grasp_v3":
            target_language_query = self.target_query_extractor(latent_tokens, latent_pad_mask, latent_modality_ids)
            grounded_target_feature, visual_selection_stats = self.sharpened_visual_grounding(
                target_language_query, latent_tokens, latent_pad_mask, latent_modality_ids
            )
            coarse_xyz = self.coarse_target_head(grounded_target_feature)
            residual_xyz = self.fine_residual_head(grounded_target_feature, coarse_xyz, current_state, target_language_query)
            target_xyz = coarse_xyz + residual_xyz  # final_target_xyz
            state_for_grasp, action_for_grasp = self._degrip_for_transition_detection(current_state_raw, clean_raw)
            gt_target_xyz, target_valid_mask = find_grasp_target(
                state_for_grasp,
                action_for_grasp,
                gripper_index=GRIPPER_INDEX_RAW,
                gripper_open_threshold=self.config.grounded_grasp_gripper_open_threshold,
                action_is_pad=action_is_pad_h,
            )
            loss_target_coarse = masked_mse(
                coarse_xyz.unsqueeze(1), gt_target_xyz.unsqueeze(1), target_valid_mask.unsqueeze(1)
            )
            loss_target_final = masked_mse(
                target_xyz.unsqueeze(1), gt_target_xyz.unsqueeze(1), target_valid_mask.unsqueeze(1)
            )
            loss_target = (
                self.config.lambda_target_coarse * loss_target_coarse
                + self.config.lambda_target_refine * loss_target_final
            )
            # Same phase-label formula as v2 (`_state_gripper_is_open`, read off the CURRENT
            # observation's own gripper channel) -- unchanged for this pilot.
            grasp_phase = (
                ~_state_gripper_is_open(state_for_grasp[..., GRIPPER_INDEX_RAW], self.config.grounded_grasp_gripper_open_threshold)
            ).long()

        loss_local_pos = clean.new_zeros(())
        loss_local_rot = clean.new_zeros(())
        loss_local_grip = clean.new_zeros(())
        loss_local_terminal_xyz = clean.new_zeros(())
        loss_local_approach_xyz = clean.new_zeros(())
        loss_local_grip_weighted = clean.new_zeros(())
        terminal_grasp_distance_m = clean.new_zeros(())
        predicted_local_residual_magnitude_m = clean.new_zeros(())
        loss_local = clean.new_zeros(())
        loss_grasp_correction = clean.new_zeros(())
        local_eligible_frac = None
        grasp_correction_eligible_frac = None
        if self.architecture == "temporal_decoder_grounded_grasp_v2_5":
            # ---- v2's OWN pipeline, verbatim (a straight copy of the `has_target_head_v2` branch
            # above, deliberately NOT shared code with it -- see this architecture's own docstring
            # in `local_grasp_refiner.py`) ----
            target_language_query = self.target_query_extractor(latent_tokens, latent_pad_mask, latent_modality_ids)
            grounded_target_feature = self.visual_grounding_cross_attn(
                target_language_query, latent_tokens, latent_pad_mask, latent_modality_ids
            )
            target_xyz = self.target_point_head(grounded_target_feature)
            state_for_grasp, action_for_grasp = self._degrip_for_transition_detection(current_state_raw, clean_raw)
            gt_target_xyz, target_valid_mask = find_grasp_target(
                state_for_grasp,
                action_for_grasp,
                gripper_index=GRIPPER_INDEX_RAW,
                gripper_open_threshold=self.config.grounded_grasp_gripper_open_threshold,
                action_is_pad=action_is_pad_h,
            )
            loss_target = masked_mse(
                target_xyz.unsqueeze(1), gt_target_xyz.unsqueeze(1), target_valid_mask.unsqueeze(1)
            )
            grasp_phase = (
                ~_state_gripper_is_open(state_for_grasp[..., GRIPPER_INDEX_RAW], self.config.grounded_grasp_gripper_open_threshold)
            ).long()

            # ---- LocalGraspRefiner supervision (new) -- see `local_grasp_refiner.py` ----
            # `find_grasp_target`'s own transition-finding formula, duplicated (not imported) so
            # that function itself is never touched by this architecture.
            first_transition_idx = _first_grasp_transition_idx(
                state_for_grasp, action_for_grasp, GRIPPER_INDEX_RAW, self.config.grounded_grasp_gripper_open_threshold, action_is_pad_h
            )
            local_eligible = target_valid_mask & (first_transition_idx <= self.config.local_grasp_window_k)  # [B]
            local_eligible_frac = local_eligible.float().mean().item()

            image_feature = masked_mean_by_modality(latent_tokens, latent_pad_mask, latent_modality_ids, IMAGE_MODALITY)
            current_ee_xyz_m = current_state_raw[..., :3] * self.state_pos_std + self.state_pos_mean
            current_ee_rot_raw = current_state_raw[..., 3:6] * self.state_rot_std + self.state_rot_mean
            predicted_target_xyz_m = target_xyz * self.action_pos_std + self.action_pos_mean
            relative_xyz = predicted_target_xyz_m - current_ee_xyz_m

            # ---- LocalGraspCorrectionHead supervision (new, opt-in) -- see `grasp_correction_
            # head.py`. `target_xyz`/`relative_xyz` stay exactly what they were above (v2's own,
            # unmodified) unless this flag is on; `refiner_target_xyz`/`refiner_relative_xyz` are
            # what actually get passed to `local_grasp_refiner` just below -- the ONLY thing this
            # flag ever changes about that call.
            refiner_target_xyz, refiner_relative_xyz = target_xyz, relative_xyz
            if self.config.local_grasp_use_correction_head:
                delta_grasp_xyz_m = self.local_grasp_correction_head(
                    grounded_target_feature, image_feature, current_state, target_xyz, relative_xyz
                )
                gt_target_xyz_m = gt_target_xyz * self.action_pos_std + self.action_pos_mean
                gt_delta_grasp_xyz_m = gt_target_xyz_m - predicted_target_xyz_m
                # Excludes frames whose OWN frozen-TargetPointHead error already looks
                # B_WRONG_OBJECT-scale (see `local_grasp_correction_outlier_clip_m`'s docstring on
                # why this is a partial, not complete, mitigation).
                not_outlier = gt_delta_grasp_xyz_m.norm(dim=-1) <= self.config.local_grasp_correction_outlier_clip_m
                grasp_correction_valid = local_eligible & not_outlier
                grasp_correction_eligible_frac = grasp_correction_valid.float().mean().item()
                loss_grasp_correction = masked_mse(
                    delta_grasp_xyz_m.unsqueeze(1), gt_delta_grasp_xyz_m.unsqueeze(1), grasp_correction_valid.unsqueeze(1)
                )
                corrected_target_xyz_m = predicted_target_xyz_m + delta_grasp_xyz_m
                refiner_target_xyz = (corrected_target_xyz_m - self.action_pos_mean) / self.action_pos_std
                refiner_relative_xyz = corrected_target_xyz_m - current_ee_xyz_m

            pred_delta_xyz, pred_delta_rot, pred_gripper = self.local_grasp_refiner(
                grounded_target_feature, image_feature, current_state, refiner_target_xyz, refiner_relative_xyz
            )  # each [B, local_grasp_action_horizon, *]

            local_h = self.config.local_grasp_action_horizon
            local_window_raw = pad_or_crop_horizon(clean_raw, local_h)  # [B, H, 7], MEAN_STD-normalized
            local_window_encoded = pad_or_crop_horizon(clean, local_h)  # [B, H, 10]
            local_pad_mask_h = (
                pad_or_crop_mask(action_is_pad_h, local_h)
                if action_is_pad_h is not None
                else clean.new_zeros((clean.shape[0], local_h), dtype=torch.bool)
            )
            local_valid_mask_h = local_eligible.unsqueeze(1) & ~local_pad_mask_h  # [B, H]

            gt_action_xyz_m = local_window_raw[..., :3] * self.action_pos_std + self.action_pos_mean
            gt_delta_xyz = gt_action_xyz_m - current_ee_xyz_m.unsqueeze(1)

            gt_action_rot_raw = local_window_raw[..., 3:6] * self.action_rot_std + self.action_rot_mean
            gt_delta_rot_raw = gt_action_rot_raw - current_ee_rot_raw.unsqueeze(1)
            # Wrap into (-pi, pi]: a DELTA (unlike an absolute angle) can always be safely wrapped
            # to its shortest-path representative without losing information -- this is what keeps
            # rotation supervision "sin/cos-convention-compatible" (avoids exactly the ~2*pi
            # spurious-error artifact `rotation_encoding.py`'s own docstring documents for absolute
            # angles) without inventing a separate sin/cos delta-composition algebra.
            gt_delta_rot = torch.remainder(gt_delta_rot_raw + torch.pi, 2 * torch.pi) - torch.pi

            gt_gripper = local_window_encoded[..., 9:10]  # already the correctly-normalized/encoded gripper channel

            loss_local_pos = masked_mse(pred_delta_xyz, gt_delta_xyz, local_valid_mask_h)
            loss_local_rot = masked_mse(pred_delta_rot, gt_delta_rot, local_valid_mask_h)
            loss_local_grip = masked_mse(pred_gripper, gt_gripper, local_valid_mask_h)

            # ---- grasp-phase-aware LocalGraspRefiner loss (new, opt-in) -- loss-ONLY ablation, no
            # new module, `LocalGraspRefiner` itself untouched (see this flag's own config
            # docstring). Adds 3 terms on top of the EXISTING `loss_local_pos`/`loss_local_rot`/
            # `loss_local_grip` above (all three still always computed, unchanged, for metrics/
            # comparability) -- `loss_local_rot` is NEVER touched by this block, per instructions.
            if self.config.local_grasp_use_phase_aware_loss:
                delta = self.config.local_grasp_huber_delta
                batch_idx = torch.arange(clean.shape[0], device=clean.device)
                # Per-sample index, WITHIN the local_h-wide predicted horizon, of "the grasp/close
                # step": the real detected transition (`first_transition_idx`, raw steps from NOW,
                # already computed above for `local_eligible` against `local_grasp_window_k=10` --
                # a DIFFERENT, usually-wider window than `local_h`, default 5) when it actually
                # falls inside this call's short horizon, else clamped to the LAST horizon step
                # (the best available proxy when the real event is farther out than this horizon
                # can see -- still `local_eligible`, since that only requires <= window_k, not <=
                # local_h). This one index drives all 3 terms below, instead of three separate ad
                # hoc conventions.
                close_idx_in_horizon = first_transition_idx.clamp(0, local_h - 1)  # [B]

                predicted_xyz_all = current_ee_xyz_m.unsqueeze(1) + pred_delta_xyz  # [B, local_h, 3], meters
                # `p_grasp_gt`: the SAME first-open->close-transition commanded xyz `loss_target`
                # already regresses `target_xyz` against (`gt_target_xyz`, see `find_grasp_target`'s
                # docstring) -- explicitly NOT object-center, per instructions -- just converted to
                # meters here to match `predicted_xyz_all`'s own space.
                p_grasp_gt_m = gt_target_xyz * self.action_pos_std + self.action_pos_mean  # [B, 3]

                # 1. Terminal grasp position loss: only the SINGLE predicted step at each sample's
                # own `close_idx_in_horizon` is supervised against `p_grasp_gt_m`.
                predicted_terminal_xyz = predicted_xyz_all[batch_idx, close_idx_in_horizon]  # [B, 3]
                terminal_err = F.huber_loss(predicted_terminal_xyz, p_grasp_gt_m, reduction="none", delta=delta).mean(-1)  # [B]
                terminal_mask = local_eligible.to(terminal_err.dtype)
                loss_local_terminal_xyz = (terminal_err * terminal_mask).sum() / terminal_mask.sum().clamp_min(1.0)

                # 2. Grasp-window approach loss: EVERY predicted horizon step (not just the terminal
                # one) regressed toward the SAME `p_grasp_gt_m`, weighted by a fixed linear-in-time
                # schedule (`local_grasp_approach_weight_min` -> `_max`, first pilot: 0.1 -> 1.0,
                # NOT swept) -- a schedule over horizon POSITION, same for every sample, deliberately
                # simpler than trying to weight by each sample's own (differing) distance to close.
                approach_weights = torch.linspace(
                    self.config.local_grasp_approach_weight_min, self.config.local_grasp_approach_weight_max,
                    local_h, device=clean.device, dtype=clean.dtype,
                )  # [local_h]
                p_grasp_gt_m_h = p_grasp_gt_m.unsqueeze(1).expand(-1, local_h, -1)  # [B, local_h, 3]
                approach_err = F.huber_loss(predicted_xyz_all, p_grasp_gt_m_h, reduction="none", delta=delta).mean(-1)  # [B, local_h]
                approach_mask = local_valid_mask_h.to(approach_err.dtype)  # same pad/eligibility mask as loss_local_pos
                approach_weight_mask = approach_weights.unsqueeze(0) * approach_mask
                loss_local_approach_xyz = (approach_err * approach_weight_mask).sum() / approach_weight_mask.sum().clamp_min(1e-8)

                # 3. Gripper timing weighting: same `gt_gripper` target as `loss_local_grip` above,
                # just weighted `local_grasp_grip_close_weight` (default 3.0) within +/-1 horizon
                # step of `close_idx_in_horizon`, else 1.0 -- samples/steps excluded by
                # `local_valid_mask_h` (ineligible or padded) contribute 0 either way, so this
                # weighting only ever redistributes emphasis among already-valid steps.
                step_idx = torch.arange(local_h, device=clean.device).unsqueeze(0)  # [1, local_h]
                near_close = (step_idx - close_idx_in_horizon.unsqueeze(1)).abs() <= 1  # [B, local_h]
                grip_weight = torch.where(
                    near_close,
                    clean.new_full((), self.config.local_grasp_grip_close_weight).expand_as(near_close),
                    clean.new_ones(()).expand_as(near_close),
                )
                grip_err = F.mse_loss(pred_gripper, gt_gripper, reduction="none").squeeze(-1)  # [B, local_h]
                grip_weight_mask = grip_weight * local_valid_mask_h.to(grip_weight.dtype)
                loss_local_grip_weighted = (grip_err * grip_weight_mask).sum() / grip_weight_mask.sum().clamp_min(1e-8)

                # ---- pure observability (never affects loss/training) -- raw-meters counterparts
                # of the Huber terms above, for offline checkpoint-to-checkpoint comparison (a
                # Huber loss value alone isn't directly interpretable as "how many cm off").
                terminal_dist_err = (predicted_terminal_xyz - p_grasp_gt_m).norm(dim=-1)  # [B], meters
                terminal_grasp_distance_m = (terminal_dist_err * terminal_mask).sum() / terminal_mask.sum().clamp_min(1.0)
                residual_norms = pred_delta_xyz.norm(dim=-1)  # [B, local_h], meters
                residual_mask = local_valid_mask_h.to(residual_norms.dtype)
                predicted_local_residual_magnitude_m = (residual_norms * residual_mask).sum() / residual_mask.sum().clamp_min(1.0)

            local_action_dim = 3 + 3 + 1
            if self.config.local_grasp_use_phase_aware_loss:
                # Gripper is pulled out into its own explicitly-coefficented term below (item 4's
                # `2.0 * L_grip_weighted`) instead of the small implicit "1/7" share it had in the
                # old blend -- pos/rot keep their EXISTING 3:3 relative weighting, renormalized over
                # 6 (not 7) now that gripper is no longer part of this particular sum.
                loss_local_action = (3 * loss_local_pos + 3 * loss_local_rot) / 6
                loss_local = (
                    loss_local_action
                    + self.config.lambda_local_terminal_xyz * loss_local_terminal_xyz
                    + self.config.lambda_local_approach_xyz * loss_local_approach_xyz
                    + self.config.lambda_local_grip_weighted * loss_local_grip_weighted
                )
            else:
                loss_local = (3 * loss_local_pos + 3 * loss_local_rot + 1 * loss_local_grip) / local_action_dim

        # `target_xyz` is always predicted and trained (above) whenever `temporal_decoder_
        # grounded_grasp`/`temporal_decoder_grounded_grasp_v2`/`_v3`/`_v2_5` is used, but for the v1
        # architecture only reaches the decoder as conditioning when
        # `grounded_grasp_condition_decoder_on_target` is True (see that field's docstring) -- the
        # ablation-B path (auxiliary loss only, no conditioning) computes `target_xyz` for the loss
        # but the decoder itself never sees it. v2/v3/v2_5 have no such ablation toggle -- target
        # conditioning is always on for them.
        decoder_target_xyz = target_xyz
        if self.architecture == "temporal_decoder_grounded_grasp" and not self.config.grounded_grasp_condition_decoder_on_target:
            decoder_target_xyz = None
        instruction_embedding = None
        if self.architecture == "temporal_decoder_instruction":
            instruction_embedding = masked_mean_by_modality(latent_tokens, latent_pad_mask, latent_modality_ids, TEXT_MODALITY)
        pred_actions = self.decoder(
            latent_tokens,
            latent_pad_mask,
            current_state,
            subgoal_state,
            target_xyz=decoder_target_xyz,
            instruction_embedding=instruction_embedding,
            latent_modality_ids=latent_modality_ids,
            grounded_target_feature=grounded_target_feature,
            grasp_phase=grasp_phase,
        )

        # xyz MSE / rotation sin-cos MSE (6-D now) / gripper MSE, in the encoded 10-D layout --
        # see `rotation_encoding.py`.
        loss_pos = masked_mse(pred_actions[..., :3], clean[..., :3], valid_mask)
        loss_rot = masked_mse(pred_actions[..., 3:9], clean[..., 3:9], valid_mask)
        loss_grip = masked_mse(pred_actions[..., 9:10], clean[..., 9:10], valid_mask)
        if self.config.lambda_smooth > 0:
            velocity = pred_actions[:, 1:] - pred_actions[:, :-1]
            acceleration = velocity[:, 1:] - velocity[:, :-1]
            loss_smooth = acceleration.square().mean()
        else:
            loss_smooth = pred_actions.new_zeros(())

        # `F.mse_loss` reduces to the mean *within* each slice, so `loss_pos`/`loss_rot`/
        # `loss_grip` alone aren't comparable to each other or to a single pooled
        # `F.mse_loss(pred_actions, clean)` over all 10 encoded dims -- that pooled mean is itself
        # the dim-count-weighted average `(3*loss_pos + 6*loss_rot + 1*loss_grip) / 10` (position
        # 3-wide, sin/cos rotation 6-wide, gripper 1-wide). Pre-multiplying by each slice's dim
        # count and dividing by `action_dim` (10) here reproduces that exactly, so
        # lambda_pos=lambda_rot=lambda_grip=1.0 (the default) is numerically identical to the old
        # single pooled MSE -- see `test_decomposed_loss_equals_old_pooled_mse_at_default_weights`.
        position_dim, rotation_dim, gripper_dim = 3, 6, 1
        action_dim = position_dim + rotation_dim + gripper_dim
        loss_action = (
            self.config.lambda_pos * position_dim * loss_pos
            + self.config.lambda_rot * rotation_dim * loss_rot
            + self.config.lambda_grip * gripper_dim * loss_grip
        ) / action_dim
        # v3's `loss_target` is already the weighted `lambda_target_coarse * loss_target_coarse +
        # lambda_target_refine * loss_target_final` combination (see above) -- applying
        # `lambda_target` to it again here would double-weight it (and silently do so only when
        # `lambda_target` is changed from its default of 1.0, since 1.0 is a no-op multiplier), so
        # v3 adds its already-weighted term directly instead of going through `lambda_target`.
        target_loss_term = (
            loss_target if self.architecture == "temporal_decoder_grounded_grasp_v3" else self.config.lambda_target * loss_target
        )
        loss = (
            loss_action
            + self.config.lambda_subgoal * loss_subgoal
            + self.config.lambda_smooth * loss_smooth
            + target_loss_term
            + self.config.lambda_local_grasp * loss_local
            + self.config.lambda_grasp_correction * loss_grasp_correction
        )
        metrics = {
            "loss": loss.item(),
            "loss_pos": loss_pos.item(),
            "loss_rot": loss_rot.item(),
            "loss_grip": loss_grip.item(),
            "loss_subgoal": loss_subgoal.item(),
            "loss_smooth": loss_smooth.item(),
            "loss_local": loss_local.item(),
            "loss_local_pos": loss_local_pos.item(),
            "loss_local_rot": loss_local_rot.item(),
            "loss_local_grip": loss_local_grip.item(),
            "loss_local_terminal_xyz": loss_local_terminal_xyz.item(),
            "loss_local_approach_xyz": loss_local_approach_xyz.item(),
            "loss_local_grip_weighted": loss_local_grip_weighted.item(),
            "terminal_grasp_distance_m": terminal_grasp_distance_m.item(),
            "predicted_local_residual_magnitude_m": predicted_local_residual_magnitude_m.item(),
            "loss_target": loss_target.item(),
            "loss_target_coarse": loss_target_coarse.item(),
            "loss_target_final": loss_target_final.item(),
            "loss_grasp_correction": loss_grasp_correction.item(),
            "action_mean": clean_raw.mean().item(),
            "action_std": clean_raw.std().item(),
        }
        if residual_xyz is not None:
            metrics["residual_xyz_magnitude_mean"] = residual_xyz.norm(dim=-1).mean().item()
        if visual_selection_stats is not None:
            metrics["visual_selection_entropy_mean"] = visual_selection_stats["entropy"].mean().item()
            metrics["visual_selection_top1_mass_mean"] = visual_selection_stats["top1_mass"].mean().item()
            metrics["visual_selection_topk_mass_mean"] = visual_selection_stats["topk_mass"].mean().item()
        if local_eligible_frac is not None:
            metrics["local_eligible_frac"] = local_eligible_frac
        if grasp_correction_eligible_frac is not None:
            metrics["grasp_correction_eligible_frac"] = grasp_correction_eligible_frac
        return loss, metrics

    def _plan_temporal_decoder(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        started = perf_counter()
        latent_tokens, latent_pad_mask, latent_modality_ids = self._encode_multimodal_latent(batch)
        current_state = self._encode_state(self._current_state(batch))
        metrics: dict[str, Tensor | float] = {}
        subgoal_state = None
        if self.architecture == "temporal_decoder_subgoal":
            subgoal_state = self._predict_subgoal(latent_tokens, latent_pad_mask, latent_modality_ids, current_state)
            # NOTE: encoded (10-D) space, while `execution.ActionExecutor`'s completion gate
            # compares this against the *raw* 7-D `current_state` it's given -- a pre-existing
            # dimension mismatch for this (unused by us; not touched per the "don't fix subgoal
            # architecture" scope of this change) gated path only.
            metrics["predicted_subgoal_state"] = subgoal_state
        target_xyz = None
        if self.architecture == "temporal_decoder_grounded_grasp":
            target_xyz = self._predict_target_xyz(latent_tokens, latent_pad_mask, latent_modality_ids)
            metrics["predicted_target_xyz"] = target_xyz
            # Cached (normalized ACTION-position space) for `_grounded_grasp_reactive_close`,
            # called every env step from `select_action` -- not just on the (much rarer) steps
            # this method itself runs, i.e. chunk replans.
            self._grounded_grasp_last_target_xyz = target_xyz.detach()
        grounded_target_feature = None
        grasp_phase = None
        if self.architecture == "temporal_decoder_grounded_grasp_v2":
            target_language_query = self.target_query_extractor(latent_tokens, latent_pad_mask, latent_modality_ids)
            grounded_target_feature = self.visual_grounding_cross_attn(
                target_language_query, latent_tokens, latent_pad_mask, latent_modality_ids
            )
            target_xyz = self.target_point_head(grounded_target_feature)
            metrics["predicted_target_xyz"] = target_xyz
            # Cached (normalized ACTION-position space) for `_grounded_grasp_v2_reactive_close`,
            # called every env step from `select_action` -- not just on the (much rarer) steps
            # this method itself runs, i.e. chunk replans.
            self._v2_last_target_xyz = target_xyz.detach()
            # The policy's own persistent per-episode phase (flipped by
            # `_grounded_grasp_v2_reactive_close`, cleared by `reset()`), broadcast to every
            # sample in the batch -- inference calls this with batch size 1 in practice, but this
            # stays correct for any batch size.
            grasp_phase = latent_tokens.new_full((latent_tokens.shape[0],), self._v2_grasp_phase, dtype=torch.long)
        if self.architecture == "temporal_decoder_grounded_grasp_v3":
            target_language_query = self.target_query_extractor(latent_tokens, latent_pad_mask, latent_modality_ids)
            grounded_target_feature, visual_selection_stats = self.sharpened_visual_grounding(
                target_language_query, latent_tokens, latent_pad_mask, latent_modality_ids
            )
            coarse_xyz = self.coarse_target_head(grounded_target_feature)
            residual_xyz = self.fine_residual_head(grounded_target_feature, coarse_xyz, current_state, target_language_query)
            target_xyz = coarse_xyz + residual_xyz  # final_target_xyz
            metrics["predicted_target_xyz"] = target_xyz
            metrics["predicted_coarse_target_xyz"] = coarse_xyz
            metrics["predicted_residual_xyz"] = residual_xyz
            metrics["visual_selection_entropy"] = visual_selection_stats["entropy"]
            metrics["visual_selection_top1_mass"] = visual_selection_stats["top1_mass"]
            metrics["visual_selection_topk_mass"] = visual_selection_stats["topk_mass"]
            # Cached (normalized ACTION-position space) for `_grounded_grasp_v3_reactive_close`,
            # called every env step from `select_action` -- not just on the (much rarer) steps this
            # method itself runs, i.e. chunk replans. This is the FINAL (coarse + residual) target,
            # exactly like `_v2_last_target_xyz` caches v2's single predicted target.
            self._v3_last_target_xyz = target_xyz.detach()
            # The policy's own persistent per-episode phase (flipped by
            # `_grounded_grasp_v3_reactive_close`, cleared by `reset()`), broadcast to every sample
            # in the batch -- same convention as v2's `_v2_grasp_phase` above.
            grasp_phase = latent_tokens.new_full((latent_tokens.shape[0],), self._v3_grasp_phase, dtype=torch.long)
        if self.architecture == "temporal_decoder_grounded_grasp_v2_5":
            # v2's OWN pipeline, verbatim -- this method only ever runs v2_5's PRE_GRASP/POST_GRASP
            # GLOBAL stage (LOCAL_GRASP bypasses it entirely via `select_action`'s own dispatch, see
            # `_plan_local_grasp_step`/`_select_action_local_grasp` below).
            target_language_query = self.target_query_extractor(latent_tokens, latent_pad_mask, latent_modality_ids)
            grounded_target_feature = self.visual_grounding_cross_attn(
                target_language_query, latent_tokens, latent_pad_mask, latent_modality_ids
            )
            target_xyz = self.target_point_head(grounded_target_feature)
            metrics["predicted_target_xyz"] = target_xyz
            # Cached (normalized ACTION-position space) for `_maybe_enter_local_grasp`'s own
            # PRE_GRASP->LOCAL_GRASP distance check, called every env step from `select_action` --
            # not just on the (much rarer) steps this method itself runs, i.e. chunk replans.
            self._v25_last_target_xyz = target_xyz.detach()
            # PRE_GRASP(0)/POST_GRASP(1) -- the only two values `TemporalActionDecoder`'s existing
            # binary `grasp_phase` embedding understands. `self._v25_stage` is never LOCAL_GRASP
            # when this method runs (see above), but this still degrades sanely (phase=1) if it
            # somehow were.
            grasp_phase = latent_tokens.new_full(
                (latent_tokens.shape[0],), 0 if self._v25_stage == PRE_GRASP else 1, dtype=torch.long
            )
        decoder_target_xyz = target_xyz
        if self.architecture == "temporal_decoder_grounded_grasp" and not self.config.grounded_grasp_condition_decoder_on_target:
            decoder_target_xyz = None
        instruction_embedding = None
        if self.architecture == "temporal_decoder_instruction":
            instruction_embedding = masked_mean_by_modality(latent_tokens, latent_pad_mask, latent_modality_ids, TEXT_MODALITY)
        actions_encoded = self.decoder(
            latent_tokens,
            latent_pad_mask,
            current_state,
            subgoal_state,
            target_xyz=decoder_target_xyz,
            instruction_embedding=instruction_embedding,
            latent_modality_ids=latent_modality_ids,
            grounded_target_feature=grounded_target_feature,
            grasp_phase=grasp_phase,
        )
        # Unit-normalize each (sin, cos) pair and `atan2` back to raw Euler, right at this
        # policy's own output boundary -- everything downstream (`execution.ActionExecutor`, the
        # postprocessor, the VLABench env) keeps receiving the original 7-D layout unchanged.
        actions = self._decode_action_prediction(actions_encoded)
        metrics["runtime_ms"] = (perf_counter() - started) * 1000
        if actions.shape[1] > 2:
            velocity = actions[:, 1:] - actions[:, :-1]
            acceleration = velocity[:, 1:] - velocity[:, :-1]
            metrics["mean_abs_delta_action"] = velocity.abs().mean().item()
            metrics["mean_abs_delta2_action"] = acceleration.abs().mean().item()
        return actions, metrics

    # ---- dispatch -----------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, float]]:
        if reduction != "mean":
            raise NotImplementedError("SafeDiff-VLA currently supports reduction='mean' only")
        if self.architecture in (
            "temporal_decoder",
            "temporal_decoder_subgoal",
            "temporal_decoder_grounded_grasp",
            "temporal_decoder_instruction",
            "temporal_decoder_text_crossattn",
            "temporal_decoder_grounded_grasp_v2",
            "temporal_decoder_grounded_grasp_v3",
            "temporal_decoder_grounded_grasp_v2_5",
        ):
            return self._forward_temporal_decoder(batch)
        if self.architecture == "smolvla_finetune":
            return self._forward_smolvla_finetune(batch)
        raise NotImplementedError(
            f"architecture={self.architecture!r} has no training objective — it's an eval-only "
            "ablation baseline (SmolVLA's own nominal action chunk, unmodified)."
        )

    def plan_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        if self.architecture in ("smolvla_nominal", "smolvla_finetune"):
            actions = self._nominal_actions(batch)
            metrics: dict[str, Tensor | float] = {}
            if actions.shape[1] > 2:
                velocity = actions[:, 1:] - actions[:, :-1]
                acceleration = velocity[:, 1:] - velocity[:, :-1]
                metrics["mean_abs_delta_action"] = velocity.abs().mean().item()
                metrics["mean_abs_delta2_action"] = acceleration.abs().mean().item()
            return actions, metrics
        return self._plan_temporal_decoder(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        actions, metrics = self.plan_action_chunk(batch)
        self.last_inference_metrics = metrics if self.config.enable_inference_metrics else {}
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        current_state = self._current_state(batch)
        if self.architecture == "temporal_decoder_grounded_grasp_v2_5" and self._v25_stage == LOCAL_GRASP:
            # Genuinely closed-loop: bypasses `self._executor`/`plan_action_chunk` entirely (no
            # chunk, no queue) -- re-observes and replans from this fresh `batch` every single call,
            # unlike every other stage/architecture here (see `local_grasp_refiner.py`'s docstring).
            return self._select_action_local_grasp(batch, current_state)
        action = self._executor.select_action(current_state, lambda: self.plan_action_chunk(batch))
        if self.architecture == "temporal_decoder_grounded_grasp":
            action = self._grounded_grasp_reactive_close(action, current_state)
        elif self.architecture == "temporal_decoder_grounded_grasp_v2":
            action = self._grounded_grasp_v2_reactive_close(action, current_state)
        elif self.architecture == "temporal_decoder_grounded_grasp_v3":
            action = self._grounded_grasp_v3_reactive_close(action, current_state)
        elif self.architecture == "temporal_decoder_grounded_grasp_v2_5":
            action = self._maybe_enter_local_grasp(action, current_state)
        return action

    def _grounded_grasp_reactive_close(self, action: Tensor, current_state: Tensor) -> Tensor:
        """`temporal_decoder_grounded_grasp` only: proximity-based reactive gripper close,
        replacing the decoder's own regressed gripper channel as the execution-time close
        trigger (see `SafeDiffVLAConfig.architecture`'s docstring for why). Uses only quantities a
        deployed policy actually has -- this step's own observed EE position
        (`current_state[:3]`) and the most recently predicted grasp target
        (`_grounded_grasp_last_target_xyz`, cached by `_plan_temporal_decoder`) -- no simulator
        privilege, unlike `examples/safediff_vla/grasp_*_oracle*.py`'s diagnostic oracles.

        One-shot latch (`_grounded_grasp_close_triggered`): once the threshold is crossed, stays
        closed for the rest of the episode; `reset()` clears it at episode boundaries. `action`'s
        rotation channels are never touched here.
        """
        if self._grounded_grasp_last_target_xyz is None:
            return action
        if not self._grounded_grasp_close_triggered:
            ee_pos_m = current_state[..., :3] * self.state_pos_std + self.state_pos_mean
            target_pos_m = self._grounded_grasp_last_target_xyz * self.action_pos_std + self.action_pos_mean
            distance_m = (ee_pos_m - target_pos_m).norm(dim=-1)
            if bool((distance_m <= self.config.grounded_grasp_close_threshold_m).any()):
                self._grounded_grasp_close_triggered = True
        if not self._grounded_grasp_close_triggered:
            return action
        closed_value = (0.0 - self.action_grip_mean) / self.action_grip_std
        action = action.clone()
        action[..., GRIPPER_INDEX_RAW] = closed_value.to(action.dtype)
        return action

    def _grounded_grasp_v2_reactive_close(self, action: Tensor, current_state: Tensor) -> Tensor:
        """`temporal_decoder_grounded_grasp_v2` only: proximity-based reactive gripper close, PLUS
        an explicit PRE_GRASP -> POST_GRASP phase transition (unlike `_grounded_grasp_reactive_close`,
        which never changes what conditioning the decoder sees). Uses only quantities a deployed
        policy actually has -- this step's own observed EE position (`current_state[:3]`) and the
        most recently predicted grasp target (`_v2_last_target_xyz`, cached by
        `_plan_temporal_decoder`) -- no simulator privilege.

        EXPERIMENTAL, `select_poker` grasp/lift-validation only (see `SafeDiffVLAConfig.
        architecture`'s docstring): on proximity trigger, this closes the gripper, flips
        `_v2_grasp_phase` to POST_GRASP, and clears the executor's remaining action queue so the
        very next `select_action` call is forced to replan (`ActionExecutor.select_action`'s
        `if not self._action_queue:` branch) with `plan_action_chunk` now reading the NEW phase --
        instead of continuing to execute the rest of a chunk planned under stale PRE_GRASP
        conditioning. A place task would need a separate release phase this does not implement.

        One-shot latch (`_v2_close_triggered`): once the threshold is crossed, the gripper stays
        forced closed (POST_GRASP never reopens it) for the rest of the episode; `reset()` clears
        both the latch and the phase at episode boundaries. `action`'s rotation channels are never
        touched here.
        """
        if self._v2_last_target_xyz is None:
            return action
        if not self._v2_close_triggered:
            ee_pos_m = current_state[..., :3] * self.state_pos_std + self.state_pos_mean
            target_pos_m = self._v2_last_target_xyz * self.action_pos_std + self.action_pos_mean
            distance_m = (ee_pos_m - target_pos_m).norm(dim=-1)
            if bool((distance_m <= self.config.grounded_grasp_close_threshold_m).any()):
                self._v2_close_triggered = True
                self._v2_grasp_phase = 1
                self._executor._action_queue.clear()
        if not self._v2_close_triggered:
            return action
        closed_value = (0.0 - self.action_grip_mean) / self.action_grip_std
        action = action.clone()
        action[..., GRIPPER_INDEX_RAW] = closed_value.to(action.dtype)
        return action

    def _grounded_grasp_v3_reactive_close(self, action: Tensor, current_state: Tensor) -> Tensor:
        """`temporal_decoder_grounded_grasp_v3` only: a straight copy of
        `_grounded_grasp_v2_reactive_close`'s proximity-trigger/phase-flip/one-shot-latch behavior
        -- same threshold (`grounded_grasp_close_threshold_m`), same one-shot latch, same
        queue-clear-forces-replan mechanism -- evaluated against `_v3_last_target_xyz` (v3's
        FINAL, coarse-plus-residual target, cached by `_plan_temporal_decoder`) instead of v2's
        single predicted target. Deliberately UNCHANGED threshold/phase/close logic for this
        pilot: v3's only experimental variable is the grounding/localization head upstream of this
        method, never this method itself (see `grounded_grasp_v3.py`'s module docstring).
        """
        if self._v3_last_target_xyz is None:
            return action
        if not self._v3_close_triggered:
            ee_pos_m = current_state[..., :3] * self.state_pos_std + self.state_pos_mean
            target_pos_m = self._v3_last_target_xyz * self.action_pos_std + self.action_pos_mean
            distance_m = (ee_pos_m - target_pos_m).norm(dim=-1)
            if bool((distance_m <= self.config.grounded_grasp_close_threshold_m).any()):
                self._v3_close_triggered = True
                self._v3_grasp_phase = 1
                self._executor._action_queue.clear()
        if not self._v3_close_triggered:
            return action
        closed_value = (0.0 - self.action_grip_mean) / self.action_grip_std
        action = action.clone()
        action[..., GRIPPER_INDEX_RAW] = closed_value.to(action.dtype)
        return action

    def _maybe_enter_local_grasp(self, action: Tensor, current_state: Tensor) -> Tensor:
        """`temporal_decoder_grounded_grasp_v2_5`'s PRE_GRASP stage only: proximity-based mode
        switch -- unlike v1/v2/v3's reactive close, this does NOT force the gripper; it only flips
        `self._v25_stage` from PRE_GRASP to LOCAL_GRASP (one-shot per episode, gated on
        `self._v25_stage == PRE_GRASP` so it can never re-fire once past it -- `reset()` is the only
        thing that puts it back to PRE_GRASP) and clears the executor's action queue so the very
        next `select_action` call routes to `_select_action_local_grasp` instead of continuing to
        pop a chunk planned under stale PRE_GRASP conditioning. `action` itself (already computed
        by v2's own global decoder for THIS step) is returned unmodified -- the switch takes effect
        starting the NEXT call, exactly like v1/v2/v3's own reactive-close pattern.
        """
        if self._v25_stage != PRE_GRASP or self._v25_last_target_xyz is None:
            return action
        ee_pos_m = current_state[..., :3] * self.state_pos_std + self.state_pos_mean
        target_pos_m = self._v25_last_target_xyz * self.action_pos_std + self.action_pos_mean
        distance_m = (ee_pos_m - target_pos_m).norm(dim=-1)
        if bool((distance_m <= self.config.local_grasp_radius_m).any()):
            self._v25_stage = LOCAL_GRASP
            self._executor._action_queue.clear()
        return action

    def _plan_local_grasp_step(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """`temporal_decoder_grounded_grasp_v2_5`'s LOCAL_GRASP stage only: one closed-loop step.
        Re-runs v2's OWN grounding pipeline fresh (frozen weights by default, but still a real
        forward pass against THIS call's own `batch` -- this IS the "re-observe" the architecture
        docstring describes, never a cached value), then `LocalGraspRefiner` for a bounded
        residual, executing only its FIRST predicted step (this pilot fixes
        `local_grasp_execute_horizon=1`) as a single RAW MEAN_STD-normalized 7-D action -- the same
        representation `_decode_action_prediction` already produces for the global stage, so it can
        be returned directly from `select_action` with no further conversion.
        """
        latent_tokens, latent_pad_mask, latent_modality_ids = self._encode_multimodal_latent(batch)
        current_state_raw = self._current_state(batch)
        current_state = self._encode_state(current_state_raw)

        target_language_query = self.target_query_extractor(latent_tokens, latent_pad_mask, latent_modality_ids)
        grounded_target_feature = self.visual_grounding_cross_attn(
            target_language_query, latent_tokens, latent_pad_mask, latent_modality_ids
        )
        predicted_target_xyz = self.target_point_head(grounded_target_feature)
        image_feature = masked_mean_by_modality(latent_tokens, latent_pad_mask, latent_modality_ids, IMAGE_MODALITY)

        current_ee_xyz_m = current_state_raw[..., :3] * self.state_pos_std + self.state_pos_mean
        current_ee_rot_raw = current_state_raw[..., 3:6] * self.state_rot_std + self.state_rot_mean
        predicted_target_xyz_m = predicted_target_xyz * self.action_pos_std + self.action_pos_mean
        relative_xyz = predicted_target_xyz_m - current_ee_xyz_m

        refiner_target_xyz, refiner_relative_xyz = predicted_target_xyz, relative_xyz
        delta_grasp_xyz_m = None
        if self.config.local_grasp_use_correction_head:
            delta_grasp_xyz_m = self.local_grasp_correction_head(
                grounded_target_feature, image_feature, current_state, predicted_target_xyz, relative_xyz
            )
            corrected_target_xyz_m = predicted_target_xyz_m + delta_grasp_xyz_m
            refiner_target_xyz = (corrected_target_xyz_m - self.action_pos_mean) / self.action_pos_std
            refiner_relative_xyz = corrected_target_xyz_m - current_ee_xyz_m

        delta_xyz, delta_rot, gripper = self.local_grasp_refiner(
            grounded_target_feature, image_feature, current_state, refiner_target_xyz, refiner_relative_xyz
        )
        # This pilot always replans every step (`local_grasp_execute_horizon=1`), so only the first
        # predicted residual step is ever used -- `local_grasp_execute_horizon>1` would need its own
        # small action queue (like `ActionExecutor`'s, just replanned far more often); not built
        # here, since sweeping this value is explicitly out of scope for this pilot.
        delta_xyz0 = delta_xyz[:, 0]
        delta_rot0 = delta_rot[:, 0]
        gripper0 = gripper[:, 0]

        local_target_xyz_m = current_ee_xyz_m + delta_xyz0
        local_target_xyz_norm = (local_target_xyz_m - self.action_pos_mean) / self.action_pos_std
        local_target_rot_raw = current_ee_rot_raw + delta_rot0
        local_target_rot_norm = (local_target_rot_raw - self.action_rot_mean) / self.action_rot_std

        action = torch.cat((local_target_xyz_norm, local_target_rot_norm, gripper0), dim=-1)
        metrics = {
            "predicted_target_xyz": predicted_target_xyz,
            "relative_xyz_m": relative_xyz,
            "delta_xyz_m": delta_xyz0,
            "delta_rot_rad": delta_rot0,
            "gripper_norm": gripper0,
        }
        if delta_grasp_xyz_m is not None:
            # Logged so eval/diagnostic scripts can plot the corrected anchor against v2's raw
            # `predicted_target_xyz` and the GT object center (see `grasp_correction_head.py`).
            metrics["delta_grasp_xyz_m"] = delta_grasp_xyz_m
            metrics["corrected_target_xyz_m"] = predicted_target_xyz_m + delta_grasp_xyz_m
        return action, metrics

    def _select_action_local_grasp(self, batch: dict[str, Tensor], current_state: Tensor) -> Tensor:
        """`temporal_decoder_grounded_grasp_v2_5`'s LOCAL_GRASP stage: one closed-loop step via
        `_plan_local_grasp_step`, then the gripper one-shot latch (prevents open/close chatter --
        see `SafeDiffVLAConfig.architecture`'s docstring) and the LOCAL_GRASP -> POST_GRASP
        transition, both driven by the SAME event (the refiner's own predicted gripper first
        reading "closed") -- no simulator privilege, matching every other reactive mechanism here.
        """
        action, metrics = self._plan_local_grasp_step(batch)
        gripper_raw_phys = metrics["gripper_norm"] * self.action_grip_std + self.action_grip_mean
        is_closed_now = bool(
            (~_action_gripper_is_open(gripper_raw_phys, self.config.grounded_grasp_gripper_open_threshold)).any()
        )
        just_closed = is_closed_now and not self._v25_gripper_closed_latch
        if just_closed:
            self._v25_gripper_closed_latch = True
        if self._v25_gripper_closed_latch:
            closed_value = (0.0 - self.action_grip_mean) / self.action_grip_std
            action = action.clone()
            action[..., GRIPPER_INDEX_RAW] = closed_value.to(action.dtype)
        if just_closed:
            # Grasp confirmed (this policy's own closed-loop signal, no simulator privilege) ->
            # exit LOCAL_GRASP. The executor's own action queue was already cleared/never touched
            # during LOCAL_GRASP (see `_maybe_enter_local_grasp`), so the very next `select_action`
            # call naturally forces a fresh `plan_action_chunk` replan under phase=POST_GRASP
            # conditioning.
            self._v25_stage = POST_GRASP
        return action
