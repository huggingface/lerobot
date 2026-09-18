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
from .rotation_encoding import ENCODED_DIM
from .rotation_encoding import decode as decode_rotation
from .rotation_encoding import encode as encode_rotation
from .state_predictor import SubgoalStatePredictor
from .temporal_decoder import TemporalActionDecoder
from .utils import pad_or_crop_horizon


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

        if self.architecture in ("temporal_decoder", "temporal_decoder_subgoal"):
            use_subgoal = self.architecture == "temporal_decoder_subgoal"
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
            )
            if use_subgoal:
                self.latent_pool_projection = nn.Linear(self._multimodal_latent_dim(), config.latent_dim)
                self.subgoal_state_predictor = SubgoalStatePredictor(
                    ENCODED_DIM, config.latent_dim, config.state_head_hidden_dim
                )
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

    @staticmethod
    def _pooled_latent(latent_tokens: Tensor, latent_pad_mask: Tensor | None) -> Tensor:
        if latent_pad_mask is None:
            return latent_tokens.mean(dim=1)
        mask = latent_pad_mask.unsqueeze(-1).to(latent_tokens.dtype)
        return (latent_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

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

    def _encode_multimodal_latent(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None]:
        """Direct multimodal-encoder path: SmolVLA's own `embed_prefix` plus one self-attention
        pass through its VLM transformer, *without* running the flow-matching action-generation
        loop at all. Neither `predict_action_chunk` nor `forward` on the backbone expose this —
        both discard exactly this tensor (`sample_actions` assigns it to `_`) — so this
        reimplements the first few lines of `sample_actions`. Entirely read-only wrt
        `modeling_smolvla.py` (no changes there, no new interface added to it): a test double can
        instead implement `encode_multimodal_latent(batch) -> (tokens, pad_mask_or_None)` and
        `multimodal_latent_dim` directly.

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
        if self.config.freeze_backbone:
            latent_tokens = latent_tokens.detach()
        return latent_tokens, prefix_pad_masks

    # ---- temporal_decoder / temporal_decoder_subgoal ---------------------------------------

    def _predict_subgoal(self, latent_tokens: Tensor, latent_pad_mask: Tensor | None, current_state: Tensor) -> Tensor:
        """`temporal_decoder_subgoal` only: a single predicted subgoal state `[B, state_dim]` from
        the pooled scene latent + current state (see `state_predictor.py`'s `SubgoalStatePredictor`)."""
        pooled = self.latent_pool_projection(self._pooled_latent(latent_tokens, latent_pad_mask))
        return self.subgoal_state_predictor(pooled, current_state)

    def _forward_temporal_decoder(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        latent_tokens, latent_pad_mask = self._encode_multimodal_latent(batch)
        current_state = self._encode_state(self._current_state(batch))
        clean_raw = pad_or_crop_horizon(batch[ACTION], self.config.action_horizon)
        clean = self._encode_action_target(clean_raw)

        subgoal_state = None
        loss_subgoal = clean.new_zeros(())
        if self.architecture == "temporal_decoder_subgoal":
            predicted_subgoal = self._predict_subgoal(latent_tokens, latent_pad_mask, current_state)
            subgoal_state = predicted_subgoal
            subgoal_target = batch.get("observation.subgoal_state")
            if subgoal_target is not None:
                loss_subgoal = F.mse_loss(predicted_subgoal, self._encode_state(subgoal_target))

        pred_actions = self.decoder(latent_tokens, latent_pad_mask, current_state, subgoal_state)

        # xyz MSE / rotation sin-cos MSE (6-D now) / gripper MSE, in the encoded 10-D layout --
        # see `rotation_encoding.py`.
        loss_pos = F.mse_loss(pred_actions[..., :3], clean[..., :3])
        loss_rot = F.mse_loss(pred_actions[..., 3:9], clean[..., 3:9])
        loss_grip = F.mse_loss(pred_actions[..., 9:10], clean[..., 9:10])
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
        loss = (
            loss_action
            + self.config.lambda_subgoal * loss_subgoal
            + self.config.lambda_smooth * loss_smooth
        )
        metrics = {
            "loss": loss.item(),
            "loss_pos": loss_pos.item(),
            "loss_rot": loss_rot.item(),
            "loss_grip": loss_grip.item(),
            "loss_subgoal": loss_subgoal.item(),
            "loss_smooth": loss_smooth.item(),
            "action_mean": clean_raw.mean().item(),
            "action_std": clean_raw.std().item(),
        }
        return loss, metrics

    def _plan_temporal_decoder(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        started = perf_counter()
        latent_tokens, latent_pad_mask = self._encode_multimodal_latent(batch)
        current_state = self._encode_state(self._current_state(batch))
        metrics: dict[str, Tensor | float] = {}
        subgoal_state = None
        if self.architecture == "temporal_decoder_subgoal":
            subgoal_state = self._predict_subgoal(latent_tokens, latent_pad_mask, current_state)
            # NOTE: encoded (10-D) space, while `execution.ActionExecutor`'s completion gate
            # compares this against the *raw* 7-D `current_state` it's given -- a pre-existing
            # dimension mismatch for this (unused by us; not touched per the "don't fix subgoal
            # architecture" scope of this change) gated path only.
            metrics["predicted_subgoal_state"] = subgoal_state
        actions_encoded = self.decoder(latent_tokens, latent_pad_mask, current_state, subgoal_state)
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
        if self.architecture in ("temporal_decoder", "temporal_decoder_subgoal"):
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
        return self._executor.select_action(self._current_state(batch), lambda: self.plan_action_chunk(batch))
