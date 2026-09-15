"""SafeDiff-VLA: pluggable action-generation architectures on a frozen (by default) SmolVLA
multimodal encoder.

## Old data flow, traced (`architecture="legacy_diffusion"`)

    image / language / state
        -> SmolVLA.predict_action_chunk()  [@torch.no_grad(), inside SmolVLA itself]
        -> a forward-pre-hook on `action_out_proj` captures its *input*, mean-pooled over the
           chunk axis -> `latent` [B, latent_dim]   (this pooling throws away all per-timestep
           structure the hook's raw capture actually had — see `_backbone_outputs`)
        -> nominal = pad_or_crop_horizon(nominal, action_horizon)   (silently truncates/replans
           more often than the backbone was ever run at if action_horizon != the backbone's own
           `chunk_size`, e.g. cropping 50 -> 16 at the old default)
        -> ConditionalDiffusionPlanner(noisy_actions, latent, nominal, state, subgoal) denoises a
           chunk *initialized from* `nominal` (`use_vla_prior_init`) -> refined action chunk

    i.e. the nominal action chunk is the primary signal end to end; the planner's job is framed
    as "correct it". Every configuration of this tried (critic-free, state-conditioned,
    subgoal-conditioned, temporal-conv-mixed across the horizon axis) matched or underperformed
    just executing `nominal` unmodified once the horizon mismatch above was fixed — refining a
    nominal action chunk post hoc never once helped. `legacy_diffusion` is kept only to reproduce
    those experiments; it is no longer the default.

## New data flow (`architecture="temporal_decoder"` / `"temporal_decoder_future_state"`, default)

    image / language / state
        -> SmolVLA.embed_prefix() + SmolVLMWithExpertModel.forward(inputs_embeds=[prefix, None])
           -- a *direct* call neither `predict_action_chunk` nor `forward` on the backbone expose
           (both discard exactly this tensor: `sample_actions` assigns it to `_`) -- see
           `_encode_multimodal_latent`. Read-only wrt `modeling_smolvla.py`; no changes there.
        -> latent_tokens [B, N_tokens, hidden_size]: the VLM's own post-self-attention
           representation of image+language+state, never mean-pooled, never touching the
           flow-matching action-generation loop at all.
        -> TemporalActionDecoder(latent_tokens, current_state, [predicted_states])
               learned per-position action-query tokens cross-attend to `latent_tokens` and
               self-attend across the horizon axis (see `temporal_decoder.py`)
        -> actions [B, action_horizon, action_dim] directly -- `nominal` (SmolVLA's own action
           head output) is never consulted; it exists elsewhere only as the `smolvla_nominal`
           ablation baseline.
"""

import math
from collections import deque
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_safediff_vla import SafeDiffVLAConfig
from .diffusion_planner import ConditionalDiffusionPlanner
from .domain_adapter import LiberoBackboneDomainAdapter, load_processor_normalization_stats
from .scheduler import DDPMScheduler
from .state_predictor import FutureStatePredictor, StatePredictor, completion_gap
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

        action_dim = config.action_feature.shape[0]
        state_dim = config.robot_state_feature.shape[0]

        if self.architecture == "legacy_diffusion":
            self.latent_projection = nn.Linear(self._latent_in_features(), config.latent_dim)
            self.planner = ConditionalDiffusionPlanner(
                action_dim,
                state_dim,
                config.latent_dim,
                config.planner_hidden_dim,
                config.timestep_embedding_dim,
                temporal_kernel_size=config.temporal_kernel_size,
                num_temporal_layers=config.num_temporal_layers,
            )
            # Replaces the old task/risk critics (see `state_predictor.py` for why): predicts the
            # subgoal state (next pick/place event), trained self-supervised against a label
            # derived offline from each episode's own gripper-transition frames.
            self.state_predictor = StatePredictor(
                action_dim, state_dim, config.latent_dim, config.state_head_hidden_dim
            )
            self.scheduler = DDPMScheduler(config.num_diffusion_steps, config.beta_schedule)
        elif self.architecture in ("temporal_decoder", "temporal_decoder_future_state"):
            use_future_state = self.architecture == "temporal_decoder_future_state"
            self.decoder = TemporalActionDecoder(
                action_dim,
                state_dim,
                self._multimodal_latent_dim(),
                config.decoder_hidden_dim,
                config.action_horizon,
                num_layers=config.decoder_num_layers,
                num_heads=config.decoder_num_heads,
                ffn_dim=config.decoder_ffn_dim,
                dropout=config.decoder_dropout,
                use_future_state=use_future_state,
            )
            if use_future_state:
                self.latent_pool_projection = nn.Linear(self._multimodal_latent_dim(), config.latent_dim)
                self.future_state_predictor = FutureStatePredictor(
                    state_dim, config.latent_dim, config.state_head_hidden_dim
                )
        elif self.architecture != "smolvla_nominal":
            raise ValueError(f"Unknown architecture {self.architecture!r}")
        # "smolvla_nominal": nothing to build -- ablation baseline, no trainable parameters.

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

    def _latent_in_features(self) -> int:
        """Hidden size of the pooled action-token latent `_backbone_outputs` (legacy_diffusion
        only) feeds into `latent_projection`.

        Matches whichever branch `_backbone_outputs` will take: the captured `action_out_proj`
        input (`expert_hidden_size`) for a plain SmolVLA backbone, or a custom backbone's own
        reported latent width. Resolved eagerly (instead of via `nn.LazyLinear`) because
        `lerobot-train` counts `policy.parameters()` before any forward pass, which raises on
        uninitialized lazy parameters.
        """
        if hasattr(self.backbone, "extract_safediff_features"):
            latent_dim = getattr(self.backbone, "safediff_latent_dim", None)
            if latent_dim is None:
                raise ValueError(
                    "Backbone exposes extract_safediff_features() but not a safediff_latent_dim "
                    "attribute; SafeDiffVLAPolicy needs it to size latent_projection up front."
                )
            return latent_dim
        return self.backbone.model.action_out_proj.in_features

    def _multimodal_latent_dim(self) -> int:
        """Token width of `_encode_multimodal_latent`'s output, to size the decoder's own input
        projection up front (see `_latent_in_features` for why this must be eager, not lazy)."""
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
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.execute_horizon)
        # Holds up to `action_horizon` past chunk predictions, oldest first, for temporal
        # ensembling: the k-th most recently appended chunk was queried k steps ago, so its
        # prediction for "now" lives at its own index k (`_ensembled_action` below).
        self._ensemble_buffer: deque[Tensor] = deque(maxlen=self.config.action_horizon)
        # Subgoal state predicted as of the last *fully committed* chunk, and the gap to it
        # measured at that same moment (see `select_action`'s completion gate). Both stay None
        # forever for architectures with no subgoal signal (`smolvla_nominal`, `temporal_decoder`).
        self._pending_target_state: Tensor | None = None
        self._last_gap: Tensor | None = None
        self._replan_retries = 0
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
        """SmolVLA's own action-head output, unmodified. Used by `architecture="smolvla_nominal"`
        and, for `legacy_diffusion`, as the `use_vla_prior_init` anchor."""
        backbone_batch = self._prepare_backbone_batch(batch)
        with torch.set_grad_enabled(not self.config.freeze_backbone):
            nominal = self.backbone.predict_action_chunk(dict(backbone_batch))
        nominal = pad_or_crop_horizon(nominal, self.config.action_horizon)
        if self.domain_adapter is not None:
            nominal = self.domain_adapter.nominal_for_target(nominal)
        return nominal.detach() if self.config.freeze_backbone else nominal

    def _backbone_outputs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return (nominal, pooled_latent) via the hook-based capture. `legacy_diffusion` only —
        see this module's docstring for why the *pooled* latent this produces isn't used by the
        newer architectures (`_encode_multimodal_latent` below)."""
        backbone_batch = self._prepare_backbone_batch(batch)
        if hasattr(self.backbone, "extract_safediff_features"):
            with torch.set_grad_enabled(not self.config.freeze_backbone):
                nominal, latent = self.backbone.extract_safediff_features(backbone_batch)
        else:
            hidden_states: list[Tensor] = []

            def capture_action_hidden(_module, inputs) -> None:
                hidden_states.append(inputs[0])

            handle = self.backbone.model.action_out_proj.register_forward_pre_hook(capture_action_hidden)
            try:
                nominal = self.backbone.predict_action_chunk(dict(backbone_batch))
            finally:
                handle.remove()
            if not hidden_states:
                raise RuntimeError("SmolVLA action-token hook did not capture a hidden state")
            latent = hidden_states[-1].mean(dim=1)

        nominal = pad_or_crop_horizon(nominal, self.config.action_horizon)
        if self.domain_adapter is not None:
            nominal = self.domain_adapter.nominal_for_target(nominal)
        return nominal.detach() if self.config.freeze_backbone else nominal, self.latent_projection(
            latent.float()
        )

    def _encode_multimodal_latent(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None]:
        """Direct multimodal-encoder path: SmolVLA's own `embed_prefix` plus one self-attention
        pass through its VLM transformer, *without* running the flow-matching action-generation
        loop at all. Neither `predict_action_chunk` nor `forward` on the backbone expose this —
        both discard exactly this tensor (`sample_actions` assigns it to `_`) — so this
        reimplements the first few lines of `sample_actions`. Entirely read-only wrt
        `modeling_smolvla.py` (no changes there, no new interface added to it): a test double can
        instead implement `encode_multimodal_latent(batch) -> (tokens, pad_mask_or_None)` and
        `multimodal_latent_dim` directly (mirrors the existing `extract_safediff_features` /
        `safediff_latent_dim` test-double convention for `legacy_diffusion`).

        If `freeze_backbone=False` this is also how gradients would actually reach the VLM,
        unlike the `legacy_diffusion` hook-based path which necessarily goes through
        `predict_action_chunk`'s `@torch.no_grad()`.
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

    # ---- legacy_diffusion -----------------------------------------------------------------

    def diffusion_loss(
        self, clean: Tensor, latent: Tensor, nominal: Tensor, state: Tensor, subgoal: Tensor
    ) -> Tensor:
        timesteps = torch.randint(self.config.num_diffusion_steps, (clean.shape[0],), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        return F.mse_loss(self.planner(noisy, timesteps, latent, nominal, state, subgoal), noise)

    def _forward_legacy_diffusion(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        nominal, latent = self._backbone_outputs(batch)
        clean = pad_or_crop_horizon(batch[ACTION], self.config.action_horizon)
        current_state = self._current_state(batch)
        # Always predicted from `nominal` (never the ground-truth `clean` chunk) so this matches
        # exactly what `plan_action_chunk` does at inference — see `state_predictor.py`.
        predicted_subgoal = self.state_predictor(latent, nominal, current_state)
        loss_diff = self.diffusion_loss(clean, latent, nominal, current_state, predicted_subgoal.detach())
        subgoal_target = batch.get("observation.subgoal_state")
        if subgoal_target is not None:
            loss_subgoal = F.mse_loss(predicted_subgoal, subgoal_target)
        else:
            # No precomputed label in this batch (`policy.subgoal_labels_path` unset) — nothing
            # to regress the state predictor against.
            loss_subgoal = predicted_subgoal.sum() * 0
        loss = self.config.lambda_diff * loss_diff + self.config.lambda_subgoal * loss_subgoal
        metrics = {
            "loss": loss.item(),
            "loss_diff": loss_diff.item(),
            "loss_subgoal": loss_subgoal.item(),
        }
        return loss, metrics

    def _sample_action_chunk(self, latent: Tensor, nominal: Tensor, state: Tensor, subgoal: Tensor) -> Tensor:
        """Single DDPM reverse pass producing one action-chunk sample, optionally anchored near
        `nominal` (`use_vla_prior_init`) instead of starting from pure noise."""
        noise = torch.randn_like(nominal)
        if self.config.use_vla_prior_init:
            last = torch.full((nominal.shape[0],), self.config.num_diffusion_steps - 1, device=nominal.device)
            sample = self.scheduler.add_noise(nominal, noise, last)
        else:
            sample = noise
        for timestep in reversed(range(self.config.num_diffusion_steps)):
            timesteps = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
            predicted_noise = self.planner(sample, timesteps, latent, nominal, state, subgoal)
            sample = self.scheduler.step(predicted_noise, timestep, sample)
        return sample

    def _plan_legacy_diffusion(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        started = perf_counter()
        nominal, latent = self._backbone_outputs(batch)
        state = self._current_state(batch)
        # Predicted once up front from (latent, nominal, state) alone, *before* diffusion
        # sampling: the subgoal (next pick/place point) doesn't depend on which candidate action
        # chunk gets sampled, only on the current situation. Detached so the diffusion loss can't
        # push the state predictor towards subgoals that merely make denoising easier.
        predicted_subgoal = self.state_predictor(latent, nominal, state).detach()
        selected = (
            self._sample_action_chunk(latent, nominal, state, predicted_subgoal)
            if self.config.use_diffusion_refinement
            else nominal
        )
        return selected, {
            "predicted_subgoal_state": predicted_subgoal,
            "runtime_ms": (perf_counter() - started) * 1000,
        }

    # ---- temporal_decoder / temporal_decoder_future_state ----------------------------------

    def _forward_temporal_decoder(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        latent_tokens, latent_pad_mask = self._encode_multimodal_latent(batch)
        current_state = self._current_state(batch)
        clean = pad_or_crop_horizon(batch[ACTION], self.config.action_horizon)

        predicted_states = None
        loss_subgoal = clean.new_zeros(())
        if self.architecture == "temporal_decoder_future_state":
            pooled = self.latent_pool_projection(self._pooled_latent(latent_tokens, latent_pad_mask))
            predicted_subgoal = self.future_state_predictor(pooled, current_state)
            predicted_states = predicted_subgoal[:, None, :].expand(-1, self.config.action_horizon, -1)
            subgoal_target = batch.get("observation.subgoal_state")
            if subgoal_target is not None:
                loss_subgoal = F.mse_loss(predicted_subgoal, subgoal_target)

        pred_actions = self.decoder(latent_tokens, latent_pad_mask, current_state, predicted_states)
        loss_action = F.mse_loss(pred_actions, clean)
        if self.config.lambda_smooth > 0:
            velocity = pred_actions[:, 1:] - pred_actions[:, :-1]
            acceleration = velocity[:, 1:] - velocity[:, :-1]
            loss_smooth = acceleration.square().mean()
        else:
            loss_smooth = pred_actions.new_zeros(())

        loss = (
            self.config.lambda_action * loss_action
            + self.config.lambda_subgoal * loss_subgoal
            + self.config.lambda_smooth * loss_smooth
        )
        metrics = {
            "loss": loss.item(),
            "loss_action": loss_action.item(),
            "loss_subgoal": loss_subgoal.item(),
            "loss_smooth": loss_smooth.item(),
        }
        return loss, metrics

    def _plan_temporal_decoder(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        started = perf_counter()
        latent_tokens, latent_pad_mask = self._encode_multimodal_latent(batch)
        current_state = self._current_state(batch)
        metrics: dict[str, Tensor | float] = {}
        predicted_states = None
        if self.architecture == "temporal_decoder_future_state":
            pooled = self.latent_pool_projection(self._pooled_latent(latent_tokens, latent_pad_mask))
            predicted_subgoal = self.future_state_predictor(pooled, current_state)
            predicted_states = predicted_subgoal[:, None, :].expand(-1, self.config.action_horizon, -1)
            metrics["predicted_subgoal_state"] = predicted_subgoal
        actions = self.decoder(latent_tokens, latent_pad_mask, current_state, predicted_states)
        metrics["runtime_ms"] = (perf_counter() - started) * 1000
        return actions, metrics

    # ---- dispatch -----------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, float]]:
        if reduction != "mean":
            raise NotImplementedError("SafeDiff-VLA currently supports reduction='mean' only")
        if self.architecture == "legacy_diffusion":
            return self._forward_legacy_diffusion(batch)
        if self.architecture in ("temporal_decoder", "temporal_decoder_future_state"):
            return self._forward_temporal_decoder(batch)
        raise NotImplementedError(
            f"architecture={self.architecture!r} has no training objective — it's an eval-only "
            "ablation baseline (SmolVLA's own nominal action chunk, unmodified)."
        )

    def plan_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        if self.architecture == "smolvla_nominal":
            return self._nominal_actions(batch), {}
        if self.architecture == "legacy_diffusion":
            return self._plan_legacy_diffusion(batch)
        return self._plan_temporal_decoder(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        actions, metrics = self.plan_action_chunk(batch)
        self.last_inference_metrics = metrics if self.config.enable_inference_metrics else {}
        return actions

    def _ensembled_action(self, chunk: Tensor) -> Tensor:
        """Blend "now"-predictions from every buffered chunk with exponential-decay weights.

        `chunk` (this step's fresh prediction) is pushed last, so iterating the buffer newest
        -> oldest via `reversed()` lines up positional age with the offset each chunk holds its
        prediction for "now" at: age 0 is `chunk` itself (offset 0), age 1 is last step's chunk
        (offset 1, since it was queried one step ago), and so on.
        """
        self._ensemble_buffer.append(chunk)
        predictions, weights = [], []
        for age, past_chunk in enumerate(reversed(self._ensemble_buffer)):
            predictions.append(past_chunk[:, age])
            weights.append(math.exp(-self.config.temporal_ensemble_coeff * age))
        weights = torch.tensor(weights, device=chunk.device, dtype=chunk.dtype)
        weights /= weights.sum()
        return (torch.stack(predictions, dim=0) * weights[:, None, None]).sum(dim=0)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.use_temporal_ensembling:
            return self._ensembled_action(self.predict_action_chunk(batch))
        if not self._action_queue:
            # A subgoal can legitimately be many chunks away (~36 steps on average across
            # `lerobot/vlabench_unified` — see `examples/safediff_vla/compute_subgoal_labels.py`),
            # so "not yet arrived after one execute_horizon" is the normal case, not a problem:
            # gating on that (as an earlier version of this method did) made the gate fire on
            # almost every commit, collapsing execution into a near-permanent single-step replan
            # loop and producing visibly jerky motion. What actually signals trouble is the gap
            # *growing* since the last check: the last chunk moved away from the target it was
            # aiming for, while still being meaningfully far from it.
            gap = (
                completion_gap(self._pending_target_state, self._current_state(batch))
                if self._pending_target_state is not None
                else None
            )
            diverging = (
                gap is not None
                and self._last_gap is not None
                and bool((gap > self._last_gap).any())
                and bool((gap > self.config.completion_threshold).any())
            )
            chunk, metrics = self.plan_action_chunk(batch)
            # Architectures with no subgoal signal (`smolvla_nominal`, `temporal_decoder`) never
            # gate: there's nothing to measure progress against, so always commit a fresh chunk.
            has_subgoal = self.config.use_completion_gate and "predicted_subgoal_state" in metrics
            if has_subgoal and diverging and self._replan_retries < self.config.max_replan_retries:
                # Take one corrective step towards the *same* still-pending target and reassess
                # on the very next call, instead of silently moving on to whatever the backbone
                # proposes next.
                self._action_queue.extend(chunk.transpose(0, 1)[:1])
                self._replan_retries += 1
                self._last_gap = gap
            else:
                if has_subgoal:
                    self._pending_target_state = metrics["predicted_subgoal_state"]
                    self._last_gap = completion_gap(self._pending_target_state, self._current_state(batch))
                self._action_queue.extend(chunk.transpose(0, 1)[: self.config.execute_horizon])
                self._replan_retries = 0
        return self._action_queue.popleft()
