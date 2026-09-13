"""SafeDiff-VLA: external diffusion refinement for SmolVLA action chunks.

The real-backbone latent is the mean final action-token hidden state captured at
SmolVLA's ``action_out_proj`` input during nominal action sampling. SmolVLA is
used unchanged; the temporary hook is owned and removed by this wrapper.
"""

import math
from collections import deque
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_safediff_vla import SafeDiffVLAConfig
from .diffusion_planner import ConditionalDiffusionPlanner
from .domain_adapter import LiberoBackboneDomainAdapter, load_processor_normalization_stats
from .scheduler import DDPMScheduler
from .state_predictor import StatePredictor, completion_gap, masked_mse_loss
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
        self.latent_projection = nn.Linear(self._latent_in_features(), config.latent_dim)
        self.planner = ConditionalDiffusionPlanner(
            action_dim, state_dim, config.latent_dim, config.planner_hidden_dim, config.timestep_embedding_dim
        )
        # Replaces the old task/risk critics (see `state_predictor.py` for why): predicts the
        # state expected `execute_horizon` steps after a candidate chunk, trained
        # self-supervised against the state the dataset actually observed there.
        self.state_predictor = StatePredictor(
            action_dim, state_dim, config.latent_dim, config.state_head_hidden_dim
        )
        self.scheduler = DDPMScheduler(config.num_diffusion_steps, config.beta_schedule)
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
        """Hidden size of the pooled action-token latent fed into ``latent_projection``.

        Matches whichever branch ``_backbone_outputs`` will take: the captured
        ``action_out_proj`` input (``expert_hidden_size``) for a plain SmolVLA backbone,
        or a custom backbone's own reported latent width. Resolved eagerly (instead of
        via ``nn.LazyLinear``) because ``lerobot-train`` counts ``policy.parameters()``
        before any forward pass, which raises on uninitialized lazy parameters.
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
        # State the state-predictor expected `execute_horizon` steps after the last *fully
        # committed* chunk (see `select_action`'s completion gate). None until the first chunk.
        self._pending_target_state: Tensor | None = None
        self._replan_retries = 0
        if hasattr(self.backbone, "reset"):
            self.backbone.reset()

    def _backbone_outputs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return target-normalized nominal [B,H,A] and pooled backbone latent [B,D]."""
        # The backbone must only ever see the *current* state: the extra future-state slice
        # `state_observation_delta_indices` adds to `batch[OBS_STATE]` (for the state-predictor's
        # training target, see `forward()`) is not a real observation SmolVLA should condition on.
        backbone_input = {**batch, OBS_STATE: self._current_state(batch)}
        backbone_batch = (
            self.domain_adapter.observation_for_backbone(backbone_input)
            if self.domain_adapter
            else backbone_input
        )
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

    @staticmethod
    def _current_state(batch: dict[str, Tensor]) -> Tensor:
        """The *current* (t=0) normalized state, whether or not `batch[OBS_STATE]` also carries
        the extra future-state slice `state_observation_delta_indices` adds for training."""
        state = batch[OBS_STATE]
        return state[:, 0] if state.ndim > 2 else state

    def diffusion_loss(self, clean: Tensor, latent: Tensor, nominal: Tensor, state: Tensor) -> Tensor:
        timesteps = torch.randint(self.config.num_diffusion_steps, (clean.shape[0],), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        return F.mse_loss(self.planner(noisy, timesteps, latent, nominal, state), noise)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, float]]:
        if reduction != "mean":
            raise NotImplementedError("SafeDiff-VLA currently supports reduction='mean' only")
        nominal, latent = self._backbone_outputs(batch)
        clean = pad_or_crop_horizon(batch[ACTION], self.config.action_horizon)
        state_seq = batch[OBS_STATE]
        current_state = state_seq[:, 0] if state_seq.ndim > 2 else state_seq
        loss_diff = self.diffusion_loss(clean, latent, nominal, current_state)
        predicted_future_state = self.state_predictor(latent, clean)
        if state_seq.ndim > 2 and state_seq.shape[1] > 1:
            future_state_target = state_seq[:, 1]
            is_pad = batch.get(f"{OBS_STATE}_is_pad")
            future_is_pad = is_pad[:, 1] if is_pad is not None and is_pad.ndim > 1 else None
            loss_state_pred = masked_mse_loss(predicted_future_state, future_state_target, future_is_pad)
        else:
            # Caller didn't wire up `state_observation_delta_indices` (no future-state slice in
            # this batch) — nothing to regress the state predictor against.
            loss_state_pred = predicted_future_state.sum() * 0
        loss = self.config.lambda_diff * loss_diff + self.config.lambda_state_pred * loss_state_pred
        metrics = {
            "loss": loss.item(),
            "loss_diff": loss_diff.item(),
            "loss_state_pred": loss_state_pred.item(),
        }
        return loss, metrics

    def _sample_action_chunk(self, latent: Tensor, nominal: Tensor, state: Tensor) -> Tensor:
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
            predicted_noise = self.planner(sample, timesteps, latent, nominal, state)
            sample = self.scheduler.step(predicted_noise, timestep, sample)
        return sample

    def plan_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        started = perf_counter()
        nominal, latent = self._backbone_outputs(batch)
        state = self._current_state(batch)
        selected = (
            self._sample_action_chunk(latent, nominal, state)
            if self.config.use_diffusion_refinement
            else nominal
        )
        predicted_future_state = self.state_predictor(latent, selected)
        return selected, {
            "predicted_future_state": predicted_future_state,
            "diffusion_runtime_ms": (perf_counter() - started) * 1000,
        }

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
            not_complete = False
            if self._pending_target_state is not None:
                gap = completion_gap(self._pending_target_state, self._current_state(batch))
                not_complete = bool((gap > self.config.completion_threshold).any())
            chunk, metrics = self.plan_action_chunk(batch)
            if not_complete and self._replan_retries < self.config.max_replan_retries:
                # The sub-goal the last committed chunk aimed for hasn't been reached: don't
                # commit to a fresh full-length chunk — take one step now, towards the *same*
                # still-pending target, and reassess on the very next call instead of silently
                # moving on to whatever the backbone proposes next.
                self._action_queue.extend(chunk.transpose(0, 1)[:1])
                self._replan_retries += 1
            else:
                self._pending_target_state = metrics["predicted_future_state"]
                self._action_queue.extend(chunk.transpose(0, 1)[: self.config.execute_horizon])
                self._replan_retries = 0
        return self._action_queue.popleft()
