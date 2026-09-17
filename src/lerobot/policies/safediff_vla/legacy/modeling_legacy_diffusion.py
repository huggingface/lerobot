"""Legacy nominal-refinement diffusion planner for SafeDiff-VLA, preserved only to reproduce past
experiments -- see `configuration_legacy_diffusion.py`'s module docstring, and
`..modeling_safediff_vla` for the current `temporal_decoder` main path.

## Data flow (`architecture` fixed to the diffusion planner, no branching)

    image / language / state
        -> SmolVLA.predict_action_chunk()  [@torch.no_grad(), inside SmolVLA itself]
        -> a forward-pre-hook on `action_out_proj` captures its *input*, mean-pooled over the
           chunk axis -> `latent` [B, latent_dim]   (this pooling throws away all per-timestep
           structure the hook's raw capture actually had — see `_backbone_outputs`)
        -> nominal = pad_or_crop_horizon(nominal, action_horizon)   (silently truncates/replans
           more often than the backbone was ever run at if action_horizon != the backbone's own
           `chunk_size`)
        -> ConditionalDiffusionPlanner(noisy_actions, latent, nominal, state, subgoal) denoises a
           chunk *initialized from* `nominal` (`use_vla_prior_init`) -> refined action chunk

i.e. the nominal action chunk is the primary signal end to end; the planner's job is framed as
"correct it". Every configuration of this tried (critic-free, state-conditioned,
subgoal-conditioned, temporal-conv-mixed across the horizon axis) matched or underperformed just
executing `nominal` unmodified once the horizon mismatch above was fixed — refining a nominal
action chunk post hoc never once helped.
"""

from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from ..domain_adapter import LiberoBackboneDomainAdapter, load_processor_normalization_stats
from ..execution import ActionExecutor
from ..utils import pad_or_crop_horizon
from .configuration_legacy_diffusion import LegacySafeDiffVLAConfig
from .diffusion_planner import ConditionalDiffusionPlanner
from .scheduler import DDPMScheduler
from .state_predictor import StatePredictor


class LegacySafeDiffVLAPolicy(PreTrainedPolicy):
    config_class = LegacySafeDiffVLAConfig
    name = "safediff_vla_legacy"

    def __init__(
        self,
        config: LegacySafeDiffVLAConfig,
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
            action_dim,
            state_dim,
            config.latent_dim,
            config.planner_hidden_dim,
            config.timestep_embedding_dim,
            temporal_kernel_size=config.temporal_kernel_size,
            num_temporal_layers=config.num_temporal_layers,
        )
        # Replaces the old task/risk critics: predicts the subgoal state (next pick/place event),
        # trained self-supervised against a label derived offline from each episode's own
        # gripper-transition frames (see `state_predictor.py`).
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
        vlm_with_expert = getattr(getattr(self.backbone, "model", None), "vlm_with_expert", None)
        if vlm_with_expert is None or not hasattr(vlm_with_expert, "get_vlm_model"):
            raise RuntimeError(
                "freeze_vision_encoder=False requires a SmolVLA-style backbone exposing "
                "model.vlm_with_expert.get_vlm_model()."
            )
        vlm_with_expert.freeze_vision_encoder = False
        vlm_with_expert.get_vlm_model().vision_model.requires_grad_(True)

    def _latent_in_features(self) -> int:
        """Hidden size of the pooled action-token latent `_backbone_outputs` feeds into
        `latent_projection`. Matches whichever branch `_backbone_outputs` will take: the captured
        `action_out_proj` input (`expert_hidden_size`) for a plain SmolVLA backbone, or a custom
        backbone's own reported latent width. Resolved eagerly (instead of via `nn.LazyLinear`)
        because `lerobot-train` counts `policy.parameters()` before any forward pass, which raises
        on uninitialized lazy parameters."""
        if hasattr(self.backbone, "extract_safediff_features"):
            latent_dim = getattr(self.backbone, "safediff_latent_dim", None)
            if latent_dim is None:
                raise ValueError(
                    "Backbone exposes extract_safediff_features() but not a safediff_latent_dim "
                    "attribute; LegacySafeDiffVLAPolicy needs it to size latent_projection up front."
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
        self._executor = ActionExecutor(self.config)
        if hasattr(self.backbone, "reset"):
            self.backbone.reset()

    # ---- shared helpers -----------------------------------------------------------------

    def _prepare_backbone_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        backbone_input = {**batch, OBS_STATE: self._current_state(batch)}
        return (
            self.domain_adapter.observation_for_backbone(backbone_input)
            if self.domain_adapter
            else backbone_input
        )

    @staticmethod
    def _current_state(batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        return state[:, 0] if state.ndim > 2 else state

    def _backbone_outputs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return (nominal, pooled_latent) via the hook-based capture."""
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

    # ---- training / planning -------------------------------------------------------------

    def diffusion_loss(
        self, clean: Tensor, latent: Tensor, nominal: Tensor, state: Tensor, subgoal: Tensor
    ) -> Tensor:
        timesteps = torch.randint(self.config.num_diffusion_steps, (clean.shape[0],), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        return F.mse_loss(self.planner(noisy, timesteps, latent, nominal, state, subgoal), noise)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, float]]:
        if reduction != "mean":
            raise NotImplementedError("SafeDiff-VLA currently supports reduction='mean' only")
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

    def plan_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
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
        metrics: dict[str, Tensor | float] = {
            "predicted_subgoal_state": predicted_subgoal,
            "runtime_ms": (perf_counter() - started) * 1000,
        }
        if selected.shape[1] > 2:
            velocity = selected[:, 1:] - selected[:, :-1]
            acceleration = velocity[:, 1:] - velocity[:, :-1]
            metrics["mean_abs_delta_action"] = velocity.abs().mean().item()
            metrics["mean_abs_delta2_action"] = acceleration.abs().mean().item()
        return selected, metrics

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
