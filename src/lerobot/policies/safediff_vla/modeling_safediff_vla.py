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
from lerobot.utils.constants import ACTION

from .configuration_safediff_vla import SafeDiffVLAConfig
from .critics import TrajectoryCritic, score_candidates
from .diffusion_planner import ConditionalDiffusionPlanner
from .domain_adapter import LiberoBackboneDomainAdapter, load_processor_normalization_stats
from .losses import optional_binary_loss
from .scheduler import DDPMScheduler
from .utils import first_available_label, pad_or_crop_horizon


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
        self.latent_projection = nn.Linear(self._latent_in_features(), config.latent_dim)
        self.planner = ConditionalDiffusionPlanner(
            action_dim, config.latent_dim, config.planner_hidden_dim, config.timestep_embedding_dim
        )
        self.task_critic = TrajectoryCritic(action_dim, config.latent_dim, config.task_critic_hidden_dim)
        self.risk_critic = TrajectoryCritic(action_dim, config.latent_dim, config.risk_critic_hidden_dim)
        self.scheduler = DDPMScheduler(config.num_diffusion_steps, config.beta_schedule)
        self.reset()
        self._set_training_mode()

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

    def _set_training_mode(self) -> None:
        self.planner.requires_grad_(self.config.training_mode in ("diffusion", "joint"))
        train_critics = self.config.training_mode in ("critics", "joint")
        self.task_critic.requires_grad_(train_critics)
        self.risk_critic.requires_grad_(train_critics)

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
        if hasattr(self.backbone, "reset"):
            self.backbone.reset()

    def _backbone_outputs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return target-normalized nominal [B,H,A] and pooled backbone latent [B,D]."""
        backbone_batch = self.domain_adapter.observation_for_backbone(batch) if self.domain_adapter else batch
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

    def diffusion_loss(self, clean: Tensor, latent: Tensor, nominal: Tensor) -> Tensor:
        timesteps = torch.randint(self.config.num_diffusion_steps, (clean.shape[0],), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        return F.mse_loss(self.planner(noisy, timesteps, latent, nominal), noise)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, float]]:
        if reduction != "mean":
            raise NotImplementedError("SafeDiff-VLA currently supports reduction='mean' only")
        nominal, latent = self._backbone_outputs(batch)
        clean = pad_or_crop_horizon(batch[ACTION], self.config.action_horizon)
        zero = latent.sum() * 0
        loss_diff = (
            self.diffusion_loss(clean, latent, nominal) if self.config.training_mode != "critics" else zero
        )
        task_labels = first_available_label(batch, ("task_success",))
        risk_labels = first_available_label(batch, ("safety_violation", "collision", "semantic_violation"))
        task_logits = self.task_critic(latent, clean)
        risk_logits = self.risk_critic(latent, clean)
        if self.config.training_mode == "diffusion":
            loss_task = loss_risk = zero
        else:
            loss_task = optional_binary_loss(task_logits, task_labels)
            loss_risk = optional_binary_loss(risk_logits, risk_labels)
        loss = (
            self.config.lambda_diff * loss_diff
            + self.config.lambda_task * loss_task
            + self.config.lambda_risk * loss_risk
        )
        metrics = {
            "loss": loss.item(),
            "loss_diff": loss_diff.item(),
            "loss_task": loss_task.item(),
            "loss_risk": loss_risk.item(),
        }
        self._add_critic_metrics(metrics, "task", task_logits, task_labels, "positive", "negative")
        self._add_critic_metrics(metrics, "risk", risk_logits, risk_labels, "unsafe", "safe")
        return loss, metrics

    @staticmethod
    def _add_critic_metrics(
        metrics: dict[str, float],
        prefix: str,
        logits: Tensor,
        labels: Tensor | None,
        true_name: str,
        false_name: str,
    ) -> None:
        if labels is None:
            return
        labels = labels.bool().view_as(logits)
        probabilities = logits.detach().sigmoid()
        metrics[f"{prefix}_score_{true_name}"] = probabilities[labels].mean().item() if labels.any() else 0.0
        metrics[f"{prefix}_score_{false_name}"] = (
            probabilities[~labels].mean().item() if (~labels).any() else 0.0
        )

    def _apply_guidance(self, sample: Tensor, latent: Tensor, nominal: Tensor) -> Tensor:
        with torch.enable_grad():
            guided = sample.detach().requires_grad_(True)
            objective = (
                self.task_critic(latent.detach(), guided).sigmoid()
                - self.config.lambda_risk * self.risk_critic(latent.detach(), guided).sigmoid()
                - self.config.lambda_prior * (guided - nominal).square().mean(dim=(-1, -2))
            )
            gradient = torch.autograd.grad(objective.sum(), guided)[0]
            norm = gradient.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-6)
            clip_scale = (self.config.critic_gradient_clip / norm).clamp(max=1.0)
            gradient = gradient * clip_scale.view(-1, 1, 1)
        return sample + self.config.critic_guidance_scale * gradient.detach()

    def generate_candidates(self, latent: Tensor, nominal: Tensor) -> Tensor:
        batch_size, horizon, action_dim = nominal.shape
        count = self.config.num_candidates
        nominal_flat = nominal[:, None].expand(-1, count, -1, -1).reshape(-1, horizon, action_dim)
        latent_flat = latent[:, None].expand(-1, count, -1).reshape(-1, latent.shape[-1])
        noise = torch.randn_like(nominal_flat)
        if self.config.use_vla_prior_init:
            last = torch.full(
                (batch_size * count,), self.config.num_diffusion_steps - 1, device=nominal.device
            )
            sample = self.scheduler.add_noise(nominal_flat, noise, last)
        else:
            sample = noise
        for timestep in reversed(range(self.config.num_diffusion_steps)):
            timesteps = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
            predicted_noise = self.planner(sample, timesteps, latent_flat, nominal_flat)
            sample = self.scheduler.step(predicted_noise, timestep, sample)
            if self.config.use_critic_guidance:
                sample = self._apply_guidance(sample, latent_flat, nominal_flat)
        return sample.reshape(batch_size, count, horizon, action_dim)

    def plan_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        started = perf_counter()
        nominal, latent = self._backbone_outputs(batch)
        nominal_risk = self.risk_critic(latent, nominal).sigmoid()
        should_plan = self.config.use_diffusion_refinement
        if self.config.adaptive_planning:
            should_plan = should_plan and bool((nominal_risk >= self.config.risk_threshold).any())
        if not should_plan:
            return nominal, {"planner_usage_rate": 0.0, "mean_risk_score": nominal_risk.mean()}

        candidates = self.generate_candidates(latent, nominal)
        shape = (nominal.shape[0], self.config.num_candidates)
        task_logits = (
            self.task_critic(latent, candidates) if self.config.use_task_critic else nominal.new_zeros(shape)
        )
        risk_logits = (
            self.risk_critic(latent, candidates)
            if self.config.use_safety_critic
            else nominal.new_full(shape, -20)
        )
        scores, prior_distance = score_candidates(
            task_logits, risk_logits, candidates, nominal, self.config.lambda_risk, self.config.lambda_prior
        )
        selected_indices = scores.argmax(dim=1)
        batch_indices = torch.arange(candidates.shape[0], device=candidates.device)
        selected = candidates[batch_indices, selected_indices]
        return selected, {
            "planner_usage_rate": 1.0,
            "mean_task_score": task_logits.sigmoid().mean(),
            "mean_risk_score": risk_logits.sigmoid().mean(),
            "mean_prior_distance": prior_distance.mean(),
            "selected_candidate_index": selected_indices,
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
            chunk = self.predict_action_chunk(batch)
            self._action_queue.extend(chunk.transpose(0, 1)[: self.config.execute_horizon])
        return self._action_queue.popleft()
