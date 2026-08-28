"""SafeDiff-VLA: external diffusion refinement for SmolVLA action chunks.

The real-backbone latent is the mean final action-token hidden state captured at
SmolVLA's ``action_out_proj`` input during nominal action sampling. SmolVLA is
used unchanged; the temporary hook is owned and removed by this wrapper.
"""

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
from .losses import optional_binary_loss
from .scheduler import DDPMScheduler
from .utils import first_available_label, pad_or_crop_horizon


class SafeDiffVLAPolicy(PreTrainedPolicy):
    config_class = SafeDiffVLAConfig
    name = "safediff_vla"

    def __init__(self, config: SafeDiffVLAConfig, backbone: nn.Module | None = None, **_: Any) -> None:
        super().__init__(config)
        config.validate_features()
        self.backbone = backbone if backbone is not None else self._make_backbone()
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
        if hasattr(self.backbone, "reset"):
            self.backbone.reset()

    def _backbone_outputs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return normalized nominal [B,H,A] and pooled action-token latent [B,D]."""
        if hasattr(self.backbone, "extract_safediff_features"):
            with torch.set_grad_enabled(not self.config.freeze_backbone):
                nominal, latent = self.backbone.extract_safediff_features(batch)
        else:
            hidden_states: list[Tensor] = []

            def capture_action_hidden(_module, inputs) -> None:
                hidden_states.append(inputs[0])

            handle = self.backbone.model.action_out_proj.register_forward_pre_hook(capture_action_hidden)
            try:
                nominal = self.backbone.predict_action_chunk(dict(batch))
            finally:
                handle.remove()
            if not hidden_states:
                raise RuntimeError("SmolVLA action-token hook did not capture a hidden state")
            latent = hidden_states[-1].mean(dim=1)

        nominal = pad_or_crop_horizon(nominal, self.config.action_horizon)
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

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)
            self._action_queue.extend(chunk.transpose(0, 1)[: self.config.execute_horizon])
        return self._action_queue.popleft()
