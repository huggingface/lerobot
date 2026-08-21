# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LeRobot policy wrapper and data adapters for LaWAM."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import (
    _diffusers_available,
    _transformers_available,
    require_package,
)

from .configuration_lawam import LaWAMConfig

_lawam_deps_available = _transformers_available and _diffusers_available

if TYPE_CHECKING or _lawam_deps_available:
    from lerobot.policies.lawam.latent_world.runtime.freeze_policy import (
        apply_policy_freeze,
        parse_policy_freeze_config,
    )
    from lerobot.policies.lawam.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
    from lerobot.policies.lawam.vlas.lawam import LatentWorldPolicyBackend, LatentWorldPolicyConfig
else:
    ConditionalFlowMatchingConfig = None
    LatentWorldPolicyBackend = None
    LatentWorldPolicyConfig = None
    apply_policy_freeze = None
    parse_policy_freeze_config = None


def _require_lawam_packages() -> None:
    """Require the optional packages used by the LaWAM implementation."""
    require_package("transformers", extra="lawam")
    require_package("diffusers", extra="lawam")


def _build_freeze_config(config: LaWAMConfig):
    """Translate LeRobot freeze settings into the native LaWAM contract."""
    return parse_policy_freeze_config(
        {
            "freeze_vision_backbone": config.freeze_vision_backbone,
            "freeze_llm_backbone": config.freeze_llm_backbone,
            "freeze_embedding": config.freeze_embedding,
            "unfreeze_vision_merger": config.unfreeze_vision_merger,
            "unfreeze_lam_decoder": config.unfreeze_lam_decoder,
            "keep_llm_first_n_layers": config.keep_llm_first_n_layers,
            "unfreeze_llm_last_n_layers": config.unfreeze_llm_last_n_layers,
        }
    )


def _build_lam_config(config: LaWAMConfig) -> dict[str, Any]:
    """Translate LeRobot fields into the latent action model configuration."""
    return {
        "dim": config.lam_dim,
        "num_heads": config.lam_num_heads,
        "ffn_expansion_factor": config.lam_ffn_expansion_factor,
        "enc_layers": config.lam_enc_layers,
        "code_dim": config.lam_code_dim,
        "max_state_dim": config.lam_max_state_dim,
        "num_frames": config.num_video_frames,
        "num_queries": config.lam_num_queries,
        "vq_kwargs": {"layer_norm": config.lam_vq_layer_norm},
        "dec_layers": config.lam_dec_layers,
        "dropout": config.lam_dropout,
        "norm_latents": config.lam_norm_latents,
        "norm_latents_type": config.lam_norm_latents_type,
        "enc_modal_mask": config.lam_enc_modal_mask,
        "latent_layer_to_use": config.lam_latent_layer_to_use,
        "num_embodiments": config.lam_num_embodiments,
        "image_hw": config.lam_image_hw,
        "patch_size": config.lam_patch_size,
        "decoder_last_ln": config.lam_decoder_last_ln,
        "dinov3_config": {
            "hidden_size": config.dinov3_hidden_size,
            "intermediate_size": config.dinov3_intermediate_size,
            "num_hidden_layers": config.dinov3_num_hidden_layers,
            "num_attention_heads": config.dinov3_num_attention_heads,
            "num_register_tokens": config.dinov3_num_register_tokens,
            "patch_size": config.lam_patch_size,
        },
    }


def _build_native_policy_config(config: LaWAMConfig) -> LatentWorldPolicyConfig:
    """Build the native LaWAM backend configuration from a LeRobot config."""
    _require_lawam_packages()
    flow_cfg = ConditionalFlowMatchingConfig(
        action_dim=int(config.flow_action_dim),
        hidden_dim=int(config.flow_hidden_dim),
        num_layers=int(config.flow_num_layers),
        attention_heads=int(config.flow_attention_heads),
        vlm_dim=int(config.flow_vlm_dim),
        vision_dim=int(config.flow_vision_dim),
        num_vision_tokens=int(config.flow_num_vision_tokens),
        num_target_vision_tokens=int(config.flow_num_target_vision_tokens),
        horizon_sec=float(config.flow_horizon_sec),
        use_state=bool(config.flow_use_state),
        state_dim=int(config.flow_state_dim),
        num_embodiments=int(config.flow_num_embodiments),
        cfg_drop_prob=float(config.flow_cfg_drop_prob),
        cfg_guidance_scale=float(config.flow_cfg_guidance_scale),
        num_inference_steps=int(config.flow_num_inference_steps),
        num_timestep_buckets=int(config.flow_num_timestep_buckets),
        interleave_self_attention=bool(config.flow_interleave_self_attention),
        use_alternate_vldit=bool(config.flow_use_alternate_vldit),
        attend_text_every_n_blocks=int(config.flow_attend_text_every_n_blocks),
        noise_beta_alpha=float(config.flow_noise_beta_alpha),
        noise_beta_beta=float(config.flow_noise_beta_beta),
        noise_s=float(config.flow_noise_s),
        token_independent_noise=bool(config.flow_token_independent_noise),
        use_action_positional_embeddings=bool(config.flow_use_action_positional_embeddings),
    )
    policy_cfg = LatentWorldPolicyConfig(flow_cfg=flow_cfg)
    policy_cfg.action_horizon = config.effective_action_horizon
    policy_cfg.lam_config = _build_lam_config(config)
    policy_cfg.latent_action_placeholder_token = str(config.latent_action_placeholder_token)
    policy_cfg.perceptual_weight = float(config.perceptual_weight)
    policy_cfg.enable_loss_distill = bool(config.enable_loss_distill)
    policy_cfg.lam_encoder_distill_weight = float(config.lam_encoder_distill_weight)
    policy_cfg.future_prediction = bool(config.future_prediction)
    policy_cfg.detach_future_feature = bool(config.detach_future_feature)
    policy_cfg.repeated_diffusion_steps = int(config.repeated_diffusion_steps)
    policy_cfg.num_action_queries = int(config.num_action_queries)
    policy_cfg.flow_action_num_queries = int(config.flow_action_num_queries)
    return policy_cfg


class LaWAMModel(nn.Module):
    """Thin module wrapper around the native LaWAM backend."""

    def __init__(self, config: LaWAMConfig) -> None:
        super().__init__()
        self.config = config
        self.policy_cfg = _build_native_policy_config(config)
        self.policy_backend = LatentWorldPolicyBackend(self.policy_cfg, vlm_model_id=str(config.base_vlm))
        apply_policy_freeze(self.policy_backend, _build_freeze_config(config))

    def forward(self, batch):
        """Run one native LaWAM training step for a prepared batch."""
        return self.policy_backend(batch=batch)

    @torch.inference_mode()
    def predict_action(self, batch, **kwargs):
        """Predict normalized action chunks from a processor-prepared batch."""
        return self.policy_backend.predict_action(batch=batch, **kwargs)


class LaWAMPolicy(PreTrainedPolicy):
    """LeRobot adapter for LaWAM SFT and evaluation.

    This class keeps LaWAM's architecture inside LeRobot while translating
    LeRobot batches into LaWAM train/eval inputs.
    """

    config_class = LaWAMConfig
    name = "lawam"

    def __init__(self, config: LaWAMConfig, **kwargs) -> None:
        _require_lawam_packages()
        super().__init__(config)
        config.resolve_runtime_config(kwargs.pop("dataset_meta", None))
        config.validate_features()
        self.config = config

        self.model = kwargs.pop("native_model", None)
        if self.model is None:
            self.model = LaWAMModel(self.config)

        if not isinstance(self.model, nn.Module):
            raise TypeError(f"`native_model` must be a torch.nn.Module, got {type(self.model)}.")

        self.model.to(config.device)
        self.reset()

    @classmethod
    def _load_as_safetensor(
        cls, model: LaWAMPolicy, model_file: str, map_location: str, strict: bool
    ) -> LaWAMPolicy:
        """Stage LaWAM checkpoint loading on CPU to avoid duplicate accelerator weights."""
        del map_location
        model.to("cpu")
        return super()._load_as_safetensor(model, model_file, "cpu", strict)

    def reset(self) -> None:
        """Clear the queued action chunk used by step-wise inference."""
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> dict:
        """Return model parameters exposed to the LeRobot optimizer factory."""
        return self.model.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Compute the LaWAM training loss and scalar logging values."""
        if "actions" not in batch:
            raise KeyError("LaWAM training requires processor-prepared `actions`.")
        output = self.model(batch)

        loss = output.get("total_loss")
        if loss is None:
            loss = output.get("loss_total")
        if loss is None:
            tensor_values = [
                value for value in output.values() if torch.is_tensor(value) and value.numel() == 1
            ]
            if not tensor_values:
                raise KeyError(f"LaWAM output does not contain a scalar loss: {sorted(output.keys())}")
            loss = sum(tensor_values)

        logs = {
            key: float(value.detach().item())
            for key, value in output.items()
            if torch.is_tensor(value) and value.numel() == 1
        }
        logs["loss"] = float(loss.detach().item())
        return loss, logs

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a normalized action chunk for each observation in the batch."""
        del noise
        self.eval()
        output = self.model.predict_action(batch)
        actions = output.get("normalized_actions") if isinstance(output, dict) else output
        if actions is None:
            raise KeyError("LaWAM inference output is missing normalized actions.")
        actions_tensor = torch.as_tensor(actions, device=self.config.device, dtype=torch.float32)
        if actions_tensor.ndim == 2:
            actions_tensor = actions_tensor.unsqueeze(0)
        action_dim = int(self.config.action_feature.shape[0])
        if actions_tensor.shape[-1] < action_dim:
            raise ValueError(
                f"LaWAM produced {actions_tensor.shape[-1]} action dims, but LeRobot expects {action_dim}."
            )
        actions_tensor = actions_tensor[..., :action_dim]
        return actions_tensor[:, : self.config.effective_action_horizon]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Return the next action, refilling the action queue when necessary."""
        del noise
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()
