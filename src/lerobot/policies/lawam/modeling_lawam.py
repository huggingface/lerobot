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

from __future__ import annotations

import logging
from collections import deque
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.policies.lawam.latent_world.batch_builder import LatentWorldPolicyInferBatchBuilder
from lerobot.policies.lawam.latent_world.processor_utils import build_latent_world_processor_spec
from lerobot.policies.lawam.latent_world.runtime.runner import LatentWorldPolicyRunner
from lerobot.policies.lawam.latent_world.train_collator import LatentWorldTrainCollator
from lerobot.policies.lawam.latent_world.vlm_adapter import LatentWorldPolicyVLMAdapter
from lerobot.policies.lawam.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
from lerobot.policies.lawam.vlas.lawam import LatentWorldPolicyBackend, LatentWorldPolicyConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_lawam import LaWAMConfig


def _build_prompt_config() -> SimpleNamespace:
    return SimpleNamespace(datasets=SimpleNamespace(vla_data={}))


def _build_native_policy_config(config: LaWAMConfig) -> LatentWorldPolicyConfig:
    if config.lam_ckpt_path is None or config.lam_yaml_path is None:
        raise ValueError(
            "LaWAM requires both `policy.lam_ckpt_path` and `policy.lam_yaml_path` to build the "
            "latent action model."
        )

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
    policy_cfg.future_action_window_size = int(config.chunk_size) - 1
    policy_cfg.past_action_window_size = 0
    policy_cfg.action_horizon = int(config.chunk_size)
    policy_cfg.hf_cache_dir = config.hf_cache_dir
    policy_cfg.lam_ckpt_path = str(config.lam_ckpt_path)
    policy_cfg.lam_yaml_path = str(config.lam_yaml_path)
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


def _normalize_lawam_checkpoint_state_dict(
    state_dict: dict[str, Tensor], model_state_dict: dict[str, Tensor]
) -> dict[str, Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key
        if key.startswith("policy_backend.lam.vision_encoder.model.layer."):
            candidate = key.replace(
                "policy_backend.lam.vision_encoder.model.layer.",
                "policy_backend.lam.vision_encoder.model.model.layer.",
                1,
            )
            if candidate in model_state_dict:
                normalized_key = candidate
        normalized[normalized_key] = value
    return normalized


def _log_checkpoint_key_mismatch(kind: str, keys: list[str], *, max_examples: int = 8) -> None:
    if not keys:
        return
    examples = keys[:max_examples]
    suffix = "" if len(keys) <= max_examples else f" ... (+{len(keys) - max_examples} more)"
    logging.getLogger(__name__).warning("%s keys when loading LaWAM checkpoint (%d): %s%s", kind, len(keys), examples, suffix)


class LaWAMModel(nn.Module):
    def __init__(self, config: LaWAMConfig) -> None:
        super().__init__()
        self.config = config
        self.policy_cfg = _build_native_policy_config(config)
        self.policy_backend = LatentWorldPolicyBackend.build(self.policy_cfg, vlm_model_id=str(config.base_vlm))
        self.policy_vlm_adapter = LatentWorldPolicyVLMAdapter(
            model=self.policy_backend.vlm,
            processor=self.policy_backend.processor,
            config=_build_prompt_config(),
            placeholder_token=self.policy_cfg.latent_action_placeholder_token,
            act_queries=int(self.policy_backend.num_action_queries),
            flow_queries=int(self.policy_backend.flow_action_query.shape[0]),
        )
        self.policy_infer_batch_builder = LatentWorldPolicyInferBatchBuilder(
            policy_cfg=self.policy_cfg,
            policy_backend=self.policy_backend,
            policy_vlm_adapter=self.policy_vlm_adapter,
            enable_primary_random_resized_crop=bool(config.enable_primary_random_resized_crop),
        )
        self.policy_runner = LatentWorldPolicyRunner(
            policy_backend=self.policy_backend,
            infer_batch_builder=self.policy_infer_batch_builder,
        )

    def forward(self, batch):
        return self.policy_runner.train_step(batch)

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        return self.policy_runner.infer_step(examples, **kwargs)


class LaWAMPolicy(PreTrainedPolicy):
    """LeRobot adapter for LaWAM SFT and evaluation.

    This class keeps LaWAM's architecture inside LeRobot while translating
    LeRobot batches into LaWAM train/eval inputs.
    """

    config_class = LaWAMConfig
    name = "lawam"

    def __init__(self, config: LaWAMConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = kwargs.pop("native_model", None)
        self._native_config = kwargs.pop("native_config", None)
        if self.model is None:
            self.model, self._native_config = self._load_native_model_and_config()

        if not isinstance(self.model, nn.Module):
            raise TypeError(f"`native_model` must be a torch.nn.Module, got {type(self.model)}.")

        self._collator = kwargs.pop("native_collator", None)
        if self._collator is None:
            self._collator = self._build_train_collator()

        self.model.to(config.device)
        self.reset()

    def _load_native_model_and_config(self) -> tuple[nn.Module, Any]:
        model = LaWAMModel(self.config)
        if self.config.lawam_checkpoint_path:
            state_dict = torch.load(self.config.lawam_checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_dict = _normalize_lawam_checkpoint_state_dict(state_dict, model.state_dict())
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            _log_checkpoint_key_mismatch("Missing", missing)
            _log_checkpoint_key_mismatch("Unexpected", unexpected)
        return model, self.config

    def _build_train_collator(self):
        policy_cfg = getattr(self.model, "policy_cfg", None)
        if policy_cfg is None:
            raise ValueError("Loaded LaWAM model does not expose `policy_cfg`; cannot build train collator.")

        processor_spec = build_latent_world_processor_spec(policy_cfg=policy_cfg, vlm_model_id=str(self.config.base_vlm))
        collator = LatentWorldTrainCollator(
            policy_cfg=policy_cfg,
            processor_spec=processor_spec,
            act_queries=int(policy_cfg.num_action_queries),
            flow_queries=int(policy_cfg.flow_action_num_queries),
            enable_primary_video_aug=self.config.enable_primary_video_aug,
            enable_primary_random_resized_crop=self.config.enable_primary_random_resized_crop,
        )
        return collator.train()

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def _image_feature_groups(self) -> tuple[list[str], list[str]]:
        image_keys = list(self.config.image_features.keys())
        wrist_keys = (
            list(self.config.wrist_image_features)
            if self.config.wrist_image_features is not None
            else [key for key in image_keys if "wrist" in key.lower()]
        )
        primary_keys = (
            list(self.config.primary_image_features)
            if self.config.primary_image_features is not None
            else [key for key in image_keys if key not in wrist_keys]
        )
        if not primary_keys:
            primary_keys = [image_keys[0]]
            wrist_keys = [key for key in wrist_keys if key != image_keys[0]]
        return primary_keys, wrist_keys

    @staticmethod
    def _to_uint8_video(tensor: Tensor) -> Tensor:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected image tensor [C,H,W] or [T,C,H,W], got {tuple(tensor.shape)}.")
        tensor = tensor.detach().float().cpu()
        if tensor.max() <= 1.0 and tensor.min() >= 0.0:
            tensor = tensor * 255.0
        return tensor.round().clamp(0, 255).to(dtype=torch.uint8).contiguous()

    @staticmethod
    def _to_uint8_frame(tensor: Tensor) -> Tensor:
        video = LaWAMPolicy._to_uint8_video(tensor)
        return video[0].contiguous()

    @staticmethod
    def _task_list(tasks: Any, batch_size: int, default_task: str) -> list[str]:
        if tasks is None:
            return [default_task] * batch_size
        if isinstance(tasks, str):
            return [tasks] * batch_size
        return [str(task) for task in tasks]

    def _state_dim(self) -> int:
        policy_cfg = getattr(self.model, "policy_cfg", None)
        flow_cfg = getattr(policy_cfg, "flow_cfg", None)
        state_dim = getattr(flow_cfg, "state_dim", None)
        if state_dim is not None:
            return int(state_dim)
        if self.config.robot_state_feature is not None:
            return int(self.config.robot_state_feature.shape[0])
        return 1

    def _prepare_train_samples(self, batch: dict[str, Tensor]) -> list[dict[str, Any]]:
        primary_keys, wrist_keys = self._image_feature_groups()
        first = batch[primary_keys[0]]
        batch_size = int(first.shape[0])
        tasks = self._task_list(batch.get("task"), batch_size, self.config.default_task)

        action_tensor = batch.get(ACTION)
        if action_tensor is None:
            raise KeyError("LaWAM training requires `action` in the batch.")
        state_tensor = batch.get(OBS_STATE)

        samples: list[dict[str, Any]] = []
        for idx in range(batch_size):
            primary_videos = torch.stack([self._to_uint8_video(batch[key][idx]) for key in primary_keys], dim=0)
            wrist_images = (
                torch.stack([self._to_uint8_frame(batch[key][idx]) for key in wrist_keys], dim=0)
                if wrist_keys
                else torch.empty((0, 3, primary_videos.shape[-2], primary_videos.shape[-1]), dtype=torch.uint8)
            )
            action = action_tensor[idx].detach().float().cpu()
            if action.ndim == 1:
                action = action.unsqueeze(0)

            if state_tensor is None:
                state = torch.zeros((1, self._state_dim()), dtype=torch.float32)
            else:
                state = state_tensor[idx].detach().float().cpu()
                if state.ndim == 1:
                    state = state.unsqueeze(0)

            samples.append(
                {
                    "primary_videos": primary_videos,
                    "wrist_images": wrist_images,
                    "lang": tasks[idx],
                    "state": state,
                    "action": action,
                    "embodiment_id": self.config.embodiment_id,
                    "action_hz": self.config.action_hz,
                }
            )
        return samples

    def _move_batch_to_device(self, batch: Any) -> Any:
        device = torch.device(str(self.config.device))
        return {
            key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        samples = self._prepare_train_samples(batch)
        lawam_batch = self._move_batch_to_device(self._collator(samples))
        output = self.model(lawam_batch)

        loss = output.get("total_loss")
        if loss is None:
            loss = output.get("loss_total")
        if loss is None:
            tensor_values = [value for value in output.values() if torch.is_tensor(value) and value.numel() == 1]
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

    def _prepare_infer_examples(self, batch: dict[str, Tensor]) -> list[dict[str, Any]]:
        primary_keys, wrist_keys = self._image_feature_groups()
        first = batch[primary_keys[0]]
        batch_size = int(first.shape[0])
        tasks = self._task_list(batch.get("task"), batch_size, self.config.default_task)
        state_tensor = batch.get(OBS_STATE)

        examples: list[dict[str, Any]] = []
        for idx in range(batch_size):
            primary_images = [
                self._to_uint8_frame(batch[key][idx]).permute(1, 2, 0).numpy() for key in primary_keys
            ]
            example: dict[str, Any] = {
                "primary_image": primary_images,
                "lang": tasks[idx],
                "embodiment_id": self.config.embodiment_id,
                "action_hz": self.config.action_hz,
            }
            if wrist_keys:
                example["wrist_image"] = [
                    self._to_uint8_frame(batch[key][idx]).permute(1, 2, 0).numpy() for key in wrist_keys
                ]
            if state_tensor is not None:
                state = state_tensor[idx].detach().float().cpu()
                if state.ndim == 1:
                    state = state.unsqueeze(0)
                example["state"] = state.numpy()
            examples.append(example)
        return examples

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        del noise
        self.eval()
        examples = self._prepare_infer_examples(batch)
        output = self.model.predict_action(
            examples,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
        )
        actions = output.get("normalized_actions")
        if actions is None:
            raise KeyError(f"LaWAM inference output missing `normalized_actions`: {sorted(output.keys())}")
        actions_tensor = torch.as_tensor(actions, device=self.config.device, dtype=torch.float32)
        if actions_tensor.ndim == 2:
            actions_tensor = actions_tensor.unsqueeze(0)
        action_dim = int(self.config.action_feature.shape[0])
        if actions_tensor.shape[-1] < action_dim:
            raise ValueError(
                f"LaWAM produced {actions_tensor.shape[-1]} action dims, but LeRobot expects {action_dim}."
            )
        actions_tensor = actions_tensor[..., :action_dim]
        return actions_tensor

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        del noise
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()
