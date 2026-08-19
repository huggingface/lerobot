#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

"""DM05 policy wrapper, checkpoint I/O, and inference helpers."""

from __future__ import annotations

import builtins
import copy
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any, Unpack

import torch
from huggingface_hub import save_torch_state_dict
from torch import Tensor

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..pretrained import ActionSelectKwargs, PreTrainedPolicy, T
from .configuration_dm05 import DM05Config
from .constants import ACTION_REFERENCE_OFFSET
from .conversion_dm05 import DM05LerobotBatchConverter
from .stats_validation_dm05 import (
    dm05_prepare_stats_command,
    dm05_stats_complete,
    validate_dm05_relative_action_stats,
)
from .utils import (
    build_action_prefix_mask,
    flatten_feature_names,
    import_dm05_core,
    relative_action_mask,
    resolve_torch_dtype,
    validate_action_prefill_pair,
)

logger = logging.getLogger(__name__)


def setup_compiled_suffix(config: Any, model: Any) -> bool:
    """Enable the compiled DM05 suffix path when the runtime supports it."""
    if not config.compile_model:
        return False
    if not hasattr(model, "setup_compiled_suffix_layers"):
        raise RuntimeError("DM05 core model does not support compiled suffix inference.")
    if not torch.cuda.is_available() or not str(config.device).startswith("cuda"):
        raise RuntimeError("DM05 compiled suffix inference requires CUDA.")
    torch.set_float32_matmul_precision("high")
    model.setup_compiled_suffix_layers(mode="reduce-overhead", dynamic=False)
    logger.info("Enabled DM05 compiled suffix inference.")
    return True


def prepare_compiled_suffix_inputs(
    config: Any,
    model: Any,
    processor: Any,
    model_inputs: dict[str, Any],
    *,
    dtype: torch.dtype,
    initial_noise: torch.Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Build the call inputs needed by compiled DM05 suffix inference."""
    input_ids = model_inputs["input_ids"]
    pad_length = config.compile_suffix_pad_length
    if pad_length is not None:
        pad_length = int(pad_length)
        seq_len = int(input_ids.shape[1])
        if seq_len > pad_length:
            raise ValueError(f"DM05 compiled suffix input length {seq_len} exceeds pad length {pad_length}.")

    call_inputs = dict(model_inputs)
    action_prefill_len = call_inputs.get("action_prefill_len")
    prefill_actions = call_inputs.get("prefill_actions")
    validate_action_prefill_pair(prefill_actions, action_prefill_len)
    if pad_length is not None:
        language_model = model.model.vlm.model.language_model
        pad_token_id = getattr(language_model, "padding_idx", None)
        if pad_token_id is None:
            pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("Unable to resolve DM05 pad token id for compiled suffix padding.")
        pad_token_id = int(pad_token_id)
        for key, value in {"input_ids": pad_token_id, "attention_mask": 0, "token_type_ids": 0}.items():
            tensor = call_inputs.get(key)
            if tensor is None:
                continue
            seq_len = int(tensor.shape[1])
            if seq_len != pad_length:
                padded = tensor.new_full((int(tensor.shape[0]), pad_length), value)
                padded[:, :seq_len] = tensor
                call_inputs[key] = padded

    input_ids = call_inputs["input_ids"]
    batch_size, device = int(input_ids.shape[0]), input_ids.device
    chunk_size, action_dim = int(model.config.chunk_size), int(model.config.action_dim)
    if initial_noise is None:
        initial_noise = torch.randn(batch_size, chunk_size, action_dim, device=device, dtype=dtype)
    else:
        initial_noise = initial_noise.to(device=device, dtype=dtype)
    action_prefix_mask = torch.zeros(batch_size, chunk_size, device=device, dtype=torch.bool)
    if action_prefill_len is not None:
        action_prefix_mask = build_action_prefix_mask(action_prefill_len, horizon=chunk_size, device=device)
    if prefill_actions is None:
        call_inputs["prefill_actions"] = torch.zeros_like(initial_noise)
    else:
        call_inputs["prefill_actions"] = prefill_actions.to(device=device, dtype=dtype)
    inference_kwargs = {
        "use_compiled_suffix": True,
        "initial_noise": initial_noise,
        "action_prefix_mask": action_prefix_mask,
    }
    return call_inputs, inference_kwargs


def _has_dm05_core_config_payload(config: DM05Config) -> bool:
    """Return whether a DM05 config carries the embedded core HF payload."""
    core_config = getattr(config, "core_config", None)
    return (
        isinstance(core_config, dict)
        and core_config.get("model_type") == "dexbotic_dm05"
        and core_config.get("vlm_config") is not None
        and core_config.get("action_config") is not None
    )


def _apply_core_overrides(core_config: Any, config: DM05Config, dtype: torch.dtype) -> Any:
    """Apply the LeRobot adapter runtime overrides to the core config."""
    core_config.bf16 = dtype is torch.bfloat16
    core_config.chunk_size = config.chunk_size
    core_config.vlm_gradient_checkpointing = bool(config.vlm_gradient_checkpointing)
    core_config.ae_gradient_checkpointing = bool(config.ae_gradient_checkpointing)
    core_config.gradient_checkpointing = bool(
        core_config.vlm_gradient_checkpointing or core_config.ae_gradient_checkpointing
    )
    core_config.ae_gradient_checkpointing_layers = int(config.ae_gradient_checkpointing_layers)
    core_config.llm_attn_implementation = config.llm_attn_implementation
    core_config.vision_attn_implementation = config.vision_attn_implementation
    core_config.action_attn_implementation = config.action_attn_implementation
    return core_config


class DM05Policy(PreTrainedPolicy):
    """LeRobot policy wrapper around the core DM05 model."""

    config_class = DM05Config
    name = "dm05"

    def __init__(
        self,
        config: DM05Config,
        *,
        checkpoint_source: str | Path | None = None,
        checkpoint_revision: str | None = None,
        processor_load_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        require_package("transformers", extra="dm05")
        super().__init__(config)
        config.validate_features()
        is_lerobot_checkpoint = checkpoint_source is not None and _has_dm05_core_config_payload(config)
        dataset_meta = kwargs.get("dataset_meta")
        if dataset_meta is not None and hasattr(dataset_meta, "features"):
            config.action_feature_names = flatten_feature_names(
                dataset_meta.features.get(ACTION, {}).get("names")
            )

        raw_source = config.pretrained_name_or_path
        if not is_lerobot_checkpoint and not raw_source:
            raise ValueError(
                "DM05 requires a standard LeRobot checkpoint or a raw checkpoint for one-time conversion."
            )

        core_config_cls, core_model_cls, tokenization_cls = import_dm05_core()

        from transformers import AutoProcessor

        processor_source = checkpoint_source if is_lerobot_checkpoint else config.processor_name_or_path
        processor_load_kwargs = dict(processor_load_kwargs or {})
        processor_load_kwargs.setdefault("revision", checkpoint_revision)
        # DM05 uses the original Gemma tokenizer regex. Transformers cannot infer
        # that from LeRobot's policy config and otherwise emits a Mistral warning.
        processor_load_kwargs.setdefault("fix_mistral_regex", False)
        self.processor = AutoProcessor.from_pretrained(
            processor_source or raw_source, **processor_load_kwargs
        )

        torch_dtype = resolve_torch_dtype(config.dtype)
        if str(config.device) == "cuda" and torch.cuda.is_available():
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                config.device = f"cuda:{int(local_rank)}"
        if is_lerobot_checkpoint:
            core_config = _apply_core_overrides(
                core_config_cls(**config.core_config),
                config,
                torch_dtype,
            )
            previous_default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch_dtype)
            try:
                self.model = core_model_cls(core_config)
            finally:
                torch.set_default_dtype(previous_default_dtype)
        else:
            self.model = core_model_cls.from_pretrained(
                raw_source,
                torch_dtype=torch_dtype,
                chunk_size=config.chunk_size,
                vlm_gradient_checkpointing=config.vlm_gradient_checkpointing,
                ae_gradient_checkpointing=config.ae_gradient_checkpointing,
                ae_gradient_checkpointing_layers=config.ae_gradient_checkpointing_layers,
                llm_attn_implementation=config.llm_attn_implementation,
                vision_attn_implementation=config.vision_attn_implementation,
                action_attn_implementation=config.action_attn_implementation,
            )
        config._validate_core_action_dim(getattr(self.model.config, "action_dim", None))
        if config.use_liger_kernel and hasattr(self.model, "_apply_liger_kernel"):
            # Disabled by default so environments without liger-kernel still load.
            self.model._apply_liger_kernel()

        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing(
                vlm_gradient_checkpointing=bool(config.vlm_gradient_checkpointing),
                ae_gradient_checkpointing=bool(config.ae_gradient_checkpointing),
                ae_layers=config.ae_gradient_checkpointing_layers,
            )
        elif (config.vlm_gradient_checkpointing or config.ae_gradient_checkpointing) and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
        if config.freeze_vlm_embedding:
            for path in (
                ("model", "vlm", "model", "language_model", "embed_tokens"),
                ("model", "language_model", "embed_tokens"),
            ):
                module = self.model
                for attr in path:
                    module = getattr(module, attr, None)
                    if module is None:
                        break
                if module is not None and hasattr(module, "parameters"):
                    for parameter in module.parameters():
                        parameter.requires_grad = False
                    break
        self.model.to(config.device)
        self._compile_suffix_active = setup_compiled_suffix(self.config, self.model)
        self._batch_converter = DM05LerobotBatchConverter(
            config=config,
            tokenization_cls=tokenization_cls,
            processor=self.processor,
        )
        self.reset()

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save the policy weights, processor, and serialized DM05 config."""
        from lerobot.distributed.checkpoint import full_model_state_dict
        from lerobot.distributed.utils import is_main_process

        policy_to_save = self.module if hasattr(self, "module") else self
        state_dict = full_model_state_dict(policy_to_save)
        if not state_dict or not is_main_process():
            return

        save_directory.mkdir(parents=True, exist_ok=True)
        self.processor.save_pretrained(save_directory)
        # Keep the processor-level chat_template.jinja emitted above for processor tokenization.
        # Also mirror the template into tokenizer_config.json for tokenizer-only consumers.
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.chat_template = getattr(self.processor, "chat_template", None)
            tokenizer.save_pretrained(save_directory, save_jinja_files=False)
        model_to_save = policy_to_save.model

        save_config = copy.deepcopy(self.config)
        prepare_config_for_save = getattr(model_to_save, "prepare_config_for_save", None)
        if callable(prepare_config_for_save):
            prepare_config_for_save()
        model_core_config = getattr(model_to_save, "config", None)
        to_dict = getattr(model_core_config, "to_dict", None)
        if callable(to_dict):
            core_config = to_dict()
        elif model_core_config is not None:
            core_config = {
                key: value
                for key, value in vars(model_core_config).items()
                if not key.startswith("_") or key in {"_name_or_path"}
            }
        else:
            core_config = None
        if core_config:
            core_config["_name_or_path"] = "."
            save_config.core_config = core_config
        save_config.pretrained_name_or_path = "."
        save_config.processor_name_or_path = "."
        save_config.pretrained_path = None
        save_config._save_pretrained(save_directory)

        # Match safetensors.save_model by omitting the tied input embedding.
        state_dict = dict(state_dict)
        state_dict.pop("model.model.vlm.model.language_model.embed_tokens.weight", None)
        total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
        save_torch_state_dict(state_dict, str(save_directory), max_shard_size=max(total_bytes, 1))

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load DM05 from a self-contained LeRobot checkpoint."""
        model_id = str(pretrained_name_or_path)
        saved_config = None
        processor_load_kwargs = {
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )
            saved_config = config
        if not isinstance(config, DM05Config):
            raise ValueError(f"Expected a DM05 config, got {type(config).__name__}.")
        config.validate_features()
        if saved_config is None:
            saved_config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )
        if not isinstance(saved_config, DM05Config):
            raise ValueError(f"Expected a saved DM05 config, got {type(saved_config).__name__}.")
        saved_config.validate_features()

        dataset_meta = kwargs.get("dataset_meta")
        dataset_stats = kwargs.get("dataset_stats")
        stats_complete = dm05_stats_complete(config, dataset_stats)
        if stats_complete:
            validate_dm05_relative_action_stats(config, dataset_stats)
        if dataset_meta is not None and not stats_complete:
            target_dims = (
                config.input_features[OBS_STATE].shape[-1],
                config.output_features[ACTION].shape[-1],
            )
            checkpoint_dims = (
                saved_config.input_features[OBS_STATE].shape[-1],
                saved_config.output_features[ACTION].shape[-1],
            )
            if target_dims != checkpoint_dims:
                command = dm05_prepare_stats_command(config, dataset_meta)
                raise ValueError(
                    "DM05 cannot reuse checkpoint statistics for different state/action dimensions "
                    f"({checkpoint_dims} -> {target_dims}). Run `{command}` before training."
                )
        if dataset_meta is not None and dataset_stats and not stats_complete:
            logger.warning(
                "Ignoring incomplete DM05 dataset statistics and retaining the checkpoint processor stats. "
                "Prepare complete target stats before fine-tuning a different embodiment or distribution."
            )
            dataset_meta.stats = None
            kwargs["dataset_stats"] = None
            dataset_stats = None
        elif dataset_meta is not None and not dataset_stats:
            logger.warning(
                "DM05 dataset statistics are missing; the checkpoint processor stats will be retained. "
                "This is valid only when the target state/action contract matches the checkpoint."
            )

        if saved_config.use_relative_actions and not config.use_relative_actions:
            raise ValueError(
                "A relative-action DM05 checkpoint cannot be loaded with "
                "use_relative_actions=False because its saved processor statistics are relative."
            )
        if config.use_relative_actions and not saved_config.use_relative_actions and not stats_complete:
            command = dm05_prepare_stats_command(config, dataset_meta)
            raise ValueError(
                "Enabling DM05 relative actions from an absolute-action checkpoint requires complete "
                f"relative-action dataset statistics. Run `{command} --force` before training."
            )
        if not _has_dm05_core_config_payload(config):
            raise ValueError(
                "DM05Policy.from_pretrained requires a converted LeRobot checkpoint with core_config. "
                "Convert raw OpenDM release assets before using them as --policy.path."
            )
        policy = super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            checkpoint_source=model_id,
            checkpoint_revision=revision,
            processor_load_kwargs=processor_load_kwargs,
            **kwargs,
        )
        return policy

    def reset(self):
        """Reset the rollout action queue and relative-action state."""
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        self._relative_generation_offset: Tensor | None = None

    def to(self, *args, **kwargs):
        """Move the policy and keep the config device in sync."""
        policy = super().to(*args, **kwargs)
        parameter = next(policy.parameters(), None)
        if parameter is not None:
            self.config.device = str(parameter.device)
        return policy

    def get_optim_params(self):
        """Return the trainable DM05 parameters."""
        return self.parameters()

    def _uses_quantile_clipping(self, feature_type: str) -> bool:
        """Return whether a normalized feature should be clipped to the model range."""
        return self.config.norm_clip and self.config.normalization_mapping.get(feature_type) in {
            NormalizationMode.QUANTILES,
            NormalizationMode.QUANTILE10,
        }

    def _prepare_policy_batch(self, batch: dict[str, Any], *, include_actions: bool) -> dict[str, Any]:
        """Apply DM05-specific clipping before the batch reaches the converter."""
        batch = dict(batch)
        if self._uses_quantile_clipping("STATE") and OBS_STATE in batch:
            batch[OBS_STATE] = torch.as_tensor(batch[OBS_STATE]).clamp(-1.0, 1.0)
        if include_actions and self._uses_quantile_clipping("ACTION") and ACTION in batch:
            batch[ACTION] = torch.as_tensor(batch[ACTION]).clamp(-1.0, 1.0)
        return batch

    def _prepare_model_inputs(self, batch: dict[str, Any], include_actions: bool) -> dict[str, Any]:
        """Convert one processed batch into the tensors consumed by DM05."""
        batch = self._prepare_policy_batch(batch, include_actions=include_actions)
        model_inputs = self._batch_converter.convert_lerobot_batch(batch, include_actions=include_actions)
        model_inputs = {
            key: value.to(self.config.device) if isinstance(value, Tensor) else value
            for key, value in model_inputs.items()
        }
        dtype = next((p.dtype for p in self.model.parameters() if p.is_floating_point()), torch.float32)
        for key in ("states", "actions", "action", "prefill_actions"):
            value = model_inputs.get(key)
            if torch.is_tensor(value) and value.is_floating_point():
                model_inputs[key] = value.to(dtype=dtype)
        if not include_actions:
            batch_size = int(model_inputs["input_ids"].shape[0])
            core_action_dim = int(self.model.config.action_dim)
            action_feature = self.config.output_features.get(ACTION) if self.config.output_features else None
            action_dim = (
                int(action_feature.shape[-1])
                if action_feature is not None and action_feature.shape
                else int(self.config.max_action_dim)
            )
            action_dim_mask = torch.zeros(
                batch_size,
                core_action_dim,
                device=self.config.device,
                dtype=torch.bool,
            )
            action_dim_mask[:, :action_dim] = True
            model_inputs["action_dim_mask"] = action_dim_mask
        return model_inputs

    def _prepare_initial_noise(
        self,
        noise: Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> Tensor | None:
        """Normalize the optional diffusion noise tensor for DM05 inference."""
        if noise is None:
            return None
        noise = noise.to(device=self.config.device, dtype=dtype)
        expected_prefix = (batch_size, int(self.model.config.chunk_size))
        if noise.ndim != 3 or noise.shape[:2] != expected_prefix:
            raise ValueError(
                f"noise must have shape [B,T,D] with B,T={expected_prefix}, got {tuple(noise.shape)}."
            )
        core_action_dim = int(self.model.config.action_dim)
        action_feature = self.config.output_features.get(ACTION) if self.config.output_features else None
        action_dim = (
            int(action_feature.shape[-1])
            if action_feature is not None and action_feature.shape
            else core_action_dim
        )
        if noise.shape[-1] not in {action_dim, core_action_dim}:
            raise ValueError(
                f"noise action dimension must be {action_dim} (policy) or {core_action_dim} (core), "
                f"got {noise.shape[-1]}."
            )
        if noise.shape[-1] == action_dim and action_dim < core_action_dim:
            noise = torch.nn.functional.pad(noise, (0, core_action_dim - action_dim))
        return noise

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the supervised DM05 training forward pass."""
        model_inputs = self._prepare_model_inputs(batch, include_actions=True)
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        return loss, {}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a full action chunk for one processed observation batch."""
        self.eval()
        model_inputs = self._prepare_model_inputs(batch, include_actions=False)
        diffusion_steps = kwargs.get("diffusion_steps", self.config.diffusion_steps)
        if diffusion_steps <= 0:
            raise ValueError(f"diffusion_steps must be positive, got {diffusion_steps}")
        model_dtype = next((p.dtype for p in self.model.parameters() if p.is_floating_point()), torch.float32)
        initial_noise = self._prepare_initial_noise(
            kwargs.get("noise"),
            batch_size=int(model_inputs["input_ids"].shape[0]),
            dtype=model_dtype,
        )
        call_inputs = model_inputs
        inference_kwargs = {"initial_noise": initial_noise}
        if self._compile_suffix_active:
            call_inputs, inference_kwargs = prepare_compiled_suffix_inputs(
                self.config,
                self.model,
                self.processor,
                model_inputs,
                dtype=model_dtype,
                initial_noise=initial_noise,
            )
        actions = self.model.inference_action(
            **call_inputs,
            **inference_kwargs,
            diffusion_steps=diffusion_steps,
        )
        action_feature = self.config.output_features.get(ACTION) if self.config.output_features else None
        action_dim = (
            int(action_feature.shape[-1])
            if action_feature is not None and action_feature.shape
            else int(self.config.max_action_dim)
        )
        return actions[:, :, :action_dim]

    def _action_reference_offset(self, batch: dict[str, Tensor]) -> Tensor:
        """Extract the cached relative-action offset from the processor output."""
        offset = batch.get(ACTION_REFERENCE_OFFSET)
        if offset is None:
            raise ValueError(
                "DM05 relative-action inference requires its checkpoint preprocessor and observation.state."
            )
        offset = torch.as_tensor(offset, device=self.config.device, dtype=torch.float32)
        return offset.unsqueeze(0) if offset.ndim == 1 else offset

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Return the next queued action slice for online DM05 inference."""
        self.eval()
        current_offset = self._action_reference_offset(batch) if self.config.use_relative_actions else None
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            if current_offset is not None:
                if current_offset.shape != (actions.shape[0], actions.shape[-1]):
                    raise ValueError(
                        "DM05 action reference shape must match predicted actions, got "
                        f"{tuple(current_offset.shape)} and {tuple(actions.shape)}."
                    )
                self._relative_generation_offset = current_offset.clone()
            self._queues[ACTION].extend(actions.transpose(0, 1))
        action = self._queues[ACTION].popleft()
        if current_offset is None:
            return action
        if self._relative_generation_offset is None:
            raise RuntimeError("DM05 relative-action queue has no generation reference.")
        if current_offset.shape != action.shape:
            raise ValueError(
                "DM05 action reference shape must match the queued action, got "
                f"{tuple(current_offset.shape)} and {tuple(action.shape)}."
            )
        mask = torch.tensor(
            relative_action_mask(
                action.shape[-1],
                self.config.action_feature_names,
                self.config.relative_exclude_joints,
            ),
            device=action.device,
            dtype=torch.float32,
        )
        generation_offset = self._relative_generation_offset.to(action.device)
        return action.float() + (generation_offset - current_offset.to(action.device)) * mask
