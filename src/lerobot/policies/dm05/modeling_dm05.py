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

from __future__ import annotations

import builtins
import copy
import json
import logging
import os
import shutil
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Unpack

import torch
from huggingface_hub import HfApi, hf_hub_download, save_torch_state_dict
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..pretrained import ActionSelectKwargs, PreTrainedPolicy, T
from ..utils import log_model_loading_keys
from .configuration_dm05 import DM05Config
from .conversion_dm05 import DM05LerobotBatchConverter
from .modeling_dm05_core import build_action_prefix_mask
from .normalization_dm05 import command_action_from_relative, resolve_dm05_normalizer
from .utils import (
    import_dm05_core,
    infer_dm05_non_delta_indices,
    resolve_torch_dtype,
)

logger = logging.getLogger(__name__)

_BUNDLED_DM05_NORM_STATS = "norm_stats.json"
_SKIP_DM05_ASSET_NAMES = {
    CONFIG_NAME,
    SAFETENSORS_SINGLE_FILE,
    "pytorch_model.bin",
    "trainer_state.json",
    "training_args.bin",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
}
_SKIP_DM05_ASSET_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_DM05_LOAD_TARGET_CORE = "core"
_DM05_LOAD_TARGET_POLICY = "policy"


def _auto_fallback(config: Any, message: str) -> bool:
    if config.compile_suffix != "auto":
        return False
    logger.warning("%s Falling back to eager inference.", message)
    return True


def setup_compiled_suffix(config: Any, model: Any) -> bool:
    if config.compile_suffix == "off":
        return False
    if not hasattr(model, "setup_compiled_suffix_layers"):
        msg = "DM05 core model does not support compiled suffix inference."
        if _auto_fallback(config, msg):
            return False
        raise RuntimeError(msg)
    if not torch.cuda.is_available() or not str(config.device).startswith("cuda"):
        msg = "DM05 compiled suffix inference requires CUDA."
        if _auto_fallback(config, msg):
            return False
        raise RuntimeError(msg)
    try:
        torch.set_float32_matmul_precision("high")
        model.setup_compiled_suffix_layers(mode="reduce-overhead", dynamic=False)
    except Exception:
        if config.compile_suffix == "auto":
            logger.exception("Failed to initialize DM05 compiled suffix inference; falling back to eager.")
            return False
        raise
    logger.info("Enabled DM05 compiled suffix inference.")
    return True


def prepare_compiled_suffix_inputs(
    config: Any,
    model: Any,
    processor: Any,
    model_inputs: dict[str, Any],
    *,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]] | None:
    input_ids = model_inputs["input_ids"]
    pad_length = config.compile_suffix_pad_length
    if pad_length is not None:
        pad_length = int(pad_length)
        seq_len = int(input_ids.shape[1])
        if seq_len > pad_length:
            msg = f"DM05 compiled suffix input length {seq_len} exceeds pad length {pad_length}."
            if _auto_fallback(config, msg):
                return None
            raise ValueError(msg)

    call_inputs = dict(model_inputs)
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
    initial_noise = torch.randn(batch_size, chunk_size, action_dim, device=device, dtype=dtype)
    action_prefill_len = call_inputs.get("action_prefill_len")
    action_prefix_mask = torch.zeros(batch_size, chunk_size, device=device, dtype=torch.bool)
    if action_prefill_len is not None:
        action_prefix_mask = build_action_prefix_mask(action_prefill_len, horizon=chunk_size, device=device)
    if (prefill_actions := call_inputs.get("prefill_actions")) is None:
        call_inputs["prefill_actions"] = torch.zeros_like(initial_noise)
    else:
        call_inputs["prefill_actions"] = prefill_actions.to(device=device, dtype=dtype)
    inference_kwargs = {
        "use_compiled_suffix": True,
        "initial_noise": initial_noise,
        "action_prefix_mask": action_prefix_mask,
    }
    return call_inputs, inference_kwargs


def warmup_compiled_suffix(
    config: Any,
    model: Any,
    processor: Any,
    model_inputs: dict[str, Any],
    *,
    diffusion_steps: int,
    dtype: torch.dtype,
) -> bool:
    compiled_inputs = prepare_compiled_suffix_inputs(config, model, processor, model_inputs, dtype=dtype)
    if compiled_inputs is None:
        return False
    call_inputs, inference_kwargs = compiled_inputs
    for _ in range(int(config.compile_suffix_warmup_steps)):
        model.inference_action(
            **call_inputs,
            **inference_kwargs,
            diffusion_steps=diffusion_steps,
        )
    return True


def _resolve_local_path(value: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        return Path(base_dir).expanduser() / path
    return path


def _same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    return left.expanduser().resolve() == right.expanduser().resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except json.JSONDecodeError:
        return {}


def _copy_dm05_reference_assets(source: str | Path | None, target: Path) -> bool:
    if source is None:
        return False
    source_path = Path(source).expanduser()
    if not source_path.is_dir():
        return False
    if source_path.resolve() == target.resolve():
        return True

    target.mkdir(parents=True, exist_ok=True)
    copied = False
    for item in source_path.iterdir():
        if item.is_dir():
            continue
        if item.name in _SKIP_DM05_ASSET_NAMES or item.name.endswith(_SKIP_DM05_ASSET_SUFFIXES):
            continue
        shutil.copy2(item, target / item.name)
        copied = True
    return copied


def _select_dm05_core_state_dict(
    model_to_save: torch.nn.Module, state_dict: dict[str, Tensor]
) -> dict[str, Tensor]:
    reference_keys = set(model_to_save.state_dict())
    if not reference_keys:
        return state_dict

    state_keys = set(state_dict)
    if state_keys == reference_keys:
        return state_dict

    for prefix in (
        "model.",
        "module.model.",
        "_fsdp_wrapped_module.model.",
        "module._fsdp_wrapped_module.model.",
    ):
        stripped = {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
            if key.startswith(prefix) and key.removeprefix(prefix) in reference_keys
        }
        if set(stripped) == reference_keys:
            return stripped

    missing = sorted(reference_keys - state_keys)
    prefixed_matches = sum(key.startswith("model.") for key in state_dict)
    raise ValueError(
        "DM05 state_dict does not match the core model state_dict. "
        f"missing={missing[:5]} total_missing={len(missing)} model_prefixed_keys={prefixed_matches}"
    )


def _has_dm05_core_config_payload(config: DM05Config) -> bool:
    core_config = getattr(config, "core_config", None)
    return (
        isinstance(core_config, dict)
        and core_config.get("model_type") == "dexbotic_dm05"
        and core_config.get("vlm_config") is not None
        and core_config.get("action_config") is not None
    )


def _configure_dm05_checkpoint_paths(
    config: DM05Config,
    checkpoint_dir: Path | None,
    *,
    use_bundled_norm_stats: bool = True,
) -> None:
    if checkpoint_dir is None:
        return

    checkpoint_config_path = checkpoint_dir / CONFIG_NAME
    if (
        _same_path(_resolve_local_path(getattr(config, "pretrained_path", None), None), checkpoint_dir)
        and checkpoint_config_path.is_file()
        and (checkpoint_dir / SAFETENSORS_SINGLE_FILE).is_file()
    ):
        config.pretrained_name_or_path = "."

    if not _has_dm05_core_config_payload(config) and checkpoint_config_path.is_file():
        core_config = _read_json(checkpoint_config_path).get("core_config")
        if isinstance(core_config, dict):
            config.core_config = core_config

    core_path = _resolve_local_path(config.pretrained_name_or_path, checkpoint_dir)
    if (not config.pretrained_name_or_path or core_path is None or not core_path.exists()) and (
        checkpoint_dir / "config.json"
    ).is_file():
        config.pretrained_name_or_path = "."

    processor_path = _resolve_local_path(config.processor_name_or_path, checkpoint_dir)
    if (config.processor_name_or_path is None or processor_path is None or not processor_path.exists()) and (
        checkpoint_dir / "processor_config.json"
    ).is_file():
        config.processor_name_or_path = "."

    if use_bundled_norm_stats:
        bundled_norm = checkpoint_dir / _BUNDLED_DM05_NORM_STATS
        norm_path = _resolve_local_path(config.norm_stats_path, checkpoint_dir)
        if (
            config.norm_stats_path is None or norm_path is None or not norm_path.exists()
        ) and bundled_norm.is_file():
            config.norm_stats_path = _BUNDLED_DM05_NORM_STATS

    for attr in ("pretrained_name_or_path", "processor_name_or_path", "norm_stats_path"):
        value = getattr(config, attr, None)
        resolved = _resolve_local_path(value, checkpoint_dir)
        if resolved is not None and resolved.exists():
            setattr(config, attr, str(resolved))

    core_path = _resolve_local_path(config.pretrained_name_or_path, checkpoint_dir)
    if (
        _has_dm05_core_config_payload(config)
        and _same_path(core_path, checkpoint_dir)
        and (checkpoint_dir / SAFETENSORS_SINGLE_FILE).is_file()
    ):
        config._dm05_core_config_only = True
        config._dm05_safetensors_load_target = _DM05_LOAD_TARGET_CORE
        return

    if core_path is not None and (core_path / "config.json").is_file():
        has_core_weights = (core_path / SAFETENSORS_SINGLE_FILE).is_file()
        if not has_core_weights:
            config._dm05_core_config_only = True
            config._dm05_safetensors_load_target = _DM05_LOAD_TARGET_POLICY


def _build_dm05_norm_dataset_from_meta(config: DM05Config, dataset_meta: Any | None) -> Any | None:
    if dataset_meta is None:
        return None

    try:
        from lerobot.datasets import LeRobotDataset, resolve_delta_timestamps

        delta_timestamps = resolve_delta_timestamps(config, dataset_meta)
        return LeRobotDataset(
            repo_id=dataset_meta.repo_id,
            root=dataset_meta.root,
            revision=dataset_meta.revision,
            delta_timestamps=delta_timestamps,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to build a LeRobotDataset for DM05 norm stats auto-computation from dataset metadata."
        ) from exc


def _patch_gemma_tokenizer_image_token_id(fallback_image_token_id: int | None) -> None:
    token_defaults = {
        "image_token": "<image_soft_token>",
        "boi_token": "<start_of_image>",
        "eoi_token": "<end_of_image>",
    }

    def token_property(token_name: str, default_token: str):
        def getter(self):
            token = getattr(self, "init_kwargs", {}).get(token_name)
            if token is not None:
                return token
            model_tokens = getattr(self, "init_kwargs", {}).get("model_specific_special_tokens") or {}
            return model_tokens.get(token_name, default_token)

        return getter

    def token_id_property(token_name: str):
        def getter(self):
            token = getattr(self, token_name)
            token_id = self.convert_tokens_to_ids(token)
            unk_token_id = getattr(self, "unk_token_id", None)
            if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
                return token_id
            if token_name == "image_token" and fallback_image_token_id is not None:  # nosec B105
                return fallback_image_token_id
            raise AttributeError(f"{self.__class__.__name__} has no attribute {token_name}_id")

        return getter

    import importlib

    for module_name in (
        "transformers.models.gemma.tokenization_gemma",
        "transformers.models.gemma.tokenization_gemma_fast",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # nosec B112 - optional tokenizer modules may be unavailable
            continue
        for class_name in ("GemmaTokenizer", "GemmaTokenizerFast"):
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            for token_name, default_token in token_defaults.items():
                if not hasattr(cls, token_name):
                    setattr(cls, token_name, property(token_property(token_name, default_token)))
                token_id_name = f"{token_name}_id"
                if not hasattr(cls, token_id_name):
                    setattr(cls, token_id_name, property(token_id_property(token_name)))


class DM05Policy(PreTrainedPolicy):
    """LeRobot policy wrapper around the core DM05 model."""

    config_class = DM05Config
    name = "dm05"

    def __init__(self, config: DM05Config, **kwargs):
        require_package("transformers", extra="dm05")
        super().__init__(config)
        config.validate_features()
        self.config = config

        dataset_meta = kwargs.get("dataset_meta")

        if not config.pretrained_name_or_path:
            raise ValueError(
                "DM05Policy requires config.pretrained_name_or_path. "
                "DM05 is currently loaded from a core HF-style checkpoint."
            )
        use_bundled_norm_stats = (
            dataset_meta is None
            or config.norm_stats_path is not None
            or getattr(config, "pretrained_path", None) is not None
        )
        checkpoint_dir = _resolve_local_path(
            config.pretrained_name_or_path,
            getattr(config, "_dm05_checkpoint_dir", None),
        )
        if checkpoint_dir is not None and not checkpoint_dir.is_dir():
            checkpoint_dir = None
        _configure_dm05_checkpoint_paths(
            config,
            checkpoint_dir,
            use_bundled_norm_stats=use_bundled_norm_stats,
        )

        core_config_cls, core_model_cls, tokenization_cls = import_dm05_core()

        from transformers import AutoProcessor

        processor_path = config.processor_name_or_path or config.pretrained_name_or_path
        processor_config = _read_json(Path(processor_path) / "config.json")
        processor_core_config = (
            processor_config.get("core_config")
            if isinstance(processor_config.get("core_config"), dict)
            else processor_config
        )
        image_token_id = processor_core_config.get("vlm_config", {}).get("image_token_index")
        _patch_gemma_tokenizer_image_token_id(image_token_id if isinstance(image_token_id, int) else None)
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=config.trust_remote_code,
        )

        torch_dtype = resolve_torch_dtype(config.dtype)
        if str(config.device) == "cuda" and torch.cuda.is_available():
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                config.device = f"cuda:{int(local_rank)}"
        if getattr(config, "_dm05_core_config_only", False):
            if _has_dm05_core_config_payload(config):
                core_config = core_config_cls(**config.core_config)
            else:
                core_config = core_config_cls.from_pretrained(
                    config.pretrained_name_or_path,
                    trust_remote_code=config.trust_remote_code,
                )
            core_config.bf16 = torch_dtype is torch.bfloat16
            core_config.chunk_size = config.chunk_size
            core_config.vlm_gradient_checkpointing = bool(config.vlm_gradient_checkpointing)
            core_config.ae_gradient_checkpointing = bool(config.ae_gradient_checkpointing)
            core_config.gradient_checkpointing = bool(
                core_config.vlm_gradient_checkpointing or core_config.ae_gradient_checkpointing
            )
            if config.ae_gradient_checkpointing_layers is not None:
                core_config.ae_gradient_checkpointing_layers = int(config.ae_gradient_checkpointing_layers)
            core_config.llm_attn_implementation = config.llm_attn_implementation
            core_config.vision_attn_implementation = config.vision_attn_implementation
            core_config.action_attn_implementation = config.action_attn_implementation
            core_checkpoint = _resolve_local_path(
                config.pretrained_name_or_path,
                getattr(config, "_dm05_checkpoint_dir", None),
            )
            if (
                getattr(config, "_dm05_safetensors_load_target", None) == _DM05_LOAD_TARGET_CORE
                and core_checkpoint is not None
                and (core_checkpoint / SAFETENSORS_SINGLE_FILE).is_file()
            ):
                core_load_kwargs = {
                    "config": core_config,
                    "torch_dtype": torch_dtype,
                    "trust_remote_code": config.trust_remote_code,
                }
                self.model = core_model_cls.from_pretrained(
                    config.pretrained_name_or_path,
                    **core_load_kwargs,
                )
                config._dm05_weights_loaded_in_init = True
            else:
                from transformers.modeling_utils import no_init_weights

                previous_default_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch_dtype)
                try:
                    with no_init_weights():
                        self.model = core_model_cls(core_config)
                finally:
                    torch.set_default_dtype(previous_default_dtype)
        else:
            self.model = core_model_cls.from_pretrained(
                config.pretrained_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=config.trust_remote_code,
                chunk_size=config.chunk_size,
                vlm_gradient_checkpointing=config.vlm_gradient_checkpointing,
                ae_gradient_checkpointing=config.ae_gradient_checkpointing,
                ae_gradient_checkpointing_layers=config.ae_gradient_checkpointing_layers,
                llm_attn_implementation=config.llm_attn_implementation,
                vision_attn_implementation=config.vision_attn_implementation,
                action_attn_implementation=config.action_attn_implementation,
            )
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
        self._compile_suffix_active = False
        self._compile_suffix_warmed = False
        self._compile_suffix_active = setup_compiled_suffix(self.config, self.model)
        if not self._compile_suffix_active:
            self._disable_compile_suffix()
        self.normalizer = resolve_dm05_normalizer(config, required=False)
        if self.normalizer is None:
            norm_dataset = _build_dm05_norm_dataset_from_meta(config, dataset_meta)
            if norm_dataset is not None:
                self.normalizer = resolve_dm05_normalizer(config, dataset=norm_dataset, required=True)
        if self.normalizer is not None and getattr(config, "_dm05_norm_stats_payload", None) is None:
            to_payload = getattr(self.normalizer, "to_payload", None)
            if callable(to_payload):
                config._dm05_norm_stats_payload = to_payload()
        self._batch_converter = DM05LerobotBatchConverter(
            config=config,
            tokenization_cls=tokenization_cls,
            processor=self.processor,
            normalizer=self.normalizer,
        )
        self.reset()

    def _disable_compile_suffix(self) -> None:
        self._compile_suffix_active = False
        self._compile_suffix_warmed = False
        if hasattr(self.model, "clear_compiled_suffix_layers"):
            self.model.clear_compiled_suffix_layers()

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        saved_core = _copy_dm05_reference_assets(self.config.pretrained_name_or_path, save_directory)
        processor_source = self.config.processor_name_or_path or self.config.pretrained_name_or_path
        saved_processor = _copy_dm05_reference_assets(processor_source, save_directory)
        if not saved_processor and (save_directory / "processor_config.json").is_file():
            saved_processor = True
        saved_norm = False
        norm_target = save_directory / _BUNDLED_DM05_NORM_STATS
        norm_path = _resolve_local_path(
            getattr(self.config, "norm_stats_path", None),
            getattr(self.config, "_dm05_checkpoint_dir", None),
        )
        if norm_path is not None and norm_path.is_file():
            if norm_path.resolve() != norm_target.resolve():
                shutil.copy2(norm_path, norm_target)
            saved_norm = True
        elif (norm_payload := getattr(self.config, "_dm05_norm_stats_payload", None)) is not None:
            norm_target.write_text(json.dumps(norm_payload, indent=2) + "\n", encoding="utf-8")
            saved_norm = True
        model_to_save = self.module.model if hasattr(self, "module") else self.model

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
        saved_core_config = False
        if core_config:
            save_config.core_config = core_config
            saved_core_config = bool(core_config.get("model_type"))
        if saved_core or saved_core_config:
            save_config.pretrained_name_or_path = "."
        if saved_processor:
            save_config.processor_name_or_path = "."
        if saved_norm:
            save_config.norm_stats_path = _BUNDLED_DM05_NORM_STATS
        for name in tuple(vars(save_config)):
            if name.startswith("_dm05_"):
                delattr(save_config, name)
        save_config._save_pretrained(save_directory)

        if state_dict is None:
            save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))
            return

        core_state_dict = _select_dm05_core_state_dict(model_to_save, state_dict)
        total_bytes = sum(t.numel() * t.element_size() for t in core_state_dict.values())
        save_torch_state_dict(core_state_dict, str(save_directory), max_shard_size=max(total_bytes, 1))

    def push_model_to_hub(self, cfg, peft_model=None):
        api = HfApi()
        repo_id = api.create_repo(
            repo_id=self.config.repo_id, private=self.config.private, exist_ok=True
        ).repo_id

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id
            if peft_model is not None:
                peft_model.save_pretrained(saved_path)
                self.config.save_pretrained(saved_path)
            else:
                self.save_pretrained(saved_path)

            card = self.generate_model_card(
                cfg.dataset.repo_id, self.config.type, self.config.license, self.config.tags
            )
            card.save(str(saved_path / "README.md"))
            cfg.save_pretrained(saved_path)
            commit_info = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message="Upload policy weights, train config and readme",
                allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md", "*.jinja"],
                ignore_patterns=["*.tmp", "*.log"],
            )

            logging.info(f"Model pushed to {commit_info.repo_url.url}")

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
        strict: bool = False,
        **kwargs,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        is_training_init = kwargs.get("dataset_meta") is not None
        if config is None:
            try:
                config = PreTrainedConfig.from_pretrained(
                    pretrained_name_or_path=pretrained_name_or_path,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    revision=revision,
                    **kwargs,
                )
            except Exception:
                config = DM05Config(pretrained_name_or_path=model_id)
        checkpoint_dir = Path(model_id) if Path(model_id).is_dir() else None
        if isinstance(config, DM05Config):
            train_policy_path_init = (
                kwargs.get("dataset_meta") is not None
                and checkpoint_dir is not None
                and str(getattr(config, "pretrained_name_or_path", "")) == "."
                and (getattr(config, "processor_name_or_path", None) in {None, "."})
                and _has_dm05_core_config_payload(config)
                and (checkpoint_dir / SAFETENSORS_SINGLE_FILE).is_file()
            )
            if not config.pretrained_name_or_path:
                config.pretrained_name_or_path = model_id
            if checkpoint_dir is not None:
                config._dm05_checkpoint_dir = str(checkpoint_dir)
                _configure_dm05_checkpoint_paths(config, checkpoint_dir)
            if train_policy_path_init:
                config.pretrained_path = None
                config.norm_stats_path = None
        policy = cls(config, **kwargs)

        if Path(model_id).is_dir():
            model_file = Path(model_id) / SAFETENSORS_SINGLE_FILE
        else:
            model_file = Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            )
        if model_file.is_file() and not getattr(config, "_dm05_weights_loaded_in_init", False):
            load_target_name = getattr(config, "_dm05_safetensors_load_target", None)
            if load_target_name is not None:
                load_target = policy.model if load_target_name == _DM05_LOAD_TARGET_CORE else policy
                missing_keys, unexpected_keys = load_model_as_safetensor(
                    load_target,
                    str(model_file),
                    strict=strict,
                    device=config.device,
                )
                log_model_loading_keys(missing_keys, unexpected_keys)
        if isinstance(config, DM05Config) and is_training_init:
            config.pretrained_path = None
        policy.to(config.device)
        policy.eval()
        return policy

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self):
        return self.parameters()

    def _prepare_model_inputs(self, batch: dict[str, Any], include_labels: bool) -> dict[str, Any]:
        model_inputs = self._batch_converter.convert_lerobot_batch(batch, include_labels=include_labels)
        model_inputs = {
            key: value.to(self.config.device) if isinstance(value, Tensor) else value
            for key, value in model_inputs.items()
        }
        dtype = next((p.dtype for p in self.model.parameters() if p.is_floating_point()), torch.float32)
        for key in ("states", "actions", "action", "prefill_actions"):
            value = model_inputs.get(key)
            if torch.is_tensor(value) and value.is_floating_point():
                model_inputs[key] = value.to(dtype=dtype)
        return model_inputs

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        model_inputs = self._prepare_model_inputs(batch, include_labels=True)
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        loss_dict = {"loss": loss.item() if loss is not None else 0.0}
        # Report only flow-matching loss; AR loss is not used here.
        for key in ("fm_loss",):
            value = getattr(outputs, key, None)
            if value is not None:
                loss_dict[key] = value.detach().float().item()
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        model_inputs = self._prepare_model_inputs(batch, include_labels=False)
        diffusion_steps = kwargs.get("diffusion_steps", self.config.diffusion_steps)
        call_inputs = model_inputs
        inference_kwargs = {}
        if self._compile_suffix_active:
            model_dtype = next(
                (p.dtype for p in self.model.parameters() if p.is_floating_point()), torch.float32
            )
            try:
                if not self._compile_suffix_warmed:
                    warmup_steps = int(getattr(self.config, "compile_suffix_warmup_steps", 0) or 0)
                    if warmup_steps <= 0:
                        self._compile_suffix_warmed = True
                    else:
                        self._compile_suffix_warmed = warmup_compiled_suffix(
                            self.config,
                            self.model,
                            self.processor,
                            model_inputs,
                            diffusion_steps=diffusion_steps,
                            dtype=model_dtype,
                        )
            except Exception:
                if self.config.compile_suffix == "auto":
                    logger.exception("DM05 compiled suffix warmup failed; falling back to eager.")
                    self._disable_compile_suffix()
                else:
                    raise
            if self._compile_suffix_active:
                compiled_inputs = prepare_compiled_suffix_inputs(
                    self.config,
                    self.model,
                    self.processor,
                    model_inputs,
                    dtype=model_dtype,
                )
                if compiled_inputs is not None:
                    call_inputs, inference_kwargs = compiled_inputs
        try:
            actions = self.model.inference_action(
                **call_inputs,
                **inference_kwargs,
                diffusion_steps=diffusion_steps,
            )
        except Exception:
            if self._compile_suffix_active and self.config.compile_suffix == "auto":
                logger.exception("DM05 compiled suffix inference failed; falling back to eager.")
                self._disable_compile_suffix()
                actions = self.model.inference_action(
                    **model_inputs,
                    diffusion_steps=diffusion_steps,
                )
            else:
                raise
        if self.normalizer is not None:
            actions = self.normalizer.denormalize_action(actions)
        action_feature = self.config.output_features.get(ACTION) if self.config.output_features else None
        action_dim = (
            int(action_feature.shape[-1])
            if action_feature is not None and action_feature.shape
            else int(self.config.max_action_dim)
        )
        if getattr(self.config, "use_absolute_action", False):
            state = batch.get(OBS_STATE)
            if state is not None:
                if not torch.is_tensor(state):
                    state = torch.as_tensor(state)
                state = state.to(device=self.config.device, dtype=torch.float32)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                if state.dim() > 2:
                    state = state.reshape(state.shape[0], -1)
                if state.shape[0] != int(actions.shape[0]):
                    state = None
            if state is None:
                raise ValueError("DM05 use_absolute_action=True requires raw observation.state.")
            actions = command_action_from_relative(
                state,
                actions,
                action_dim=action_dim,
                non_delta_indices=infer_dm05_non_delta_indices(self.config),
            )
        return actions[:, :, :action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()
