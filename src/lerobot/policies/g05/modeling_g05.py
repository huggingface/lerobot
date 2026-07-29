# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""First-class LeRobot wrapper around the community-licensed G0.5 implementation."""

from __future__ import annotations

import importlib
import json
import shutil
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import OptimizerParams
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_g05 import (
    G05_POLICY_PARTS,
    G05Config,
    make_g05_cot_prompt_template,
    make_g05_prompt_template,
)


def _author_backend(config: G05Config) -> nn.Module:
    if not config.author_model_config:
        raise ValueError(
            "G0.5 author_model_config is empty. Load a packaged checkpoint, or "
            "inject a backend explicitly for testing."
        )
    try:
        from omegaconf import OmegaConf

        module = importlib.import_module("g05.models.g05.g05_policy_qwen35")
    except ImportError as exc:
        raise ImportError(
            "The OpenGalaxea G0.5 author package is required for real model execution. "
            "Clone the pinned GalaxeaVLA source, accept LICENSE-G0.5, and install its "
            "runtime dependencies in a compatible environment. LeRobot does not vendor "
            "or silently download that non-commercial code."
        ) from exc
    backend_cls = module.G05PolicyQwen35
    return backend_cls(**OmegaConf.to_container(OmegaConf.create(config.author_model_config)))


class G05Policy(PreTrainedPolicy):
    """LeRobot policy surface for G0.5's unified CoT and action stream."""

    config_class = G05Config
    name = "g05"

    def __init__(self, config: G05Config, backend: nn.Module | None = None):
        super().__init__(config)
        config.validate_features()
        self.backend = backend if backend is not None else _author_backend(config)
        if not isinstance(self.backend, nn.Module):
            raise TypeError(f"G0.5 backend must be an nn.Module, got {type(self.backend)}.")
        self._action_queue: deque[Tensor] = deque()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: G05Config | None = None,
        **kwargs,
    ) -> G05Policy:
        resolved_path = Path(pretrained_name_or_path)
        if not resolved_path.is_dir():
            resolved_path = Path(
                snapshot_download(
                    repo_id=str(pretrained_name_or_path),
                    token=kwargs.get("token"),
                    cache_dir=kwargs.get("cache_dir"),
                    local_files_only=kwargs.get("local_files_only", False),
                    revision=kwargs.get("revision"),
                )
            )
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                resolved_path,
                token=kwargs.get("token"),
                cache_dir=kwargs.get("cache_dir"),
                local_files_only=kwargs.get("local_files_only", False),
                revision=kwargs.get("revision"),
            )
        if not isinstance(config, G05Config):
            raise TypeError(f"Expected a G05Config, got {type(config).__name__}.")
        author_config = dict(config.author_model_config)
        author_config["hf_processor_path"] = str(resolved_path / "hf_processor")
        at_config = dict(author_config.get("AT_CONFIG") or {})
        at_config["ckpt_dir"] = str(resolved_path / "action_tokenizer.pt")
        author_config["AT_CONFIG"] = at_config
        author_config["pretrained_model_path"] = None
        config.author_model_config = author_config
        return super().from_pretrained(
            resolved_path,
            config=config,
            **kwargs,
        )

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        super()._save_pretrained(save_directory, state_dict=state_dict)
        author_config = dict(self.config.author_model_config)
        processor_value = author_config.get("hf_processor_path")
        processor_path = Path(str(processor_value)) if processor_value else None
        at_config = dict(author_config.get("AT_CONFIG") or {})
        tokenizer_value = at_config.get("ckpt_dir")
        tokenizer_path = Path(str(tokenizer_value)) if tokenizer_value else None
        roots = [
            path.parent for path in (processor_path, tokenizer_path) if path is not None and path.exists()
        ]

        if (
            processor_path is not None
            and processor_path.is_dir()
            and processor_path.resolve() != (save_directory / "hf_processor").resolve()
        ):
            shutil.copytree(processor_path, save_directory / "hf_processor", dirs_exist_ok=True)
        if (
            tokenizer_path is not None
            and tokenizer_path.is_file()
            and tokenizer_path.resolve() != (save_directory / "action_tokenizer.pt").resolve()
        ):
            shutil.copy2(tokenizer_path, save_directory / "action_tokenizer.pt")
        for name in (
            "g05_dataset_stats.json",
            "author_config.yaml",
            "LICENSE-G0.5",
            "LICENSE_QWEN3_5.txt",
            "THIRD_PARTY_NOTICES.md",
            "NOTICE",
            "README.md",
        ):
            source = next((root / name for root in roots if (root / name).is_file()), None)
            if source is not None and source.resolve() != (save_directory / name).resolve():
                shutil.copy2(source, save_directory / name)

        # Serialized paths are portable sidecar names. Local/Hub loading resolves them
        # against the downloaded checkpoint directory before constructing the author model.
        if (processor_path is not None and processor_path.exists()) or (
            tokenizer_path is not None and tokenizer_path.exists()
        ):
            portable = dict(author_config)
            portable["hf_processor_path"] = "hf_processor"
            portable_at = dict(portable.get("AT_CONFIG") or {})
            portable_at["ckpt_dir"] = "action_tokenizer.pt"
            portable["AT_CONFIG"] = portable_at
            portable["pretrained_model_path"] = None
            runtime_config = self.config.author_model_config
            self.config.author_model_config = portable
            self.config._save_pretrained(save_directory)
            self.config.author_model_config = runtime_config

    def reset(self) -> None:
        self._action_queue.clear()
        reset = getattr(self.backend, "reset", None)
        if callable(reset):
            reset()

    def _apply_author_inference_precision(self) -> None:
        """Match the released serving path's BF16 weights with declared FP32 islands."""

        self.backend.to(dtype=torch.bfloat16)
        apply_fp32_params = getattr(self.backend, "apply_fp32_params", None)
        if callable(apply_fp32_params):
            apply_fp32_params()

    def to(self, *args, **kwargs) -> G05Policy:
        """Apply author inference precision and move the ActionCodec sidecar."""

        result = super().to(*args, **kwargs)
        explicit_dtype = "dtype" in kwargs or any(isinstance(arg, torch.dtype | Tensor) for arg in args)
        if (
            self.config.model_weights_to_bf16
            and not explicit_dtype
            and next(self.backend.parameters()).device.type == "cuda"
        ):
            self._apply_author_inference_precision()
        action_tokenizer = getattr(self.backend, "action_tokenizer", None)
        move_tokenizer = getattr(action_tokenizer, "to", None)
        if callable(move_tokenizer):
            device = next(self.backend.parameters()).device
            move_tokenizer(device)
        return result

    def get_optim_params(self) -> OptimizerParams:
        get_param_groups = getattr(self.backend, "get_optim_param_groups", None)
        if callable(get_param_groups):
            return get_param_groups(
                lr=self.config.optimizer_lr,
                weight_decay=self.config.optimizer_weight_decay,
                apply_decay_on_norm_and_bias=self.config.optimizer_apply_decay_on_norm_and_bias,
                backbone_lr_multiplier=self.config.optimizer_backbone_lr_multiplier,
                vision_lr_multiplier=self.config.optimizer_vision_lr_multiplier,
            )
        get_params = getattr(self.backend, "get_optim_params", None)
        if callable(get_params):
            params = get_params()
            return [params] if isinstance(params, dict) and "params" in params else params
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @staticmethod
    def _task_values(batch: Mapping[str, Any], task: str | None, batch_size: int) -> list[str]:
        if task is not None:
            return [task] * batch_size
        value = batch.get("task")
        if isinstance(value, str):
            return [value] * batch_size
        if isinstance(value, list | tuple) and len(value) == batch_size:
            return [str(item) for item in value]
        raise ValueError(
            "G0.5 requires the already-selected LeRobot task string; no task augmentation "
            "or model-local sampling is performed."
        )

    @staticmethod
    def _batch_item(value: Any, index: int, batch_size: int) -> Any:
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            return value[index]
        if isinstance(value, list | tuple) and len(value) == batch_size:
            return value[index]
        return value

    def _recipe_cot_targets(
        self,
        batch: Mapping[str, Any],
        index: int,
        batch_size: int,
    ) -> tuple[str | None, str | None]:
        """Read the selected recipe's supervised Subtask/BBox messages."""

        messages = batch.get("messages")
        target_indices = batch.get("target_message_indices")
        if messages is None or target_indices is None:
            return None, None

        sample_messages = messages
        if (
            isinstance(messages, list | tuple)
            and len(messages) == batch_size
            and (not messages or isinstance(messages[0], list | tuple))
        ):
            sample_messages = messages[index]
        sample_target_indices = target_indices
        has_batched_target_indices = (isinstance(target_indices, Tensor) and target_indices.ndim > 1) or (
            isinstance(target_indices, list | tuple)
            and len(target_indices) == batch_size
            and (not target_indices or isinstance(target_indices[0], list | tuple | Tensor))
        )
        if has_batched_target_indices:
            sample_target_indices = target_indices[index]
        if isinstance(sample_messages, Mapping):
            sample_messages = [sample_messages]
        if isinstance(sample_target_indices, Tensor):
            sample_target_indices = sample_target_indices.detach().cpu().tolist()
        if not isinstance(sample_messages, list | tuple) or not isinstance(
            sample_target_indices, list | tuple
        ):
            return None, None

        subtask: str | None = None
        bbox_json: str | None = None
        for target_index in sample_target_indices:
            message = sample_messages[int(target_index)]
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str):
                continue
            if content.startswith("Subtask:"):
                value = content.removeprefix("Subtask:").strip()
                if value:
                    subtask = value
            elif content.startswith("BBoxJSON:"):
                value = content.removeprefix("BBoxJSON:").strip()
                if value:
                    bbox_json = value
        return subtask, bbox_json

    @staticmethod
    def _format_bbox_target(bbox_json: str | None, image_size: tuple[int, int]) -> str | None:
        """Convert LeRobot grounded-VQA JSON into G0.5's location-token format."""

        if not bbox_json:
            return None
        try:
            payload = json.loads(bbox_json)
            if isinstance(payload, str):
                payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        if isinstance(payload.get("answer"), Mapping):
            payload = payload["answer"]

        height, width = image_size
        boxes: list[tuple[str, list[float]]] = []
        detections = payload.get("detections")
        if isinstance(detections, list):
            for detection in detections:
                if not isinstance(detection, Mapping) or detection.get("bbox_format", "xyxy") != "xyxy":
                    continue
                coords = detection.get("bbox")
                if not isinstance(coords, list | tuple) or len(coords) != 4:
                    continue
                label = str(detection.get("label") or "object")
                boxes.append((label, [float(value) for value in coords]))
        else:
            for label, coords in payload.items():
                if isinstance(coords, list | tuple) and len(coords) == 4:
                    boxes.append((str(label), [float(value) for value in coords]))
        if not boxes:
            return None

        def normalize(coords: list[float]) -> list[float]:
            if max(abs(value) for value in coords) <= 1.0:
                return coords
            x1, y1, x2, y2 = coords
            return [x1 / width, y1 / height, x2 / width, y2 / height]

        def location_token(value: float) -> str:
            location = max(0, min(1023, round(value * 1024)))
            return f"<loc{location:04d}>"

        formatted = []
        for label, raw_coords in boxes:
            x1, y1, x2, y2 = normalize(raw_coords)
            locations = "".join(location_token(value) for value in (y1, x1, y2, x2))
            formatted.append(f"{label} {locations}")
        return "BBox: " + "; ".join(formatted)

    def _apply_recipe_cot(
        self,
        sample: dict[str, Any],
        batch: Mapping[str, Any],
        index: int,
        batch_size: int,
    ) -> bool:
        """Populate one author sample from recipe-rendered CoT targets."""

        subtask, bbox_json = self._recipe_cot_targets(batch, index, batch_size)
        image_size = batch.get("g05_bbox_image_size")
        if (
            isinstance(image_size, list | tuple)
            and len(image_size) == batch_size
            and image_size
            and isinstance(image_size[0], list | tuple | Tensor)
        ):
            image_size = image_size[index]
        if isinstance(image_size, Tensor):
            image_size = image_size.detach().cpu().tolist()
        if not isinstance(image_size, list | tuple) or len(image_size) != 2:
            camera = self.config.cot_bbox_camera or self.config.camera_order[0]
            image_size = self.config.camera_sizes[camera]
        bbox = self._format_bbox_target(bbox_json, (int(image_size[0]), int(image_size[1])))

        fields = tuple(field for field, value in (("bbox", bbox), ("subtask", subtask)) if value)
        if not fields:
            return False
        flow_only = "<action_action" not in self.config.prompt_template
        sample["template"] = make_g05_cot_prompt_template(
            self.config.num_prompt_images,
            fields=fields,
            flow_only=flow_only,
        )
        if bbox is not None:
            sample["bbox"] = bbox
        if subtask is not None:
            sample["atomic_task"] = f"Subtask: {subtask}"
        sample["prompt"] = {
            ("bbox",): "predict bbox",
            ("subtask",): "predict subtask",
            ("bbox", "subtask"): "predict bbox, subtask and action",
        }[fields]
        return True

    def _prepare_author_batch(self, batch: Mapping[str, Any], task: str | None = None) -> dict[str, Any]:
        prepare = getattr(self.backend, "prepare_lerobot_batch", None)
        if callable(prepare):
            return prepare(batch, task=task, config=self.config)

        state = batch.get(OBS_STATE)
        if not isinstance(state, Tensor):
            raise ValueError(f"G0.5 requires tensor {OBS_STATE!r}.")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        tasks = self._task_values(batch, task, batch_size)
        state_mask = batch.get("proprio_dim_is_pad")
        if state_mask is None:
            state_mask = torch.zeros(
                batch_size, self.config.policy_state_dim, dtype=torch.bool, device=state.device
            )
        elif isinstance(state_mask, Tensor) and state_mask.ndim == 1:
            state_mask = state_mask.unsqueeze(0).expand(batch_size, -1)

        pixel_values: dict[str, Tensor] = {}
        for key in self.config.camera_order:
            image = batch.get(key)
            if not isinstance(image, Tensor):
                raise ValueError(f"G0.5 requires camera {key!r}; camera order is checkpoint state.")
            if image.ndim == 4:
                image = image.unsqueeze(1)
            pixel_values[key] = image
        image_count = sum(image.shape[1] for image in pixel_values.values())
        if image_count != self.config.num_input_images:
            raise ValueError(
                f"G0.5 received {image_count} camera/history frames, but the checkpoint "
                f"template requires {self.config.num_input_images}."
            )

        samples = []
        for index, raw_task in enumerate(tasks):
            proprio = state[index]
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            sample = {
                "template": self.config.prompt_template,
                # This is the author InputPreprocessor command slot. Keep it byte-for-byte
                # unchanged; checkpoint-specific chat formatting occurs downstream.
                "command": raw_task,
                "embodiment": self.config.embodiment,
                "proprio": {
                    "value": proprio,
                    "proprio_dim_is_pad": state_mask[index],
                },
            }
            frequency = self.config.processor_metadata.get("frequency")
            if frequency is not None:
                sample["frequency"] = frequency
            if self.config.predict_cot:
                rendered_recipe = "messages" in batch
                applied_recipe_cot = rendered_recipe and self._apply_recipe_cot(
                    sample, batch, index, batch_size
                )
                if not applied_recipe_cot:
                    # During mixed-recipe training an applicable no-CoT branch is a
                    # genuine target format. At inference, where actions are absent,
                    # retain the checkpoint's configured System 2 prompt.
                    if rendered_recipe and isinstance(batch.get(ACTION), Tensor):
                        sample["template"] = make_g05_prompt_template(
                            self.config.num_prompt_images,
                            predict_cot=False,
                            flow_only="<action_action" not in self.config.prompt_template,
                        )
                    else:
                        sample["prompt"] = "predict subtask"
                        atomic_task = batch.get("atomic_task")
                        if atomic_task is not None:
                            atomic_task = str(self._batch_item(atomic_task, index, batch_size))
                            sample["atomic_task"] = (
                                atomic_task
                                if atomic_task.startswith("Subtask:")
                                else f"Subtask: {atomic_task}"
                            )
            for image_index in range(self.config.num_prompt_images):
                camera = self.config.camera_order[image_index % len(self.config.camera_order)]
                sample[f"image{image_index}"] = self.config.camera_sizes[camera]
            action = batch.get(ACTION)
            if "<action_action" in self.config.prompt_template:
                if not isinstance(action, Tensor):
                    action = state.new_zeros(
                        batch_size, self.config.chunk_size, self.config.policy_action_dim
                    )
                action_dim_is_pad = batch.get("action_dim_is_pad")
                if action_dim_is_pad is None:
                    action_dim_is_pad = torch.zeros(
                        batch_size,
                        self.config.policy_action_dim,
                        dtype=torch.bool,
                        device=action.device,
                    )
                elif action_dim_is_pad.ndim == 1:
                    action_dim_is_pad = action_dim_is_pad.unsqueeze(0).expand(batch_size, -1)
                action_payload = {
                    "value": action[index],
                    "action_dim_is_pad": action_dim_is_pad[index],
                }
                action_op_mask = batch.get("action_op_mask")
                if isinstance(action_op_mask, Tensor):
                    action_payload["action_op_mask"] = (
                        action_op_mask[index] if action_op_mask.ndim > 1 else action_op_mask
                    )
                else:
                    action_payload["action_op_mask"] = ~action_dim_is_pad[index]
                action_payload["parts_meta"] = batch.get(
                    "action_parts_meta", G05_POLICY_PARTS[self.config.policy_action_dim]
                )
                sample["action"] = action_payload
            samples.append(sample)
        prepared = dict(batch)
        prepared["samples"] = samples
        prepared["pixel_values"] = pixel_values
        return prepared

    def _run_inference(
        self, batch: Mapping[str, Any], *, task: str | None = None
    ) -> tuple[Tensor, dict[str, Any]]:
        prepared = self._prepare_author_batch(batch, task=task)
        predict = getattr(self.backend, "predict_action", None)
        device = next(self.backend.parameters()).device
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.model_weights_to_bf16 and device.type == "cuda",
        ):
            result = predict(prepared) if callable(predict) else self.backend(prepared)
        if isinstance(result, Tensor):
            result = {ACTION: result}
        if not isinstance(result, Mapping):
            raise TypeError("G0.5 backend inference must return a tensor or mapping.")

        if self.config.action_head == "actioncodec":
            action = result.get("ar_action", result.get(ACTION))
        else:
            action = result.get(ACTION)
        if not isinstance(action, Tensor):
            raise ValueError(f"G0.5 {self.config.action_head} output is missing its action tensor.")
        metadata = {
            key: result[key]
            for key in ("cot_text", "generated_ids", "decoded_action_tokens", "ar_absent_keys", "_timing")
            if key in result
        }
        return action, metadata

    def predict_action_chunk_with_runtime(
        self, batch: dict[str, Any], *, task: str
    ) -> tuple[Tensor, dict[str, Any]]:
        """Return System 1 actions and same-pass System 2 telemetry atomically."""

        return self._run_inference(batch, task=task)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> Tensor:
        action, _ = self._run_inference(batch)
        return action

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            if chunk.ndim != 3:
                raise ValueError(f"G0.5 action chunk must be [B,T,D], got {tuple(chunk.shape)}.")
            # LeRobot's synchronous select_action queue is intentionally batch-size one.
            if chunk.shape[0] != 1:
                raise ValueError(
                    "G0.5 select_action requires batch size 1; use predict_action_chunk for B>1."
                )
            self._action_queue.extend(chunk[0, : self.config.n_action_steps])
        return self._action_queue.popleft().unsqueeze(0)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any] | None]:
        prepared = self._prepare_author_batch(batch)
        device = next(self.backend.parameters()).device
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.model_weights_to_bf16 and device.type == "cuda",
        ):
            result = self.backend(prepared)
        if isinstance(result, tuple) and len(result) == 2:
            loss, loss_dict = result
        elif isinstance(result, Mapping) and "loss" in result:
            loss = result["loss"]
            loss_dict = {key: value for key, value in result.items() if key != "loss"}
        else:
            raise TypeError("G0.5 training backend must return (loss, loss_dict) or {'loss': ...}.")
        if not isinstance(loss, Tensor):
            raise TypeError("G0.5 training loss must be a torch.Tensor.")
        logging_values = {
            key: value.detach().item() if isinstance(value, Tensor) and value.numel() == 1 else value
            for key, value in (loss_dict or {}).items()
        }
        return loss, logging_values
