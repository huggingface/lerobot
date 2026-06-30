# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""MolmoAct2 pre/post processing pipeline.

Builds the multimodal prompt (images, discretised state, task text),
tokenises it via the vendored MolmoAct2 processor, and handles quantile
normalisation with optional per-dimension gripper masking.
"""

from __future__ import annotations

import json
import logging
import math
import re
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _scipy_available, _transformers_available, require_package

from .configuration_molmoact2 import MolmoAct2Config
from .modeling_molmoact2 import _hf_token, _resolve_checkpoint_location

logger = logging.getLogger(__name__)

MOLMOACT2_DEFAULT_NUM_IMAGES = 2
MOLMOACT2_IMAGE_TOKENS_PER_IMAGE = 196
MOLMOACT2_FIXED_PROMPT_TOKEN_BUDGET = 80
MOLMOACT2_TASK_TOKEN_BUDGET = 32
MOLMOACT2_SEQUENCE_LENGTH_MARGIN = 32
MOLMOACT2_SEQUENCE_LENGTH_MULTIPLE = 64
MOLMOACT2_DISCRETE_ACTION_WRAPPER_TOKENS = 4
MOLMOACT2_MIN_DISCRETE_ACTION_TOKENS_PER_STEP = 6
MOLMOACT2_DISCRETE_ACTION_TOKENS_PER_DIM = 0.95


def _round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def infer_molmoact2_max_sequence_length(
    *,
    num_images: int,
    state_dim: int,
    action_dim: int,
    action_horizon: int,
    include_discrete_action: bool,
) -> int:
    """Infer the padded text/image sequence cap from MolmoAct2's fixed token layout."""
    if num_images < 1:
        num_images = MOLMOACT2_DEFAULT_NUM_IMAGES
    if state_dim < 0:
        state_dim = 0
    if action_dim < 1:
        action_dim = 1
    if action_horizon < 1:
        action_horizon = 1

    image_tokens = num_images * MOLMOACT2_IMAGE_TOKENS_PER_IMAGE
    prompt_tokens = (
        MOLMOACT2_FIXED_PROMPT_TOKEN_BUDGET
        + MOLMOACT2_TASK_TOKEN_BUDGET
        + state_dim
        + MOLMOACT2_SEQUENCE_LENGTH_MARGIN
    )
    action_tokens = 0
    if include_discrete_action:
        action_tokens_per_step = max(
            MOLMOACT2_MIN_DISCRETE_ACTION_TOKENS_PER_STEP,
            math.ceil(action_dim * MOLMOACT2_DISCRETE_ACTION_TOKENS_PER_DIM),
        )
        action_tokens = MOLMOACT2_DISCRETE_ACTION_WRAPPER_TOKENS + action_horizon * action_tokens_per_step

    return _round_up(
        image_tokens + prompt_tokens + action_tokens,
        MOLMOACT2_SEQUENCE_LENGTH_MULTIPLE,
    )


if TYPE_CHECKING or _transformers_available:
    from transformers import Qwen2Tokenizer

    from .molmoact2_hf_model.image_processing_molmoact2 import MolmoAct2ImageProcessor
    from .molmoact2_hf_model.processing_molmoact2 import MolmoAct2Processor
    from .molmoact2_hf_model.video_processing_molmoact2 import MolmoAct2VideoProcessor
else:
    Qwen2Tokenizer = None
    MolmoAct2ImageProcessor = None
    MolmoAct2Processor = None
    MolmoAct2VideoProcessor = None

if TYPE_CHECKING or (_transformers_available and _scipy_available):
    from .molmoact2_hf_model.action_tokenizer import UniversalActionProcessor
else:
    UniversalActionProcessor = None

ACTION_OUTPUT_TOKEN = "<action_output>"  # nosec B105
ACTION_START_TOKEN = "<action_start>"  # nosec B105
ACTION_END_TOKEN = "<action_end>"  # nosec B105
ACTION_TOKEN_PREFIX = "<action_"  # nosec B105
STATE_START_TOKEN = "<state_start>"  # nosec B105
STATE_END_TOKEN = "<state_end>"  # nosec B105
STATE_TOKEN_PREFIX = "<state_"  # nosec B105
SETUP_START_TOKEN = "<setup_start>"  # nosec B105
SETUP_END_TOKEN = "<setup_end>"  # nosec B105
CONTROL_START_TOKEN = "<control_start>"  # nosec B105
CONTROL_END_TOKEN = "<control_end>"  # nosec B105

_QUESTION_TRAILING_SENTENCE_PUNCTUATION = ".,!?;:,\u2026"
_QUESTION_TRAILING_CLOSERS = "\"'\u201d\u2019)]}"
_QUESTION_SURROUNDING_DELIMITERS = "\"'`\u201c\u201d\u2018\u2019[](){}"
_QUESTION_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def _load_hf_norm_stats_for_tag(
    checkpoint_path: str,
    *,
    revision: str | None,
    force_download: bool,
    norm_tag: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    norm_tag = str(norm_tag or "").strip()
    if not norm_tag:
        raise ValueError("MolmoAct2 HF checkpoint inference requires `policy.norm_tag` for normalization.")

    checkpoint_location = Path(
        _resolve_checkpoint_location(
            checkpoint_path,
            revision=revision,
            force_download=force_download,
        )
    )
    config_path = checkpoint_location / "config.json"
    norm_stats_filename = "norm_stats.json"
    if config_path.exists():
        with suppress(OSError, json.JSONDecodeError):
            norm_stats_filename = str(
                json.loads(config_path.read_text()).get("norm_stats_filename") or norm_stats_filename
            )

    stats_path = checkpoint_location / norm_stats_filename
    if not stats_path.exists():
        raise FileNotFoundError(
            f"MolmoAct2 HF checkpoint is missing {norm_stats_filename!r}; cannot resolve norm_tag={norm_tag!r}."
        )
    payload = json.loads(stats_path.read_text())
    metadata_by_tag = payload.get("metadata_by_tag")
    if not isinstance(metadata_by_tag, dict):
        raise ValueError(f"MolmoAct2 norm stats file {stats_path} has no metadata_by_tag mapping.")
    metadata = metadata_by_tag.get(norm_tag)
    if metadata is None:
        available = sorted(str(tag) for tag in metadata_by_tag)
        raise ValueError(f"Unknown MolmoAct2 norm_tag={norm_tag!r}. Available tags: {available}.")
    if not isinstance(metadata, dict):
        raise ValueError(f"MolmoAct2 norm_tag={norm_tag!r} metadata must be a mapping.")

    def numeric_stats(raw_stats: dict[str, Any]) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for key, value in raw_stats.items():
            if key == "names":
                continue
            if isinstance(value, (list, tuple)) and any(isinstance(item, str) for item in value):
                continue
            stats[key] = deepcopy(value)
        return stats

    action_stats = metadata.get("action_stats")
    state_stats = metadata.get("state_stats")
    if not isinstance(action_stats, dict) or not isinstance(state_stats, dict):
        raise ValueError(f"MolmoAct2 norm_tag={norm_tag!r} must define action_stats and state_stats.")
    return {ACTION: numeric_stats(action_stats), OBS_STATE: numeric_stats(state_stats)}, metadata


def _strip_processor_config(config: dict[str, Any], *metadata_keys: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"auto_map", "processor_class", *metadata_keys}
    }


def _load_local_molmoact2_processor(checkpoint_location: str) -> Any:
    if (
        Qwen2Tokenizer is None
        or MolmoAct2ImageProcessor is None
        or MolmoAct2Processor is None
        or MolmoAct2VideoProcessor is None
    ):
        raise RuntimeError("transformers is required to load MolmoAct2 processor.")

    checkpoint_path = Path(checkpoint_location)
    processor_config_path = checkpoint_path / "processor_config.json"
    if not processor_config_path.exists():
        raise FileNotFoundError(f"MolmoAct2 checkpoint is missing {processor_config_path}.")
    processor_config = json.loads(processor_config_path.read_text())

    image_config = _strip_processor_config(
        dict(processor_config.get("image_processor") or {}),
        "image_processor_type",
    )
    video_config = _strip_processor_config(
        dict(processor_config.get("video_processor") or {}),
        "video_processor_type",
    )
    image_processor = MolmoAct2ImageProcessor(**image_config)
    video_processor = MolmoAct2VideoProcessor(**video_config)
    tokenizer = Qwen2Tokenizer.from_pretrained(
        checkpoint_location,
        token=_hf_token(),
    )

    chat_template_path = checkpoint_path / "chat_template.jinja"
    chat_template = chat_template_path.read_text() if chat_template_path.exists() else None
    return MolmoAct2Processor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
        image_use_col_tokens=processor_config.get("image_use_col_tokens", True),
        use_single_crop_col_tokens=processor_config.get("use_single_crop_col_tokens"),
        use_single_crop_start_token=processor_config.get("use_single_crop_start_token", True),
        video_use_col_tokens=processor_config.get("video_use_col_tokens", False),
        use_frame_special_tokens=processor_config.get("use_frame_special_tokens", True),
    )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_image(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    while arr.ndim > 3 and int(arr.shape[0]) == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in {3, 4}:
        raise ValueError(f"Unsupported image shape for MolmoAct2: {arr.shape}.")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype in (np.float16, np.float32, np.float64):
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _normalize_question_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip(_QUESTION_SURROUNDING_DELIMITERS).strip()
        for pattern in _QUESTION_PREFIX_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_CLOSERS).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
    chunks = [chunk.strip() for chunk in re.split(r"[.!?]+", normalized) if chunk.strip()]
    if len(chunks) > 1:
        normalized = "; ".join(chunks)
    return normalized.lower()


def _wrap_setup_text(setup_type: str, add_setup_tokens: bool) -> str:
    setup_type = str(setup_type or "")
    if setup_type.startswith(SETUP_START_TOKEN) and setup_type.endswith(SETUP_END_TOKEN):
        return setup_type
    if not setup_type or not add_setup_tokens:
        return setup_type
    return f"{SETUP_START_TOKEN}{setup_type}{SETUP_END_TOKEN}"


def _wrap_control_text(control_mode: str, add_control_tokens: bool) -> str:
    control_mode = str(control_mode or "")
    if control_mode.startswith(CONTROL_START_TOKEN) and control_mode.endswith(CONTROL_END_TOKEN):
        return control_mode
    if not control_mode or not add_control_tokens:
        return control_mode
    return f"{CONTROL_START_TOKEN}{control_mode}{CONTROL_END_TOKEN}"


def _build_discrete_state_string(state: np.ndarray, num_state_tokens: int) -> str:
    if num_state_tokens <= 0:
        raise ValueError(f"num_state_tokens must be > 0, got {num_state_tokens}.")
    arr = np.asarray(state, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    scaled = (arr + 1.0) / 2.0 * float(num_state_tokens - 1)
    token_ids = np.clip(np.rint(scaled).astype(np.int64), 0, int(num_state_tokens) - 1).reshape(-1)
    return f"{STATE_START_TOKEN}{''.join(f'{STATE_TOKEN_PREFIX}{int(token_id)}>' for token_id in token_ids)}{STATE_END_TOKEN}"


def _build_robot_text(
    *,
    task: str,
    discrete_state_string: str,
    setup_type: str,
    control_mode: str,
    add_setup_tokens: bool,
    add_control_tokens: bool,
    num_images: int,
) -> str:
    setup_text = _wrap_setup_text(setup_type, add_setup_tokens=add_setup_tokens)
    control_text = _wrap_control_text(control_mode, add_control_tokens=add_control_tokens)
    state_clause = (
        f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
    )
    prompt = (
        f"The task is to {task}. The setup is {setup_text}.{state_clause} "
        f"The expected control mode is {control_text}. Given these, what action should the robot take to complete the task?"
    )
    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(f"Image {idx + 1}<|image|>" for idx in range(num_images))
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{ACTION_OUTPUT_TOKEN}"


def _as_text_list(value: Any, batch_size: int) -> list[str]:
    if value is None:
        return [""] * batch_size
    if isinstance(value, str):
        return [value] * batch_size
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [str(value.item())] * batch_size
        flat = value.detach().cpu().reshape(-1).tolist()
        texts = [str(item) for item in flat]
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [str(value.item())] * batch_size
        texts = [str(item) for item in value.reshape(-1).tolist()]
    elif isinstance(value, (list, tuple)):
        texts = [str(item) for item in value]
    else:
        texts = [str(value)]
    if len(texts) == batch_size:
        return texts
    if len(texts) == 1:
        return texts * batch_size
    raise ValueError(f"Expected {batch_size} task strings, got {len(texts)}.")


def _tokenize_discrete_action(action: np.ndarray, processor: Any) -> list[int]:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 1:
        arr = arr[None, None, :]
    tokens_out = processor(arr)
    if isinstance(tokens_out, dict):
        tokens_out = tokens_out.get("input_ids", next(iter(tokens_out.values())))
    if isinstance(tokens_out, np.ndarray):
        tokens_out = tokens_out.tolist()
    if torch.is_tensor(tokens_out):
        tokens_out = tokens_out.detach().cpu().tolist()
    if not isinstance(tokens_out, list):
        raise TypeError(f"Unexpected discrete action tokenizer output type: {type(tokens_out)}")
    if tokens_out and isinstance(tokens_out[0], (list, tuple, np.ndarray)):
        tokens_out = tokens_out[0]
    return [int(token_id) for token_id in tokens_out]


def _build_discrete_action_string(action: np.ndarray, processor: Any) -> str:
    token_ids = _tokenize_discrete_action(action, processor)
    pieces = "".join(f"{ACTION_TOKEN_PREFIX}{int(token_id)}>" for token_id in token_ids)
    return f"{ACTION_START_TOKEN}{pieces}{ACTION_END_TOKEN}"


def _single_token_id(tokenizer: Any, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"MolmoAct2 token {token!r} must encode to one token, got {token_ids}.")
    return int(token_ids[0])


def _flatten_feature_names(raw_names: Any) -> list[str] | None:
    if raw_names is None:
        return None
    if isinstance(raw_names, dict):
        names: list[str] = []
        for value in raw_names.values():
            if isinstance(value, (list, tuple)):
                names.extend(str(item) for item in value)
            elif value is not None:
                names.append(str(value))
        return names or None
    if isinstance(raw_names, (list, tuple)):
        names = [str(item) for item in raw_names]
        return names or None
    return [str(raw_names)]


def _feature_dim(stats: dict[str, Any] | None) -> int | None:
    if not isinstance(stats, dict):
        return None
    for key in ("mean", "std", "min", "max", "q01", "q99", "q10", "q90", "mask"):
        value = stats.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            return int(value.shape[-1]) if value.ndim > 0 else None
        arr = np.asarray(value)
        return int(arr.shape[-1]) if arr.ndim > 0 else None
    return None


def _stats_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy() if value.ndim > 0 else None
    arr = np.asarray(value)
    return arr if arr.ndim > 0 else None


def _validate_masked_passthrough_stats(feature_stats: dict[str, Any], mask: list[bool], key: str) -> None:
    min_values = _stats_array(feature_stats.get("min"))
    max_values = _stats_array(feature_stats.get("max"))
    if min_values is None or max_values is None:
        return

    mask_array = np.asarray(mask, dtype=bool)
    if (
        mask_array.ndim != 1
        or min_values.shape[-1] != mask_array.shape[0]
        or max_values.shape[-1] != mask_array.shape[0]
        or not bool((~mask_array).any())
    ):
        return

    passthrough_min = min_values[..., ~mask_array]
    passthrough_max = max_values[..., ~mask_array]
    if bool(((passthrough_min < -1.0) | (passthrough_max > 1.0)).any()):
        raise ValueError(
            f"MolmoAct2 {key} gripper values are not under [-1, 1]. Please set normalize_gripper=True."
        )


def _feature_names_from_meta(dataset_meta: Any | None, feature_key: str) -> list[str] | None:
    if dataset_meta is None:
        return None

    root = getattr(dataset_meta, "root", None)
    candidate_roots = []
    if root is not None:
        repo_id = str(getattr(dataset_meta, "repo_id", "") or "").strip()
        if repo_id:
            candidate_roots.append(Path(root) / repo_id)
        candidate_roots.append(Path(root))
    for candidate_root in candidate_roots:
        info_path = candidate_root / "meta" / "info.json"
        if info_path.exists():
            try:
                with info_path.open("r", encoding="utf-8") as f:
                    info = json.load(f)
                names = _flatten_feature_names((info.get("features") or {}).get(feature_key, {}).get("names"))
                if names:
                    return names
            except (OSError, json.JSONDecodeError, AttributeError):
                pass

    for container in (
        getattr(getattr(dataset_meta, "info", None), "features", None),
        getattr(dataset_meta, "features", None),
    ):
        if not isinstance(container, dict):
            continue
        feature = container.get(feature_key)
        if not isinstance(feature, dict):
            continue
        names = _flatten_feature_names(feature.get("names"))
        if names:
            return names
    return None


def _add_gripper_masks_to_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
    dataset_meta: Any | None,
    *,
    normalize_gripper: bool,
    dataset_feature_names: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]] | None:
    if not dataset_stats:
        return dataset_stats

    stats = deepcopy(dataset_stats)
    for key in (ACTION, OBS_STATE):
        feature_stats = stats.get(key)
        if not isinstance(feature_stats, dict):
            continue
        dim = _feature_dim(feature_stats)
        if dim is None:
            continue

        if normalize_gripper:
            feature_stats["mask"] = [True] * dim
            continue

        names = _flatten_feature_names((dataset_feature_names or {}).get(key))
        if names is None:
            names = _feature_names_from_meta(dataset_meta, key)
        if names is None:
            names = _flatten_feature_names(feature_stats.get("names"))
        if names is None:
            continue
        if len(names) != dim:
            continue
        mask = ["gripper" not in name.lower() for name in names]
        _validate_masked_passthrough_stats(feature_stats, mask, key)
        feature_stats["mask"] = mask
    return stats


def _normalization_masks_from_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
) -> dict[str, list[bool]]:
    masks: dict[str, list[bool]] = {}
    for key in (ACTION, OBS_STATE):
        feature_stats = (dataset_stats or {}).get(key)
        if not isinstance(feature_stats, dict):
            continue
        mask = feature_stats.get("mask")
        if isinstance(mask, Tensor):
            mask = mask.detach().cpu().tolist()
        if isinstance(mask, list) and all(isinstance(value, bool) for value in mask):
            masks[key] = mask
    return masks


class _MolmoAct2MaskedNormalizationMixin:
    @staticmethod
    def _broadcast_feature_mask(mask: Tensor, tensor: Tensor) -> Tensor | None:
        mask = mask.to(device=tensor.device, dtype=torch.bool)
        if mask.ndim != 1 or tensor.shape[-1] != mask.shape[0]:
            return None
        while mask.ndim < tensor.ndim:
            mask = mask.unsqueeze(0)
        return mask

    @staticmethod
    def _validate_masked_passthrough_range(tensor: Tensor, mask: Tensor, key: str) -> None:
        passthrough_mask = ~mask.expand_as(tensor)
        if not bool(passthrough_mask.any()):
            return
        passthrough_values = tensor[passthrough_mask]
        if bool(((passthrough_values < -1.0) | (passthrough_values > 1.0)).any()):
            raise ValueError(
                f"MolmoAct2 {key} gripper values are not under [-1, 1]. Please set normalize_gripper=True."
            )

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: Any, *, inverse: bool = False
    ) -> Tensor:
        transformed = super()._apply_transform(tensor, key, feature_type, inverse=inverse)
        stats = getattr(self, "_tensor_stats", {}).get(key, {})
        mask = stats.get("mask") if isinstance(stats, dict) else None
        if mask is None:
            return transformed
        mask = self._broadcast_feature_mask(mask, tensor)
        if mask is None:
            return transformed
        if not inverse:
            self._validate_masked_passthrough_range(tensor, mask, key)
        return torch.where(mask, transformed, tensor)


@ProcessorStepRegistry.register(name="molmoact2_masked_normalizer")
@dataclass
class MolmoAct2MaskedNormalizerProcessorStep(_MolmoAct2MaskedNormalizationMixin, NormalizerProcessorStep):
    pass


@ProcessorStepRegistry.register(name="molmoact2_masked_unnormalizer")
@dataclass
class MolmoAct2MaskedUnnormalizerProcessorStep(_MolmoAct2MaskedNormalizationMixin, UnnormalizerProcessorStep):
    pass


@ProcessorStepRegistry.register(name="molmoact2_clamp_normalized")
@dataclass
class MolmoAct2ClampNormalizedProcessorStep(ProcessorStep):
    """Clamp q01/q99-normalized state and action to the range used by the old trainer."""

    normalization_masks: dict[str, list[bool]] | None = None

    @staticmethod
    def _broadcast_feature_mask(mask: list[bool], tensor: Tensor) -> Tensor | None:
        tensor_mask = torch.tensor(mask, device=tensor.device, dtype=torch.bool)
        if tensor_mask.ndim != 1 or tensor.shape[-1] != tensor_mask.shape[0]:
            return None
        while tensor_mask.ndim < tensor.ndim:
            tensor_mask = tensor_mask.unsqueeze(0)
        return tensor_mask

    @staticmethod
    def _validate_masked_passthrough_range(tensor: Tensor, mask: Tensor, key: str) -> None:
        passthrough_mask = ~mask.expand_as(tensor)
        if not bool(passthrough_mask.any()):
            return
        passthrough_values = tensor[passthrough_mask]
        if bool(((passthrough_values < -1.0) | (passthrough_values > 1.0)).any()):
            raise ValueError(
                f"MolmoAct2 {key} gripper values are not under [-1, 1]. Please set normalize_gripper=True."
            )

    def _clamp_tensor(self, tensor: Tensor, key: str) -> Tensor:
        mask = (self.normalization_masks or {}).get(key)
        if mask is None:
            return tensor.clamp(-1.0, 1.0)
        tensor_mask = self._broadcast_feature_mask(mask, tensor)
        if tensor_mask is None:
            return tensor.clamp(-1.0, 1.0)
        self._validate_masked_passthrough_range(tensor, tensor_mask, key)
        return torch.where(tensor_mask, tensor.clamp(-1.0, 1.0), tensor)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict) and OBS_STATE in observation:
            observation = observation.copy()
            observation[OBS_STATE] = self._clamp_tensor(torch.as_tensor(observation[OBS_STATE]), OBS_STATE)
            transition[TransitionKey.OBSERVATION] = observation
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = self._clamp_tensor(torch.as_tensor(action), ACTION)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="molmoact2_pack_inputs")
@dataclass
class MolmoAct2PackInputsProcessorStep(ProcessorStep):
    checkpoint_path: str
    checkpoint_revision: str | None = None
    checkpoint_force_download: bool = False
    action_mode: str = "both"
    discrete_action_tokenizer: str = "allenai/MolmoAct2-FAST-Tokenizer"
    image_keys: list[str] = field(default_factory=list)
    allow_image_key_fallback: bool = False
    setup_type: str = ""
    control_mode: str = ""
    normalize_language: bool = True
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    num_state_tokens: int = 256
    max_sequence_length: int | None = None
    chunk_size: int = 30
    max_action_dim: int = 32
    env_action_dim: int | None = None

    def __post_init__(self) -> None:
        require_package("transformers", extra="molmoact2")

        checkpoint_location = _resolve_checkpoint_location(
            self.checkpoint_path,
            revision=self.checkpoint_revision,
            force_download=bool(self.checkpoint_force_download),
        )
        self.processor = _load_local_molmoact2_processor(checkpoint_location)
        self.action_processor = None
        if self.action_mode in {"discrete", "both"}:
            require_package("scipy", extra="molmoact2")
            if UniversalActionProcessor is None:
                raise RuntimeError("transformers and scipy are required to load MolmoAct2 action tokenizer.")
            self.action_processor = UniversalActionProcessor.from_pretrained_local(
                self.discrete_action_tokenizer,
            )
        self._action_start_id = _single_token_id(self.processor.tokenizer, ACTION_START_TOKEN)
        self._action_end_id = _single_token_id(self.processor.tokenizer, ACTION_END_TOKEN)
        self._eos_token = self.processor.tokenizer.eos_token or ""
        self._eos_token_id = self.processor.tokenizer.eos_token_id

    def get_config(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_force_download": self.checkpoint_force_download,
            "action_mode": self.action_mode,
            "discrete_action_tokenizer": self.discrete_action_tokenizer,
            "image_keys": list(self.image_keys),
            "allow_image_key_fallback": self.allow_image_key_fallback,
            "setup_type": self.setup_type,
            "control_mode": self.control_mode,
            "normalize_language": self.normalize_language,
            "add_setup_tokens": self.add_setup_tokens,
            "add_control_tokens": self.add_control_tokens,
            "num_state_tokens": self.num_state_tokens,
            "max_sequence_length": self.max_sequence_length,
            "chunk_size": self.chunk_size,
            "max_action_dim": self.max_action_dim,
            "env_action_dim": self.env_action_dim,
        }

    def _resolve_max_sequence_length(
        self,
        *,
        num_images: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        include_discrete_action: bool,
    ) -> int:
        if self.max_sequence_length is not None:
            return int(self.max_sequence_length)
        return infer_molmoact2_max_sequence_length(
            num_images=num_images,
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            include_discrete_action=include_discrete_action,
        )

    def _batch_size(self, observation: dict[str, Any], action: Tensor | None) -> int:
        if action is not None:
            return int(action.shape[0])
        state = observation.get(OBS_STATE)
        if torch.is_tensor(state) or isinstance(state, np.ndarray):
            return int(state.shape[0]) if getattr(state, "ndim", 0) > 1 else 1
        for key in self._resolve_image_keys(observation):
            value = observation[key]
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                return int(value.shape[0]) if getattr(value, "ndim", 0) == 4 else 1
        return 1

    @staticmethod
    def _observation_image_keys(observation: dict[str, Any]) -> list[str]:
        keys = [key for key in observation if str(key).startswith(f"{OBS_IMAGES}.")]
        if not keys:
            keys = [key for key in observation if str(key).startswith("observation.image")]
        return sorted(keys)

    def _resolve_image_keys(self, observation: dict[str, Any]) -> list[str]:
        if self.image_keys:
            missing = [key for key in self.image_keys if key not in observation]
            if missing:
                fallback_keys = self._observation_image_keys(observation)
                if self.allow_image_key_fallback and fallback_keys:
                    return fallback_keys
                raise ValueError(f"MolmoAct2 image_keys missing from observation: {missing}.")
            return list(self.image_keys)
        keys = self._observation_image_keys(observation)
        if not keys:
            raise ValueError("MolmoAct2 requires at least one image observation.")
        return sorted(keys)

    def _extract_images(self, observation: dict[str, Any], batch_size: int) -> list[list[np.ndarray]]:
        images_by_example: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
        for key in self._resolve_image_keys(observation):
            value = observation[key]
            for batch_idx in range(batch_size):
                item = value
                if (torch.is_tensor(value) or isinstance(value, np.ndarray)) and getattr(
                    value, "ndim", 0
                ) >= 4:
                    item = value[batch_idx]
                images_by_example[batch_idx].append(_normalize_image(item))
        return images_by_example

    def _extract_state(self, observation: dict[str, Any], batch_size: int) -> Tensor:
        if OBS_STATE not in observation:
            raise ValueError("MolmoAct2 requires observation.state for discrete state prompting.")
        state = torch.as_tensor(observation[OBS_STATE], dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if int(state.shape[0]) != batch_size:
            raise ValueError(f"State batch size {state.shape[0]} does not match batch size {batch_size}.")
        return state

    def _pad_action(self, action: Tensor, action_is_pad: Any | None) -> tuple[Tensor, Tensor, Tensor]:
        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.ndim != 3:
            raise ValueError(f"MolmoAct2 expected action shape [B, T, D], got {tuple(action.shape)}.")
        if action.shape[-1] > self.max_action_dim:
            raise ValueError(
                f"Action dim {action.shape[-1]} exceeds MolmoAct2 max_action_dim={self.max_action_dim}."
            )
        padded = torch.zeros(
            (*action.shape[:-1], self.max_action_dim),
            device=action.device,
            dtype=torch.float32,
        )
        padded[..., : action.shape[-1]] = action.to(dtype=torch.float32)
        action_dim_is_pad = torch.ones(
            (action.shape[0], self.max_action_dim), device=action.device, dtype=torch.bool
        )
        action_dim_is_pad[:, : action.shape[-1]] = False
        if action_is_pad is None:
            action_horizon_is_pad = torch.zeros(action.shape[:2], device=action.device, dtype=torch.bool)
        else:
            action_horizon_is_pad = torch.as_tensor(action_is_pad, device=action.device, dtype=torch.bool)
            if action_horizon_is_pad.ndim == 1:
                action_horizon_is_pad = action_horizon_is_pad.unsqueeze(0)
            if tuple(action_horizon_is_pad.shape) != tuple(action.shape[:2]):
                raise ValueError(
                    "action_is_pad must match action horizon shape: "
                    f"got {tuple(action_horizon_is_pad.shape)} for action {tuple(action.shape)}."
                )
        return padded, action_horizon_is_pad, action_dim_is_pad

    def _build_labels(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        labels = torch.full_like(input_ids, -100)
        for batch_idx in range(input_ids.shape[0]):
            valid = attention_mask[batch_idx].to(dtype=torch.bool)
            row = input_ids[batch_idx]
            starts = (row == self._action_start_id).nonzero(as_tuple=False).flatten().tolist()
            ends = (row == self._action_end_id).nonzero(as_tuple=False).flatten().tolist()
            end_ptr = 0
            for start in starts:
                while end_ptr < len(ends) and ends[end_ptr] < start:
                    end_ptr += 1
                if end_ptr >= len(ends):
                    raise ValueError(
                        "Found <action_start> without matching <action_end> in MolmoAct2 labels."
                    )
                end = int(ends[end_ptr])
                label_end = end + 1
                if (
                    self._eos_token_id is not None
                    and label_end < int(row.shape[0])
                    and int(row[label_end]) == int(self._eos_token_id)
                ):
                    label_end += 1
                labels[batch_idx, start:label_end] = row[start:label_end]
                end_ptr += 1
            if not starts:
                raise ValueError("No discrete action span found in MolmoAct2 training text.")
            labels[batch_idx] = torch.where(
                valid, labels[batch_idx], torch.full_like(labels[batch_idx], -100)
            )
        return labels

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        if not isinstance(observation, dict):
            raise ValueError("MolmoAct2 expected an observation dictionary.")
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        raw_action = transition.get(TransitionKey.ACTION)
        action = torch.as_tensor(raw_action, dtype=torch.float32) if raw_action is not None else None
        batch_size = self._batch_size(observation, action)
        state = self._extract_state(observation, batch_size)
        images_by_example = self._extract_images(observation, batch_size)

        task_source = complementary.get("task")
        if task_source is None:
            task_source = observation.get("task")
        if task_source is None:
            task_source = observation.get("observation.language")
        if task_source is None:
            task_source = complementary.get("language_instruction")
        tasks = _as_text_list(task_source, batch_size)
        if self.normalize_language:
            tasks = [_normalize_question_text(task) for task in tasks]
        complementary["task"] = tasks

        action_padded = None
        action_horizon_is_pad = None
        action_dim_is_pad = torch.ones((batch_size, self.max_action_dim), dtype=torch.bool)
        real_action_dim = int(self.env_action_dim or 0)
        if action is not None:
            action_is_pad = complementary.get("action_is_pad")
            if action_is_pad is None:
                action_is_pad = complementary.get("action_horizon_is_pad")
            action_padded, action_horizon_is_pad, action_dim_is_pad = self._pad_action(action, action_is_pad)
            real_action_dim = int(action.shape[-1])
        elif real_action_dim > 0:
            action_dim_is_pad[:, :real_action_dim] = False

        prompt_texts: list[str] = []
        full_texts: list[str] = []
        flat_images: list[np.ndarray] = []
        state_np = state.detach().cpu().numpy()
        build_action_labels = action is not None and self.action_mode in {"discrete", "both"}
        for batch_idx in range(batch_size):
            images = images_by_example[batch_idx]
            flat_images.extend(images)
            discrete_state = _build_discrete_state_string(state_np[batch_idx], self.num_state_tokens)
            prompt = _build_robot_text(
                task=tasks[batch_idx],
                discrete_state_string=discrete_state,
                setup_type=self.setup_type,
                control_mode=self.control_mode,
                add_setup_tokens=self.add_setup_tokens,
                add_control_tokens=self.add_control_tokens,
                num_images=len(images),
            )
            prompt_texts.append(prompt)
            if build_action_labels:
                if self.action_processor is None:
                    raise ValueError("Discrete MolmoAct2 training requires an action tokenizer.")
                answer = _build_discrete_action_string(
                    action[batch_idx].detach().cpu().numpy(), self.action_processor
                )
                full_texts.append(f"{prompt}{answer}{self._eos_token}")
            else:
                full_texts.append(prompt)

        text = full_texts if build_action_labels else prompt_texts
        inputs = self.processor(text=text, images=flat_images, return_tensors="pt", padding=True)
        if action is None:
            action_horizon = self.chunk_size
        elif action.ndim == 2:
            action_horizon = 1
        else:
            action_horizon = int(action.shape[1])
        max_sequence_length = self._resolve_max_sequence_length(
            num_images=max((len(images) for images in images_by_example), default=0),
            state_dim=int(state.shape[-1]),
            action_dim=max(real_action_dim, 1),
            action_horizon=action_horizon,
            include_discrete_action=build_action_labels,
        )
        if int(inputs["input_ids"].shape[1]) > max_sequence_length:
            raise ValueError(
                f"MolmoAct2 sequence length {int(inputs['input_ids'].shape[1])} exceeds "
                f"max_sequence_length={max_sequence_length}."
            )

        if build_action_labels:
            inputs["labels"] = self._build_labels(inputs["input_ids"], inputs["attention_mask"])

        complementary.update(dict(inputs))
        complementary["action_dim_is_pad"] = action_dim_is_pad
        if action_horizon_is_pad is not None:
            complementary["action_horizon_is_pad"] = action_horizon_is_pad

        if action_padded is not None:
            transition[TransitionKey.ACTION] = action_padded
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="molmoact2_state_frame_transform")
@dataclass
class MolmoAct2StateFrameTransformStep(ProcessorStep):
    """Convert robot state from arm frame to model frame before normalization.

    Required for zero-shot deployment of MolmoAct2-SO100_101 on SO-100/101
    arms calibrated with LeRobot >= 0.5.0 (v3.0 convention). The checkpoint
    was trained on data using a different joint convention (sign flip on
    shoulder_lift, 90 deg offset on shoulder_lift and elbow_flex).

    No-op when joint_signs and joint_offsets are None (default), so this
    step has no effect on fine-tuned models or other embodiments.

    state_model = signs * arm_state + offsets

    See: https://huggingface.co/docs/lerobot/backwardcomp
    """

    joint_signs: list[float] | None = None
    joint_offsets: list[float] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.joint_signs is None or self.joint_offsets is None:
            return transition
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict) or OBS_STATE not in observation:
            return transition
        transition = transition.copy()
        observation = observation.copy()
        state = torch.as_tensor(observation[OBS_STATE], dtype=torch.float32).clone()
        n = len(self.joint_signs)
        signs = torch.tensor(self.joint_signs, dtype=torch.float32, device=state.device)
        offsets = torch.tensor(self.joint_offsets, dtype=torch.float32, device=state.device)
        state[..., :n] = signs * state[..., :n] + offsets
        observation[OBS_STATE] = state
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"joint_signs": self.joint_signs, "joint_offsets": self.joint_offsets}


@ProcessorStepRegistry.register(name="molmoact2_action_frame_transform")
@dataclass
class MolmoAct2ActionFrameTransformStep(ProcessorStep):
    """Convert model action from model frame back to arm frame after unnormalization.

    Inverse of MolmoAct2StateFrameTransformStep. Required for zero-shot
    MolmoAct2-SO100_101 on SO-100/101 arms. No-op when both fields are None.

    action_arm = signs * (model_action - offsets)

    See: https://huggingface.co/docs/lerobot/backwardcomp
    """

    joint_signs: list[float] | None = None
    joint_offsets: list[float] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.joint_signs is None or self.joint_offsets is None:
            return transition
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        transition = transition.copy()
        action = torch.as_tensor(action, dtype=torch.float32).clone()
        n = len(self.joint_signs)
        signs = torch.tensor(self.joint_signs, dtype=torch.float32, device=action.device)
        offsets = torch.tensor(self.joint_offsets, dtype=torch.float32, device=action.device)
        action[..., :n] = signs * (action[..., :n] - offsets)
        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"joint_signs": self.joint_signs, "joint_offsets": self.joint_offsets}


@ProcessorStepRegistry.register(name="molmoact2_clamp_action")
@dataclass
class MolmoAct2ClampActionProcessorStep(ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = torch.as_tensor(action).clamp(-1.0, 1.0)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_molmoact2_pre_post_processors(
    config: MolmoAct2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    env_action_dim = None
    if config.output_features and ACTION in config.output_features:
        env_action_dim = int(config.output_features[ACTION].shape[0])

    hf_metadata: dict[str, Any] = {}
    if dataset_stats is None and str(config.norm_tag or "").strip():
        dataset_stats, hf_metadata = _load_hf_norm_stats_for_tag(
            config.checkpoint_path,
            revision=config.checkpoint_revision,
            force_download=bool(config.checkpoint_force_download),
            norm_tag=config.norm_tag,
        )

    image_keys = list(config.image_keys)
    visual_feature_keys = [
        key for key, feature in config.input_features.items() if feature.type == FeatureType.VISUAL
    ]
    if not image_keys and isinstance(hf_metadata.get("camera_keys"), list):
        metadata_image_keys = [str(key) for key in hf_metadata["camera_keys"]]
        if not visual_feature_keys or all(key in config.input_features for key in metadata_image_keys):
            image_keys = metadata_image_keys
    if not image_keys:
        image_keys = visual_feature_keys
    setup_type = config.setup_type or str(hf_metadata.get("setup_type") or "")
    control_mode = config.control_mode or str(hf_metadata.get("control_mode") or "")
    chunk_size = int(hf_metadata.get("action_horizon") or config.chunk_size)

    masked_dataset_stats = _add_gripper_masks_to_stats(
        dataset_stats,
        dataset_meta,
        normalize_gripper=config.normalize_gripper,
        dataset_feature_names=config.dataset_feature_names,
    )
    normalization_masks = _normalization_masks_from_stats(masked_dataset_stats)

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        MolmoAct2StateFrameTransformStep(
            joint_signs=config.joint_signs,
            joint_offsets=config.joint_offsets,
        ),
        MolmoAct2MaskedNormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=masked_dataset_stats,
        ),
        MolmoAct2ClampNormalizedProcessorStep(normalization_masks=normalization_masks),
        MolmoAct2PackInputsProcessorStep(
            checkpoint_path=config.checkpoint_path,
            checkpoint_revision=config.checkpoint_revision,
            checkpoint_force_download=config.checkpoint_force_download,
            action_mode=config.action_mode,
            discrete_action_tokenizer=config.discrete_action_tokenizer,
            image_keys=image_keys,
            allow_image_key_fallback=not bool(config.image_keys),
            setup_type=setup_type,
            control_mode=control_mode,
            normalize_language=config.normalize_language,
            add_setup_tokens=config.add_setup_tokens,
            add_control_tokens=config.add_control_tokens,
            num_state_tokens=config.num_state_tokens,
            max_sequence_length=config.max_sequence_length,
            chunk_size=chunk_size,
            max_action_dim=config.expected_max_action_dim,
            env_action_dim=env_action_dim,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        MolmoAct2ClampActionProcessorStep(),
        MolmoAct2MaskedUnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=masked_dataset_stats,
        ),
        MolmoAct2ActionFrameTransformStep(
            joint_signs=config.joint_signs,
            joint_offsets=config.joint_offsets,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
