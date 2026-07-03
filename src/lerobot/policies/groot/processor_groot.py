#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

import logging
import random
from copy import copy, deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as tv_functional
from einops import rearrange
from torchvision.transforms import InterpolationMode

from lerobot.utils.import_utils import _datasets_available, _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import (
        AutoTokenizer,
        ProcessorMixin,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )
else:
    AutoTokenizer = None
    ProcessorMixin = object
    Qwen2VLImageProcessor = None
    Qwen3VLProcessor = None
    Qwen3VLVideoProcessor = None

if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None

from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    to_relative_actions,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.device_utils import get_safe_torch_device

from .configuration_groot import (
    GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
    GROOT_N1_5_REMOVAL_GUIDANCE,
    GROOT_N1_7_BACKBONE_MODEL,
    N1_7_DEFAULT_IMAGE_CROP_SIZE,
    N1_7_DEFAULT_IMAGE_TARGET_SIZE,
    GrootConfig,
    is_raw_groot_n1_7_checkpoint,
)
from .utils import (
    as_int_pair,
    as_optional_float,
    as_optional_int,
    config_value,
    flatten_n1_7_modality_stats,
    has_modality_stats,
    infer_n1_7_batch_size_and_device,
    prepare_n1_7_language_batch,
    read_json,
    relative_eef_to_absolute,
    stat_dim_from_entry,
)

# Native GR00T N1.7 action horizon: checkpoints are trained to predict 40-step
# action chunks, so processor-side horizons are capped at this value.
N1_7_NATIVE_ACTION_HORIZON = 40

N1_7_EMBODIMENT_MAPPING = {
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
}


@dataclass
class _GrootN17CheckpointProcessorAssets:
    """Processor metadata loaded from a raw Isaac-GR00T N1.7 checkpoint.

    Public N1.7 checkpoints store preprocessing and action-decoding choices next
    to the model weights. Keeping those values together avoids falling back to
    LeRobot defaults that are valid for older GR00T variants but change N1.7
    inputs or decoded actions.
    """

    stats: dict[str, dict[str, Any]]
    raw_stats: dict[str, Any]
    modality_config: dict[str, Any]
    embodiment_mapping: dict[str, int]
    formalize_language: bool
    valid_action_horizon: int | None
    max_action_horizon: int | None
    video_horizon: int | None
    use_percentiles: bool
    use_relative_action: bool
    state_dropout_prob: float
    clip_outliers: bool
    video_modality_keys: list[str] | None
    image_crop_size: list[int] | None
    image_target_size: list[int] | None
    shortest_image_edge: int | None
    crop_fraction: float | None
    use_albumentations: bool
    letter_box_transform: bool


@dataclass(frozen=True)
class _GrootN17ActionGroup:
    key: str
    indices: list[int]
    relative: bool


def _load_n1_7_checkpoint_processor_assets(config: GrootConfig) -> _GrootN17CheckpointProcessorAssets | None:
    """Load N1.7 processor settings from checkpoint sidecar JSON files.

    Returns ``None`` for non-raw N1.7 checkpoints so the generic GR00T pipeline
    can keep using caller-provided dataset stats and config values.
    """

    if not is_raw_groot_n1_7_checkpoint(config.base_model_path):
        return None

    checkpoint_path = Path(config.base_model_path).expanduser()
    processor_config = read_json(checkpoint_path / "processor_config.json")
    processor_kwargs = processor_config.get("processor_kwargs", {})
    if not isinstance(processor_kwargs, dict):
        processor_kwargs = {}

    all_stats = read_json(checkpoint_path / "statistics.json")
    raw_stats = all_stats.get(config.embodiment_tag)
    if not isinstance(raw_stats, dict):
        raw_stats = {}

    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        modality_configs = {}
    modality_config = modality_configs.get(config.embodiment_tag)
    if not isinstance(modality_config, dict):
        modality_config = {}

    use_relative_action = bool(processor_kwargs.get("use_relative_action", False))
    state_dropout_prob = as_optional_float(processor_kwargs.get("state_dropout_prob"))
    if state_dropout_prob is None:
        state_dropout_prob = 0.0
    stats = _load_n1_7_checkpoint_stats(
        checkpoint_path,
        processor_kwargs,
        config.embodiment_tag,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_relative_action=use_relative_action,
    )
    embodiment_mapping = _load_n1_7_embodiment_mapping(checkpoint_path) or dict(N1_7_EMBODIMENT_MAPPING)
    formalize_language = processor_kwargs.get("formalize_language", True)
    if not isinstance(formalize_language, bool):
        formalize_language = True
    clip_outliers = processor_kwargs.get("clip_outliers", True)
    if not isinstance(clip_outliers, bool):
        clip_outliers = True
    use_albumentations = processor_kwargs.get("use_albumentations", False)
    if not isinstance(use_albumentations, bool):
        use_albumentations = False
    letter_box_transform = processor_kwargs.get("letter_box_transform", False)
    if not isinstance(letter_box_transform, bool):
        letter_box_transform = False

    valid_action_horizon = _load_n1_7_checkpoint_action_horizon(processor_kwargs, config.embodiment_tag)
    video_horizon = _load_n1_7_checkpoint_video_horizon(processor_kwargs, config.embodiment_tag)
    video_modality_keys = _load_n1_7_checkpoint_video_modality_keys(processor_kwargs, config.embodiment_tag)
    max_action_horizon = processor_kwargs.get("max_action_horizon")
    if not isinstance(max_action_horizon, int):
        max_action_horizon = None

    return _GrootN17CheckpointProcessorAssets(
        stats=stats,
        raw_stats=raw_stats,
        modality_config=modality_config,
        embodiment_mapping=embodiment_mapping,
        formalize_language=formalize_language,
        valid_action_horizon=valid_action_horizon,
        max_action_horizon=max_action_horizon,
        video_horizon=video_horizon,
        use_percentiles=bool(processor_kwargs.get("use_percentiles", False)),
        use_relative_action=use_relative_action,
        state_dropout_prob=state_dropout_prob,
        clip_outliers=clip_outliers,
        video_modality_keys=video_modality_keys,
        image_crop_size=as_int_pair(processor_kwargs.get("image_crop_size")),
        image_target_size=as_int_pair(processor_kwargs.get("image_target_size")),
        shortest_image_edge=as_optional_int(processor_kwargs.get("shortest_image_edge")),
        crop_fraction=as_optional_float(processor_kwargs.get("crop_fraction")),
        use_albumentations=use_albumentations,
        letter_box_transform=letter_box_transform,
    )


def _load_n1_7_embodiment_mapping(checkpoint_path: Path) -> dict[str, int] | None:
    mapping = read_json(checkpoint_path / "embodiment_id.json")
    if not mapping:
        return None
    parsed: dict[str, int] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        try:
            parsed[key] = int(value)
        except (TypeError, ValueError):
            continue
    return parsed or None


def _load_n1_7_checkpoint_stats(
    checkpoint_path: Path,
    processor_kwargs: dict[str, Any],
    embodiment_tag: str,
    *,
    raw_stats: dict[str, Any] | None = None,
    modality_config: dict[str, Any] | None = None,
    use_relative_action: bool = False,
) -> dict[str, dict[str, Any]]:
    """Convert checkpoint modality-group stats into LeRobot flat tensor stats.

    Isaac-GR00T keeps statistics keyed by semantic groups such as EEF pose and
    joints. LeRobot normalizers operate over a single vector, so this function
    preserves checkpoint group order while flattening each selected statistic.
    """

    if raw_stats is None:
        all_stats = read_json(checkpoint_path / "statistics.json")
        raw_stats = all_stats.get(embodiment_tag)
    if not isinstance(raw_stats, dict):
        return {}

    if modality_config is None:
        modality_configs = processor_kwargs.get("modality_configs", {})
        if not isinstance(modality_configs, dict):
            return {}
        modality_config = modality_configs.get(embodiment_tag)
    if not isinstance(modality_config, dict):
        return {}

    use_percentiles = processor_kwargs.get("use_percentiles", False)
    return {
        OBS_STATE: flatten_n1_7_modality_stats(
            embodiment_stats=raw_stats,
            embodiment_config=modality_config,
            modality="state",
            use_percentiles=bool(use_percentiles),
            use_relative_action=use_relative_action,
        ),
        ACTION: flatten_n1_7_modality_stats(
            embodiment_stats=raw_stats,
            embodiment_config=modality_config,
            modality="action",
            use_percentiles=bool(use_percentiles),
            use_relative_action=use_relative_action,
        ),
    }


def _load_n1_7_checkpoint_action_horizon(
    processor_kwargs: dict[str, Any],
    embodiment_tag: str,
) -> int | None:
    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None
    embodiment_config = modality_configs.get(embodiment_tag, {})
    if not isinstance(embodiment_config, dict):
        return None
    action_config = embodiment_config.get("action", {})
    if not isinstance(action_config, dict):
        return None
    delta_indices = action_config.get("delta_indices", [])
    if not isinstance(delta_indices, list):
        return None
    return len(delta_indices) or None


def _load_n1_7_checkpoint_video_horizon(
    processor_kwargs: dict[str, Any],
    embodiment_tag: str,
) -> int | None:
    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None
    embodiment_config = modality_configs.get(embodiment_tag, {})
    if not isinstance(embodiment_config, dict):
        return None
    video_config = embodiment_config.get("video", {})
    if not isinstance(video_config, dict):
        return None
    delta_indices = video_config.get("delta_indices", [])
    if not isinstance(delta_indices, list):
        return None
    return len(delta_indices) or None


def _load_n1_7_checkpoint_video_modality_keys(
    processor_kwargs: dict[str, Any],
    embodiment_tag: str,
) -> list[str] | None:
    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None
    embodiment_config = modality_configs.get(embodiment_tag, {})
    if not isinstance(embodiment_config, dict):
        return None
    video_config = embodiment_config.get("video", {})
    if not isinstance(video_config, dict):
        return None
    modality_keys = video_config.get("modality_keys", [])
    if not isinstance(modality_keys, list):
        return None
    keys = [key for key in modality_keys if isinstance(key, str)]
    return keys or None


# GR00T normalizes and represents actions inside its own processor steps, so it deliberately has no
# standard NormalizerProcessorStep/UnnormalizerProcessorStep or generic relative/absolute action steps.
# ``lerobot-train`` can still emit those generic override keys; for a GR00T pipeline they legitimately
# match no step, so drop them up front without masking unrelated typo keys.
_GROOT_ABSENT_STANDARD_OVERRIDE_KEYS = frozenset(
    {
        "absolute_actions_processor",
        "normalizer_processor",
        "relative_actions_processor",
        "unnormalizer_processor",
    }
)


def _drop_groot_absent_standard_overrides(overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip standard override keys that a GR00T pipeline has no step for."""

    if not overrides:
        return overrides

    filtered: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _GROOT_ABSENT_STANDARD_OVERRIDE_KEYS:
            logging.debug(
                "Ignoring override key '%s': GR00T normalizes inside its own processor steps and has "
                "no matching step (see GrootConfig.normalization_mapping).",
                key,
            )
            continue
        filtered[key] = value
    return filtered


def _apply_groot_step_overrides(
    pipeline: PolicyProcessorPipeline,
    overrides: dict[str, Any] | None,
) -> None:
    """Apply ``from_pretrained``-style step overrides to a freshly built pipeline.

    Raw N1.7 checkpoints build their processors from scratch instead of
    deserializing them, so caller overrides must be applied to the constructed
    steps. Override keys match a step's registry name or, as a convenience, its
    class name (``PolicyProcessorPipeline.from_pretrained`` matches registered
    steps by registry name only — prefer registry names so overrides keep
    working after the checkpoint is converted and reloaded from a serialized
    pipeline). Keys or fields that match nothing raise instead of being dropped
    silently (standard normalization keys GR00T has no step for are removed
    beforehand by ``_drop_groot_absent_standard_overrides``).
    """

    if not overrides:
        return

    def _step_keys(step: ProcessorStep) -> set[str]:
        keys = {type(step).__name__}
        registry_name = getattr(type(step), "_registry_name", None)
        if registry_name:
            keys.add(registry_name)
        return keys

    for override_key, step_overrides in overrides.items():
        matched_steps = [step for step in pipeline.steps if override_key in _step_keys(step)]
        if not matched_steps:
            available = [
                getattr(type(step), "_registry_name", None) or type(step).__name__ for step in pipeline.steps
            ]
            raise KeyError(
                f"Override key '{override_key}' does not match any step of the GR00T processor pipeline "
                f"built for this raw N1.7 checkpoint. Available step keys: {available}."
            )
        for step in matched_steps:
            if not is_dataclass(step):
                raise TypeError(
                    f"Cannot apply overrides to step '{override_key}': it is not a dataclass step."
                )
            init_field_names = {f.name for f in fields(step) if f.init}
            for field_name, value in dict(step_overrides).items():
                if field_name not in init_field_names:
                    raise TypeError(
                        f"Override field '{field_name}' is not a config field of step '{override_key}'. "
                        f"Available fields: {sorted(init_field_names)}."
                    )
                setattr(step, field_name, value)
            # Re-derive attributes computed from the overridden config (e.g.
            # DeviceProcessorStep resolves its torch.device in __post_init__).
            post_init = getattr(step, "__post_init__", None)
            if callable(post_init):
                post_init()


def _set_groot_preprocessor_training(
    preprocessor: PolicyProcessorPipeline,
    *,
    training: bool,
) -> None:
    """Set the runtime-only mode of GR00T stochastic processor steps.

    Any dataclass step exposing a ``training`` field participates, so processor
    steps can opt into train-time-only behavior (dropout, augmentation) without
    this helper enumerating them.
    """
    for step in preprocessor.steps:
        if is_dataclass(step) and any(f.name == "training" for f in fields(step)):
            step.training = training


def make_groot_pre_post_processors_from_pretrained(
    config: GrootConfig,
    pretrained_path: str,
    *,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load Groot processors for a raw N1.7 checkpoint or a serialized LeRobot pipeline."""

    # Drop the standard normalizer/unnormalizer override keys lerobot-train emits unconditionally:
    # GR00T has no such steps, so they would make both the raw-checkpoint and serialized override
    # paths raise. This must happen before either branch below.
    preprocessor_overrides = _drop_groot_absent_standard_overrides(preprocessor_overrides)
    postprocessor_overrides = _drop_groot_absent_standard_overrides(postprocessor_overrides)

    if is_raw_groot_n1_7_checkpoint(pretrained_path):
        processor_cfg = copy(config)
        processor_cfg.base_model_path = str(pretrained_path)
        preprocessor, postprocessor = make_groot_pre_post_processors(
            config=processor_cfg,
            dataset_stats=dataset_stats,
            dataset_meta=dataset_meta,
        )
        # Raw checkpoints have no serialized pipelines to load overrides into,
        # so apply the caller overrides (e.g. device and rename_map from
        # lerobot-eval or the policy server) to the freshly built steps.
        _apply_groot_step_overrides(preprocessor, preprocessor_overrides)
        _apply_groot_step_overrides(postprocessor, postprocessor_overrides)
        _apply_groot_action_decode_transform(postprocessor, config.action_decode_transform)
        return preprocessor, postprocessor

    preprocessor, postprocessor = _load_groot_processor_pipelines(
        pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        preprocessor_config_filename=preprocessor_config_filename,
        postprocessor_config_filename=postprocessor_config_filename,
    )
    _reconnect_groot_relative_absolute_steps(preprocessor, postprocessor)
    _reconnect_groot_n1_7_pack_decode_steps(preprocessor, postprocessor)
    _apply_groot_action_decode_transform(postprocessor, config.action_decode_transform)
    _set_groot_preprocessor_training(preprocessor, training=dataset_meta is not None)
    return preprocessor, postprocessor


def _load_groot_processor_pipelines(
    pretrained_path: str,
    *,
    preprocessor_overrides: dict[str, Any],
    postprocessor_overrides: dict[str, Any],
    preprocessor_config_filename: str,
    postprocessor_config_filename: str,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # Register the GR00T N1.5 rejection stubs before deserializing, so a saved N1.5 pipeline
    # referencing their registry names fails with the canonical removal guidance.
    _register_removed_n1_5_step_stubs()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=preprocessor_config_filename,
        overrides=preprocessor_overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=postprocessor_config_filename,
        overrides=postprocessor_overrides,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def _reconnect_groot_relative_absolute_steps(
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> None:
    relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)),
        None,
    )
    if relative_step is None:
        return

    for step in postprocessor.steps:
        if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
            step.relative_step = relative_step


def _reconnect_groot_n1_7_pack_decode_steps(
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> None:
    """Re-link a deserialized N1.7 action decode step to its pack step.

    The pack step holds the per-instance raw-state cache that relative-action
    decoding reads its reference state from; the link itself is not serialized.
    """

    pack_step = next(
        (step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep)),
        None,
    )
    if pack_step is None:
        return

    for step in postprocessor.steps:
        if isinstance(step, GrootN17ActionDecodeStep) and step.pack_step is None:
            step.pack_step = pack_step


def _apply_groot_action_decode_transform(
    postprocessor: PolicyProcessorPipeline,
    action_decode_transform: str | None,
) -> None:
    use_libero_transform = action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO

    for step in postprocessor.steps:
        if isinstance(step, GrootN17ActionDecodeStep):
            step.action_decode_transform = action_decode_transform
        elif isinstance(step, GrootActionUnpackUnnormalizeStep):
            step.libero_gripper_action = use_libero_transform
            if use_libero_transform:
                step.libero_gripper_binarize = True


def _resolve_feature_names_from_dataset_meta(dataset_meta: Any | None, feature_key: str) -> list[str] | None:
    features = getattr(dataset_meta, "features", {}) or {}
    feature = features.get(feature_key) if isinstance(features, dict) else None
    names = feature.get("names") if isinstance(feature, dict) else getattr(feature, "names", None)
    return list(names) if names is not None else None


def _resolve_action_feature_names_from_dataset_meta(dataset_meta: Any | None) -> list[str] | None:
    return _resolve_feature_names_from_dataset_meta(dataset_meta, ACTION)


def _resolve_visual_modality_keys_from_dataset_meta(dataset_meta: Any | None) -> list[str] | None:
    features = getattr(dataset_meta, "features", {}) or {}
    if not isinstance(features, dict):
        return None

    keys: list[str] = []
    for key, value in features.items():
        dtype = value.get("dtype") if isinstance(value, dict) else getattr(value, "dtype", None)
        feature_type = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
        is_visual = dtype in {"image", "video"} or str(feature_type).upper().endswith("VISUAL")
        if not is_visual or not isinstance(key, str) or not key.startswith(f"{OBS_IMAGES}."):
            continue
        keys.append(key.removeprefix(f"{OBS_IMAGES}."))
    return keys or None


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    item = getattr(value, "item", None)
    if callable(item):
        return int(item())
    return int(value)


def _to_float_tensor(value: Any, *, key: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"Cannot compute relative action statistics: sample is missing '{key}'.")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _state_reference_batch(state: torch.Tensor) -> torch.Tensor:
    if state.ndim == 1:
        return state.unsqueeze(0)
    if state.ndim == 2:
        return state
    if state.ndim > 2:
        return state.reshape(-1, state.shape[-1])[-1:].contiguous()
    raise ValueError(f"observation.state must have at least 1 dimension, got shape {tuple(state.shape)}.")


def _action_training_batch(action: torch.Tensor, state_batch: torch.Tensor) -> torch.Tensor:
    if action.ndim == 1:
        return action.unsqueeze(0)
    if action.ndim == 2:
        if state_batch.shape[0] == action.shape[0] and state_batch.shape[0] > 1:
            return action
        return action.unsqueeze(0)
    if action.ndim == 3:
        return action
    raise ValueError(f"action must be (D,), (T, D), (B, D), or (B, T, D), got {tuple(action.shape)}.")


def _relative_action_chunks_by_horizon(
    relative_action: torch.Tensor, pad_mask: Any | None
) -> list[list[np.ndarray]]:
    if relative_action.ndim == 2:
        relative_action = relative_action.unsqueeze(0)
    if relative_action.ndim != 3:
        raise ValueError(
            "Cannot compute horizon-preserving relative action statistics from "
            f"shape {tuple(relative_action.shape)}."
        )

    batch_size, horizon, _action_dim = relative_action.shape
    keep = torch.ones(batch_size, horizon, dtype=torch.bool)
    if pad_mask is not None:
        mask = torch.as_tensor(pad_mask, dtype=torch.bool).cpu()
        if mask.ndim == 1 and batch_size == 1 and mask.numel() == horizon:
            keep[0, :] = not bool(mask.any())
        elif mask.ndim == 2 and tuple(mask.shape) == (batch_size, horizon):
            complete_chunks = ~mask.any(dim=1)
            keep = complete_chunks[:, None].expand(batch_size, horizon).clone()

    chunks: list[list[np.ndarray]] = [[] for _ in range(horizon)]
    relative_np = relative_action.detach().cpu().numpy()
    for batch_idx in range(batch_size):
        for horizon_idx in range(horizon):
            if keep[batch_idx, horizon_idx]:
                chunks[horizon_idx].append(relative_np[batch_idx, horizon_idx])
    return chunks


def _compute_horizon_relative_action_stats(
    chunks_by_horizon: list[list[np.ndarray]],
) -> dict[str, np.ndarray]:
    if not chunks_by_horizon or not any(chunks_by_horizon):
        raise ValueError("Cannot compute relative action statistics without unpadded action vectors.")

    stats: dict[str, list[np.ndarray]] = {key: [] for key in ("min", "max", "mean", "std", "q01", "q99")}
    counts: list[int] = []
    for horizon_idx, vectors in enumerate(chunks_by_horizon):
        if len(vectors) < 2:
            raise ValueError(
                "Cannot compute horizon-preserving relative action statistics from fewer than 2 "
                f"unpadded vectors at action timestep {horizon_idx}."
            )
        values = np.stack(vectors, axis=0).astype(np.float32)
        stats["min"].append(np.min(values, axis=0))
        stats["max"].append(np.max(values, axis=0))
        stats["mean"].append(np.mean(values, axis=0))
        stats["std"].append(np.std(values, axis=0))
        stats["q01"].append(np.quantile(values, 0.01, axis=0).astype(np.float32))
        stats["q99"].append(np.quantile(values, 0.99, axis=0).astype(np.float32))
        counts.append(len(vectors))

    computed = {key: np.stack(values, axis=0) for key, values in stats.items()}
    computed["count"] = np.asarray(counts, dtype=np.int64)
    return computed


def _iter_action_state_training_samples(dataset: Any):
    ensure_reader = getattr(dataset, "_ensure_reader", None)
    if callable(ensure_reader):
        reader = ensure_reader()
        if reader.hf_dataset is None:
            reader.load_and_activate()
        delta_indices = getattr(reader, "delta_indices", None)
        for idx in range(len(dataset)):
            item = reader.hf_dataset[idx]
            action = item.get(ACTION)
            state = item.get(OBS_STATE)
            pad_mask = None
            if delta_indices is not None and ACTION in delta_indices:
                ep_idx = _as_int(item["episode_index"])
                abs_idx = _as_int(item["index"])
                query_indices, padding = reader._get_query_indices(abs_idx, ep_idx)
                action = reader._query_hf_dataset({ACTION: query_indices[ACTION]})[ACTION]
                pad_mask = padding.get(f"{ACTION}_is_pad")
            yield action, state, pad_mask
        return

    for idx in range(len(dataset)):
        item = dataset[idx]
        yield item.get(ACTION), item.get(OBS_STATE), item.get(f"{ACTION}_is_pad")


def _make_relative_action_training_stats(
    dataset: Any,
    *,
    exclude_joints: list[str] | None,
    action_names: list[str] | None,
    preserve_action_horizon: bool = True,
) -> dict[str, dict[str, Any]]:
    try:
        dataset_len = len(dataset)
    except TypeError as exc:
        raise ValueError(
            "Cannot compute relative action statistics for a dataset without a finite length. "
            "Disable streaming or provide precomputed relative action statistics."
        ) from exc

    if dataset_len == 0:
        raise ValueError("Cannot compute relative action statistics for an empty dataset.")

    relative_step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=list(exclude_joints or []),
        action_names=action_names,
    )
    stats = deepcopy(getattr(getattr(dataset, "meta", None), "stats", {}) or {})
    chunks_by_horizon: list[list[np.ndarray]] | None = None
    num_vectors = 0

    for action_value, state_value, pad_mask in _iter_action_state_training_samples(dataset):
        action = _to_float_tensor(action_value, key=ACTION)
        state = _to_float_tensor(state_value, key=OBS_STATE)
        state_batch = _state_reference_batch(state)
        action_batch = _action_training_batch(action, state_batch)
        if action_batch.shape[0] != state_batch.shape[0]:
            if state_batch.shape[0] == 1:
                state_batch = state_batch.expand(action_batch.shape[0], -1)
            else:
                raise ValueError(
                    "Cannot compute relative action statistics: action and state batch sizes differ "
                    f"({action_batch.shape[0]} vs {state_batch.shape[0]})."
                )

        relative_action = to_relative_actions(
            action_batch,
            state_batch,
            relative_step._build_mask(action_batch.shape[-1]),
        )
        if not preserve_action_horizon:
            relative_action = relative_action.reshape(-1, relative_action.shape[-1]).unsqueeze(0)
            pad_mask = None
        sample_chunks = _relative_action_chunks_by_horizon(relative_action, pad_mask)
        if chunks_by_horizon is None:
            chunks_by_horizon = [[] for _ in range(len(sample_chunks))]
        if len(sample_chunks) != len(chunks_by_horizon):
            raise ValueError(
                "Cannot compute horizon-preserving relative action statistics from samples with "
                f"different action horizons ({len(sample_chunks)} vs {len(chunks_by_horizon)})."
            )
        for horizon_idx, vectors in enumerate(sample_chunks):
            chunks_by_horizon[horizon_idx].extend(vectors)
            num_vectors += len(vectors)

    if num_vectors < 2:
        raise ValueError(
            "Cannot compute relative action statistics from fewer than 2 unpadded action vectors."
        )

    stats[ACTION] = _compute_horizon_relative_action_stats(chunks_by_horizon or [])
    return stats


def _relative_stats_action_horizon(action_stats: dict[str, Any]) -> int | None:
    """Return the chunk horizon of horizon-preserving relative action stats, if any."""
    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        value = action_stats.get(stat_name)
        if value is None:
            continue
        tensor = torch.as_tensor(value)
        return tensor.shape[0] if tensor.ndim >= 2 else None
    return None


def _stats_preserve_action_horizon(stats: dict[str, dict[str, Any]] | None) -> bool:
    if not stats or ACTION not in stats:
        return False
    action_stats = stats.get(ACTION) or {}
    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        value = action_stats.get(stat_name)
        if value is None:
            continue
        return torch.as_tensor(value).ndim >= 2
    return False


def _make_relative_action_training_stats_from_dataset_meta(
    config: GrootConfig, dataset_meta: Any | None
) -> dict[str, dict[str, Any]] | None:
    repo_id = getattr(dataset_meta, "repo_id", None)
    root = getattr(dataset_meta, "root", None)
    fps = getattr(dataset_meta, "fps", None)
    if dataset_meta is None or repo_id is None or root is None or fps is None:
        return None

    require_package("datasets", extra="groot")

    # Relative stats are computed per chunk timestep at the native N1.7 horizon, so the
    # stats dataset must yield native-length action windows even when config.chunk_size
    # executes fewer steps.
    delta_timestamps = {ACTION: [index / fps for index in range(N1_7_NATIVE_ACTION_HORIZON)]}
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        revision=getattr(dataset_meta, "revision", None),
        download_videos=False,
        return_uint8=True,
    )
    return _make_relative_action_training_stats(
        dataset,
        exclude_joints=list(config.relative_exclude_joints or []),
        action_names=_resolve_action_feature_names_from_dataset_meta(dataset_meta),
        preserve_action_horizon=True,
    )


def _slice_stats_entry(stats: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    if not indices:
        return {}

    max_index = max(indices)
    sliced: dict[str, Any] = {}
    for stat_name, value in stats.items():
        if stat_name == "count":
            sliced[stat_name] = torch.as_tensor(value).flatten().tolist()
            continue
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim >= 2:
            if tensor.shape[-1] <= max_index:
                continue
            sliced[stat_name] = tensor[..., indices].tolist()
        else:
            tensor = tensor.flatten()
            if tensor.numel() <= max_index:
                continue
            sliced[stat_name] = [float(tensor[index].item()) for index in indices]

    if "min" in sliced and "max" in sliced:
        min_arr = np.asarray(sliced["min"], dtype=np.float32)
        max_arr = np.asarray(sliced["max"], dtype=np.float32)
        if "mean" not in sliced:
            sliced["mean"] = ((min_arr + max_arr) * 0.5).tolist()
        if "std" not in sliced:
            sliced["std"] = (np.abs(max_arr - min_arr) * 0.5).tolist()
    return sliced


def _feature_group_key(name: str) -> str:
    base = name.removesuffix(".pos").split(".")[-1]
    return base.replace(" ", "_") or "action"


def _infer_n1_7_action_groups(
    action_names: list[str],
    *,
    action_dim: int,
    exclude_joints: list[str],
) -> list[_GrootN17ActionGroup]:
    if not action_names or action_dim <= 0:
        return []

    names = list(action_names[:action_dim])
    exclude_tokens = [str(token).lower() for token in exclude_joints if token]
    groups: list[_GrootN17ActionGroup] = []
    current_indices: list[int] = []

    def flush_relative_group() -> None:
        if not current_indices:
            return
        key = (
            "single_arm"
            if not any(group.key == "single_arm" for group in groups)
            else f"single_arm_{len(groups)}"
        )
        groups.append(_GrootN17ActionGroup(key=key, indices=list(current_indices), relative=True))
        current_indices.clear()

    for index, name in enumerate(names):
        lowered = str(name).lower()
        is_excluded = any(token == lowered or token in lowered for token in exclude_tokens)
        if is_excluded:
            flush_relative_group()
            groups.append(
                _GrootN17ActionGroup(key=_feature_group_key(str(name)), indices=[index], relative=False)
            )
        else:
            current_indices.append(index)

    flush_relative_group()
    return groups


def _group_stats_by_action_groups(
    stats: dict[str, Any], groups: list[_GrootN17ActionGroup]
) -> dict[str, dict[str, list[float]]]:
    return {group.key: _slice_stats_entry(stats, group.indices) for group in groups}


def _grouped_stats_support_percentiles(
    raw_stats: dict[str, Any],
    modality_config: dict[str, Any],
    *,
    use_relative_action: bool,
) -> bool:
    state_keys = modality_config.get("state", {}).get("modality_keys", [])
    for key in state_keys:
        stats = raw_stats.get("state", {}).get(key, {})
        if "q01" not in stats or "q99" not in stats:
            return False

    action_cfg = modality_config.get("action", {})
    action_keys = action_cfg.get("modality_keys", [])
    action_configs = action_cfg.get("action_configs", [])
    for idx, key in enumerate(action_keys):
        cfg = action_configs[idx] if idx < len(action_configs) else {}
        is_relative = (
            use_relative_action and isinstance(cfg, dict) and config_value(cfg.get("rep")) == "relative"
        )
        if is_relative:
            continue
        stats = raw_stats.get("action", {}).get(key, {})
        if "q01" not in stats or "q99" not in stats:
            return False
    return True


def _build_n1_7_relative_action_processor_assets(
    config: GrootConfig,
    dataset_stats: dict[str, dict[str, Any]] | None,
    dataset_meta: Any | None,
    *,
    base_assets: _GrootN17CheckpointProcessorAssets | None = None,
) -> _GrootN17CheckpointProcessorAssets | None:
    if not config.use_relative_actions or not dataset_stats:
        return None

    try:
        action_dim = int(config.output_features[ACTION].shape[0])
    except Exception:
        return None

    action_names = _resolve_action_feature_names_from_dataset_meta(dataset_meta)
    if not action_names:
        return None

    groups = _infer_n1_7_action_groups(
        action_names,
        action_dim=action_dim,
        exclude_joints=list(config.relative_exclude_joints or []),
    )
    if not groups or not any(group.relative for group in groups):
        return None

    meta_stats = getattr(dataset_meta, "stats", None) or {}
    state_stats = (meta_stats.get(OBS_STATE) if isinstance(meta_stats, dict) else None) or dataset_stats.get(
        OBS_STATE, {}
    )
    absolute_action_stats = (
        meta_stats.get(ACTION) if isinstance(meta_stats, dict) else None
    ) or dataset_stats.get(ACTION, {})
    relative_action_stats = dataset_stats.get(ACTION, {})
    if not state_stats or not absolute_action_stats or not relative_action_stats:
        return None

    raw_stats: dict[str, Any] = {
        "state": _group_stats_by_action_groups(state_stats, groups),
        "action": _group_stats_by_action_groups(absolute_action_stats, groups),
        "relative_action": {
            group.key: _slice_stats_entry(relative_action_stats, group.indices)
            for group in groups
            if group.relative
        },
    }

    action_configs = [
        {
            "rep": "RELATIVE" if group.relative else "ABSOLUTE",
            "type": "NON_EEF",
            "format": "DEFAULT",
            "state_key": None,
        }
        for group in groups
    ]
    # Horizon-preserving relative stats are computed per chunk timestep at the native
    # chunk length of the dataset samples, so they dictate the processor horizon even
    # when config.chunk_size asks for fewer executed steps.
    action_horizon = _relative_stats_action_horizon(relative_action_stats) or min(
        config.chunk_size, N1_7_NATIVE_ACTION_HORIZON
    )
    modality_config: dict[str, Any] = {
        "state": {"modality_keys": [group.key for group in groups]},
        "action": {
            "modality_keys": [group.key for group in groups],
            "action_configs": action_configs,
            "delta_indices": list(range(action_horizon)),
        },
    }
    video_modality_keys = (
        base_assets.video_modality_keys if base_assets is not None else None
    ) or _resolve_visual_modality_keys_from_dataset_meta(dataset_meta)
    if video_modality_keys:
        modality_config["video"] = {
            "modality_keys": list(video_modality_keys),
            "delta_indices": [0],
        }

    if config.chunk_size > action_horizon:
        logging.warning(
            "GrootConfig.chunk_size=%d exceeds the relative-action stats horizon %d; clamping the "
            "valid action horizon to %d. The GR00T N1.7 action head decodes at most the horizon "
            "baked into the relative-action statistics.",
            config.chunk_size,
            action_horizon,
            action_horizon,
        )

    use_percentiles = _grouped_stats_support_percentiles(raw_stats, modality_config, use_relative_action=True)
    flat_stats = {
        OBS_STATE: flatten_n1_7_modality_stats(
            embodiment_stats=raw_stats,
            embodiment_config=modality_config,
            modality="state",
            use_percentiles=use_percentiles,
            use_relative_action=True,
        ),
        ACTION: flatten_n1_7_modality_stats(
            embodiment_stats=raw_stats,
            embodiment_config=modality_config,
            modality="action",
            use_percentiles=use_percentiles,
            use_relative_action=True,
        ),
    }

    return _GrootN17CheckpointProcessorAssets(
        stats=flat_stats,
        raw_stats=raw_stats,
        modality_config=modality_config,
        embodiment_mapping=base_assets.embodiment_mapping
        if base_assets is not None
        else dict(N1_7_EMBODIMENT_MAPPING),
        formalize_language=base_assets.formalize_language if base_assets is not None else True,
        valid_action_horizon=min(config.chunk_size, action_horizon),
        max_action_horizon=action_horizon,
        video_horizon=base_assets.video_horizon if base_assets is not None else None,
        use_percentiles=use_percentiles,
        use_relative_action=True,
        state_dropout_prob=base_assets.state_dropout_prob if base_assets is not None else 0.0,
        clip_outliers=base_assets.clip_outliers if base_assets is not None else True,
        video_modality_keys=video_modality_keys,
        image_crop_size=base_assets.image_crop_size if base_assets is not None else None,
        image_target_size=base_assets.image_target_size if base_assets is not None else None,
        shortest_image_edge=base_assets.shortest_image_edge if base_assets is not None else None,
        crop_fraction=base_assets.crop_fraction if base_assets is not None else None,
        use_albumentations=base_assets.use_albumentations if base_assets is not None else False,
        letter_box_transform=base_assets.letter_box_transform if base_assets is not None else False,
    )


def make_groot_pre_post_processors(
    config: GrootConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create preprocessor and postprocessor for Groot policy.

    This creates a processing pipeline that transforms LeRobot data format into
    the format expected by Isaac-GR00T models:

    Preprocessing steps:
    1. Optional key renaming (dataset-specific key mapping)
    2. Add batch dimension to unbatched data
    3. Pack video/state/action/language/embodiment and apply optional min-max normalization before padding
    4. Encode video+language with the GR00T N1.7 VLM backbone (Qwen3-VL) into intermediate VLM content
    5. Collate the VLM content into batched backbone input tensors
    6. Move tensors to device (GPU)

    NOTE: We optionally apply min-max normalization to STATE and ACTION using
    dataset-provided statistics prior to padding, mapping values to [-1, 1].
    This mirrors SO100-style preprocessing and keeps scales consistent with GR00T.

    Args:
        config: Groot configuration containing data_config, embodiment_tag, etc.
        dataset_stats: Optional per-key min/max statistics for normalization before padding.

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """

    dataset_meta = dataset_meta or getattr(config, "_runtime_dataset_meta", None)
    checkpoint_assets = _load_n1_7_checkpoint_processor_assets(config)
    checkpoint_stats = checkpoint_assets.stats if checkpoint_assets is not None else None
    checkpoint_has_stats = has_modality_stats(checkpoint_stats)
    if config.use_relative_actions and not checkpoint_has_stats:
        relative_dataset_stats = dataset_stats
        if not _stats_preserve_action_horizon(relative_dataset_stats):
            relative_dataset_stats = _make_relative_action_training_stats_from_dataset_meta(
                config, dataset_meta
            )
        relative_assets = _build_n1_7_relative_action_processor_assets(
            config,
            relative_dataset_stats,
            dataset_meta,
            base_assets=checkpoint_assets,
        )
        if relative_assets is None:
            raise ValueError(
                "GR00T relative-action training requires horizon-preserving relative action statistics. "
                "Pass dataset_meta with a local LeRobot dataset root, or pass precomputed relative dataset_stats."
            )
        checkpoint_assets = relative_assets
        checkpoint_stats = checkpoint_assets.stats
        checkpoint_has_stats = has_modality_stats(checkpoint_stats)

    action_horizon = (
        checkpoint_assets.max_action_horizon
        if checkpoint_assets is not None and checkpoint_assets.max_action_horizon is not None
        else min(config.chunk_size, N1_7_NATIVE_ACTION_HORIZON)
    )
    valid_action_horizon = (
        checkpoint_assets.valid_action_horizon
        if checkpoint_assets is not None and checkpoint_assets.valid_action_horizon is not None
        else action_horizon
    )
    padded_stats = checkpoint_stats if checkpoint_has_stats else (dataset_stats or {})
    embodiment_mapping = (
        checkpoint_assets.embodiment_mapping
        if checkpoint_assets is not None
        else dict(N1_7_EMBODIMENT_MAPPING)
    )
    formalize_language = checkpoint_assets.formalize_language if checkpoint_assets is not None else True
    clip_outliers = checkpoint_assets.clip_outliers if checkpoint_assets is not None else True
    video_modality_keys = checkpoint_assets.video_modality_keys if checkpoint_assets is not None else None
    try:
        env_action_dim = int(config.output_features[ACTION].shape[0])
    except Exception:
        env_action_dim = 0
    pack_step = GrootN17PackInputsStep(
        state_horizon=1,
        action_horizon=action_horizon,
        valid_action_horizon=valid_action_horizon,
        video_horizon=checkpoint_assets.video_horizon if checkpoint_assets is not None else None,
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        language_key="task",
        formalize_language=formalize_language,
        embodiment_tag=config.embodiment_tag,
        embodiment_mapping=embodiment_mapping,
        normalize_min_max=True,
        training=dataset_meta is not None,
        state_dropout_prob=(checkpoint_assets.state_dropout_prob if checkpoint_assets is not None else 0.0),
        stats=padded_stats,
        clip_outliers=clip_outliers,
        video_modality_keys=video_modality_keys,
        raw_stats=checkpoint_assets.raw_stats if checkpoint_assets is not None else None,
        use_percentiles=checkpoint_assets.use_percentiles if checkpoint_assets is not None else False,
        modality_config=checkpoint_assets.modality_config if checkpoint_assets is not None else None,
    )

    # Resolve the image preprocessing geometry. Honor the checkpoint's processor_config
    # when it provides an image_target_size; otherwise fall back to the geometry the
    # N1.7 backbone was trained on. Without this fallback a raw base checkpoint with no
    # processor_config image sizing (e.g. fine-tuning nvidia/GR00T-N1.7-3B with a new
    # embodiment, where checkpoint_assets is None) would patchify full-resolution camera
    # frames, inflating the VLM token count and feeding the model a resolution it was not trained on.
    if checkpoint_assets is not None and checkpoint_assets.image_target_size is not None:
        image_target_size = checkpoint_assets.image_target_size
        image_crop_size = checkpoint_assets.image_crop_size
        shortest_image_edge = checkpoint_assets.shortest_image_edge
        crop_fraction = checkpoint_assets.crop_fraction
    else:
        image_target_size = list(N1_7_DEFAULT_IMAGE_TARGET_SIZE)
        image_crop_size = list(N1_7_DEFAULT_IMAGE_CROP_SIZE)
        shortest_image_edge = None
        crop_fraction = None
    use_albumentations = checkpoint_assets.use_albumentations if checkpoint_assets is not None else False
    letter_box_transform = checkpoint_assets.letter_box_transform if checkpoint_assets is not None else False

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        pack_step,
        GrootN17VLMEncodeStep(
            model_name=GROOT_N1_7_BACKBONE_MODEL,
            image_crop_size=image_crop_size,
            image_target_size=image_target_size,
            shortest_image_edge=shortest_image_edge,
            crop_fraction=crop_fraction,
            use_albumentations=use_albumentations,
            letter_box_transform=letter_box_transform,
            training=dataset_meta is not None,
            device=config.device,
        ),
        DeviceProcessorStep(device=config.device),
    ]
    uses_native_relative_actions = bool(
        checkpoint_assets is not None and checkpoint_assets.use_relative_action
    )
    relative_step: RelativeActionsProcessorStep | None = None
    if config.use_relative_actions and not uses_native_relative_actions:
        logging.warning(
            "GR00T relative actions are using the generic RelativeActionsProcessorStep fallback because "
            "the checkpoint already carries non-relative statistics. Relative deltas will be normalized "
            "with absolute action stats rather than Isaac-GR00T's per-horizon relative stats. For "
            "OSS-faithful relative normalization, build from a checkpoint without baked-in stats (or "
            "pass dataset_meta) so native relative stats are computed."
        )
        relative_step = RelativeActionsProcessorStep(
            enabled=True,
            exclude_joints=list(config.relative_exclude_joints or []),
            action_names=_resolve_action_feature_names_from_dataset_meta(dataset_meta),
        )
        input_steps.insert(2, relative_step)

    if checkpoint_assets is not None and not checkpoint_has_stats and not has_modality_stats(padded_stats):
        raise ValueError(
            f"GR00T N1.7 checkpoint '{config.base_model_path}' has no statistics for embodiment tag "
            f"'{config.embodiment_tag}', and no dataset stats were provided to fall back to, so "
            "actions cannot be normalized or decoded. Pass dataset_stats, or set "
            "config.embodiment_tag to an embodiment present in the checkpoint's statistics.json."
        )
    if checkpoint_assets is None or not checkpoint_has_stats:
        # When the checkpoint sidecars have no stats for the configured
        # embodiment tag (e.g. finetuning a raw base checkpoint with the
        # default 'new_embodiment' tag), the pack step above normalized with
        # the dataset stats; the decode step must invert with the same stats
        # instead of using a checkpoint decoder whose empty stats would
        # silently return normalized [-1, 1] actions.
        action_decode_step: ProcessorStep = GrootActionUnpackUnnormalizeStep(
            env_action_dim=env_action_dim,
            stats=padded_stats,
            normalize_min_max=True,
            clip_normalized_action=True,
            libero_gripper_action=config.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
        )
    else:
        action_decode_step = GrootN17ActionDecodeStep(
            env_action_dim=env_action_dim,
            raw_stats=checkpoint_assets.raw_stats,
            modality_config=checkpoint_assets.modality_config,
            use_percentiles=checkpoint_assets.use_percentiles,
            use_relative_action=checkpoint_assets.use_relative_action,
            pack_step=pack_step,
            action_decode_transform=config.action_decode_transform,
        )

    output_steps: list[ProcessorStep] = [action_decode_step]
    if relative_step is not None:
        output_steps.append(AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step))
    output_steps.append(DeviceProcessorStep(device="cpu"))

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


# GR00T specific processor steps


def _to_uint8_np_bthwc(img_t: torch.Tensor) -> np.ndarray:
    # img_t: (B, C, H, W) or (B, T, C, H, W), float in [0,1] or uint8
    if img_t.dtype.is_floating_point:
        img_t = (img_t.clamp(0, 1) * 255.0).to(torch.uint8)
    if img_t.dim() == 4:
        return rearrange(img_t.cpu().numpy(), "b c h w -> b 1 h w c")
    if img_t.dim() == 5:
        return rearrange(img_t.cpu().numpy(), "b t c h w -> b t h w c")
    raise ValueError(f"Expected image tensor shape (B, C, H, W) or (B, T, C, H, W), got {tuple(img_t.shape)}")


def _align_video_horizon(video: np.ndarray, horizon: int | None) -> np.ndarray:
    """Match the checkpoint video horizon by truncating or left-padding frames."""

    if horizon is None or horizon <= 0:
        return video
    current = video.shape[1]
    if current == horizon:
        return video
    if current > horizon:
        return video[:, -horizon:]
    pad = np.repeat(video[:, :1], horizon - current, axis=1)
    return np.concatenate([pad, video], axis=1)


def _build_n1_7_processor(model_name: str = GROOT_N1_7_BACKBONE_MODEL) -> ProcessorMixin:
    require_package("transformers", extra="groot")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    video_processor = Qwen3VLVideoProcessor.from_pretrained(model_name, trust_remote_code=True)
    proc = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )
    proc.tokenizer.padding_side = "left"
    return proc


def _transform_n1_7_image_for_vlm_albumentations(
    image: np.ndarray,
    *,
    image_crop_size: list[int] | None,
    image_target_size: list[int] | None,
    shortest_image_edge: int | None,
    crop_fraction: float | None,
    letter_box_transform: bool = False,
    crop_position: tuple[float, float] | None = None,
) -> np.ndarray:
    """cv2/INTER_AREA eval transform mirroring Isaac-GR00T's albumentations preprocessing.

    Used only for checkpoints saved with ``use_albumentations=True``. cv2 is
    CPU/numpy-only so this path cannot run on GPU; the default (non-albumentations)
    geometry is handled on-device by :func:`_transform_n1_7_image_for_vlm_torch`. The
    cv2/INTER_AREA resize and floored center-crop here intentionally differ from that
    torch path and must stay bit-exact to the upstream reference. The hot path accepts
    and returns numpy arrays to avoid per-frame PIL round-trips.

    ``crop_position`` selects where the ``crop_fraction`` window sits: ``None``
    keeps the deterministic center crop (eval contract), while ``(y, x)``
    fractions in [0, 1] place the window for Isaac's train-time random crop
    (0.5, 0.5 == center). Training samples one position per sample and reuses
    it across camera views.
    """
    if image_target_size is None:
        return image

    target_h, target_w = image_target_size

    image_np = np.asarray(image)
    if image_np.ndim == 2:
        image_np = np.repeat(image_np[:, :, None], 3, axis=2)
    elif image_np.ndim == 3 and image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]

    if not image_np.flags.c_contiguous:
        image_np = np.ascontiguousarray(image_np)

    if letter_box_transform:
        height, width = image_np.shape[:2]
        if height != width:
            square_edge = max(height, width)
            pad_h = square_edge - height
            pad_w = square_edge - width
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            image_np = cv2.copyMakeBorder(image_np, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    resize_edge = shortest_image_edge or target_h

    def resize_shortest_edge(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        shortest_edge = min(height, width)
        if shortest_edge == resize_edge:
            return frame
        scale = resize_edge / float(shortest_edge)
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))
        return cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    image_np = resize_shortest_edge(image_np)

    if crop_fraction is None and image_crop_size is not None:
        crop_fraction = image_crop_size[0] / float(target_h)
    if crop_fraction is not None and 0.0 < crop_fraction < 1.0:
        height, width = image_np.shape[:2]
        crop_h = max(1, int(height * crop_fraction))
        crop_w = max(1, int(width * crop_fraction))
        if crop_position is None:
            top = max(0, (height - crop_h) // 2)
            left = max(0, (width - crop_w) // 2)
        else:
            pos_y, pos_x = crop_position
            top = int(round((height - crop_h) * min(max(pos_y, 0.0), 1.0)))
            left = int(round((width - crop_w) * min(max(pos_x, 0.0), 1.0)))
        image_np = image_np[top : top + crop_h, left : left + crop_w]

    return resize_shortest_edge(image_np)


def _transform_n1_7_image_for_vlm_torch(
    image: torch.Tensor,
    *,
    image_crop_size: list[int] | None,
    image_target_size: list[int] | None,
    shortest_image_edge: int | None,
    crop_fraction: float | None,
    letter_box_transform: bool = False,
) -> torch.Tensor:
    """Default (non-albumentations) N1.7 image transform.

    Optionally pads to square, then resizes to ``shortest_image_edge``, center-crops
    by ``crop_fraction``, and resizes to ``image_target_size``.

    Operates on a ``(C, H, W)`` uint8 tensor and keeps the result on the input
    tensor's device so the resize/crop run on GPU when the tensor is. Bicubic
    interpolation with antialiasing matches PIL's ``Image.Resampling.BICUBIC``
    closely (sub-``2/255`` per-pixel on worst-case inputs). The ``use_albumentations``
    cv2/INTER_AREA path has no torch equivalent and stays on
    :func:`_transform_n1_7_image_for_vlm_albumentations`.
    """
    if image_target_size is None:
        return image

    target_h, target_w = image_target_size
    _, height, width = image.shape

    if letter_box_transform:
        square_edge = max(height, width)
        if height != width:
            left = (square_edge - width) // 2
            top = (square_edge - height) // 2
            image = tv_functional.pad(
                image, [left, top, square_edge - width - left, square_edge - height - top], fill=0
            )

    resize_edge = shortest_image_edge or target_h
    image = tv_functional.resize(
        image, [resize_edge, resize_edge], interpolation=InterpolationMode.BICUBIC, antialias=True
    )

    if crop_fraction is None and image_crop_size is not None:
        crop_fraction = image_crop_size[0] / float(target_h)
    if crop_fraction is not None and 0.0 < crop_fraction < 1.0:
        # Match the PIL helper's center crop exactly: round() the crop size but
        # floor() the offset (torchvision.center_crop rounds the offset, which
        # shifts the region by 1px when (edge - crop) is odd).
        crop_h = max(1, int(round(image.shape[-2] * crop_fraction)))
        crop_w = max(1, int(round(image.shape[-1] * crop_fraction)))
        top = max(0, (image.shape[-2] - crop_h) // 2)
        left = max(0, (image.shape[-1] - crop_w) // 2)
        image = image[..., top : top + crop_h, left : left + crop_w]

    if tuple(image.shape[-2:]) != (target_h, target_w):
        image = tv_functional.resize(
            image, [target_h, target_w], interpolation=InterpolationMode.BICUBIC, antialias=True
        )
    return image


@dataclass
@ProcessorStepRegistry.register(name="groot_n1_7_pack_inputs_v1")
class GrootN17PackInputsStep(ProcessorStep):
    """Pack LeRobot transitions into the raw tensor layout expected by N1.7.

    This step preserves the checkpoint's camera order, video horizon, language
    formatting, normalization statistics, action mask semantics, and embodiment
    id mapping before the Qwen3-VL processor sees the sample.
    """

    state_horizon: int = 1
    action_horizon: int = N1_7_NATIVE_ACTION_HORIZON
    valid_action_horizon: int = N1_7_NATIVE_ACTION_HORIZON
    video_horizon: int | None = None
    max_state_dim: int = 132
    max_action_dim: int = 132
    language_key: str = "task"
    formalize_language: bool = True
    embodiment_tag: str = "new_embodiment"
    embodiment_mapping: dict[str, int] = field(default_factory=lambda: dict(N1_7_EMBODIMENT_MAPPING))
    normalize_min_max: bool = True
    training: bool = False
    state_dropout_prob: float = 0.0
    stats: dict[str, dict[str, Any]] | None = None
    clip_outliers: bool = True
    use_percentiles: bool = False
    video_modality_keys: list[str] | None = None
    raw_stats: dict[str, Any] | None = None
    modality_config: dict[str, Any] | None = None
    _last_raw_state: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _warned_image_keys: bool = field(default=False, init=False, repr=False)

    def _ordered_image_keys(self, obs: dict[str, Any]) -> list[str]:
        available = {key for key in obs if key.startswith(OBS_IMAGES)}
        if not available and OBS_IMAGE in obs:
            return [OBS_IMAGE]
        if not self.video_modality_keys:
            return sorted(available)

        ordered: list[str] = []
        unmatched: list[str] = []
        for modality_key in self.video_modality_keys:
            candidates = [f"{OBS_IMAGES}.{modality_key}"]
            # Alias for datasets converted with generic camera names (e.g. the
            # LIBERO conversions expose the wrist camera as
            # `observation.images.image2`), so raw N1.7 LIBERO checkpoints
            # match those datasets out of the box.
            if modality_key == "wrist_image":
                candidates.append(f"{OBS_IMAGES}.image2")

            match = next((candidate for candidate in candidates if candidate in available), None)
            if match is None:
                unmatched.append(modality_key)
            else:
                ordered.append(match)

        if not ordered:
            if not self._warned_image_keys:
                self._warned_image_keys = True
                logging.warning(
                    "None of the GR00T N1.7 checkpoint video modality keys %s match a camera among %s; "
                    "falling back to feeding all cameras in alphabetical order, which is unlikely to be "
                    "the layout the checkpoint was trained with. Rename the dataset cameras (e.g. via "
                    "--rename_map) to match the checkpoint keys.",
                    self.video_modality_keys,
                    sorted(available),
                )
            return sorted(available)
        unused = sorted(available - set(ordered))
        if (unmatched or unused) and not self._warned_image_keys:
            self._warned_image_keys = True
            if unmatched:
                logging.warning(
                    "GR00T N1.7 checkpoint video modality keys %s have no matching camera among %s; "
                    "the model will receive %d view(s) instead of the %d it was trained with. Rename "
                    "the dataset cameras (e.g. via --rename_map) to match the checkpoint keys %s.",
                    unmatched,
                    sorted(available),
                    len(ordered),
                    len(self.video_modality_keys),
                    self.video_modality_keys,
                )
            if unused:
                logging.warning(
                    "Dropping camera(s) %s: the GR00T N1.7 checkpoint only consumes the video modality "
                    "keys %s, which matched %s.",
                    unused,
                    self.video_modality_keys,
                    ordered,
                )
        return ordered

    def _state_groups_from_tensor(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.modality_config is None or self.raw_stats is None:
            return {}
        state_config = self.modality_config.get("state", {})
        if not isinstance(state_config, dict):
            return {}
        state_keys = state_config.get("modality_keys", [])
        if not isinstance(state_keys, list):
            return {}

        grouped: dict[str, torch.Tensor] = {}
        start_idx = 0
        for key in state_keys:
            if not isinstance(key, str):
                continue
            key_stats = self.raw_stats.get("state", {}).get(key, {})
            dim = stat_dim_from_entry(key_stats) if isinstance(key_stats, dict) else 0
            if dim <= 0:
                continue
            grouped[key] = state[:, start_idx : start_idx + dim]
            start_idx += dim
        return grouped

    def _convert_relative_action_groups_for_training(
        self, action: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        if self.modality_config is None or self.raw_stats is None:
            return action

        action_config = self.modality_config.get("action", {})
        if not isinstance(action_config, dict):
            return action
        action_keys = action_config.get("modality_keys", [])
        action_configs = action_config.get("action_configs", [])
        if not isinstance(action_keys, list) or not isinstance(action_configs, list):
            return action

        state_groups = self._state_groups_from_tensor(state)
        if not state_groups:
            return action

        converted = action
        start_idx = 0
        cloned = False
        for idx, key in enumerate(action_keys):
            if not isinstance(key, str):
                continue
            key_stats = self.raw_stats.get("action", {}).get(key, {})
            dim = stat_dim_from_entry(key_stats) if isinstance(key_stats, dict) else 0
            if dim <= 0:
                continue
            end_idx = start_idx + dim
            if end_idx > action.shape[-1]:
                break

            cfg = (
                action_configs[idx]
                if idx < len(action_configs) and isinstance(action_configs[idx], dict)
                else {}
            )
            if config_value(cfg.get("rep")) == "relative":
                action_type = config_value(cfg.get("type"))
                if action_type != "non_eef":
                    raise ValueError(f"Unsupported relative N1.7 action config for '{key}': {cfg}")
                state_key = cfg.get("state_key") or key
                reference = state_groups.get(state_key)
                if reference is None:
                    raise KeyError(f"Missing raw state group '{state_key}' for relative N1.7 action '{key}'")
                if reference.shape[-1] != dim:
                    raise ValueError(
                        f"Relative N1.7 action group '{key}' has dim {dim}, but state group "
                        f"'{state_key}' has dim {reference.shape[-1]}."
                    )
                if not cloned:
                    converted = action.clone()
                    cloned = True
                converted[..., start_idx:end_idx] -= reference[:, None, :]

            start_idx = end_idx

        return converted

    def _normalize_action_groups_for_training(self, action: torch.Tensor) -> torch.Tensor | None:
        if self.modality_config is None or self.raw_stats is None:
            return None

        action_config = self.modality_config.get("action", {})
        if not isinstance(action_config, dict):
            return None
        action_keys = action_config.get("modality_keys", [])
        action_configs = action_config.get("action_configs", [])
        if not isinstance(action_keys, list) or not isinstance(action_configs, list):
            return None

        normalized_groups: list[torch.Tensor] = []
        start_idx = 0
        for idx, key in enumerate(action_keys):
            if not isinstance(key, str):
                continue
            cfg = (
                action_configs[idx]
                if idx < len(action_configs) and isinstance(action_configs[idx], dict)
                else {}
            )
            is_relative = config_value(cfg.get("rep")) == "relative"
            stats_modality = "relative_action" if is_relative else "action"
            key_stats = self.raw_stats.get(stats_modality, {}).get(key, {})
            dim = stat_dim_from_entry(key_stats) if isinstance(key_stats, dict) else 0
            if dim <= 0:
                continue
            end_idx = start_idx + dim
            if end_idx > action.shape[-1]:
                return None

            min_v, max_v = _n1_7_decode_stats_for_action(
                self.raw_stats,
                key,
                cfg,
                use_relative_action=True,
                use_percentiles=self.use_percentiles,
            )
            group = action[..., start_idx:end_idx]
            min_t = torch.as_tensor(min_v, dtype=group.dtype, device=group.device)
            max_t = torch.as_tensor(max_v, dtype=group.dtype, device=group.device)
            if min_t.ndim == 1:
                min_t = min_t.view(1, 1, -1)
                max_t = max_t.view(1, 1, -1)
            elif min_t.ndim == 2:
                if group.shape[1] > min_t.shape[0]:
                    return None
                min_t = min_t[: group.shape[1]].unsqueeze(0)
                max_t = max_t[: group.shape[1]].unsqueeze(0)
            else:
                return None

            denom = max_t - min_t
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            normalized = torch.where(mask, 2 * (group - min_t) / safe_denom - 1, torch.zeros_like(group))
            if self.clip_outliers:
                normalized = normalized.clamp(-1.0, 1.0)
            normalized_groups.append(normalized)
            start_idx = end_idx

        if not normalized_groups or start_idx != action.shape[-1]:
            return None
        return torch.cat(normalized_groups, dim=-1)

    def _uses_relative_action_groups(self) -> bool:
        """True when the action modality declares at least one relative group.

        Relative groups normalize with per-chunk-timestep (2D) ``relative_action`` stats, which the
        flat ``_min_max_norm`` fallback cannot honor, so a relative config that fails grouped
        normalization must fail loudly rather than silently wrongly scale every timestep.
        """
        if not isinstance(self.modality_config, dict):
            return False
        action_config = self.modality_config.get("action", {})
        if not isinstance(action_config, dict):
            return False
        action_configs = action_config.get("action_configs", [])
        if not isinstance(action_configs, list):
            return False
        return any(
            isinstance(cfg, dict) and config_value(cfg.get("rep")) == "relative" for cfg in action_configs
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        raw_state_for_action: torch.Tensor | None = None

        def _align_vec(vec: Any, target_dim: int, *, default: float) -> torch.Tensor:
            t = torch.as_tensor(vec)
            t = t.flatten().to(
                dtype=torch.float32,
                device=next(
                    (v.device for v in obs.values() if isinstance(v, torch.Tensor)), torch.device("cpu")
                ),
            )
            d = int(t.shape[-1]) if t.numel() > 0 else 0
            if d == target_dim:
                return t
            if d < target_dim:
                pad = torch.full((target_dim - d,), default, dtype=t.dtype, device=t.device)
                return torch.cat([t, pad], dim=0)
            return t[:target_dim]

        def _min_max_norm(x: torch.Tensor, key: str) -> torch.Tensor:
            if not self.normalize_min_max or self.stats is None or key not in self.stats:
                return x
            stats_k = self.stats[key]
            last_dim = x.shape[-1]
            min_v = _align_vec(stats_k.get("min", torch.zeros(last_dim)), last_dim, default=0.0)
            max_v = _align_vec(stats_k.get("max", torch.ones(last_dim)), last_dim, default=1.0)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            mapped = 2 * (x - min_v) / safe_denom - 1
            normalized = torch.where(mask, mapped, torch.zeros_like(mapped))
            if self.clip_outliers:
                normalized = normalized.clamp(-1.0, 1.0)
            return normalized

        def _cache_raw_state(state: torch.Tensor) -> None:
            if self.modality_config is None or self.raw_stats is None:
                return
            state_config = self.modality_config.get("state", {})
            if not isinstance(state_config, dict):
                return
            state_keys = state_config.get("modality_keys", [])
            if not isinstance(state_keys, list):
                return

            raw_state = state.detach().cpu().float().numpy()
            start_idx = 0
            grouped: dict[str, np.ndarray] = {}
            for key in state_keys:
                if not isinstance(key, str):
                    continue
                key_stats = self.raw_stats.get("state", {}).get(key, {})
                dim = len(key_stats.get("mean") or key_stats.get("min") or key_stats.get("q01") or [])
                if dim <= 0:
                    continue
                grouped[key] = raw_state[:, start_idx : start_idx + dim]
                start_idx += dim
            if grouped:
                self._last_raw_state = grouped

        img_keys = self._ordered_image_keys(obs)
        if img_keys:
            cams = [_align_video_horizon(_to_uint8_np_bthwc(obs[k]), self.video_horizon) for k in img_keys]
            video = np.stack(cams, axis=2)  # (B, T, V, H, W, C)
            obs["video"] = video
            image_keys_to_remove = [key for key in obs if key.startswith(OBS_IMAGES)]
            if OBS_IMAGE in obs:
                image_keys_to_remove.append(OBS_IMAGE)
            for k in image_keys_to_remove:
                obs.pop(k, None)

        bsz, _device = infer_n1_7_batch_size_and_device(obs, transition.get(TransitionKey.ACTION))
        comp["language"] = prepare_n1_7_language_batch(
            comp.get(self.language_key),
            bsz,
            formalize_language=self.formalize_language,
        )

        if OBS_STATE in obs:
            state = obs[OBS_STATE]
            if state.dim() != 2:
                raise ValueError(f"state must be (B, D), got {tuple(state.shape)}")
            bsz, dim = state.shape
            if dim > self.max_state_dim:
                raise ValueError(f"State dimension {dim} exceeds max_state_dim {self.max_state_dim}.")
            _cache_raw_state(state)
            raw_state_for_action = state
            if self.normalize_min_max:
                state = _min_max_norm(state, OBS_STATE)
            state = state.unsqueeze(1)
            if dim < self.max_state_dim:
                pad = torch.zeros(bsz, 1, self.max_state_dim - dim, dtype=state.dtype, device=state.device)
                state = torch.cat([state, pad], dim=2)
            if self.training and torch.is_grad_enabled() and self.state_dropout_prob > 0:
                drop_state = torch.tensor(
                    [random.random() < self.state_dropout_prob for _ in range(bsz)],
                    dtype=torch.bool,
                    device=state.device,
                ).view(bsz, 1, 1)
                state = state.masked_fill(drop_state, 0)
            obs["state"] = state

        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            if action.dim() == 2:
                action = action.unsqueeze(1)
            elif action.dim() == 3:
                pass
            else:
                raise ValueError(f"action must be (B, D) or (B, T, D), got {tuple(action.shape)}")

            bsz, horizon, dim = action.shape
            if horizon > self.action_horizon:
                raise ValueError(f"Action horizon {horizon} exceeds action_horizon {self.action_horizon}.")
            if dim > self.max_action_dim:
                raise ValueError(f"Action dimension {dim} exceeds max_action_dim {self.max_action_dim}.")
            if raw_state_for_action is not None:
                action = self._convert_relative_action_groups_for_training(action, raw_state_for_action)
            if self.normalize_min_max:
                normalized_action = self._normalize_action_groups_for_training(action)
                if normalized_action is not None:
                    action = normalized_action
                elif self._uses_relative_action_groups():
                    raise ValueError(
                        "GrootN17PackInputsStep could not apply native grouped normalization to a "
                        "relative-action chunk: the action layout or horizon does not match the "
                        f"checkpoint relative_action stats (action shape {tuple(action.shape)}). The flat "
                        "min/max fallback cannot honor per-chunk-timestep relative stats, so refusing to "
                        "silently wrongly normalize. Recompute the relative action stats so their horizon and "
                        "dimensions match the action chunk."
                    )
                else:
                    flat = _min_max_norm(action.reshape(bsz * horizon, dim), ACTION)
                    action = flat.view(bsz, horizon, dim)
            valid_dim = min(dim, self.max_action_dim)
            valid_horizon = min(horizon, self.valid_action_horizon, self.action_horizon)
            if dim < self.max_action_dim:
                pad = torch.zeros(
                    bsz, horizon, self.max_action_dim - dim, dtype=action.dtype, device=action.device
                )
                action = torch.cat([action, pad], dim=2)
            if horizon < self.action_horizon:
                pad = torch.zeros(
                    bsz,
                    self.action_horizon - horizon,
                    self.max_action_dim,
                    dtype=action.dtype,
                    device=action.device,
                )
                action = torch.cat([action, pad], dim=1)
                horizon = self.action_horizon
            horizon_valid = torch.zeros(bsz, horizon, dtype=torch.bool, device=action.device)
            horizon_valid[:, :valid_horizon] = True
            action_is_pad = comp.get(f"{ACTION}_is_pad")
            if action_is_pad is None:
                action_is_pad = comp.get("action_horizon_is_pad")
            if action_is_pad is not None:
                action_pad = torch.as_tensor(action_is_pad, dtype=torch.bool, device=action.device)
                if action_pad.ndim == 1:
                    if bsz == 1 and action_pad.numel() == horizon:
                        action_pad = action_pad.unsqueeze(0)
                    elif horizon == 1 and action_pad.numel() == bsz:
                        action_pad = action_pad.view(bsz, 1)
                if action_pad.ndim != 2 or action_pad.shape[0] != bsz:
                    raise ValueError(
                        "action_is_pad must have shape (B, T) matching the action batch; "
                        f"got {tuple(action_pad.shape)} for action {tuple(action.shape)}."
                    )
                pad_horizon = min(horizon, action_pad.shape[1])
                horizon_valid[:, :pad_horizon] &= ~action_pad[:, :pad_horizon]

            if valid_horizon < horizon or action_is_pad is not None:
                action = action.clone()
                action[:, valid_horizon:, :] = 0
                action = action * horizon_valid.unsqueeze(-1).to(dtype=action.dtype)
            action_mask = torch.zeros(
                bsz, horizon, self.max_action_dim, dtype=torch.float32, device=action.device
            )
            action_mask[:, :, :valid_dim] = horizon_valid.unsqueeze(-1).to(dtype=action_mask.dtype)
            transition[TransitionKey.ACTION] = action
            comp["action_mask"] = action_mask

        emb_id = self.embodiment_mapping.get(self.embodiment_tag, 0)
        bsz, device = infer_n1_7_batch_size_and_device(obs, transition.get(TransitionKey.ACTION))
        if "action_mask" not in comp:
            action_mask = torch.zeros(bsz, self.action_horizon, dtype=torch.float32, device=device)
            valid_horizon = min(self.valid_action_horizon, self.action_horizon)
            action_mask[:, :valid_horizon] = 1.0
            comp["action_mask"] = action_mask
        comp["embodiment_id"] = torch.full((bsz,), emb_id, dtype=torch.int32, device=device)

        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "state_horizon": self.state_horizon,
            "action_horizon": self.action_horizon,
            "valid_action_horizon": self.valid_action_horizon,
            "video_horizon": self.video_horizon,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "language_key": self.language_key,
            "formalize_language": self.formalize_language,
            "embodiment_tag": self.embodiment_tag,
            "embodiment_mapping": self.embodiment_mapping,
            "normalize_min_max": self.normalize_min_max,
            "state_dropout_prob": self.state_dropout_prob,
            "clip_outliers": self.clip_outliers,
            "use_percentiles": self.use_percentiles,
            "video_modality_keys": self.video_modality_keys,
            "raw_stats": self.raw_stats,
            "modality_config": self.modality_config,
        }

    def get_cached_raw_state(self) -> dict[str, np.ndarray] | None:
        """Return the latest unnormalized state split by checkpoint modality key."""

        return self._last_raw_state

    def state_dict(self) -> dict[str, torch.Tensor]:
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                flat[f"{key}.{stat_name}"] = torch.as_tensor(value).cpu()
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if not state:
            return
        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                reconstructed.setdefault(key, {})[stat_name] = tensor
        if reconstructed:
            self.stats = reconstructed


@dataclass
@ProcessorStepRegistry.register(name="groot_n1_7_vlm_encode_v1")
class GrootN17VLMEncodeStep(ProcessorStep):
    """Tokenize N1.7's packed video-language prompt with the Qwen3-VL processor.

    The packed video has shape ``(B, T, V, H, W, C)``. Each frame/view becomes
    an image item in the same chat message so the resulting image tokens match
    the temporal VLM packing used by Isaac-GR00T.

    Images are handed to the torchvision-backed Qwen3-VL processor as ``(C, H, W)``
    uint8 tensors (no per-frame PIL roundtrip), and, when ``device`` resolves to a
    CUDA device, the resize/rescale/normalize/patchify run there. This keeps the
    output bit-identical on CPU and moves the dominant preprocessing cost off
    the critical path on GPU.
    """

    model_name: str = GROOT_N1_7_BACKBONE_MODEL
    image_crop_size: list[int] | None = None
    image_target_size: list[int] | None = None
    shortest_image_edge: int | None = None
    crop_fraction: float | None = None
    use_albumentations: bool = False
    letter_box_transform: bool = False
    # Runtime-only train/eval mode: True enables Isaac's train-time random crop
    # (one window per sample, replayed across views); False keeps the
    # deterministic center crop. Never serialized - reloaded pipelines default
    # to eval and are re-enabled only when processors are built with dataset_meta.
    training: bool = False
    device: str | None = None
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_n1_7_processor(self.model_name)
        return self._proc

    def _target_device(self) -> torch.device | None:
        # The albumentations path is cv2/numpy only, so it cannot run on GPU.
        if self.device is None or self.use_albumentations:
            return None
        try:
            return get_safe_torch_device(self.device)
        except (AssertionError, RuntimeError):
            # A device serialized at train time (e.g. "cuda") may be unavailable
            # when the processor is reloaded elsewhere (e.g. CPU-only eval), and
            # this step is not in the standard device-override set. Fall back to
            # the CPU path, which is bit-identical, instead of crashing.
            return None

    def _build_sample_images(
        self, video: Any, batch_size: int, target_device: torch.device | None
    ) -> list[list[Any]]:
        """Return, per batch item, its ordered ``(timestep, view)`` frames.

        ``use_albumentations`` keeps the legacy per-frame cv2/INTER_AREA transform;
        otherwise frames are ``(C, H, W)`` uint8 tensors (moved to
        ``target_device`` when set) for the torchvision-backed Qwen processor.
        """
        if self.use_albumentations:
            video_np = np.asarray(video)
            train_crop = self.training and torch.is_grad_enabled()
            sample_images: list[list[Any]] = []
            for batch_idx in range(batch_size):
                # Isaac-GR00T samples ONE crop window per sample and replays it
                # across every (timestep, view) frame of that sample, keeping
                # cross-view geometry consistent. Eval keeps the center crop.
                crop_position = (random.random(), random.random()) if train_crop else None
                sample_images.append(
                    [
                        _transform_n1_7_image_for_vlm_albumentations(
                            video_np[batch_idx, timestep, view_idx],
                            image_crop_size=self.image_crop_size,
                            image_target_size=self.image_target_size,
                            shortest_image_edge=self.shortest_image_edge,
                            crop_fraction=self.crop_fraction,
                            letter_box_transform=self.letter_box_transform,
                            crop_position=crop_position,
                        )
                        for timestep in range(video_np.shape[1])
                        for view_idx in range(video_np.shape[2])
                    ]
                )
            return sample_images

        video_t = video if torch.is_tensor(video) else torch.from_numpy(np.ascontiguousarray(video))
        # (B, T, V, H, W, C) uint8 -> (B, T, V, C, H, W)
        video_t = video_t.permute(0, 1, 2, 5, 3, 4).contiguous()
        if target_device is not None and video_t.device != target_device:
            video_t = video_t.to(target_device, non_blocking=(target_device.type == "cuda"))

        frames_per_sample: list[list[Any]] = []
        for batch_idx in range(batch_size):
            sample = video_t[batch_idx]  # (T, V, C, H, W)
            frames_per_sample.append(
                [
                    _transform_n1_7_image_for_vlm_torch(
                        sample[timestep, view_idx],
                        image_crop_size=self.image_crop_size,
                        image_target_size=self.image_target_size,
                        shortest_image_edge=self.shortest_image_edge,
                        crop_fraction=self.crop_fraction,
                        letter_box_transform=self.letter_box_transform,
                    )
                    for timestep in range(sample.shape[0])
                    for view_idx in range(sample.shape[1])
                ]
            )
        return frames_per_sample

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        video = obs.get("video")
        if video is None:
            return transition

        batch_size = int(video.shape[0])
        languages = prepare_n1_7_language_batch(
            comp.get("language"),
            batch_size,
            formalize_language=False,
        )

        target_device = self._target_device()
        sample_images = self._build_sample_images(video, batch_size, target_device)

        texts: list[str] = []
        images: list[Any] = []
        for batch_idx in range(batch_size):
            frames = sample_images[batch_idx]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in frames],
                        {"type": "text", "text": languages[batch_idx]},
                    ],
                }
            ]
            texts.append(
                self.proc.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            images.extend(frames)

        proc_kwargs: dict[str, Any] = {
            "text": texts,
            "images": images,
            "return_tensors": "pt",
            "padding": True,
        }
        if target_device is not None:
            proc_kwargs["device"] = str(target_device)
        encoded = self.proc(**proc_kwargs)
        for key, value in encoded.items():
            comp[key] = value
        obs.pop("video", None)
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "image_crop_size": self.image_crop_size,
            "image_target_size": self.image_target_size,
            "shortest_image_edge": self.shortest_image_edge,
            "crop_fraction": self.crop_fraction,
            "use_albumentations": self.use_albumentations,
            "letter_box_transform": self.letter_box_transform,
            "device": self.device,
        }


def _n1_7_decode_stats_for_action(
    raw_stats: dict[str, Any],
    key: str,
    action_config: dict[str, Any],
    *,
    use_relative_action: bool,
    use_percentiles: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the min/max arrays needed to decode one checkpoint action group."""

    is_relative = use_relative_action and config_value(action_config.get("rep")) == "relative"
    modality = "relative_action" if is_relative else "action"
    stats = raw_stats.get(modality, {}).get(key, {})
    if not isinstance(stats, dict):
        raise KeyError(f"Missing N1.7 statistics for {modality}.{key}")
    min_name = "min" if is_relative else ("q01" if use_percentiles else "min")
    max_name = "max" if is_relative else ("q99" if use_percentiles else "max")
    if min_name not in stats or max_name not in stats:
        raise KeyError(f"Missing '{min_name}'/'{max_name}' statistics for {modality}.{key}")
    return np.asarray(stats[min_name], dtype=np.float32), np.asarray(stats[max_name], dtype=np.float32)


def _unnormalize_min_max(action: np.ndarray, min_v: np.ndarray, max_v: np.ndarray) -> np.ndarray:
    return (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * (max_v - min_v) + min_v


def _n1_7_decode_valid_horizon(action_config: dict[str, Any], action_np: np.ndarray) -> int | None:
    if action_np.ndim != 3:
        return None
    delta_indices = action_config.get("delta_indices", [])
    if not isinstance(delta_indices, list) or not delta_indices:
        return None
    return max(1, min(action_np.shape[1], len(delta_indices)))


def _n1_7_action_group_slice(
    action_keys: list[Any], decoded_groups: dict[str, np.ndarray], target_key: str
) -> slice:
    start_idx = 0
    for key in action_keys:
        if not isinstance(key, str) or key not in decoded_groups:
            continue
        dim = decoded_groups[key].shape[-1]
        end_idx = start_idx + dim
        if key == target_key:
            return slice(start_idx, end_idx)
        start_idx = end_idx

    raise KeyError(f"Missing N1.7 action group '{target_key}' required by action decode transform.")


def _apply_n1_7_action_decode_transform(
    decoded: np.ndarray,
    *,
    transform: str | None,
    action_keys: list[Any],
    decoded_groups: dict[str, np.ndarray],
) -> np.ndarray:
    if transform is None:
        return decoded

    if transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO:
        gripper_slice = _n1_7_action_group_slice(action_keys, decoded_groups, "gripper")
        if gripper_slice.stop is None or gripper_slice.stop > decoded.shape[-1]:
            raise ValueError(
                "N1.7 LIBERO action decode transform requested, but the decoded gripper action "
                "is outside the sliced environment action."
            )
        if gripper_slice.stop - gripper_slice.start != 1:
            raise ValueError("N1.7 LIBERO action decode transform expects a scalar gripper action.")

        transformed = decoded.copy()
        gripper = transformed[..., gripper_slice]
        transformed[..., gripper_slice] = -np.sign(2.0 * gripper - 1.0)
        return transformed

    raise ValueError(f"Unsupported N1.7 action decode transform '{transform}'.")


@dataclass
@ProcessorStepRegistry.register(name="groot_n1_7_action_decode_v1")
class GrootN17ActionDecodeStep(ProcessorStep):
    """Decode the full 132-D N1.7 model action back to environment actions.

    N1.7 predicts checkpoint-order action groups. This step unnormalizes each
    group with the checkpoint stats, converts relative groups to absolute values
    using the raw state cached during packing, concatenates groups in checkpoint
    order, and finally slices to the environment action dimension.

    Relative-action decoding reads the reference state from the connected
    ``pack_step`` (re-linked after ``from_pretrained`` by
    ``_reconnect_groot_n1_7_pack_decode_steps``), i.e. the state seen by the
    most recent preprocess call. Engines that decode the whole chunk right
    after prediction (RTC, async policy server) therefore use the
    prediction-time state, matching Isaac-GR00T. The sync per-step queue path
    instead decodes each popped (B, D) action against the latest observation:
    the reference can be newer than the observation the chunk was predicted
    from, and per-timestep relative stats are applied as if the popped action
    were chunk step 0. Fixing that would require carrying the reference state
    and chunk index alongside each queued action through the postprocessor.
    """

    env_action_dim: int = 0
    raw_stats: dict[str, Any] | None = None
    modality_config: dict[str, Any] | None = None
    use_percentiles: bool = False
    use_relative_action: bool = False
    action_decode_transform: str | None = None
    pack_step: GrootN17PackInputsStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition
        if self.raw_stats is None or self.modality_config is None:
            return transition

        action_config = self.modality_config.get("action", {})
        if not isinstance(action_config, dict):
            return transition
        action_keys = action_config.get("modality_keys", [])
        action_configs = action_config.get("action_configs", [])
        if not isinstance(action_keys, list) or not isinstance(action_configs, list):
            return transition

        action_np = action.detach().cpu().float().numpy()
        if self.use_relative_action and action_np.ndim != 3:
            raise NotImplementedError(
                "GrootN17ActionDecodeStep cannot decode native relative actions one step at a time. "
                "Decode the full action chunk returned by predict_action_chunk while the matching "
                "GrootN17PackInputsStep state is still cached, then queue the decoded absolute actions."
            )
        # The sync action queue postprocesses popped actions as (B, D); decode
        # them as single-step (B, 1, D) chunks and squeeze the horizon back at
        # the end so both ranks share the chunk decode logic below.
        squeeze_horizon = action_np.ndim == 2
        if squeeze_horizon:
            action_np = action_np[:, None, :]
        valid_horizon = _n1_7_decode_valid_horizon(action_config, action_np)
        if valid_horizon is not None:
            action_np = action_np[:, :valid_horizon]
        decoded_groups: dict[str, np.ndarray] = {}
        start_idx = 0
        for idx, key in enumerate(action_keys):
            if not isinstance(key, str):
                continue
            stats_entry = self.raw_stats.get("action", {}).get(key, {})
            if not isinstance(stats_entry, dict):
                continue
            dim = stat_dim_from_entry(stats_entry)
            if dim <= 0:
                continue
            cfg = (
                action_configs[idx]
                if idx < len(action_configs) and isinstance(action_configs[idx], dict)
                else {}
            )
            normalized = action_np[..., start_idx : start_idx + dim]
            min_v, max_v = _n1_7_decode_stats_for_action(
                self.raw_stats,
                key,
                cfg,
                use_relative_action=self.use_relative_action,
                use_percentiles=self.use_percentiles,
            )
            # Per-timestep stats carry one row per chunk step; align them with
            # the decoded horizon (chunks always start at step 0, and a popped
            # (B, D) action is decoded as step 0).
            if min_v.ndim == 2 and normalized.shape[1] <= min_v.shape[0]:
                min_v = min_v[: normalized.shape[1]]
                max_v = max_v[: normalized.shape[1]]
            decoded_groups[key] = _unnormalize_min_max(normalized, min_v, max_v)
            start_idx += dim

        if self.use_relative_action:
            raw_state = self.pack_step.get_cached_raw_state() if self.pack_step is not None else None
            if raw_state is None:
                raise RuntimeError(
                    "GrootN17ActionDecodeStep requires the raw state cached by its connected "
                    "GrootN17PackInputsStep to convert relative N1.7 actions back to absolute actions. "
                    "Build both pipelines through make_groot_pre_post_processors (or load them together "
                    "via make_groot_pre_post_processors_from_pretrained) and run the preprocessor on an "
                    "observation before decoding actions."
                )
            for idx, key in enumerate(action_keys):
                if not isinstance(key, str) or key not in decoded_groups or idx >= len(action_configs):
                    continue
                cfg = action_configs[idx]
                if not isinstance(cfg, dict) or config_value(cfg.get("rep")) != "relative":
                    continue
                state_key = cfg.get("state_key") or key
                if state_key not in raw_state:
                    raise KeyError(f"Missing cached raw state '{state_key}' for relative N1.7 action '{key}'")
                reference = raw_state[state_key]
                action_type = config_value(cfg.get("type"))
                action_format = config_value(cfg.get("format"))
                if action_type == "non_eef":
                    decoded_groups[key] = decoded_groups[key] + reference[:, None, :]
                elif action_type == "eef" and action_format == "xyz+rot6d":
                    decoded_groups[key] = relative_eef_to_absolute(decoded_groups[key], reference)
                else:
                    raise ValueError(f"Unsupported relative N1.7 action config for '{key}': {cfg}")

        if not decoded_groups:
            return transition

        decoded = np.concatenate(
            [decoded_groups[key] for key in action_keys if isinstance(key, str) and key in decoded_groups],
            axis=-1,
        )
        if self.env_action_dim and decoded.shape[-1] > self.env_action_dim:
            decoded = decoded[..., : self.env_action_dim]
        decoded = _apply_n1_7_action_decode_transform(
            decoded,
            transform=self.action_decode_transform,
            action_keys=action_keys,
            decoded_groups=decoded_groups,
        )
        if squeeze_horizon:
            decoded = decoded[:, 0]
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = torch.as_tensor(
            decoded, dtype=action.dtype, device=action.device
        )
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "env_action_dim": self.env_action_dim,
            "raw_stats": self.raw_stats,
            "modality_config": self.modality_config,
            "use_percentiles": self.use_percentiles,
            "use_relative_action": self.use_relative_action,
            "action_decode_transform": self.action_decode_transform,
        }


# v2: unlike the N1.5-era v1 step, this step no longer collapses (B, T, D)
# action chunks to the last timestep, so old serialized v1 pipelines must not
# silently load into it (v1 is stubbed below with the removal guidance).
@dataclass
@ProcessorStepRegistry.register(name="groot_action_unpack_unnormalize_v2")
class GrootActionUnpackUnnormalizeStep(ProcessorStep):
    env_action_dim: int = 0
    # Apply inverse of min-max normalization if it was used in preprocessor
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None
    clip_normalized_action: bool = False
    libero_gripper_action: bool = False
    libero_gripper_binarize: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Expect model outputs to be in TransitionKey.ACTION as (B, T, D_model)
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition

        # Slice to env dimension while preserving an optional action horizon.
        # Sync rollout postprocesses selected actions as (B, D); RTC postprocesses
        # chunks as (B, T, D), matching Isaac-GR00T's decode_action contract.
        if self.env_action_dim and action.shape[-1] >= self.env_action_dim:
            action = action[..., : self.env_action_dim]

        # Inverse min-max normalization mirroring _min_max_norm:
        # forward: y = 2 * (x - min) / denom - 1, with y=0 when denom==0
        # inverse: x = (y+1)/2 * denom + min, and when denom==0 -> x = min
        if self.normalize_min_max and self.stats is not None:
            if self.clip_normalized_action:
                action = action.clamp(-1.0, 1.0)
            stats_k = self.stats.get(ACTION, {})
            d = action.shape[-1]
            min_v = torch.as_tensor(
                stats_k.get("min", torch.zeros(d)), dtype=action.dtype, device=action.device
            )
            max_v = torch.as_tensor(
                stats_k.get("max", torch.ones(d)), dtype=action.dtype, device=action.device
            )
            if min_v.numel() != d:
                min_v = torch.nn.functional.pad(min_v.flatten()[:d], (0, max(0, d - min_v.numel())))
                min_v = min_v.to(action.device, dtype=action.dtype)
            if max_v.numel() != d:
                max_v = torch.nn.functional.pad(max_v.flatten()[:d], (0, max(0, d - max_v.numel())))
                max_v = max_v.to(action.device, dtype=action.dtype)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            inv = (action + 1.0) * 0.5 * safe_denom + min_v
            action = torch.where(mask, inv, min_v)

        if self.libero_gripper_action and action.shape[-1] >= 7:
            gripper = action[..., -1]
            if self.libero_gripper_binarize:
                gripper = -torch.sign(2.0 * gripper - 1.0)
            else:
                gripper = -(2.0 * gripper - 1.0)
            action = action.clone()
            action[..., -1] = gripper

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        Excludes 'stats' since they are saved separately via state_dict().
        """
        return {
            "env_action_dim": self.env_action_dim,
            "normalize_min_max": self.normalize_min_max,
            "clip_normalized_action": self.clip_normalized_action,
            "libero_gripper_action": self.libero_gripper_action,
            "libero_gripper_binarize": self.libero_gripper_binarize,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns normalization statistics as a flat state dictionary.

        This enables saving stats to safetensors files, similar to normalizer_processor.
        """
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        Loads normalization statistics from a flat state dictionary.

        This enables loading stats from safetensors files during from_pretrained.
        """
        if not state:
            return

        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor

        if reconstructed:
            self.stats = reconstructed


# Registry names that only GR00T N1.5 processor pipelines serialize. Saved N1.5 checkpoints
# reference these in their processor JSON, so deserializing one must fail with the canonical N1.5
# removal guidance instead of an opaque registry KeyError (or, for
# ``groot_action_unpack_unnormalize_v1``, silently loading the v2 step whose action-chunk
# semantics changed).
_REMOVED_N1_5_STEP_NAMES = (
    "groot_pack_inputs_v3",
    "groot_eagle_encode_v3",
    "groot_eagle_collate_v3",
    "groot_action_unpack_unnormalize_v1",
)


def _register_removed_n1_5_step_stub(registry_name: str) -> None:
    """Register a single rejecting stub for a removed GR00T N1.5 processor step name.

    Idempotent: ``ProcessorStepRegistry.register`` raises on a duplicate name, so already-registered
    names are skipped. This lets the caller re-run on every processor load without a run-once guard.
    """
    if registry_name in ProcessorStepRegistry.list():
        return

    @ProcessorStepRegistry.register(name=registry_name)
    class _RemovedGrootN15ProcessorStep(ProcessorStep):
        def __init__(self, **_kwargs: Any) -> None:
            raise ValueError(
                f"Processor step '{registry_name}' belongs to a GR00T N1.5 processor pipeline. "
                f"{GROOT_N1_5_REMOVAL_GUIDANCE}"
            )

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            raise NotImplementedError

        def transform_features(self, features):
            raise NotImplementedError


def _register_removed_n1_5_step_stubs() -> None:
    """Register the GR00T N1.5 removal stubs, lazily.

    Deferred from import time so importing this module has no global side effects; invoked just
    before a GR00T processor pipeline is deserialized (the only point at which a saved N1.5 pipeline
    could reference these registry names). Idempotent via :func:`_register_removed_n1_5_step_stub`.
    """
    for registry_name in _REMOVED_N1_5_STEP_NAMES:
        _register_removed_n1_5_step_stub(registry_name)
