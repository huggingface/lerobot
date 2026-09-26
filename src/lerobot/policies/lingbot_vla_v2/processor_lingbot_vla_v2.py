# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""LingBot-VLA 2.0 policy processor.

The pipeline is composed of LeRobot's standard ``ProcessorStep`` building blocks
(rename → batch-dim → relative actions → normalization → slot mapping → Qwen3-VL
image processing → chat template → tokenization → device), plus one policy-specific
step: :class:`LingbotVLAV2SlotMappingProcessorStep` maps the raw dataset features
onto the unified 55-D canonical state/action layout (with joint masks). It only
remaps/pads — normalization and relative actions are handled by the standard steps
in raw feature space, so slicing the raw stats with the same offsets reproduces the
per-slot statistics bit-for-bit.

The postprocessor inverts that pipeline: canonical → raw action dims, unnormalize,
relative → absolute actions, and a final move to CPU.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms.v2.functional import resize as tv_resize

from lerobot.configs import (
    FeatureType,
    NormalizationMode,
    PipelineFeatureType,
    PolicyFeature,
)
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    batch_to_transition,
    make_policy_processor_pipelines,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

from .configuration_lingbot_vla_v2 import LingbotVLAV2Config, resolve_robot_config_and_stats
from .preprocessing.data_transform import prepare_images

if _transformers_available:
    from transformers import AutoImageProcessor, AutoTokenizer
else:  # pragma: no cover - transformers is an optional dependency
    AutoImageProcessor = None
    AutoTokenizer = None

logger = logging.getLogger(__name__)

DEFAULT_TASK = "Execute the robot action."

# Registry names of the custom steps (override keys for from_pretrained).
SLOT_MAPPING_STEP = "lingbot_vla_v2_slot_mapping"
INVERSE_SLOT_MAPPING_STEP = "lingbot_vla_v2_inverse_slot_mapping"
IMAGE_STEP = "lingbot_vla_v2_image"
CHAT_TEMPLATE_STEP = "lingbot_vla_v2_chat_template"

# Upstream per-slot norm modes → standard NormalizationMode. Only the modes the
# released checkpoints train with are mapped; anything else falls back with a
# warning (see ``_resolve_norm_map``).
_LINGBOT_NORM_MODE_TO_STANDARD = {
    "meanstd": NormalizationMode.MEAN_STD,
    "bounds_99_woclip": NormalizationMode.QUANTILES,
    "identity": NormalizationMode.IDENTITY,
}


def _future_video_fps(dataset_fps: float, offset: int, action_is_pad=None):
    """Use real spacing when LeRobot clamps a requested future to the episode end.

    The default offset 49 is covered by the 50-step action pad mask. An infinite
    FPS for a duplicated final frame encodes a zero temporal RoPE step.
    """
    if isinstance(action_is_pad, torch.Tensor) and offset < action_is_pad.shape[-1]:
        valid_tail = (~action_is_pad.bool()).sum(dim=-1).sub(1).clamp(min=0, max=offset)
        return float(dataset_fps) / valid_tail.float()
    return float(dataset_fps) / max(1, offset)


# ---------------------------------------------------------------------------
# Slot-mapping helpers (shared by the forward and inverse steps)
# ---------------------------------------------------------------------------


def _canonical_joint_name(slot_key: str, category: str) -> str:
    """Strip the ``observation.state.`` / ``action.`` prefix from a robot-config slot key.

    Robot configs may key slots either by the bare canonical joint name
    (``arm.position``, what the typed config fields serialize) or by the full
    feature name (``observation.state.arm.position``, the upstream YAML format).
    """
    prefix = f"{OBS_STATE}." if category == "states" else f"{ACTION}."
    return slot_key.removeprefix(prefix)


def _build_slot_plans(
    robot_config: dict | None,
    canonical_joints: dict[str, int],
) -> tuple[list[tuple[str, int, list[tuple[str, int, int]] | None]], ...]:
    """Parse a robot config into per-canonical-joint raw-span plans.

    Returns ``(state_plan, action_plan)``; each plan is a list aligned with
    ``canonical_joints`` of ``(joint, canonical_dim, spans)`` where ``spans`` is
    ``[(raw_key, start, end), ...]`` or ``None`` when the slot is unmapped
    (zero-filled at canonical layout).
    """
    raw_plans: dict[str, dict[str, list[tuple[str, int, int]]]] = {"states": {}, "actions": {}}
    for category in ("states", "actions"):
        for entry in (robot_config or {}).get(category, []):
            for slot_key, slot_cfg in entry.items():
                joint = _canonical_joint_name(slot_key, category)
                spans: list[tuple[str, int, int]] = []
                for origin in slot_cfg.get("origin_keys", []):
                    for raw_key, span in origin.items():
                        spans.append((raw_key, int(span["start"]), int(span["end"])))
                raw_plans[category][joint] = spans

    plans = []
    for category in ("states", "actions"):
        plan = []
        for joint, dim in canonical_joints.items():
            spans = raw_plans[category].get(joint)
            if spans is not None and sum(e - s for _, s, e in spans) > dim:
                raise ValueError(
                    f"robot-config slot {joint!r} spans {sum(e - s for _, s, e in spans)} dims, "
                    f"wider than its canonical dimension {dim}."
                )
            plan.append((joint, dim, spans))
        plans.append(plan)
    return tuple(plans)


def _camera_rename_map(robot_config: dict | None) -> dict[str, str]:
    """Canonical camera key → raw dataset camera key."""
    mapping: dict[str, str] = {}
    for entry in (robot_config or {}).get("images", []):
        for canonical_key, slot_cfg in entry.items():
            raw_key = slot_cfg["origin_keys"] if isinstance(slot_cfg, dict) else slot_cfg
            mapping[canonical_key] = raw_key
    return mapping


def _prepare_camera_frame(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize one camera frame and scale it to the [0, 255] range Qwen3-VL expects.

    LeRobot images are float CHW in [0, 1]; the HF image processor rescales by
    1/255 itself, so the pixel values are scaled only when clearly normalized.
    """
    img = tv_resize(img, list(size), antialias=True)
    if img.dtype.is_floating_point and float(img.max()) <= 1.0 + 1e-4:
        img = img * 255.0
    return img


def _split_camera_frames(
    img: torch.Tensor,
    size: tuple[int, int],
    use_future_image: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pick the current (and optional future) frame from one camera tensor.

    Training samples with future-frame deltas are [T,C,H,W]; the policy consumes
    the current frame while the distillation teachers additionally get the last
    sampled (future) frame. Inference supplies a plain [C,H,W] image even for a
    depth-aligned checkpoint. Both frames are resized to the target size.
    """
    if use_future_image and img.ndim == 4:
        return _prepare_camera_frame(img[0], size), _prepare_camera_frame(img[-1], size)
    return _prepare_camera_frame(img, size), None


def _normalize_task_at(task: Any, index: int) -> str:
    """Extract item ``index`` of a task that arrived as str, list or tensor."""
    if isinstance(task, torch.Tensor):
        t = task[index].item() if task.ndim > 0 else task.item()
    elif isinstance(task, (list, tuple)):
        t = task[index] if index < len(task) else task[-1]
    else:
        t = task
    return t if isinstance(t, str) else str(t)


def _make_identity_robot_config(config: LingbotVLAV2Config) -> dict:
    """Build an identity robot_config for format-only converted checkpoints.

    When no slot mappings are provided (e.g. a checkpoint converted without
    --robot-config-path), this generates an identity passthrough: raw features are
    assumed to already be in the canonical layout — canonical slot *j* reads the
    cumulative raw span ``[offset_j, offset_j + dim_j)`` of ``observation.state`` /
    ``action``. This lets users evaluate the checkpoint without training first;
    fine-tuning overrides with the user's own slot mappings.
    """
    robot_config: dict = {}

    state_offset = 0
    states = []
    for joint, dim in config.canonical_joints.items():
        states.append(
            {
                f"{OBS_STATE}.{joint}": {
                    "origin_keys": [{OBS_STATE: {"start": state_offset, "end": state_offset + dim}}]
                }
            }
        )
        state_offset += dim
    robot_config["states"] = states

    action_offset = 0
    actions = []
    for joint, dim in config.canonical_joints.items():
        actions.append(
            {
                f"{ACTION}.{joint}": {
                    "origin_keys": [{ACTION: {"start": action_offset, "end": action_offset + dim}}],
                    "subtract_state": False,
                }
            }
        )
        action_offset += dim
    robot_config["actions"] = actions

    robot_config["images"] = [
        {f"{OBS_IMAGES}.{cam}": {"origin_keys": f"{OBS_IMAGES}.{cam}"}} for cam in config.canonical_cameras
    ]
    return robot_config


def _resolve_norm_map(canonical_norm_type: dict[str, str]) -> dict[str, NormalizationMode]:
    """Map the per-slot upstream norm modes onto the standard per-type modes.

    ``meanstd`` → ``MEAN_STD`` and ``bounds_99_woclip`` → ``QUANTILES`` (the
    standard q01/q99 mapping does not clip, exactly like the woclip variant).
    The standard ``NormalizerProcessorStep`` applies one mode per feature type in
    raw feature space; when a config mixes modes per slot the mode of the first
    canonical joint wins and a warning is logged.
    """
    modes = [mode for mode in canonical_norm_type.values() if mode != "identity"]
    if not modes:
        mode = NormalizationMode.IDENTITY
    else:
        standard = {_LINGBOT_NORM_MODE_TO_STANDARD.get(mode) for mode in modes}
        if None in standard or len(modes) > 1 and len(standard) > 1:
            first = modes[0]
            logger.warning(
                "canonical_norm_type %s mixes or uses modes without a standard equivalent; "
                "normalizing STATE/ACTION with the first joint's mode %r mapped to %s. "
                "Set canonical_norm_type to a single mode for exact parity.",
                dict(canonical_norm_type),
                first,
                _LINGBOT_NORM_MODE_TO_STANDARD.get(first, NormalizationMode.MEAN_STD),
            )
            mode = _LINGBOT_NORM_MODE_TO_STANDARD.get(first, NormalizationMode.MEAN_STD)
        else:
            mode = standard.pop()
    return {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": mode,
        "ACTION": mode,
    }


def _raw_stats_from_slot_stats(
    robot_config: dict,
    norm_stats: dict | None,
) -> dict[str, dict[str, list]] | None:
    """Assemble raw-feature stats from the checkpoint's per-slot ``norm_stats``.

    The embedded stats are keyed by canonical slot (``observation.state.arm.position``)
    with per-slot vectors; normalization now happens in raw feature space, so each
    slot stat slice is written back at its raw span offset. Dims covered by no slot
    get identity stats (they never reach the model). Returns ``None`` when no stats
    are embedded (identity passthrough — normalization becomes a no-op).
    """
    slot_stats = (norm_stats or {}).get("norm_stats") or {}
    if not slot_stats:
        return None

    raw_stats: dict[str, dict[str, list]] = {}
    total_dims: dict[str, int] = {}

    for category in ("states", "actions"):
        for entry in robot_config.get(category, []):
            for slot_key, slot_cfg in entry.items():
                stats = slot_stats.get(slot_key)
                if stats is None:
                    continue
                offset = 0
                for origin in slot_cfg.get("origin_keys", []):
                    for raw_key, span in origin.items():
                        start, end = int(span["start"]), int(span["end"])
                        width = end - start
                        target = raw_stats.setdefault(raw_key, {})
                        total_dims[raw_key] = max(total_dims.get(raw_key, 0), end)
                        for stat_name, values in stats.items():
                            if stat_name == "count":
                                continue
                            row = target.setdefault(stat_name, [None] * 0)
                            if len(row) < end:
                                row.extend([None] * (end - len(row)))
                            row[start:end] = values[offset : offset + width]
                        offset += width

    # Fill dims that no slot covers with identity stats so normalization is a
    # no-op there (those dims are dropped by the slot mapping anyway).
    for raw_key, stats in raw_stats.items():
        dim = total_dims[raw_key]
        for stat_name, values in list(stats.items()):
            fill = 1.0 if stat_name in ("std", "q99", "max") else (0.0 if stat_name == "mean" else -1.0)
            values.extend([fill] * (dim - len(values)))
            stats[stat_name] = [fill if value is None else value for value in values]
    return raw_stats


def _relative_actions_settings(
    config: LingbotVLAV2Config,
) -> dict[str, Any]:
    """Derive the standard relative-action settings from the typed slot mappings.

    Slots flagged ``subtract_state`` map onto the standard all-relative mode with
    per-dimension names synthesized from the slot spans (zero-padded indices so
    exclude-joint name matching cannot collide); the non-subtracting slots are
    listed in ``exclude_joints`` to stay absolute. An explicit
    ``config.relative_exclude_joints`` always wins over the derived list.
    """
    slots = config.action_slots or {}
    subtracting = {name for name, mapping in slots.items() if mapping.subtract_state}
    enabled = bool(config.use_relative_actions) or bool(subtracting)
    if not enabled:
        return {"enabled": False}
    if config.relative_exclude_joints:
        return {"enabled": True, "exclude_joints": list(config.relative_exclude_joints)}
    if not slots or not subtracting:
        return {"enabled": True}

    # Order the per-dim names by raw span offset so the mask aligns with the raw
    # action layout the standard step subtracts against.
    named_dims: list[tuple[int, str, bool]] = []
    for name, mapping in slots.items():
        offset = 0
        for origin in mapping.origin_keys:
            for _raw_key, span in origin.items():
                width = int(span["end"]) - int(span["start"])
                for i in range(width):
                    named_dims.append((int(span["start"]) + i, f"{name}.{i:03d}", name in subtracting))
                offset += width
    named_dims.sort(key=lambda item: item[0])
    action_names = [name for _, name, _ in named_dims]
    exclude_joints = [name for _, name, is_relative in named_dims if not is_relative]
    return {"enabled": True, "exclude_joints": exclude_joints, "action_names": action_names}


# ---------------------------------------------------------------------------
# Custom processor steps
# ---------------------------------------------------------------------------


@dataclass
@ProcessorStepRegistry.register(name=SLOT_MAPPING_STEP)
class LingbotVLAV2SlotMappingProcessorStep(ProcessorStep):
    """Map raw dataset features onto the unified canonical layout (mapping only).

    This is the single LingBot-specific step: it concatenates the raw feature
    spans each canonical slot declares (``origin_keys``) into the canonical
    state/action vectors, zero-pads them to ``max_state_dim`` / ``max_action_dim``,
    and emits the per-dim validity masks (``state_joint_mask`` / ``action_joint_mask``
    / ``joint_mask``) the model uses to mask padded slots. It also renames raw
    camera keys onto the canonical camera names. No normalization, no relative
    actions — those are the standard steps running before this one.
    """

    robot_config: dict | None = None
    # Ordered canonical joint vocabulary (name -> dim); defines the layout.
    canonical_joints: dict = field(default_factory=dict)
    # Canonical camera names (observation.images.<name> keys after the step).
    cameras: list = field(default_factory=list)
    chunk_size: int = 50
    max_state_dim: int = 55
    max_action_dim: int = 55
    use_future_image: bool = False

    _state_plan: Any = field(default=None, init=False, repr=False)
    _action_plan: Any = field(default=None, init=False, repr=False)
    _camera_map: Any = field(default=None, init=False, repr=False)
    _warned_missing_slots: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._state_plan, self._action_plan = _build_slot_plans(self.robot_config, self.canonical_joints)
        self._camera_map = _camera_rename_map(self.robot_config)
        self._warned_missing_slots = set()

    def _warn_missing(self, joint: str, raw_key: str) -> None:
        if joint not in self._warned_missing_slots:
            self._warned_missing_slots.add(joint)
            logger.warning(
                "Canonical slot %r left unfilled — source key %r absent from the batch. "
                "Check the slot mapping against the dataset features; the state/action "
                "dims of this slot will be zero-padded.",
                joint,
                raw_key,
            )

    def _map_vector(self, source: dict[str, Any], joint: str, dim: int, spans) -> torch.Tensor | None:
        """Concatenate a slot's raw spans into a (…, dim) canonical vector."""
        pieces = []
        for raw_key, start, end in spans:
            tensor = source.get(raw_key)
            if tensor is None:
                self._warn_missing(joint, raw_key)
                return None
            piece = tensor[..., start:end]
            if piece.shape[-1] < end - start:
                # An over-spanning mapping (e.g. identity on a short raw feature)
                # zero-pads the tail instead of failing.
                piece = F.pad(piece, (0, end - start - piece.shape[-1]))
            pieces.append(piece)
        vector = torch.cat(pieces, dim=-1).to(torch.float32)
        return F.pad(vector, (0, dim - vector.shape[-1]))

    def _joint_mask(self, spans, dim: int) -> torch.Tensor:
        real = sum(end - start for _, start, end in spans) if spans else 0
        mask = torch.zeros(dim, dtype=torch.bool)
        mask[:real] = True
        return mask

    def __call__(self, transition):
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("LingbotVLAV2SlotMappingProcessorStep requires an observation dict.")
        new_obs = dict(observation)

        state = new_obs.get(OBS_STATE)
        if state is None:
            raise ValueError("LingbotVLAV2SlotMappingProcessorStep requires 'observation.state'.")
        # Future-frame sampling stacks T frames on every observation key; the
        # policy state is the current frame only.
        if self.use_future_image and state.ndim == 3:
            state = state[:, 0]
        state_source = {OBS_STATE: state}

        state_vecs, state_masks = [], []
        for joint, dim, spans in self._state_plan:
            if spans is None:
                state_vecs.append(torch.zeros(*state.shape[:-1], dim))
                state_masks.append(torch.zeros(dim, dtype=torch.bool))
                continue
            vector = self._map_vector(state_source, joint, dim, spans)
            if vector is None:
                state_vecs.append(torch.zeros(*state.shape[:-1], dim))
                state_masks.append(torch.zeros(dim, dtype=torch.bool))
                continue
            state_vecs.append(vector)
            state_masks.append(self._joint_mask(spans, dim))
        canonical_width = self._canonical_width()
        canonical_state = F.pad(torch.cat(state_vecs, dim=-1), (0, self.max_state_dim - canonical_width))
        state_joint_mask = F.pad(torch.cat(state_masks, dim=-1), (0, self.max_state_dim - canonical_width))

        # Cameras: rename raw keys onto canonical camera names.
        for canonical_key, raw_key in self._camera_map.items():
            if raw_key in new_obs and raw_key != canonical_key:
                new_obs[canonical_key] = new_obs[raw_key]

        action = transition.get(TransitionKey.ACTION)
        action_joint_mask = F.pad(
            torch.cat([self._joint_mask(spans, dim) for _, dim, spans in self._action_plan], dim=-1),
            (0, self.max_action_dim - canonical_width),
        )
        if action is not None:
            action_source = {ACTION: action}
            action_vecs = []
            for joint, dim, spans in self._action_plan:
                if spans is None:
                    action_vecs.append(torch.zeros(*action.shape[:-1], dim))
                    continue
                vector = self._map_vector(action_source, joint, dim, spans)
                if vector is None:
                    vector = torch.zeros(*action.shape[:-1], dim)
                action_vecs.append(vector)
            canonical_action = F.pad(
                torch.cat(action_vecs, dim=-1), (0, self.max_action_dim - canonical_width)
            ).to(torch.float32)
            transition[TransitionKey.ACTION] = canonical_action
            batch_size = canonical_state.shape[0]
            chunk = canonical_action.shape[-2] if canonical_action.ndim == 3 else self.chunk_size
            joint_mask = action_joint_mask.unsqueeze(0).expand(batch_size, chunk, -1)
            new_obs["joint_mask"] = joint_mask

        new_obs[OBS_STATE] = canonical_state.to(torch.float32)
        new_obs["state_joint_mask"] = state_joint_mask.unsqueeze(0).expand(canonical_state.shape[0], -1)
        new_obs["action_joint_mask"] = action_joint_mask.unsqueeze(0).expand(canonical_state.shape[0], -1)
        transition[TransitionKey.OBSERVATION] = new_obs
        return transition

    def _canonical_width(self) -> int:
        return sum(self.canonical_joints.values())

    def get_config(self) -> dict[str, Any]:
        return {
            "robot_config": self.robot_config,
            "canonical_joints": self.canonical_joints,
            "cameras": self.cameras,
            "chunk_size": self.chunk_size,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "use_future_image": self.use_future_image,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE, shape=(self.max_state_dim,)
        )
        features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(self.chunk_size, self.max_action_dim)
        )
        return features


@dataclass
@ProcessorStepRegistry.register(name=INVERSE_SLOT_MAPPING_STEP)
class LingbotVLAV2InverseSlotMappingProcessorStep(PolicyActionProcessorStep):
    """Invert the canonical slot mapping on predicted action chunks.

    ``(B, [chunk,] max_action_dim)`` canonical actions are sliced per canonical
    joint group and written back to the raw action dims each slot was sourced
    from, producing the raw-dim action the unnormalizer and the robot expect.
    """

    robot_config: dict | None = None
    canonical_joints: dict = field(default_factory=dict)

    _action_plan: Any = field(default=None, init=False, repr=False)
    # Live reference to the preprocessor's slot-mapping step (for introspection;
    # the inverse mapping itself is static — derived from the robot config).
    slot_mapping_step: Any = field(default=None, repr=False)

    def __post_init__(self):
        _state_plan, self._action_plan = _build_slot_plans(self.robot_config, self.canonical_joints)

    def action(self, action: PolicyAction) -> PolicyAction:
        spans = [(joint, dim, slot_spans) for joint, dim, slot_spans in self._action_plan if slot_spans]
        if not spans:
            raise ValueError(
                "LingbotVLAV2InverseSlotMappingProcessorStep has no action slot mapping; "
                "cannot map canonical actions back to raw action dims."
            )
        raw_dim = max(end for _, _, slot_spans in spans for _, _, end in slot_spans)
        raw = action.new_zeros(*action.shape[:-1], raw_dim)
        offset = 0
        for _joint, dim, slot_spans in spans:
            real = sum(end - start for _, start, end in slot_spans)
            segment = action[..., offset : offset + real]
            position = 0
            for raw_key, start, end in slot_spans:
                if raw_key != ACTION:
                    raise ValueError(
                        f"LingBot-VLA 2.0 supports a single '{ACTION}' raw action feature, "
                        f"got span on {raw_key!r}."
                    )
                raw[..., start:end] = segment[..., position : position + (end - start)]
                position += end - start
            offset += dim
        return raw

    def get_config(self) -> dict[str, Any]:
        return {"robot_config": self.robot_config, "canonical_joints": self.canonical_joints}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name=IMAGE_STEP)
class LingbotVLAV2ImageProcessorStep(ProcessorStep):
    """Run the standard Qwen3-VL image processor over the canonical cameras.

    Per item and camera: resize to ``resize_imgs_with_padding``, rescale to
    [0, 255], then the HF image processor patchifies at native resolution and
    returns ``pixel_values`` plus the ``image_grid_thw`` patch grid. Missing
    canonical views are zero-filled with ``img_masks=False``. With the depth /
    DINO-video distillation branch enabled, the pre-Qwen frames are also carried
    as ``pil_images`` (and ``future_pil_images`` for the future frame).
    """

    processor_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    cameras: list = field(default_factory=list)
    resize_imgs_with_padding: tuple = (224, 224)
    # Cap the Qwen3-VL image processor's dynamic-resolution token budget. Left
    # uncapped, a native 1080x1920 frame explodes to ~8k vision tokens -> an
    # O(N^2) eager-attention tensor that OOMs and does not match the checkpoint's
    # training resolution. Qwen3-VL uses 16px patches + 2x2 merge (=1024 px/token),
    # so 1,048,576 px ~= 1024 tokens.
    image_max_pixels: int = 262144
    image_min_pixels: int = 131072
    # When set (e.g. "cuda"), camera images are uploaded to this device and run
    # through the HF image processor in one batched call, with the outputs staying
    # on-device for the vision tower. None keeps the per-camera CPU path.
    preprocess_device: str | None = None
    return_image_grid_thw: bool = True
    use_depth_align: bool = False
    use_future_image: bool = False
    dataset_fps: int | None = None
    future_frame_offset: int | None = None
    chunk_size: int = 50

    _image_processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "transformers is required for LingbotVLAV2ImageProcessorStep. "
                "Install it with `pip install 'lerobot[lingbot_vla2]'`."
            )
        self._image_processor = AutoImageProcessor.from_pretrained(
            self.processor_path,
            max_pixels=self.image_max_pixels,
            min_pixels=self.image_min_pixels,
        )

    def __call__(self, transition):
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("LingbotVLAV2ImageProcessorStep requires an observation dict.")
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        new_obs = dict(observation)

        state = new_obs[OBS_STATE]
        batch_size = state.shape[0]
        image_keys = [f"{OBS_IMAGES}.{cam}" for cam in self.cameras]
        pad_mask = complementary.get("action_is_pad")

        images, img_masks, grids, pil_images, future_pil = [], [], [], [], []
        for i in range(batch_size):
            image_dict: dict[str, torch.Tensor] = {}
            future_dict: dict[str, torch.Tensor] = {}
            for key in image_keys:
                img = new_obs.get(key)
                if img is None:
                    continue
                current, future = _split_camera_frames(
                    img[i], self.resize_imgs_with_padding, self.use_future_image
                )
                image_dict[key] = current
                if future is not None:
                    future_dict[key] = future

            item_obs = {"image": image_dict, "state": state[i]}
            item_images, item_masks, item_pil, item_grid = prepare_images(
                self._image_processor,
                item_obs,
                image_keys=image_keys,
                use_depth_align=self.use_depth_align,
                return_image_grid_thw=self.return_image_grid_thw,
                preprocess_device=self.preprocess_device,
            )
            images.append(item_images)
            img_masks.append(item_masks)
            grids.append(item_grid)
            pil_images.append(item_pil)
            if self.use_future_image and future_dict:
                _future_images, _future_masks, future_pil_i, _future_grid = prepare_images(
                    self._image_processor,
                    {"image": future_dict, "state": state[i]},
                    image_keys=image_keys,
                    use_depth_align=self.use_depth_align,
                    return_image_grid_thw=False,
                    augment_params=None,
                )
                future_pil.append(future_pil_i)

        new_obs["images"] = torch.stack(images, dim=0)
        new_obs["img_masks"] = torch.stack(img_masks, dim=0)
        if self.return_image_grid_thw:
            new_obs["image_grid_thw"] = torch.stack(grids, dim=0)
        if self.use_depth_align:
            new_obs["pil_images"] = torch.stack(pil_images, dim=0)
            if future_pil:
                new_obs["future_pil_images"] = torch.stack(future_pil, dim=0)
                if self.dataset_fps is not None:
                    offset = (
                        self.future_frame_offset
                        if self.future_frame_offset is not None
                        else max(1, self.chunk_size - 1)
                    )
                    new_obs["future_video_effective_fps"] = _future_video_fps(
                        self.dataset_fps, offset, pad_mask
                    )
        transition[TransitionKey.OBSERVATION] = new_obs
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_path": self.processor_path,
            "cameras": self.cameras,
            "resize_imgs_with_padding": list(self.resize_imgs_with_padding),
            # Must round-trip: these cap the Qwen3-VL vision-token budget and change
            # the effective input resolution. Omitting them silently reset the reload
            # to defaults and mismatched the checkpoint's training resolution.
            "image_max_pixels": self.image_max_pixels,
            "image_min_pixels": self.image_min_pixels,
            "preprocess_device": self.preprocess_device,
            "return_image_grid_thw": self.return_image_grid_thw,
            "use_depth_align": self.use_depth_align,
            "use_future_image": self.use_future_image,
            "dataset_fps": self.dataset_fps,
            "future_frame_offset": self.future_frame_offset,
            "chunk_size": self.chunk_size,
        }

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        """Save the image processor so the step reloads without a hub lookup."""
        artifact_path = Path("image_processor")
        self._image_processor.save_pretrained(save_directory / artifact_path)
        return {"processor_path": artifact_path.as_posix()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name=CHAT_TEMPLATE_STEP)
class LingbotVLAV2ChatTemplateProcessorStep(ProcessorStep):
    """Format the task with the Qwen3 chat template, Pi0.5-style.

    Mirrors ``Pi05PrepareStateTokenizerProcessorStep``: the raw task string is
    rewritten into the model's prompt format (Qwen3 chat template, or the
    ``<bos>…\n`` wrapping for non-chat-template checkpoints) so the downstream
    standard ``TokenizerProcessorStep`` can tokenize it as-is.
    """

    tokenizer_name: str | None = None
    task_key: str = "task"
    use_qwen3_chat_template: bool = True

    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "transformers is required for LingbotVLAV2ChatTemplateProcessorStep. "
                "Install it with `pip install 'lerobot[lingbot_vla2]'`."
            )
        if self.tokenizer_name is None:
            raise ValueError("LingbotVLAV2ChatTemplateProcessorStep requires a tokenizer_name.")
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def __call__(self, transition):
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("LingbotVLAV2ChatTemplateProcessorStep requires an observation dict.")
        state = observation.get(OBS_STATE)
        batch_size = state.shape[0] if state is not None else 1

        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        task = complementary.get(self.task_key, DEFAULT_TASK)
        prompts = [_normalize_task_at(task, i) for i in range(batch_size)]
        if self.use_qwen3_chat_template:
            prompts = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for prompt in prompts
            ]
        else:
            prompts = [(f"<bos>{prompt}" if not prompt.startswith("<bos>") else prompt) for prompt in prompts]
            prompts = [f"{prompt}\n" if not prompt.endswith("\n") else prompt for prompt in prompts]
        complementary[self.task_key] = prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "use_qwen3_chat_template": self.use_qwen3_chat_template,
            "tokenizer_name": self.tokenizer_name,
        }

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        """Save the tokenizer so the step reloads without a hub lookup."""
        artifact_path = Path("chat_template_tokenizer")
        self._tokenizer.save_pretrained(save_directory / artifact_path)
        return {"tokenizer_name": artifact_path.as_posix()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def make_lingbot_vla_v2_pre_post_processors(
    config: LingbotVLAV2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the LingBot-VLA 2.0 pre- and post-processing pipelines.

    Standard steps carry rename / batch-dim / relative actions / normalization /
    tokenization / device placement; the only custom step is the slot mapping
    onto the canonical layout. Stats precedence:

    1. ``dataset_stats`` (raw-feature stats from LeRobot's standard mechanism)
       — the recommended path for fine-tuning.
    2. ``config.norm_stats`` (per-slot stats embedded in the checkpoint),
       rewritten into raw-feature space via the slot-mapping offsets.
    3. Neither → identity passthrough: the steps are present but normalization
       is a no-op.
    """
    resolve_robot_config_and_stats(config)
    if not config.robot_config:
        # No slot mappings provided (format-only converted checkpoint): use identity
        # passthrough — raw features are assumed to already be in canonical layout.
        config.robot_config = _make_identity_robot_config(config)

    norm_map = _resolve_norm_map(config.canonical_norm_type)
    stats = (
        dataset_stats
        if dataset_stats is not None
        else _raw_stats_from_slot_stats(config.robot_config, config.norm_stats)
    )
    features = {**config.input_features, **config.output_features}

    relative_step = RelativeActionsProcessorStep(
        **_relative_actions_settings(config),
    )

    slot_step = LingbotVLAV2SlotMappingProcessorStep(
        robot_config=config.robot_config,
        canonical_joints=config.canonical_joints,
        cameras=config.canonical_cameras,
        chunk_size=config.chunk_size,
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        use_future_image=config.use_future_image,
    )

    image_step = LingbotVLAV2ImageProcessorStep(
        processor_path=config.processor_path or config.tokenizer_path,
        cameras=config.canonical_cameras,
        resize_imgs_with_padding=tuple(config.resize_imgs_with_padding),
        image_max_pixels=config.image_max_pixels,
        image_min_pixels=config.image_min_pixels,
        preprocess_device=config.preprocess_device,
        return_image_grid_thw=config.return_image_grid_thw,
        use_depth_align=config.use_depth_align,
        use_future_image=config.use_future_image,
        dataset_fps=config.dataset_fps,
        future_frame_offset=config.future_frame_offset,
        chunk_size=config.chunk_size,
    )

    tokenizer_name = config.processor_path or config.tokenizer_path

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats),
        slot_step,
        image_step,
        LingbotVLAV2ChatTemplateProcessorStep(
            tokenizer_name=tokenizer_name,
            use_qwen3_chat_template=config.use_qwen3_chat_template,
        ),
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps: list[ProcessorStep] = [
        LingbotVLAV2InverseSlotMappingProcessorStep(
            robot_config=config.robot_config,
            canonical_joints=config.canonical_joints,
            slot_mapping_step=slot_step,
        ),
        UnnormalizerProcessorStep(features=config.output_features, norm_map=norm_map, stats=stats),
        AbsoluteActionsProcessorStep(enabled=relative_step.enabled, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def _reconnect_lingbot_steps(
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> None:
    """Re-establish the cross-pipeline step references after deserialization.

    The inverse slot-mapping step's reference to the preprocessor's slot-mapping
    step and ``AbsoluteActionsProcessorStep.relative_step`` are not serializable;
    rewire them the same way ``factory._reconnect_relative_absolute_steps`` does.
    """
    slot_step = next(
        (s for s in preprocessor.steps if isinstance(s, LingbotVLAV2SlotMappingProcessorStep)), None
    )
    relative_step = next((s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)), None)
    for step in postprocessor.steps:
        if isinstance(step, LingbotVLAV2InverseSlotMappingProcessorStep) and step.slot_mapping_step is None:
            step.slot_mapping_step = slot_step
        if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
            step.relative_step = relative_step


def make_lingbot_vla_v2_pre_post_processors_from_pretrained(
    config: LingbotVLAV2Config,
    pretrained_path: str,
    *,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    pretrained_revision: str | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load the processors saved alongside a LeRobot checkpoint.

    The saved steps carry the slot mapping / normalization stats of the
    checkpoint's *source* embodiment. When fine-tuning on a new embodiment the
    policy config's assets must win here too — explicit slot-mapping fields first,
    the config's embedded contents as fallback (same rule as
    ``resolve_robot_config_and_stats``) — forwarded as per-step overrides.
    """
    preprocessor_overrides = dict(preprocessor_overrides or {})
    postprocessor_overrides = dict(postprocessor_overrides or {})
    if "device_processor" not in postprocessor_overrides and "device_processor" in preprocessor_overrides:
        postprocessor_overrides["device_processor"] = preprocessor_overrides["device_processor"]

    resolve_robot_config_and_stats(config)

    # Same config → step parameter set as ``make_lingbot_vla_v2_pre_post_processors``
    # (paths excluded: their resolved contents are forwarded instead). Only fields the
    # config actually carries (non-None) override the checkpoint's saved values.
    def _overrides(step_key: str, params: dict[str, Any]) -> None:
        resolved = {key: value for key, value in params.items() if value is not None}
        if resolved:
            preprocessor_overrides.setdefault(step_key, {}).update(resolved)

    _overrides(
        SLOT_MAPPING_STEP,
        {
            "robot_config": config.robot_config,
            "canonical_joints": config.canonical_joints,
            "cameras": config.canonical_cameras,
            "chunk_size": config.chunk_size,
            "max_state_dim": config.max_state_dim,
            "max_action_dim": config.max_action_dim,
            "use_future_image": config.use_future_image,
        },
    )
    processor_path = config.processor_path or config.tokenizer_path
    _overrides(
        IMAGE_STEP,
        {
            "processor_path": processor_path,
            "cameras": config.canonical_cameras,
            "resize_imgs_with_padding": tuple(config.resize_imgs_with_padding)
            if config.resize_imgs_with_padding
            else None,
            "image_max_pixels": config.image_max_pixels,
            "image_min_pixels": config.image_min_pixels,
            "return_image_grid_thw": config.return_image_grid_thw,
            "use_depth_align": config.use_depth_align,
            "use_future_image": config.use_future_image,
            "dataset_fps": config.dataset_fps,
            "future_frame_offset": config.future_frame_offset,
            "chunk_size": config.chunk_size,
        },
    )
    if config.use_qwen3_chat_template is not None:
        preprocessor_overrides.setdefault(CHAT_TEMPLATE_STEP, {})["use_qwen3_chat_template"] = (
            config.use_qwen3_chat_template
        )
    if processor_path is not None:
        preprocessor_overrides.setdefault("tokenizer_processor", {})["tokenizer_name"] = processor_path
        preprocessor_overrides.setdefault(CHAT_TEMPLATE_STEP, {})["tokenizer_name"] = processor_path

    # GPU preprocessing default: when the rollout inference device is CUDA and nobody
    # explicitly configured preprocess_device (policy config, saved checkpoint, or an
    # override), default it to that device. The fast path (prepare_images_on_device)
    # is pure torch/torchvision — bit-exact vs the CPU path (bench/check_gpu_preprocess.py)
    # — and saves ~171ms of per-tick host preprocessing on the measured 4090 setup
    # (x86 shared-host CPU contention; on GB10 both paths measure ~5ms). Explicit
    # config always wins, and ``preprocess_device="cpu"`` is the documented opt-out
    # (keeps the original per-camera HF processor path).
    if config.preprocess_device is None:
        dev_override = (preprocessor_overrides.get("device_processor") or {}).get("device")
        target_dev = dev_override or getattr(config, "device", None)
        if (
            target_dev is not None
            and str(target_dev).startswith("cuda")
            and torch.cuda.is_available()
            and "preprocess_device" not in preprocessor_overrides.get(IMAGE_STEP, {})
        ):
            preprocessor_overrides.setdefault(IMAGE_STEP, {})["preprocess_device"] = target_dev

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=preprocessor_config_filename,
        overrides=preprocessor_overrides,
        # Same standard converters as ``make_policy_processor_pipelines`` /
        # the generic loader in ``policies.factory``.
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        revision=pretrained_revision,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=postprocessor_config_filename,
        overrides=postprocessor_overrides,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        revision=pretrained_revision,
    )
    _reconnect_lingbot_steps(preprocessor, postprocessor)
    return preprocessor, postprocessor
