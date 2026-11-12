#!/usr/bin/env python

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

"""Pre- and post-processor pipelines of the Praxis-1 policy, and the DROID pose conversions.

The preprocessor renames the observations, adds the batch dimension, prepares the cameras
(`PraxisTileViewsProcessorStep`: each view becomes its tile of the canvas, with the rule of the
checkpoint), converts the DROID pose to the praxis pose8 (`PraxisPoseProcessorStep`), tokenizes
the instruction for the text encoder (`PraxisTokenizerProcessorStep`), moves the batch to the
device and passes the features through the stock normalizer (identity). The policy keeps the
tiles on the device they arrive on until the autoencoder encodes them. The postprocessor converts
the policy's command, the praxis wire pose and the gripper, to the DROID
``[x, y, z, roll, pitch, yaw, gripper]`` (`PraxisCommandProcessorStep`) on the CPU, where the
policy returns its commands.

The q99 action normalization, the mean/std proprio normalization and the relative-action decode
stay in the praxis session inside the policy: they are anchored to the pose at the start of each
executed chunk, which the session tracks over the rollout. The statistics are those of the
checkpoint (`PraxisConfig.model_config["normalization"]`). The frozen text encoder and its
embedding cache stay in the policy too; the tokenizer step gives it the token ids.

The DROID pose conversions (`droid7_to_pose8`, `pose8_to_droid7`, `pose12_grip_to_droid7`) use the
scipy-exact rotation helpers of `lerobot.utils.rotation`, so the pose rows have the bits of the
praxis training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.rotation import Rotation

from .configuration_praxis import PRAXIS_ACTION_DIM, PRAXIS_COMMAND_DIM, PRAXIS_POSE_DIM, PraxisConfig
from .praxis_core import bimanual
from .praxis_core.robot_io import ASPECT_TOL, aspect_crop, text_encoder_prompt, unit_quat_wxyz_to_matrix
from .praxis_core.single_arm import TILE_CHUNK_FRAMES, _fit_view, canvas_size
from .praxis_core.text_encoder import PAD_TO_MULTIPLE_OF

__all__ = [
    "PraxisCommandProcessorStep",
    "PraxisPoseProcessorStep",
    "PraxisTileViewsProcessorStep",
    "PraxisTokenizerProcessorStep",
    "droid7_to_pose8",
    "make_praxis_pre_post_processors",
    "pose8_to_droid7",
    "pose12_grip_to_droid7",
    "tile_views",
]


# ---------------------------------------------------------------------------
# DROID pose <-> praxis pose
# ---------------------------------------------------------------------------


def droid7_to_pose8(rows: np.ndarray) -> np.ndarray:
    """Convert ``(..., 7)`` ``[xyz, roll, pitch, yaw, grip]`` rows to float32 pose8 rows.

    The quaternion comes from the Euler angles, the matrix and back (`Rotation.matrix_to_quat_xyzw`), with
    the sign of ``scipy.spatial.transform``, as in the praxis DROID data code. The gripper is
    clipped to [0, 1].
    """
    a = np.asarray(rows, dtype=np.float64)
    flat = a.reshape(-1, a.shape[-1])
    if flat.shape[-1] != PRAXIS_ACTION_DIM:
        raise ValueError(f"expected rows of 7 values [xyz, rpy, grip], got {a.shape}")
    rot = Rotation.euler_xyz_to_matrix(flat[:, 3:6])
    quat = Rotation.matrix_to_quat_xyzw(rot)[:, [3, 0, 1, 2]]
    grip = np.clip(flat[:, 6:7], 0.0, 1.0)
    out = np.concatenate([flat[:, :3], quat, grip], axis=1).astype(np.float32)
    return out.reshape(*a.shape[:-1], PRAXIS_POSE_DIM)


def pose8_to_droid7(pose8: np.ndarray) -> np.ndarray:
    """Convert ``(..., 8)`` pose8 rows to float32 ``[xyz, roll, pitch, yaw, grip]`` rows."""
    a = np.asarray(pose8, dtype=np.float64)
    flat = a.reshape(-1, a.shape[-1])
    rot = np.stack([unit_quat_wxyz_to_matrix(q) for q in flat[:, 3:7]])
    rpy = Rotation.matrix_to_euler_xyz(rot)
    out = np.concatenate([flat[:, :3], rpy, flat[:, 7:8]], axis=1).astype(np.float32)
    return out.reshape(*a.shape[:-1], PRAXIS_ACTION_DIM)


def pose12_grip_to_droid7(pose12: np.ndarray, grip: float) -> np.ndarray:
    """Convert one praxis command ``[xyz, R_flat9]`` (row-major) and gripper to a 7-value row."""
    p = np.asarray(pose12, dtype=np.float64).reshape(12)
    rpy = Rotation.matrix_to_euler_xyz(p[3:].reshape(1, 3, 3))[0]
    return np.concatenate([p[:3], rpy, [float(grip)]]).astype(np.float32)


def _arms(rows: Tensor | np.ndarray, per_arm: int) -> int:
    """The arms of rows whose last dim is `per_arm` values per arm (1 or 2), else 0."""
    dim = int(rows.shape[-1])
    return dim // per_arm if dim in (per_arm, 2 * per_arm) else 0


def _feature_arms(feature: PolicyFeature) -> int:
    """The arms a state or action feature describes: 2 for 14 or 16 values, else 1."""
    dim = int(feature.shape[-1]) if feature.shape else 0
    return 2 if dim in (2 * PRAXIS_ACTION_DIM, 2 * PRAXIS_POSE_DIM) else 1


def _droid_rows_to_pose_tensor(rows: Tensor) -> Tensor:
    """``(..., 7 * n)`` DROID rows of ``n`` arms as ``(..., 8 * n)`` pose8 rows (float32)."""
    n = _arms(rows, PRAXIS_ACTION_DIM)
    if n == 0:
        raise ValueError(f"expected rows of 7 or 14 values, got {tuple(rows.shape)}")
    arms = [_droid7_to_pose8_tensor(rows[..., 7 * i : 7 * (i + 1)]) for i in range(n)]
    return torch.cat(arms, dim=-1)


def _droid7_to_pose8_tensor(rows: Tensor) -> Tensor:
    """``(..., 7)`` DROID rows to ``(..., 8)`` pose8 rows, float32 on the same device."""
    values = rows.detach().float().cpu().numpy().reshape(-1, PRAXIS_ACTION_DIM)
    pose8 = droid7_to_pose8(values).reshape(*rows.shape[:-1], PRAXIS_POSE_DIM)
    return torch.from_numpy(pose8).to(rows.device)


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


def tile_views(
    frames: Tensor, tile_height: int, tile_width: int, aspect_tol: float = ASPECT_TOL, antialias: bool = False
) -> Tensor:
    """Return the praxis tile of camera frames: ``(..., 3, tile_height, tile_width)`` float in [0, 1].

    The frames are ``(..., 3, H, W)``, float in [0, 1] or uint8. The operations are those of the
    praxis canvas (``single_arm.tile_clip``), so the policy's canvas of a tile is the tile itself,
    bit for bit. ``antialias`` is the ``video.antialias`` of the checkpoint config:

    * off (the earlier DROID checkpoint): a centre crop to the tile aspect (none when the aspects agree within
      ``aspect_tol``), one bilinear resize without antialias in float32 and a round to uint8 levels.
    * on (`praxis-1-droid`, the fit rule): a centre crop to the native 720 x 1280 aspect, one antialiased
      bilinear resize to 234 x 416 (none for a view already at that size) and a centre trim to the
      tile, 224 x 416.

    A view already at the tile size passes through: the fit rule is not idempotent, so a tile must
    not go through the canvas twice. The resize runs on the CPU whatever the device of the frames,
    as the praxis serve and training paths do: the CUDA antialiased bilinear kernel rounds differently
    (one level on about a third of the pixels). The result is on the device of the frames.
    """
    if frames.ndim < 3 or frames.shape[-3] != 3:
        raise ValueError(f"camera frames must be (..., 3, H, W), got {tuple(frames.shape)}")
    lead = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:]).cpu()
    if flat.dtype != torch.uint8:
        flat = (flat.float() * 255.0).round().clamp_(0, 255).to(torch.uint8)
    if tuple(flat.shape[-2:]) == (tile_height, tile_width):
        tiles = flat.float()
    elif antialias:
        tiles = _fit_view(flat, tile_height, tile_width, aspect_tol).float()
    else:
        cropped = aspect_crop(flat, tile_height, tile_width, aspect_tol)
        tiles = torch.cat(
            [
                F.interpolate(
                    chunk.float(), size=(tile_height, tile_width), mode="bilinear", align_corners=False
                )
                for chunk in cropped.split(TILE_CHUNK_FRAMES, dim=0)
            ]
        ).round()
    tiles = tiles.clamp_(0, 255) / 255.0
    return tiles.reshape(*lead, 3, tile_height, tile_width).to(frames.device)


def tile_views_bimanual(frames: dict[str, Tensor], tile_height: int, tile_width: int) -> dict[str, Tensor]:
    """The tiles of the bimanual rig's views (`bimanual.canvas_clip`'s fit rule), one tensor per key.

    The views share one fitted height (`fit_row_sizes`: each view scaled to fit a 416 px box with
    its aspect kept, the taller ones to the smallest height, then trimmed to the 32 px grid); the
    result must be the tile. Frames are ``(..., 3, H, W)`` float in [0, 1] or uint8; the resize
    runs on the CPU in float32 with antialias, as the praxis session does, and the tiles come back
    as float in [0, 1] on the device of the frames. Views already at the tile size pass through.
    """
    keys = list(frames)
    flat: dict[str, Tensor] = {}
    for key in keys:
        value = frames[key]
        if value.ndim < 3 or value.shape[-3] != 3:
            raise ValueError(f"{key}: camera frames must be (..., 3, H, W), got {tuple(value.shape)}")
        f = value.reshape(-1, *value.shape[-3:]).cpu()
        if f.dtype != torch.uint8:
            f = (f.float() * 255.0).round().clamp_(0, 255).to(torch.uint8)
        flat[key] = f
    native = [(int(f.shape[-2]), int(f.shape[-1])) for f in flat.values()]
    if all(n == (tile_height, tile_width) for n in native):
        tiles = dict(flat.items())
    else:
        resize, trimmed = bimanual.fit_row_sizes(native, bimanual.RESTRICT_SIDE, bimanual.SPATIAL_PATCH)
        if any(tr != (tile_height, tile_width) for tr in trimmed):
            raise ValueError(
                f"views of {native} fit to {trimmed}, not the {tile_height}x{tile_width} tiles of the "
                "checkpoint; send the views at their native aspect (16:10 for the YAM rig)"
            )
        tiles = {
            k: bimanual._trim(bimanual._resize(f, *r), tile_height, tile_width)
            for (k, f), r in zip(flat.items(), resize, strict=True)
        }
    out = {}
    for key in keys:
        lead = frames[key].shape[:-3]
        out[key] = (
            (tiles[key].float() / 255.0).reshape(*lead, 3, tile_height, tile_width).to(frames[key].device)
        )
    return out


@dataclass
@ProcessorStepRegistry.register(name="praxis_tile_views_processor")
class PraxisTileViewsProcessorStep(ObservationProcessorStep):
    """Prepare the cameras: each view becomes its tile of the praxis canvas (`tile_views`).

    Runs on the CPU before the device step, with the operations of the praxis canvas, so the
    tiles are the bytes the policy would compute itself. A view that is a tile already passes
    through unchanged.

    Attributes:
        camera_keys: Observation keys of the views (exterior, wrist).
        tile_height: Tile height of the checkpoint (224 for `praxis-1-droid`, 256 for the earlier DROID checkpoint).
        tile_width: Tile width of the checkpoint (416).
        aspect_tol: Aspect tolerance of the centre crop.
        antialias: The ``video.antialias`` of the checkpoint: the fit rule with an antialiased
            resize (`praxis-1-droid`) instead of the plain resize to the tile (the earlier DROID checkpoint).
        fit_rule: ``single_arm`` (the DROID fit rule of `tile_views`) or ``bimanual`` (the YAM
            rule of `tile_views_bimanual`: the views fit a 416 px box together).
    """

    camera_keys: list[str] = field(default_factory=list)
    tile_height: int = 256
    tile_width: int = 416
    aspect_tol: float = ASPECT_TOL
    antialias: bool = False
    fit_rule: str = "single_arm"

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        out = dict(observation)
        present = [k for k in self.camera_keys if k in out and isinstance(out[k], Tensor)]
        if self.fit_rule == "bimanual" and present:
            if len(present) != len(self.camera_keys):
                raise ValueError(f"the bimanual canvas needs every view of {self.camera_keys}, got {present}")
            out.update(tile_views_bimanual({k: out[k] for k in present}, self.tile_height, self.tile_width))
            return out
        for key in present:
            out[key] = tile_views(
                out[key], self.tile_height, self.tile_width, self.aspect_tol, self.antialias
            )
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_keys": list(self.camera_keys),
            "tile_height": self.tile_height,
            "tile_width": self.tile_width,
            "aspect_tol": self.aspect_tol,
            "antialias": self.antialias,
            "fit_rule": self.fit_rule,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observations = features.get(PipelineFeatureType.OBSERVATION, {})
        for key in self.camera_keys:
            if key in observations:
                observations[key] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, self.tile_height, self.tile_width)
                )
        return features


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------


@dataclass
@ProcessorStepRegistry.register(name="praxis_pose_processor")
class PraxisPoseProcessorStep(ProcessorStep):
    """Convert the DROID pose to the praxis pose8 ``[x, y, z, qw, qx, qy, qz, gripper]``.

    Applies to the state (``observation.state``) and, in training batches, to the commanded poses
    (``action``): the quaternion comes from the Euler angles with the operations of
    ``scipy.spatial.transform``, the gripper is clipped to [0, 1] (`droid7_to_pose8`). Rows that
    are pose8 already pass through.
    """

    state_key: str = OBS_STATE
    action_key: str = ACTION

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        out = transition.copy()
        observation = out.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict):
            state = observation.get(self.state_key)
            if isinstance(state, Tensor) and _arms(state, PRAXIS_ACTION_DIM):
                observation = dict(observation)
                observation[self.state_key] = _droid_rows_to_pose_tensor(state)
                out[TransitionKey.OBSERVATION] = observation
        action = out.get(TransitionKey.ACTION)
        if isinstance(action, Tensor) and _arms(action, PRAXIS_ACTION_DIM):
            out[TransitionKey.ACTION] = _droid_rows_to_pose_tensor(action)
        return out

    def get_config(self) -> dict[str, Any]:
        return {"state_key": self.state_key, "action_key": self.action_key}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observations = features.get(PipelineFeatureType.OBSERVATION, {})
        if self.state_key in observations:
            n = _feature_arms(observations[self.state_key])
            observations[self.state_key] = PolicyFeature(type=FeatureType.STATE, shape=(PRAXIS_POSE_DIM * n,))
        actions = features.get(PipelineFeatureType.ACTION, {})
        if self.action_key in actions:
            n = _feature_arms(actions[self.action_key])
            actions[self.action_key] = PolicyFeature(type=FeatureType.ACTION, shape=(PRAXIS_POSE_DIM * n,))
        return features


# ---------------------------------------------------------------------------
# Instruction tokens
# ---------------------------------------------------------------------------


@dataclass
@ProcessorStepRegistry.register(name="praxis_tokenizer_processor")
class PraxisTokenizerProcessorStep(ProcessorStep):
    """Tokenize the instruction for the praxis text encoder (Qwen3.5-2B).

    The tokenizer is the ``AutoProcessor`` of the text encoder, called as the praxis text encoder
    calls it: no chat template, the tokenizer's special tokens, right padding to a multiple of
    ``pad_to_multiple_of``, no truncation. The instruction is stripped of surrounding whitespace
    as the praxis session strips it; a blank instruction becomes one space. The step writes
    ``observation.language.tokens`` ``(B, L)`` int64 and ``observation.language.attention_mask``
    ``(B, L)`` bool; the policy strips the padding, pads each instruction on its own as the text
    encoder does, encodes it with the frozen Qwen3.5-2B and caches the embedding per token sequence.

    Attributes:
        tokenizer_id: The Hub repo id or local directory of the text encoder.
        pad_to_multiple_of: The padding of the praxis text encoder.
        task_key: The key of the instruction in the transition's complementary data.
    """

    tokenizer_id: str = "Qwen/Qwen3.5-2B"
    pad_to_multiple_of: int = PAD_TO_MULTIPLE_OF
    task_key: str = "task"
    tokenizer: Any = field(default=None, init=False, repr=False)

    def _tokenizer(self) -> Any:
        if self.tokenizer is None:
            from transformers import AutoProcessor

            self.tokenizer = AutoProcessor.from_pretrained(self.tokenizer_id)
        return self.tokenizer

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        task = complementary.get(self.task_key)
        if task is None:
            return transition
        # The praxis session and data module strip the instruction before the text encoder.
        tasks = [task.strip()] if isinstance(task, str) else [str(t).strip() for t in task]
        encoded = self._tokenizer()(
            text=[text_encoder_prompt(t) for t in tasks],
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        out = transition.copy()
        observation = dict(out.get(TransitionKey.OBSERVATION) or {})
        observation[OBS_LANGUAGE_TOKENS] = encoded["input_ids"].to(torch.int64)
        observation[OBS_LANGUAGE_ATTENTION_MASK] = encoded["attention_mask"].to(torch.bool)
        out[TransitionKey.OBSERVATION] = observation
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "tokenizer_id": self.tokenizer_id,
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "task_key": self.task_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observations = features.setdefault(PipelineFeatureType.OBSERVATION, {})
        # The length follows the instruction (a multiple of ``pad_to_multiple_of``).
        observations.setdefault(
            OBS_LANGUAGE_TOKENS, PolicyFeature(type=FeatureType.LANGUAGE, shape=(self.pad_to_multiple_of,))
        )
        observations.setdefault(
            OBS_LANGUAGE_ATTENTION_MASK,
            PolicyFeature(type=FeatureType.LANGUAGE, shape=(self.pad_to_multiple_of,)),
        )
        return features


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@dataclass
@ProcessorStepRegistry.register(name="praxis_command_processor")
class PraxisCommandProcessorStep(PolicyActionProcessorStep):
    """Convert the policy's command to the DROID ``[x, y, z, roll, pitch, yaw, gripper]``.

    The policy emits the praxis wire pose ``[x, y, z, rotation (9, row-major)]`` and the gripper
    (``PRAXIS_COMMAND_DIM`` values) on the CPU; the Euler angles come from the rotation matrix with
    the operations of ``scipy.spatial.transform`` (`pose12_grip_to_droid7`). The result stays on
    the CPU, where the commands go next.
    """

    n_arms: int = 1

    def action(self, action: PolicyAction) -> PolicyAction:
        width = PRAXIS_COMMAND_DIM * self.n_arms
        if action.shape[-1] != width:
            raise ValueError(
                f"a praxis command of {self.n_arms} arm(s) has {width} values, got {tuple(action.shape)}"
            )
        rows = action.detach().float().cpu().numpy().reshape(-1, width)
        out = np.stack(
            [
                np.concatenate(
                    [
                        pose12_grip_to_droid7(row[13 * i : 13 * i + 12], float(row[13 * i + 12]))
                        for i in range(self.n_arms)
                    ]
                )
                for row in rows
            ]
        ).astype(np.float32)
        return torch.from_numpy(out.reshape(*action.shape[:-1], PRAXIS_ACTION_DIM * self.n_arms))

    def get_config(self) -> dict[str, Any]:
        return {"n_arms": self.n_arms}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        actions = features.get(PipelineFeatureType.ACTION, {})
        if ACTION in actions:
            actions[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(PRAXIS_ACTION_DIM * self.n_arms,))
        return features


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def make_praxis_pre_post_processors(
    config: PraxisConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the Praxis pipelines (see the module docstring).

    Args:
        config: The policy configuration; the tile size, the aspect tolerance and the canvas rule
            come from the checkpoint's ``model_config``, the tokenizer from ``text_encoder_id``
            (no tokenizer step when it is empty: the policy then tokenizes the task itself).
        dataset_stats: Dataset statistics. Praxis normalizes with the statistics of its checkpoint,
            so they are not used; the argument keeps the factory signature of every policy.
    """
    steps = make_default_policy_processor_steps(config, dataset_stats)
    video = (config.model_config or {}).get("video", {})
    bimanual_rig = config.embodiment == "bimanual"
    if config.model_config and video.get("canvas") == "tiles":
        tile_height, canvas_width = (bimanual.canvas_size if bimanual_rig else canvas_size)(
            config.model_config
        )
        tile_width = canvas_width // len(config.image_keys)
    else:
        tile_height, tile_width = int(video.get("tile_height", 256)), int(video.get("tile_width", 416))
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        PraxisTileViewsProcessorStep(
            camera_keys=list(config.image_keys),
            tile_height=tile_height,
            tile_width=tile_width,
            aspect_tol=float(video.get("aspect_tol", ASPECT_TOL)),
            antialias=bool(video.get("antialias", False)),
            fit_rule="bimanual" if bimanual_rig else "single_arm",
        ),
        PraxisPoseProcessorStep(),
    ]
    if config.text_encoder_id:
        input_steps.append(
            PraxisTokenizerProcessorStep(
                tokenizer_id=config.text_encoder_id, pad_to_multiple_of=config.text_pad_to_multiple_of
            )
        )
    input_steps += [steps.to_device, steps.normalize]
    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        PraxisCommandProcessorStep(n_arms=config.n_arms),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
