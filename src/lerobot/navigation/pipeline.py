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

"""Keyframe integration loop core.

Ported from the dyna360 research stack (viz-free). One keyframe is
carved then added into the voxel map — carve first so we never remove
voxels we just created this frame. This is the shared step behind live
mapping on the robot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lerobot.navigation.voxel_map import CarveResult, VoxelMap, VoxelMapStats

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyframeContext:
    """Everything one keyframe needs to contribute to the voxel map.

    ``rgb_uint8`` is RGB order (same layout fed to the geometry model and
    the feature extractor). ``points_world`` / ``local_points`` come from
    the geometry runner; ``feat_map`` is the bilinearly-upsampled patch
    grid at ``(H, W, D)`` fp16, or ``None`` for a geometry-only frame.
    """

    frame_idx: int
    t_sec: float
    rgb_uint8: np.ndarray  # (H, W, 3) RGB uint8
    points_world: np.ndarray  # (H, W, 3) float32
    local_points: np.ndarray  # (H, W, 3) float32
    conf: np.ndarray  # (H, W) in [0, 1]
    pose: np.ndarray  # (4, 4) cam-to-world
    feat_map: np.ndarray | None  # (H, W, D) fp16, L2-normalized per pixel


@dataclass(frozen=True)
class PipelineConfig:
    """Knobs that change per-run but not per-keyframe."""

    conf_thresh: float = 0.5
    carve_margin: float = 0.05
    focal_px: float = 100.0


def integrate_keyframe(
    voxel_map: VoxelMap,
    ctx: KeyframeContext,
    pcfg: PipelineConfig | None = None,
) -> tuple[CarveResult, VoxelMapStats]:
    """Carve observed free space, then add this keyframe's points.

    Carve runs before add. Returns the carve result + add stats so callers
    can surface them in their own progress UI.
    """
    pcfg = pcfg or PipelineConfig()
    carve = voxel_map.carve(
        local_points=ctx.local_points,
        conf=ctx.conf,
        pose=ctx.pose,
        focal_px=pcfg.focal_px,
        frame=ctx.frame_idx,
        t=ctx.t_sec,
        conf_thresh=pcfg.conf_thresh,
        margin=pcfg.carve_margin,
    )
    stats = voxel_map.add(
        points=ctx.points_world,
        rgb=ctx.rgb_uint8,
        conf=ctx.conf,
        frame=ctx.frame_idx,
        t=ctx.t_sec,
        conf_thresh=pcfg.conf_thresh,
        feat_map=ctx.feat_map,
    )
    return carve, stats


def upsample_features_to_view(
    patch_feats_one_view: np.ndarray,
    view_h: int,
    view_w: int,
) -> np.ndarray:
    """Bilinearly upsample one keyframe's ``(Hp, Wp, D)`` patch features to
    view resolution ``(H, W, D)`` fp16."""
    import torch

    fp = torch.from_numpy(patch_feats_one_view).permute(2, 0, 1).unsqueeze(0).float()  # (1, D, Hp, Wp)
    fp_up = torch.nn.functional.interpolate(fp, size=(view_h, view_w), mode="bilinear", align_corners=False)
    return fp_up.squeeze(0).permute(1, 2, 0).to(torch.float16).numpy()
