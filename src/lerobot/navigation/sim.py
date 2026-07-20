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

"""Synthetic scenes for hardware-free dry-runs and tests.

Ported from the dyna360 eval harness. A :class:`SyntheticScene` is a
deterministic hand-crafted :class:`~lerobot.navigation.voxel_map.VoxelMap`
— a navigable floor plus labelled objects each carrying a unit feature
vector — paired with a
:class:`~lerobot.navigation.features.BasisVectorFeatureExtractor` whose
text encodings live in the same space. This lets ``dog_cli --dry-run``
(and the tests) exercise the full locate/goto/explore stack with no
models, camera, or robot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lerobot.navigation.features import BasisVectorFeatureExtractor
from lerobot.navigation.voxel_map import VoxelMap

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticObject:
    """One labelled object. ``feature_vec`` lives in the same space as the
    text embeddings fed to ``VoxelMap.query`` (one-hot basis vectors, so a
    query hits the right cluster cleanly)."""

    name: str
    xyz: tuple[float, float, float]
    half_extent_m: float
    feature_vec: np.ndarray


@dataclass(frozen=True)
class SyntheticScene:
    """A ground-truth scene: voxel map + object metadata."""

    voxel_map: VoxelMap
    objects: list[SyntheticObject]
    floor_extent_m: float
    voxel_size: float
    feature_dim: int

    def name_to_xyz(self) -> dict[str, tuple[float, float, float]]:
        return {o.name: o.xyz for o in self.objects}

    def object(self, name: str) -> SyntheticObject | None:
        for o in self.objects:
            if o.name == name:
                return o
        return None

    def feature_extractor(self) -> BasisVectorFeatureExtractor:
        """A text encoder whose vectors match this scene's object features."""
        table = {o.name: o.feature_vec for o in self.objects}
        return BasisVectorFeatureExtractor(table, self.feature_dim)


@dataclass(frozen=True)
class SceneSpec:
    """Declarative recipe used by :func:`build_scene`."""

    objects: list[SyntheticObject]
    floor_extent_m: float = 6.0
    voxel_size: float = 0.1
    feature_dim: int = 8
    ground_y: float = 1.0
    object_density_per_dim: int = 5
    wall_xz_range: tuple[float, float, float, float] | None = None
    """Optional axis-aligned wall ``(x_min, z_min, x_max, z_max)`` of
    obstacle voxels at robot height — to test ``goto`` against a block."""
    feature_noise: float = 0.0
    rng_seed: int = 0


def basis_vec(dim: int, idx: int) -> np.ndarray:
    """A unit basis vector of length ``dim`` with a 1 at ``idx``."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def build_scene(spec: SceneSpec) -> SyntheticScene:
    """Construct a deterministic :class:`SyntheticScene` from a spec."""
    rng = np.random.default_rng(spec.rng_seed)
    vm = VoxelMap(voxel_size=spec.voxel_size)

    # ----- floor (NAVIGABLE) -----
    half = spec.voxel_size / 2.0
    floor_pts: list[tuple[float, float, float]] = []
    for x in np.arange(-spec.floor_extent_m + half, spec.floor_extent_m + half, spec.voxel_size):
        for z in np.arange(-spec.floor_extent_m + half, spec.floor_extent_m + half, spec.voxel_size):
            floor_pts.append((float(x), spec.ground_y, float(z)))
    arr = np.asarray(floor_pts, dtype=np.float64).reshape(-1, 1, 3)
    rgb = np.full((len(floor_pts), 1, 3), 180, dtype=np.uint8)
    conf = np.ones((len(floor_pts), 1), dtype=np.float32)
    if spec.feature_dim >= 1:
        floor_vec = np.zeros(spec.feature_dim, dtype=np.float16)
        floor_vec[-1] = 1.0
        floor_feat = np.tile(floor_vec, (len(floor_pts), 1, 1))
        vm.add(arr, rgb, conf, frame=0, t=0.0, feat_map=floor_feat)
    else:
        vm.add(arr, rgb, conf, frame=0, t=0.0)

    # ----- objects -----
    for i, obj in enumerate(spec.objects, start=1):
        d = obj.half_extent_m
        n = spec.object_density_per_dim
        coords = np.linspace(-d + half, d - half, n)
        pts = np.array(
            [
                (float(obj.xyz[0] + dx), float(obj.xyz[1] + dy), float(obj.xyz[2] + dz))
                for dx in coords
                for dy in coords
                for dz in coords
            ],
            dtype=np.float64,
        ).reshape(-1, 1, 3)
        rgb_o = np.full((pts.shape[0], 1, 3), 100 + (i * 30) % 156, dtype=np.uint8)
        conf_o = np.ones((pts.shape[0], 1), dtype=np.float32)

        if obj.feature_vec.shape != (spec.feature_dim,):
            raise ValueError(
                f"object {obj.name!r} feature_vec has shape {obj.feature_vec.shape}, "
                f"expected ({spec.feature_dim},) to match SceneSpec.feature_dim"
            )
        base = obj.feature_vec.astype(np.float32).reshape(1, 1, -1)
        feats = np.tile(base, (pts.shape[0], 1, 1))
        if spec.feature_noise > 0:
            noise = rng.normal(scale=spec.feature_noise, size=feats.shape).astype(np.float32)
            feats = feats + noise
            norms = np.linalg.norm(feats, axis=-1, keepdims=True)
            feats = feats / np.maximum(norms, 1e-6)
        vm.add(pts, rgb_o, conf_o, frame=i, t=float(i), feat_map=feats.astype(np.float16))

    # ----- optional wall (OBSTACLE) -----
    if spec.wall_xz_range is not None:
        wx0, wz0, wx1, wz1 = spec.wall_xz_range
        wall_pts = [
            (float(x), float(y), float(z))
            for x in np.arange(wx0 + half, wx1, spec.voxel_size)
            for z in np.arange(wz0 + half, wz1, spec.voxel_size)
            for y in np.arange(spec.ground_y - 1.0, spec.ground_y - 0.1, spec.voxel_size)
        ]
        if wall_pts:
            pts = np.asarray(wall_pts, dtype=np.float64).reshape(-1, 1, 3)
            rgb_w = np.full((len(wall_pts), 1, 3), 80, dtype=np.uint8)
            conf_w = np.ones((len(wall_pts), 1), dtype=np.float32)
            vm.add(pts, rgb_w, conf_w, frame=99, t=99.0)

    LOG.info(
        "built scene: %d voxels, %d objects, floor extent %.1f m, D=%d",
        len(vm),
        len(spec.objects),
        spec.floor_extent_m,
        spec.feature_dim,
    )
    return SyntheticScene(
        voxel_map=vm,
        objects=list(spec.objects),
        floor_extent_m=spec.floor_extent_m,
        voxel_size=spec.voxel_size,
        feature_dim=spec.feature_dim,
    )


_KITCHEN_DIM = 64  # Feature dim sized so the random-direction noise floor
# (≈1/sqrt(D) ≈ 0.125) sits well below a sane locate threshold, so an absent
# object reliably ABSTAINS instead of hitting a known basis vector.


def kitchen_scene(wall: tuple[float, float, float, float] | None = None) -> SyntheticScene:
    """A 6×6 m floor with four labelled objects at distinctive corners."""
    spec = SceneSpec(
        objects=[
            SyntheticObject("couch", (3.0, 0.5, 2.0), 0.3, basis_vec(_KITCHEN_DIM, 0)),
            SyntheticObject("chair", (-2.0, 0.5, -1.5), 0.2, basis_vec(_KITCHEN_DIM, 1)),
            SyntheticObject("lamp", (2.5, 0.5, -2.0), 0.15, basis_vec(_KITCHEN_DIM, 2)),
            SyntheticObject("plant", (-2.5, 0.5, 2.5), 0.25, basis_vec(_KITCHEN_DIM, 3)),
        ],
        floor_extent_m=6.0,
        voxel_size=0.1,
        feature_dim=_KITCHEN_DIM,
        ground_y=1.0,
        wall_xz_range=wall,
    )
    return build_scene(spec)
