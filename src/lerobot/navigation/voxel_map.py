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

"""Sparse-hash voxel memory with free-space carving + semantic features.

Ported from the dyna360 research stack. Per occupied voxel: voxel index,
running-mean xyz (count-weighted), running-mean rgb (count-weighted),
count, last_frame, last_time, and — once vision-language features have
been fed in — a conf-weighted running-mean feature in fp16 plus the
weight sum.

Storage is hybrid: a Python dict maps voxel index ``(ix, iy, iz)`` to a
row in column-stored numpy arrays so lookup is O(1) and bulk arithmetic
stays vectorized. ``carve`` removes voxels that fall inside a view's
observed free space (DynaMem-style dynamic updates); ``query`` returns
the top-k cosine matches against a text embedding.

Default voxel size is 5 cm. The map is geometry-only until
``add(..., feat_map=...)`` supplies per-pixel features; occupancy /
planning use only the geometry, so they work without any features.
"""

# ruff: noqa: N806  — H, W, D are conventional array-dimension names (and appear verbatim in error strings)
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass
class VoxelMapStats:
    """Per-keyframe deltas, surfaced to scalar logs."""

    n_voxels: int
    n_added: int
    n_updated: int
    n_removed: int = 0


@dataclass(frozen=True)
class VoxelSnapshot:
    """Current voxel map state, materialized for visualization / export."""

    xyz: np.ndarray  # (M, 3) float32 — count-weighted mean position
    rgb: np.ndarray  # (M, 3) uint8 — count-weighted mean color (RGB)
    count: np.ndarray  # (M,)  int64
    last_frame: np.ndarray  # (M,)  int64
    last_time: np.ndarray  # (M,)  float64
    feat: np.ndarray | None = None  # (M, D) fp16 — L2-normalized per-voxel mean


@dataclass(frozen=True)
class CarveResult:
    """Output of one ``carve`` pass."""

    n_removed: int
    removed_xyz: np.ndarray  # (K, 3) float32 — centres of removed voxels, for viz


@dataclass(frozen=True)
class QueryResult:
    """Top-k cosine matches against a text embedding."""

    xyz: np.ndarray  # (k, 3) float32
    score: np.ndarray  # (k,) float32 — cosine similarity in [-1, 1]
    voxel_indices: np.ndarray  # (k,) int64 — row indices into the map


_MAX_ABS_VOXEL_INDEX = 1 << 20
_FEAT_CHUNK_PIXELS = 16384  # bound peak per-keyframe feature contribution memory


class VoxelMap:
    """Sparse-hash voxel grid with count-weighted means and semantic features."""

    def __init__(self, voxel_size: float = 0.05) -> None:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")
        self.voxel_size = float(voxel_size)

        self._lookup: dict[tuple[int, int, int], int] = {}
        self._idx = np.zeros((0, 3), dtype=np.int64)
        self._count = np.zeros(0, dtype=np.int64)
        self._xyz_sum = np.zeros((0, 3), dtype=np.float64)
        self._rgb_sum = np.zeros((0, 3), dtype=np.float64)
        self._last_frame = np.zeros(0, dtype=np.int64)
        self._last_time = np.zeros(0, dtype=np.float64)

        # Lazily allocated on first add() with feat_map.
        self._feature_dim: int | None = None
        self._feat_sum: np.ndarray | None = None  # (M, D) fp16
        self._feat_weight: np.ndarray | None = None  # (M,) fp32

    def __len__(self) -> int:
        return int(self._count.shape[0])

    @property
    def feature_dim(self) -> int | None:
        return self._feature_dim

    # ------------------------------------------------------------------ add

    def add(
        self,
        points: np.ndarray,
        rgb: np.ndarray,
        conf: np.ndarray,
        frame: int,
        t: float,
        conf_thresh: float = 0.5,
        feat_map: np.ndarray | None = None,
    ) -> VoxelMapStats:
        """Insert / update voxels from a per-pixel observation.

        ``points``:   ``(..., 3)`` world xyz, fp32.
        ``rgb``:      ``(..., 3)`` uint8 (RGB order).
        ``conf``:     ``(...,)`` in [0, 1].
        ``feat_map``: optional ``(..., D)`` fp16 per-pixel feature, already
                      bilinearly upsampled to the points/conf grid. First
                      call with features locks the feature dimension;
                      subsequent calls must match.
        """
        pts = np.asarray(points).reshape(-1, 3)
        cols = np.asarray(rgb).reshape(-1, 3)
        cnf = np.asarray(conf).reshape(-1)
        if not (len(pts) == len(cols) == len(cnf)):
            raise ValueError(f"length mismatch: points={len(pts)}, rgb={len(cols)}, conf={len(cnf)}")

        features: np.ndarray | None = None
        if feat_map is not None:
            features = np.asarray(feat_map).reshape(-1, feat_map.shape[-1])
            if len(features) != len(pts):
                raise ValueError(f"feat_map length {len(features)} != points length {len(pts)}")
            D = features.shape[-1]
            if self._feature_dim is None:
                self._feature_dim = int(D)
                # Pad pre-existing voxels (added before features arrived) with zeros.
                self._feat_sum = np.zeros((len(self), D), dtype=np.float16)
                self._feat_weight = np.zeros(len(self), dtype=np.float32)
                LOG.info("VoxelMap features enabled: D=%d (fp16 storage)", D)
            elif self._feature_dim != D:
                raise ValueError(f"feature dim mismatch: existing={self._feature_dim}, got={D}")

        mask = (cnf >= conf_thresh) & np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        cols = cols[mask]
        cnf_kept = cnf[mask]
        if features is not None:
            features = features[mask]
        if pts.size == 0:
            return VoxelMapStats(n_voxels=len(self), n_added=0, n_updated=0)

        idx = np.floor(pts / self.voxel_size).astype(np.int64)
        sane = (np.abs(idx) < _MAX_ABS_VOXEL_INDEX).all(axis=1)
        if not sane.all():
            n_drop = int((~sane).sum())
            LOG.debug("dropping %d points with extreme voxel index", n_drop)
            idx = idx[sane]
            pts = pts[sane]
            cols = cols[sane]
            cnf_kept = cnf_kept[sane]
            if features is not None:
                features = features[sane]
        if idx.size == 0:
            return VoxelMapStats(n_voxels=len(self), n_added=0, n_updated=0)

        unique_idx, inverse = np.unique(idx, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        n_unique = unique_idx.shape[0]
        kf_count = np.bincount(inverse, minlength=n_unique).astype(np.int64)
        kf_xyz_sum = np.zeros((n_unique, 3), dtype=np.float64)
        kf_rgb_sum = np.zeros((n_unique, 3), dtype=np.float64)
        np.add.at(kf_xyz_sum, inverse, pts.astype(np.float64))
        np.add.at(kf_rgb_sum, inverse, cols.astype(np.float64))

        kf_feat_sum: np.ndarray | None = None
        kf_feat_weight: np.ndarray | None = None
        if features is not None:
            kf_feat_sum = np.zeros((n_unique, self._feature_dim), dtype=np.float32)
            kf_feat_weight = np.zeros(n_unique, dtype=np.float32)
            cnf_f = cnf_kept.astype(np.float32)
            # Chunked accumulation — keeps the (chunk, D) intermediate small.
            for s in range(0, features.shape[0], _FEAT_CHUNK_PIXELS):
                e = s + _FEAT_CHUNK_PIXELS
                w = cnf_f[s:e]
                contrib = w[:, None] * features[s:e].astype(np.float32)
                np.add.at(kf_feat_sum, inverse[s:e], contrib)
                np.add.at(kf_feat_weight, inverse[s:e], w)

        existing_rows: list[int] = []
        existing_local: list[int] = []
        new_local: list[int] = []
        new_keys: list[tuple[int, int, int]] = []
        for i in range(n_unique):
            key = (int(unique_idx[i, 0]), int(unique_idx[i, 1]), int(unique_idx[i, 2]))
            row = self._lookup.get(key)
            if row is None:
                new_local.append(i)
                new_keys.append(key)
            else:
                existing_rows.append(row)
                existing_local.append(i)

        if existing_rows:
            rows = np.asarray(existing_rows, dtype=np.int64)
            local = np.asarray(existing_local, dtype=np.int64)
            self._count[rows] += kf_count[local]
            self._xyz_sum[rows] += kf_xyz_sum[local]
            self._rgb_sum[rows] += kf_rgb_sum[local]
            self._last_frame[rows] = frame
            self._last_time[rows] = t
            if kf_feat_sum is not None:
                assert self._feat_sum is not None and self._feat_weight is not None
                # fp32 accumulator -> fp16 storage; cast on store to match storage dtype.
                self._feat_sum[rows] = (self._feat_sum[rows].astype(np.float32) + kf_feat_sum[local]).astype(
                    np.float16
                )
                self._feat_weight[rows] += kf_feat_weight[local]

        if new_local:
            base = len(self)
            local = np.asarray(new_local, dtype=np.int64)
            self._idx = np.concatenate([self._idx, unique_idx[local]], axis=0)
            self._count = np.concatenate([self._count, kf_count[local]])
            self._xyz_sum = np.concatenate([self._xyz_sum, kf_xyz_sum[local]], axis=0)
            self._rgb_sum = np.concatenate([self._rgb_sum, kf_rgb_sum[local]], axis=0)
            self._last_frame = np.concatenate(
                [self._last_frame, np.full(len(new_local), frame, dtype=np.int64)]
            )
            self._last_time = np.concatenate([self._last_time, np.full(len(new_local), t, dtype=np.float64)])
            if self._feature_dim is not None:
                assert self._feat_sum is not None and self._feat_weight is not None
                if kf_feat_sum is not None:
                    new_feats = kf_feat_sum[local].astype(np.float16)
                    new_weights = kf_feat_weight[local]
                else:
                    # Features enabled, but this call didn't bring any — pad zeros
                    # so array sizes stay aligned with _count.
                    new_feats = np.zeros((len(new_local), self._feature_dim), dtype=np.float16)
                    new_weights = np.zeros(len(new_local), dtype=np.float32)
                self._feat_sum = np.concatenate([self._feat_sum, new_feats], axis=0)
                self._feat_weight = np.concatenate([self._feat_weight, new_weights])
            for offset, key in enumerate(new_keys):
                self._lookup[key] = base + offset

        return VoxelMapStats(
            n_voxels=len(self),
            n_added=len(new_local),
            n_updated=len(existing_rows),
        )

    # ------------------------------------------------------- hard-delete
    def remove_voxels_in_box(
        self,
        xyz_min: tuple[float, float, float],
        xyz_max: tuple[float, float, float],
    ) -> int:
        """Surgical hard-delete of every voxel whose mean position lies inside
        the axis-aligned bounding box.

        Different from :meth:`carve` (DynaMem-style free-space removal from a
        camera frustum + depth). This one is for simulated scene mutation:
        "the couch moved away" is removing the box around the old couch then
        ``add()``-ing one at the new position.
        """
        if len(self) == 0:
            return 0
        cnt = self._count.astype(np.float64).reshape(-1, 1)
        means = self._xyz_sum / cnt
        in_box = (
            (means[:, 0] >= xyz_min[0])
            & (means[:, 0] <= xyz_max[0])
            & (means[:, 1] >= xyz_min[1])
            & (means[:, 1] <= xyz_max[1])
            & (means[:, 2] >= xyz_min[2])
            & (means[:, 2] <= xyz_max[2])
        )
        if not in_box.any():
            return 0
        keep = ~in_box
        n_removed = int(in_box.sum())
        for k in self._idx[in_box]:
            del self._lookup[(int(k[0]), int(k[1]), int(k[2]))]
        self._idx = self._idx[keep]
        self._count = self._count[keep]
        self._xyz_sum = self._xyz_sum[keep]
        self._rgb_sum = self._rgb_sum[keep]
        self._last_frame = self._last_frame[keep]
        self._last_time = self._last_time[keep]
        if self._feat_sum is not None and self._feat_weight is not None:
            self._feat_sum = self._feat_sum[keep]
            self._feat_weight = self._feat_weight[keep]
        # Row indices shifted — rebuild the lookup.
        self._lookup = {
            (int(self._idx[i, 0]), int(self._idx[i, 1]), int(self._idx[i, 2])): i
            for i in range(len(self._idx))
        }
        return n_removed

    # ---------------------------------------------------------------- carve

    def carve(
        self,
        local_points: np.ndarray,
        conf: np.ndarray,
        pose: np.ndarray,
        focal_px: float,
        frame: int,
        t: float,
        conf_thresh: float = 0.5,
        margin: float = 0.05,
    ) -> CarveResult:
        """Remove voxels inside this view's observed free space.

        A voxel is carved when it projects into the image, sits in front of
        the camera, and lies closer than the observed depth (minus a margin)
        at that pixel — i.e. we can see through where it claims to be. Carve
        runs before ``add`` each keyframe so moved/removed objects disappear.
        """
        if len(self) == 0:
            return CarveResult(0, np.zeros((0, 3), dtype=np.float32))

        if local_points.ndim != 3 or local_points.shape[-1] != 3:
            raise ValueError(f"expected (H, W, 3), got {local_points.shape}")
        if conf.shape != local_points.shape[:2]:
            raise ValueError(f"conf shape {conf.shape} != local_points (H, W) {local_points.shape[:2]}")
        if pose.shape != (4, 4):
            raise ValueError(f"pose must be (4, 4); got {pose.shape}")

        H, W = local_points.shape[:2]
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        depth_map = local_points[..., 2]

        cnt = self._count.astype(np.float64).reshape(-1, 1)
        xyz_world = self._xyz_sum / cnt

        R = pose[:3, :3].astype(np.float64)
        t_vec = pose[:3, 3].astype(np.float64)
        xyz_cam = (xyz_world - t_vec[None, :]) @ R

        d_voxel = xyz_cam[:, 2]
        front = d_voxel > 1e-3

        d_safe = np.where(front, d_voxel, 1.0)
        u = focal_px * xyz_cam[:, 0] / d_safe + cx
        v = focal_px * xyz_cam[:, 1] / d_safe + cy
        in_bounds = (u >= 0.0) & (u < W) & (v >= 0.0) & (v < H)
        valid = front & in_bounds

        u_i = np.clip(np.floor(u).astype(np.int64), 0, W - 1)
        v_i = np.clip(np.floor(v).astype(np.int64), 0, H - 1)
        D_at = depth_map[v_i, u_i]
        C_at = conf[v_i, u_i]

        finite_D = np.isfinite(D_at) & (D_at > 0.0)
        free_space = valid & finite_D & (C_at >= conf_thresh) & (d_voxel < (D_at - margin))

        n_removed = int(free_space.sum())
        if n_removed == 0:
            return CarveResult(0, np.zeros((0, 3), dtype=np.float32))

        removed_xyz = xyz_world[free_space].astype(np.float32)
        removed_keys = self._idx[free_space]
        for k in removed_keys:
            del self._lookup[(int(k[0]), int(k[1]), int(k[2]))]

        keep = ~free_space
        self._idx = self._idx[keep]
        self._count = self._count[keep]
        self._xyz_sum = self._xyz_sum[keep]
        self._rgb_sum = self._rgb_sum[keep]
        self._last_frame = self._last_frame[keep]
        self._last_time = self._last_time[keep]
        if self._feat_sum is not None and self._feat_weight is not None:
            self._feat_sum = self._feat_sum[keep]
            self._feat_weight = self._feat_weight[keep]

        self._lookup = {
            (int(self._idx[i, 0]), int(self._idx[i, 1]), int(self._idx[i, 2])): i
            for i in range(len(self._idx))
        }
        LOG.debug("carve frame=%d t=%.3fs removed=%d", frame, t, n_removed)
        return CarveResult(n_removed=n_removed, removed_xyz=removed_xyz)

    # ------------------------------------------------------------- snapshot

    def snapshot(self, include_features: bool = False) -> VoxelSnapshot:
        """Materialize the current map.

        ``include_features``: pay the cost of normalizing the per-voxel
        feature mean. Off by default — visualization doesn't need features.
        """
        if len(self) == 0:
            return VoxelSnapshot(
                xyz=np.zeros((0, 3), dtype=np.float32),
                rgb=np.zeros((0, 3), dtype=np.uint8),
                count=np.zeros(0, dtype=np.int64),
                last_frame=np.zeros(0, dtype=np.int64),
                last_time=np.zeros(0, dtype=np.float64),
                feat=None,
            )
        cnt = self._count.astype(np.float64).reshape(-1, 1)
        xyz = (self._xyz_sum / cnt).astype(np.float32)
        rgb = np.clip(self._rgb_sum / cnt, 0, 255).astype(np.uint8)

        feat = None
        if include_features and self._feat_sum is not None and self._feat_weight is not None:
            feat = self._normalized_features()

        return VoxelSnapshot(
            xyz=xyz,
            rgb=rgb,
            count=self._count.copy(),
            last_frame=self._last_frame.copy(),
            last_time=self._last_time.copy(),
            feat=feat,
        )

    def _normalized_features(self) -> np.ndarray:
        """Per-voxel L2-normalized feature mean. (M, D) fp16."""
        assert self._feat_sum is not None and self._feat_weight is not None
        w = np.maximum(self._feat_weight, 1e-6).reshape(-1, 1)
        mean = self._feat_sum.astype(np.float32) / w
        norms = np.linalg.norm(mean, axis=1, keepdims=True)
        mean = mean / np.maximum(norms, 1e-6)
        return mean.astype(np.float16)

    # ----------------------------------------------------------------- query

    def query(self, text_embedding: np.ndarray, top_k: int = 32) -> QueryResult:
        """Top-k cosine matches against ``text_embedding``.

        ``text_embedding``: ``(D,)`` array — does NOT need to be unit norm;
        we re-normalize.
        """
        if self._feat_sum is None or self._feature_dim is None:
            raise RuntimeError("VoxelMap has no semantic features yet — call add(..., feat_map=...) first")
        if len(self) == 0:
            return QueryResult(
                xyz=np.zeros((0, 3), dtype=np.float32),
                score=np.zeros(0, dtype=np.float32),
                voxel_indices=np.zeros(0, dtype=np.int64),
            )
        if text_embedding.shape != (self._feature_dim,):
            raise ValueError(f"text_embedding shape {text_embedding.shape} != ({self._feature_dim},)")

        voxel_feat = self._normalized_features().astype(np.float32)
        text_unit = text_embedding.astype(np.float32)
        text_unit = text_unit / max(float(np.linalg.norm(text_unit)), 1e-6)

        # fp16 feature storage can carry the odd inf/nan from a saturated
        # running sum; the cosine stays well-defined, so don't warn on it.
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            scores = np.nan_to_num(voxel_feat @ text_unit)  # (M,)
        k = min(int(top_k), len(scores))
        # Partition-and-sort for the top-k.
        top_idx = np.argpartition(scores, -k)[-k:]
        order = np.argsort(-scores[top_idx])
        top_idx = top_idx[order]

        snap_xyz = (self._xyz_sum[top_idx] / self._count[top_idx].astype(np.float64).reshape(-1, 1)).astype(
            np.float32
        )
        return QueryResult(
            xyz=snap_xyz,
            score=scores[top_idx].astype(np.float32),
            voxel_indices=top_idx.astype(np.int64),
        )

    # --------------------------------------------------------- introspection

    def memory_bytes(self) -> dict[str, int]:
        """Return per-array memory footprint."""
        out = {
            "idx": self._idx.nbytes,
            "count": self._count.nbytes,
            "xyz_sum": self._xyz_sum.nbytes,
            "rgb_sum": self._rgb_sum.nbytes,
            "last_frame": self._last_frame.nbytes,
            "last_time": self._last_time.nbytes,
            "lookup_dict": _approx_dict_bytes(self._lookup),
        }
        if self._feat_sum is not None:
            out["feat_sum"] = self._feat_sum.nbytes
            assert self._feat_weight is not None
            out["feat_weight"] = self._feat_weight.nbytes
        out["total"] = sum(v for k, v in out.items() if k != "total")
        return out


def _approx_dict_bytes(d: dict) -> int:
    """Rough lower-bound estimate; ~100 bytes/entry is a fine ballpark."""
    return 100 * len(d)
