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

"""DynaMem-style value maps for exploration.

Ported from the dyna360 research stack. Two scalar fields over the same
occupancy grid as :mod:`occupancy`:

  - **V_T (time-recency)** — sigmoid of "how long ago was this cell last
    observed?" Cells not seen in a while (or never) score high; freshly
    observed cells score low. This biases exploration away from
    just-covered territory.
  - **V_S (query-similarity)** — sigmoid of the cosine between the cell's
    aggregated feature and a text query. Only defined when a query is
    given AND the voxel map carries features.

Linear combination ``V = (1 − α)·V_T + α·V_S`` gates exploration. With no
query it is a pure recency-driven frontier walk; with a query it biases
toward regions semantically consistent with the target (DynaMem §3.4).
Maps are derived per-call from ``VoxelMap.snapshot`` so they inherit
carving for free.
"""

# ruff: noqa: N806  — H, W, D are conventional array-dimension names
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from lerobot.navigation.occupancy import OccupancyGrid

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValueMapConfig:
    """Knobs shared between recency and similarity value maps."""

    recency_mid_s: float = 10.0
    """Age (s) at which V_T crosses 0.5 — older = more interesting."""

    recency_scale_s: float = 8.0
    """How sharply V_T transitions around the mid age. Smaller = sharper."""

    similarity_mid: float = 0.15
    """Cosine score at which V_S crosses 0.5."""

    similarity_scale: float = 0.05
    """How sharply V_S transitions around the mid cosine."""

    alpha_similarity: float = 0.6
    """Weight of V_S in the combined value when a query is given.
    0.0 = pure recency, 1.0 = pure similarity."""

    unknown_value: float = 1.0
    """V_T for UNOBSERVED cells — they are maximally interesting."""

    distance_discount_per_meter: float = 0.05
    """Multiplicative discount on far frontiers so the base does not
    ping-pong across the map. 0 disables."""


@dataclass(frozen=True)
class ValueMaps:
    """The scalar fields, all shaped ``(H, W)`` like the occupancy grid."""

    last_time: np.ndarray  # float64 — −inf where UNOBSERVED
    recency: np.ndarray  # float32 V_T in [0, 1]
    similarity: np.ndarray | None  # float32 V_S in [0, 1], None when no query
    combined: np.ndarray  # float32 V — what explore() optimizes


# --------------------------------------------------------------------- #


def _eps_for_cell(cell_size: float) -> float:
    """Same float32-drift epsilon as :mod:`occupancy` so the two
    projections agree on which voxels land in which cells."""
    return cell_size * 1e-3


def _project_voxels_to_cells(voxel_map, grid: OccupancyGrid, want_features: bool):
    """Project every voxel into its XZ cell.

    Returns ``(last_time_per_cell, feat_per_cell)`` where last_time is
    (H, W) float64 (−inf for empty cells) and feat_per_cell is
    (H, W, D) float32 or None.
    """
    snap = voxel_map.snapshot(include_features=want_features)
    H, W = grid.shape
    last_time = np.full((H, W), -math.inf, dtype=np.float64)
    if snap.xyz.size == 0:
        return last_time, None

    x = snap.xyz[:, 0].astype(np.float64)
    z = snap.xyz[:, 2].astype(np.float64)
    eps = _eps_for_cell(grid.cell_size)
    ix = np.clip(np.floor((x - grid.origin_x) / grid.cell_size + eps).astype(np.int32), 0, W - 1)
    iz = np.clip(np.floor((z - grid.origin_z) / grid.cell_size + eps).astype(np.int32), 0, H - 1)

    # Per-cell max last_time. `np.maximum.at` is the unbuffered ufunc version,
    # which correctly handles duplicate (iz, ix) targets.
    np.maximum.at(last_time, (iz, ix), snap.last_time.astype(np.float64))

    feat_per_cell: np.ndarray | None = None
    if want_features and snap.feat is not None and snap.feat.size > 0:
        D = snap.feat.shape[1]
        feat_sum = np.zeros((H, W, D), dtype=np.float32)
        np.add.at(feat_sum, (iz, ix), snap.feat.astype(np.float32))
        counts = np.zeros((H, W), dtype=np.int32)
        np.add.at(counts, (iz, ix), 1)
        # Normalize per-cell — count is the number of CONTRIBUTING voxels.
        denom = np.maximum(counts, 1).astype(np.float32)[..., None]
        feat_per_cell = feat_sum / denom

    return last_time, feat_per_cell


def _recency_value(last_time_per_cell: np.ndarray, now_t: float, cfg: ValueMapConfig) -> np.ndarray:
    """V_T per cell. Unobserved cells get ``cfg.unknown_value``."""
    out = np.full(last_time_per_cell.shape, cfg.unknown_value, dtype=np.float32)
    observed = last_time_per_cell > -math.inf
    if not observed.any():
        return out
    age = (now_t - last_time_per_cell[observed]).astype(np.float32)
    out[observed] = 1.0 / (1.0 + np.exp(-(age - cfg.recency_mid_s) / cfg.recency_scale_s))
    return out


def _similarity_value(
    feat_per_cell: np.ndarray | None,
    text_emb: np.ndarray | None,
    cfg: ValueMapConfig,
) -> np.ndarray | None:
    """V_S per cell. ``None`` when there are no features or no query."""
    if feat_per_cell is None or text_emb is None:
        return None
    text = text_emb.astype(np.float32)
    text = text / max(float(np.linalg.norm(text)), 1e-6)
    # Per-cell mean feat may not be unit-norm — renormalize so the dot product
    # behaves like a cosine. Empty cells stay a 0 vector, so renorm clamps to 0.
    norms = np.linalg.norm(feat_per_cell, axis=-1, keepdims=True)
    feat_normed = feat_per_cell / np.maximum(norms, 1e-6)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        cosine = np.nan_to_num((feat_normed @ text).astype(np.float32))
    sim = 1.0 / (1.0 + np.exp(-(cosine - cfg.similarity_mid) / cfg.similarity_scale))
    sim = np.where(norms.squeeze(-1) > 1e-6, sim, 0.0).astype(np.float32)
    return sim


def compute_value_maps(
    voxel_map,
    grid: OccupancyGrid,
    *,
    text_emb: np.ndarray | None = None,
    now_t: float | None = None,
    cfg: ValueMapConfig | None = None,
) -> ValueMaps:
    """Build the full value-map bundle for one ``explore`` call."""
    cfg = cfg or ValueMapConfig()
    last_time, feat_per_cell = _project_voxels_to_cells(voxel_map, grid, want_features=(text_emb is not None))
    if now_t is None:
        observed_mask = last_time > -math.inf
        now_t = float(last_time[observed_mask].max()) if observed_mask.any() else 0.0

    v_t = _recency_value(last_time, now_t, cfg)
    v_s = _similarity_value(feat_per_cell, text_emb, cfg)

    if v_s is not None:
        combined = ((1.0 - cfg.alpha_similarity) * v_t + cfg.alpha_similarity * v_s).astype(np.float32)
    else:
        combined = v_t

    return ValueMaps(last_time=last_time, recency=v_t, similarity=v_s, combined=combined)


def pick_best_frontier_cell(
    grid: OccupancyGrid,
    frontier_cells: np.ndarray,
    values: ValueMaps,
    robot_position_xz: tuple[float, float],
    cfg: ValueMapConfig | None = None,
) -> tuple[int, tuple[float, float], float, float]:
    """Score every frontier cell by ``values.combined`` (with a distance
    discount) and return the winner.

    Returns ``(index_into_frontier_cells, (x, z), distance_m, score)``.
    """
    if frontier_cells.shape[0] == 0:
        raise ValueError("frontier_cells is empty")
    cfg = cfg or ValueMapConfig()

    iz_f = frontier_cells[:, 0]
    ix_f = frontier_cells[:, 1]
    raw = values.combined[iz_f, ix_f]

    xs = grid.origin_x + (ix_f.astype(np.float64) + 0.5) * grid.cell_size
    zs = grid.origin_z + (iz_f.astype(np.float64) + 0.5) * grid.cell_size
    rx, rz = robot_position_xz
    d = np.hypot(xs - rx, zs - rz)
    discount = 1.0 / (1.0 + cfg.distance_discount_per_meter * d)
    scored = raw * discount

    best = int(np.argmax(scored))
    return best, (float(xs[best]), float(zs[best])), float(d[best]), float(scored[best])
