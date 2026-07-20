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

"""SpatialSkills tool layer.

Ported from the dyna360 research stack. The agent calls these as a fixed
toolset:

  - :meth:`SpatialSkills.locate` — text → 3D position (or NOT_FOUND)
  - :meth:`SpatialSkills.goto` — base navigation to a 3D target
  - :meth:`SpatialSkills.explore` — pick a frontier to drive toward

The skills compose a :class:`~lerobot.navigation.voxel_map.VoxelMap`
(geometry + semantic features) with a
:class:`~lerobot.navigation.base_controller.BaseController` (motion) and a
text encoder. Stateless-per-call: each call snapshots the world, does its
work, and hands control back. The agent decides what to call next.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from lerobot.navigation.occupancy import (
    OccupancyGrid,
    astar,
    find_frontier_cells,
    project_voxel_map_to_grid,
)
from lerobot.navigation.value_map import (
    ValueMapConfig,
    compute_value_maps,
    pick_best_frontier_cell,
)

if TYPE_CHECKING:
    from lerobot.navigation.base_controller import BaseController
    from lerobot.navigation.features import FeatureExtractor
    from lerobot.navigation.voxel_map import VoxelMap

LOG = logging.getLogger(__name__)


# ----- typed results returned to the agent -------------------------------


@dataclass(frozen=True)
class LocateResult:
    """Output of :meth:`SpatialSkills.locate`.

    ``found=False`` is load-bearing — the signal the agent uses to pick
    :meth:`explore` over :meth:`goto`. Don't fabricate an ``xyz`` when
    abstaining.
    """

    found: bool
    xyz: tuple[float, float, float] | None
    confidence: float  # top cosine score; -1.0 if no features
    n_voxels: int  # how many voxels supported the cluster
    text: str


@dataclass(frozen=True)
class GotoResult:
    """Output of :meth:`SpatialSkills.goto`."""

    reached: bool
    final_xyz: tuple[float, float, float]
    distance_to_target: float
    n_steps: int
    reason: str  # "ok" | "no path" | "max steps" | "blocked"
    path_xyz: list[tuple[float, float, float]]  # for viz / debugging


@dataclass(frozen=True)
class ExploreResult:
    """Output of :meth:`SpatialSkills.explore`."""

    target_xyz: tuple[float, float, float] | None
    found_frontier: bool
    distance_to_target: float  # 0.0 when no frontier
    reason: str  # "ok" | "no frontier" | ...
    value: float = 0.0
    """Combined V_T + α·V_S value of the chosen frontier — useful for
    debugging exploration bias and as a give-up signal for the agent."""


# ----- configuration ------------------------------------------------------


@dataclass(frozen=True)
class SkillsConfig:
    """Knobs shared across the skills."""

    # Occupancy projection
    cell_size: float = 0.1
    ground_y: float | None = None  # None ⇒ auto-estimate from voxels
    obstacle_y_range: tuple[float, float] = (-2.0, -0.1)  # m above ground (y-down)
    obstacle_inflate_cells: int = 1

    # locate()
    locate_top_k: int = 128
    locate_threshold: float = 0.15  # min cosine for found=True
    locate_outlier_quantile: float = 0.5
    locate_outlier_scale: float = 2.0

    # goto()
    goto_threshold: float = 0.3
    goto_step_size: float = 0.2  # m advanced per controller tick
    goto_max_steps: int = 500
    goto_replan_every: int = 5
    goto_dt: float = 0.1

    # explore()
    explore_max_frontiers: int = 256
    value_cfg: ValueMapConfig = field(default_factory=ValueMapConfig)
    """DynaMem-style V_T (recency) + V_S (similarity) knobs."""


# ----- the skills layer ---------------------------------------------------


class SpatialSkills:
    """Composes the voxel memory + base + text encoder into the agent toolset."""

    def __init__(
        self,
        voxel_map: VoxelMap,
        base: BaseController,
        siglip: FeatureExtractor | None = None,
        cfg: SkillsConfig | None = None,
    ) -> None:
        self.voxel_map = voxel_map
        self.base = base
        self.siglip = siglip
        self.cfg = cfg or SkillsConfig()

    # ----- shared helper ---------------------------------------------------

    def occupancy(self) -> OccupancyGrid:
        """Project the *current* voxel map into a 2D occupancy grid."""
        return project_voxel_map_to_grid(
            self.voxel_map,
            cell_size=self.cfg.cell_size,
            ground_y=self.cfg.ground_y,
            obstacle_y_range=self.cfg.obstacle_y_range,
            inflate_cells=self.cfg.obstacle_inflate_cells,
        )

    # ----- locate(text) ----------------------------------------------------

    def locate(self, text: str) -> LocateResult:
        text = text.strip()
        if not text:
            return LocateResult(False, None, -1.0, 0, text)
        if self.siglip is None:
            return LocateResult(False, None, -1.0, 0, text)
        if self.voxel_map.feature_dim is None:
            return LocateResult(False, None, -1.0, 0, text)

        text_emb = self.siglip.encode_text(text)
        qr = self.voxel_map.query(text_emb, top_k=self.cfg.locate_top_k)
        if qr.score.size == 0:
            return LocateResult(False, None, -1.0, 0, text)
        top_score = float(qr.score.max())
        if top_score < self.cfg.locate_threshold:
            LOG.info(
                "locate(%r): top score %.3f < threshold %.3f → NOT_FOUND",
                text,
                top_score,
                self.cfg.locate_threshold,
            )
            return LocateResult(False, None, top_score, 0, text)

        # Score-weighted centroid, then outlier rejection (anchor against the
        # cluster median distance so a couple of stray voxels in the top-k
        # can't drag the centroid into empty space).
        scores = qr.score.astype(np.float64)
        weights = scores - scores.min() + 1e-6
        centroid = (qr.xyz * weights[:, None]).sum(axis=0) / weights.sum()
        d = np.linalg.norm(qr.xyz - centroid, axis=1)
        thresh = max(
            self.cfg.cell_size * 4,
            float(np.quantile(d, self.cfg.locate_outlier_quantile)) * self.cfg.locate_outlier_scale,
        )
        inliers = d <= thresh
        if inliers.sum() >= 3:
            inlier_xyz = qr.xyz[inliers]
            inlier_w = weights[inliers]
            centroid = (inlier_xyz * inlier_w[:, None]).sum(axis=0) / inlier_w.sum()
        return LocateResult(
            True,
            (float(centroid[0]), float(centroid[1]), float(centroid[2])),
            top_score,
            int(inliers.sum()),
            text,
        )

    # ----- goto(xyz) -------------------------------------------------------

    def goto(
        self,
        target_xyz: tuple[float, float, float],
        *,
        max_steps: int | None = None,
        threshold: float | None = None,
    ) -> GotoResult:
        """Closed-loop nav: A* → step a few cells → replan → repeat.

        The replan cadence makes this a staleness governor — a moving
        obstacle (or a previously-mapped one that got carved out) is picked
        up at the next replan.
        """
        max_steps = max_steps if max_steps is not None else self.cfg.goto_max_steps
        threshold = threshold if threshold is not None else self.cfg.goto_threshold

        path_xyz_global: list[tuple[float, float, float]] = []
        n_steps = 0
        last_path: list[tuple[float, float]] = []

        for step in range(max_steps):
            pos = self.base.position()
            d = math.hypot(pos[0] - target_xyz[0], pos[2] - target_xyz[2])
            if d <= threshold:
                return GotoResult(True, pos, d, n_steps, "ok", path_xyz_global)

            if step % self.cfg.goto_replan_every == 0 or not last_path:
                grid = self.occupancy()
                last_path = (
                    astar(
                        grid,
                        start_world=(pos[0], pos[2]),
                        goal_world=(target_xyz[0], target_xyz[2]),
                    )
                    or []
                )
                if not last_path or len(last_path) < 2:
                    return GotoResult(False, pos, d, n_steps, "no path", path_xyz_global)

            # Head toward the next-but-one cell to smooth corners.
            next_idx = min(2, len(last_path) - 1)
            target_xz = last_path[next_idx]
            dx = target_xz[0] - pos[0]
            dz = target_xz[1] - pos[2]
            n = math.hypot(dx, dz)
            if n < 1e-6:
                last_path.pop(0)
                continue
            vx = self.cfg.goto_step_size / max(self.cfg.goto_dt, 1e-6) * dx / n
            vz = self.cfg.goto_step_size / max(self.cfg.goto_dt, 1e-6) * dz / n
            self.base.move(vx=vx, vz=vz, dt=self.cfg.goto_dt)
            pos = self.base.position()
            path_xyz_global.append(pos)
            n_steps += 1

            # Pop waypoint when we've crossed it.
            if math.hypot(target_xz[0] - pos[0], target_xz[1] - pos[2]) < self.cfg.cell_size:
                last_path.pop(0)
                if not last_path:
                    last_path = []  # force replan

        pos = self.base.position()
        d = math.hypot(pos[0] - target_xyz[0], pos[2] - target_xyz[2])
        return GotoResult(False, pos, d, n_steps, "max steps", path_xyz_global)

    # ----- explore() -------------------------------------------------------

    def explore(self, query: str | None = None) -> ExploreResult:
        """Pick a frontier to drive toward via the DynaMem §3.4 value map.

        With no query this is pure recency (visit oldest-observed or
        UNOBSERVED frontiers first); with a query + features it biases
        toward semantic matches.
        """
        grid = self.occupancy()
        cells = find_frontier_cells(grid)
        if cells.shape[0] == 0:
            return ExploreResult(None, False, 0.0, "no frontier")

        # Subsample if huge so the loop stays fast even on big maps.
        if cells.shape[0] > self.cfg.explore_max_frontiers:
            idx = np.random.default_rng(0).choice(
                cells.shape[0], self.cfg.explore_max_frontiers, replace=False
            )
            cells = cells[idx]

        text_emb = None
        if query is not None and self.siglip is not None and self.voxel_map.feature_dim is not None:
            text_emb = self.siglip.encode_text(query)

        values = compute_value_maps(self.voxel_map, grid, text_emb=text_emb, cfg=self.cfg.value_cfg)

        pos = self.base.position()
        _, (xt, zt), dist, score = pick_best_frontier_cell(
            grid, cells, values, robot_position_xz=(pos[0], pos[2]), cfg=self.cfg.value_cfg
        )
        return ExploreResult(
            target_xyz=(xt, grid.ground_y, zt),
            found_frontier=True,
            distance_to_target=dist,
            reason="ok",
            value=score,
        )
