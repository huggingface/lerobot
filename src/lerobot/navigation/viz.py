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

"""Live Rerun visualization of the spatial-memory map.

Shows the voxel map as it is built and updated: the point cloud (colored
by RGB or by observation recency), the robot pose, the top-down occupancy
grid, the planned path, query hits, and — the dynamic part — voxels that
were carved out this keyframe (moved/removed objects), flashed in red.

Because the full current voxel snapshot is re-logged under one entity path
each keyframe, carved voxels simply disappear from the cloud on the next
frame, so DynaMem-style dynamic updates are visible in real time. Rerun
(`rerun-sdk`) is imported lazily — ``pip install 'lerobot[viz]'`` — so the
rest of the stack never depends on it.

Requires ``rerun-sdk``; install with ``pip install 'lerobot[viz]'``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

_TIMELINE = "t"


def _recency_colors(last_time: np.ndarray, now: float, horizon_s: float = 30.0) -> np.ndarray:
    """Map per-voxel age to an (M, 3) uint8 color: recent = cyan, old = red."""
    age = np.clip((now - last_time.astype(np.float64)) / max(horizon_s, 1e-6), 0.0, 1.0)
    r = (60 + 195 * age).astype(np.uint8)
    g = (200 * (1.0 - age)).astype(np.uint8)
    b = (200 * (1.0 - age) + 40).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


class MapVisualizer:
    """Rerun visualizer for the navigation map. Lazily starts the viewer."""

    def __init__(
        self,
        app_id: str = "dog-nav",
        spawn: bool = True,
        color_mode: str = "rgb",
        voxel_radius: float = 0.03,
    ) -> None:
        self.app_id = app_id
        self.spawn = spawn
        self.color_mode = color_mode  # "rgb" | "recency"
        self.voxel_radius = float(voxel_radius)
        self._rr: Any | None = None

    def _ensure_started(self):
        if self._rr is not None:
            return self._rr
        import rerun as rr

        rr.init(self.app_id, spawn=self.spawn)
        # OpenCV world convention: X right, Y down, Z forward (RDF).
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        self._rr = rr
        return rr

    def set_time(self, t_sec: float) -> None:
        rr = self._ensure_started()
        rr.set_time(_TIMELINE, timestamp=float(t_sec))

    # ----- map + dynamics --------------------------------------------------

    def log_map(self, snapshot, now: float | None = None) -> None:
        """Log the current voxel cloud. Re-logging replaces the previous
        frame, so carved voxels vanish — that's the dynamic update."""
        rr = self._ensure_started()
        xyz = snapshot.xyz
        if xyz.size == 0:
            rr.log("world/map", rr.Clear(recursive=False))
            return
        if self.color_mode == "recency" and now is not None:
            colors = _recency_colors(snapshot.last_time, now)
        else:
            colors = snapshot.rgb
        rr.log(
            "world/map",
            rr.Points3D(xyz.astype(np.float32), colors=colors, radii=self.voxel_radius),
        )

    def log_removed(self, xyz: np.ndarray, radius: float | None = None) -> None:
        """Flash this keyframe's carved (removed) voxels in red — the
        moved/vanished objects DynaMem carves out."""
        rr = self._ensure_started()
        r = radius if radius is not None else self.voxel_radius * 1.6
        if xyz is None or len(xyz) == 0:
            rr.log("world/carved", rr.Clear(recursive=False))
            return
        red = np.tile(np.array([[230, 40, 40]], dtype=np.uint8), (len(xyz), 1))
        rr.log("world/carved", rr.Points3D(xyz.astype(np.float32), colors=red, radii=r))

    # ----- robot + planning ------------------------------------------------

    def log_robot(self, pose: np.ndarray, body_radius: float = 0.15) -> None:
        rr = self._ensure_started()
        rr.log(
            "world/robot",
            rr.Transform3D(
                translation=pose[:3, 3].astype(np.float32), mat3x3=pose[:3, :3].astype(np.float32)
            ),
        )
        rr.log(
            "world/robot/body",
            rr.Points3D(
                np.zeros((1, 3), dtype=np.float32),
                colors=np.array([[60, 140, 255]], dtype=np.uint8),
                radii=body_radius,
            ),
        )

    def log_occupancy(self, grid) -> None:
        rr = self._ensure_started()
        from lerobot.navigation.occupancy import occupancy_to_rgb

        rr.log("plan/occupancy", rr.Image(occupancy_to_rgb(grid)))

    def log_path(self, path_xyz: list[tuple[float, float, float]], radius: float = 0.02) -> None:
        rr = self._ensure_started()
        if not path_xyz:
            rr.log("world/path", rr.Clear(recursive=False))
            return
        pts = np.asarray(path_xyz, dtype=np.float32)
        rr.log("world/path", rr.LineStrips3D([pts], radii=radius, colors=[[255, 210, 60]]))

    def log_target(self, xyz: tuple[float, float, float] | None) -> None:
        """Highlight the located target (green) or clear it when not found."""
        rr = self._ensure_started()
        if xyz is None:
            rr.log("world/target", rr.Clear(recursive=False))
            return
        rr.log(
            "world/target",
            rr.Points3D(
                np.asarray([xyz], dtype=np.float32),
                colors=np.array([[40, 230, 90]], dtype=np.uint8),
                radii=0.12,
            ),
        )
