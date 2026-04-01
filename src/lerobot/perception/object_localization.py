# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Helpers for 3D object localization: camera→robot transform, focus selection, temporal smoothing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.agent.agentic_flow import SceneObject, SceneObservation


def build_scene_summary_renumbered(observation: SceneObservation) -> str:
    """Like ``build_scene_summary`` but uses 0..N-1 indices aligned with filtered object lists."""
    lines = [f"Task: {observation.task}", ""]
    if not observation.objects:
        lines.append("No objects detected in the scene.")
        return "\n".join(lines)
    lines.append("Detected objects (use index 0 to N-1):")
    for j, obj in enumerate(observation.objects):
        cx, cy, cz = obj.center_xyz
        sx, sy, sz = obj.size_xyz
        lines.append(
            f"  [{j}] {obj.label}: center=({cx:.3f}, {cy:.3f}, {cz:.3f}) m, "
            f"size=({sx:.3f}, {sy:.3f}, {sz:.3f}) m, distance={obj.distance_m:.3f} m"
        )
    return "\n".join(lines)


LOCALIZE_DISAMBIGUATION_SYSTEM_PROMPT = """You match the user's OBJECT_FOCUS text to exactly one detected object in the scene list.

Reply with exactly one JSON line:
{"action": "pick", "object_index": N}
where N is the 0-based index from the scene list, or
{"action": "fail", "reason": "brief reason"}
if nothing matches the focus query.

Rules: Use only indices shown in the scene. No markdown, no extra text."""


def transform_xyz_cam_to_robot(cam_to_robot_4x4: np.ndarray, xyz_cam: np.ndarray) -> np.ndarray:
    """Apply rigid transform to a 3D point in camera frame (homogeneous linear part + translation)."""
    R = cam_to_robot_4x4[:3, :3]
    t = cam_to_robot_4x4[:3, 3]
    p = np.asarray(xyz_cam, dtype=np.float64).reshape(3)
    return (R @ p + t).astype(np.float64)


def select_focus_detection_index(
    detections_with_state: list[tuple[Any, dict[str, np.ndarray]]],
    strategy: str,
    image_hw: tuple[int, int],
) -> int | None:
    """Choose one detection index when multiple boxes match the query.

    Args:
        detections_with_state: Pairs ``(detection, state)`` with ``state`` from
            ``compute_object_state`` and ``detection.bbox_xyxy``.
        strategy: ``closest`` | ``largest_bbox`` | ``central`` | ``first``.
        image_hw: ``(H, W)`` for the central-pixel heuristic.

    Returns:
        Index into ``detections_with_state``, or ``None`` if empty.
    """
    if not detections_with_state:
        return None
    if len(detections_with_state) == 1:
        return 0

    strategy = (strategy or "closest").lower()
    h, w = image_hw
    cx_img = w * 0.5
    cy_img = h * 0.5

    if strategy == "first":
        return 0

    if strategy == "closest":
        return int(
            np.argmin(
                [float(s[1]["obj_distance"][0]) for s in detections_with_state],
            )
        )

    if strategy == "largest_bbox":

        def area(i: int) -> float:
            det = detections_with_state[i][0]
            x1, y1, x2, y2 = det.bbox_xyxy
            return float(max(0, x2 - x1) * max(0, y2 - y1))

        return int(max(range(len(detections_with_state)), key=area))

    if strategy == "central":

        def dist_sq(i: int) -> float:
            det = detections_with_state[i][0]
            x1, y1, x2, y2 = det.bbox_xyxy
            bx = 0.5 * (x1 + x2)
            by = 0.5 * (y1 + y2)
            return float((bx - cx_img) ** 2 + (by - cy_img) ** 2)

        return int(min(range(len(detections_with_state)), key=dist_sq))

    return 0


@dataclass
class LocalizedObjectSnapshot:
    """One object: 3D in camera frame and optionally robot base frame."""

    label: str
    detection_index: int
    center_cam_m: tuple[float, float, float]
    size_cam_m: tuple[float, float, float]
    distance_m: float
    bbox_xyxy: tuple[int, int, int, int]
    center_base_m: tuple[float, float, float] | None = None
    size_base_m: tuple[float, float, float] | None = None


def snapshots_from_scene(
    observation: SceneObservation,
    detections_with_state: list[tuple[Any, dict[str, np.ndarray]]],
    cam_to_robot_4x4: np.ndarray | None = None,
) -> list[LocalizedObjectSnapshot]:
    """Pair ``SceneObject`` entries with bbox from detections; optional base-frame pose."""
    out: list[LocalizedObjectSnapshot] = []
    for obj, (det, state) in zip(observation.objects, detections_with_state, strict=True):
        c = np.array(obj.center_xyz, dtype=np.float64)
        s = np.array(obj.size_xyz, dtype=np.float64)
        cb = sb = None
        if cam_to_robot_4x4 is not None:
            half = 0.5 * s
            cmin = c - half
            cmax = c + half
            corners_cam = np.array(
                [
                    [cmin[0], cmin[1], cmin[2]],
                    [cmax[0], cmin[1], cmin[2]],
                    [cmin[0], cmax[1], cmin[2]],
                    [cmax[0], cmax[1], cmin[2]],
                    [cmin[0], cmin[1], cmax[2]],
                    [cmax[0], cmin[1], cmax[2]],
                    [cmin[0], cmax[1], cmax[2]],
                    [cmax[0], cmax[1], cmax[2]],
                ],
                dtype=np.float64,
            )
            corners_base = np.stack([transform_xyz_cam_to_robot(cam_to_robot_4x4, p) for p in corners_cam])
            c_b = corners_base.min(axis=0)
            c_bb = corners_base.max(axis=0)
            cb = tuple(float(x) for x in (0.5 * (c_b + c_bb)).tolist())
            sb = tuple(float(x) for x in (c_bb - c_b).tolist())
        bbox = getattr(det, "bbox_xyxy", (0, 0, 0, 0))
        out.append(
            LocalizedObjectSnapshot(
                label=obj.label,
                detection_index=obj.index,
                center_cam_m=tuple(float(x) for x in c.tolist()),
                size_cam_m=tuple(float(x) for x in s.tolist()),
                distance_m=float(obj.distance_m),
                bbox_xyxy=tuple(int(x) for x in bbox),
                center_base_m=cb,
                size_base_m=sb,
            )
        )
    return out


@dataclass
class _SmoothEntry:
    center_cam: np.ndarray
    size_cam: np.ndarray
    center_base: np.ndarray | None = None
    size_base: np.ndarray | None = None


@dataclass
class TemporalObjectMap:
    """Per-label exponential moving average of 3D pose (camera and optional base frame)."""

    alpha: float = 0.35
    _entries: dict[str, _SmoothEntry] = field(default_factory=dict)

    def _key(self, label: str, slot: int) -> str:
        return f"{label.strip().lower()}#{slot}"

    def smooth_snapshots(self, snapshots: list[LocalizedObjectSnapshot]) -> list[LocalizedObjectSnapshot]:
        """Return snapshots with EMA-smoothed centers/sizes; slot is list index."""
        result: list[LocalizedObjectSnapshot] = []
        a = float(np.clip(self.alpha, 0.0, 1.0))
        for slot, snap in enumerate(snapshots):
            key = self._key(snap.label, slot)
            c = np.array(snap.center_cam_m, dtype=np.float64)
            s = np.array(snap.size_cam_m, dtype=np.float64)
            cb = (
                np.array(snap.center_base_m, dtype=np.float64) if snap.center_base_m is not None else None
            )
            sb = np.array(snap.size_base_m, dtype=np.float64) if snap.size_base_m is not None else None

            prev = self._entries.get(key)
            if prev is None:
                new_e = _SmoothEntry(
                    center_cam=c.copy(),
                    size_cam=s.copy(),
                    center_base=cb.copy() if cb is not None else None,
                    size_base=sb.copy() if sb is not None else None,
                )
            else:
                new_c = a * c + (1.0 - a) * prev.center_cam
                new_s = a * s + (1.0 - a) * prev.size_cam
                if cb is not None and sb is not None and prev.center_base is not None and prev.size_base is not None:
                    new_cb = a * cb + (1.0 - a) * prev.center_base
                    new_sb = a * sb + (1.0 - a) * prev.size_base
                elif cb is not None:
                    new_cb = cb.copy()
                    new_sb = sb.copy() if sb is not None else None
                else:
                    new_cb = prev.center_base.copy() if prev.center_base is not None else None
                    new_sb = prev.size_base.copy() if prev.size_base is not None else None
                new_e = _SmoothEntry(center_cam=new_c, size_cam=new_s, center_base=new_cb, size_base=new_sb)
            self._entries[key] = new_e

            cb_out = sb_out = None
            if new_e.center_base is not None:
                cb_out = tuple(float(x) for x in new_e.center_base.tolist())
            if new_e.size_base is not None:
                sb_out = tuple(float(x) for x in new_e.size_base.tolist())

            result.append(
                LocalizedObjectSnapshot(
                    label=snap.label,
                    detection_index=snap.detection_index,
                    center_cam_m=tuple(float(x) for x in new_e.center_cam.tolist()),
                    size_cam_m=tuple(float(x) for x in new_e.size_cam.tolist()),
                    distance_m=snap.distance_m,
                    bbox_xyxy=snap.bbox_xyxy,
                    center_base_m=cb_out,
                    size_base_m=sb_out,
                )
            )
        return result
