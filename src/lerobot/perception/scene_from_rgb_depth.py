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

"""Build a structured scene (2D VLM detections + 3D from depth) from RGB + depth."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import numpy as np

from lerobot.agent.agentic_flow import SceneObject, SceneObservation
from lerobot.perception.detection_filters import filter_detections_by_query_color
from lerobot.processor.depth_perception_processor import compute_object_state_from_bbox_roi

logger = logging.getLogger(__name__)


def build_scene_from_rgb_depth(
    rgb_np: np.ndarray,
    depth_np: np.ndarray,
    detector: Any,
        compute_object_state: Callable[..., dict[str, np.ndarray]],
        intrinsics: dict[str, float],
    query: str,
    *,
    color_filter_min_fraction: float | None = None,
) -> tuple[SceneObservation, list[tuple[Any, dict[str, np.ndarray]]]]:
    """Run VLM detection with ``query`` and lift each masked region to 3D using depth.

    Args:
        rgb_np: RGB image (H, W, 3).
        depth_np: Depth aligned to RGB, uint16 mm or per intrinsics ``depth_scale``.
        detector: Object with ``detect(rgb, query) -> list[Detection]``.
        compute_object_state: ``compute_object_state(depth, mask, intrinsics)`` from
            ``depth_perception_processor``.
        intrinsics: ``fx, fy, cx, cy, depth_scale``.
        query: Natural-language prompt for the VLM (e.g. "the red cube", "pen on table").
        color_filter_min_fraction: If set and > 0, drop detections whose bbox does not
            contain enough pixels matching color words in ``query`` (HSV heuristic).

    Returns:
        ``SceneObservation`` and parallel list of ``(detection, state_dict)`` for viz / IK.
    """
    detections = detector.detect(rgb_np, query)
    if color_filter_min_fraction is not None and color_filter_min_fraction > 0:
        n_vlm = len(detections)
        filtered = filter_detections_by_query_color(
            rgb_np, detections, query, color_filter_min_fraction
        )
        if filtered:
            detections = filtered
        elif n_vlm > 0:
            logger.warning(
                "Color filter would remove all %d VLM detection(s); keeping unfiltered boxes.",
                n_vlm,
            )
    n_vlm = len(detections)
    try:
        detections_json = [
            {
                "label": det.label,
                "bbox_xyxy": list(det.bbox_xyxy),
                "has_mask": det.mask is not None,
            }
            for det in detections
        ]
        logger.debug("VLM raw detections JSON: %s", json.dumps(detections_json))
    except Exception:
        pass

    objects: list[SceneObject] = []
    detections_with_state: list[tuple[Any, dict[str, np.ndarray]]] = []
    _min_side, _min_area = 10, 400
    out_idx = 0

    for det in detections:
        if hasattr(det, "bbox_xyxy"):
            x1, y1, x2, y2 = det.bbox_xyxy
            bw, bh = int(x2) - int(x1), int(y2) - int(y1)
            if bw < _min_side or bh < _min_side or bw * bh < _min_area:
                logger.debug(
                    "Skip %r: bbox too small %s",
                    getattr(det, "label", "?"),
                    det.bbox_xyxy,
                )
                continue
        mask = det.mask
        if mask is None:
            continue
        state = compute_object_state(depth_np, mask, intrinsics)
        center = state["obj_center_xyz"]
        thr = float(intrinsics.get("bbox_fallback_if_mask_far_m", 0.0) or 0.0)
        if thr > 0 and hasattr(det, "bbox_xyxy"):
            n_center = float(np.linalg.norm(center))
            use_fb = np.allclose(center, 0) or n_center > thr
            if use_fb:
                fb = compute_object_state_from_bbox_roi(depth_np, det.bbox_xyxy, intrinsics)
                if not np.allclose(fb["obj_center_xyz"], 0):
                    n_fb = float(np.linalg.norm(fb["obj_center_xyz"]))
                    cap = float(intrinsics.get("max_depth_m", 5.0)) + 0.5
                    if np.allclose(center, 0):
                        take_fb = n_fb <= cap
                    else:
                        take_fb = n_fb < n_center and n_fb <= cap
                    if take_fb:
                        logger.debug(
                            "BBox depth fallback %r: mask |c|=%.2fm → bbox_roi %.2fm",
                            det.label,
                            n_center if not np.allclose(center, 0) else 0.0,
                            n_fb,
                        )
                        state = fb
                        center = state["obj_center_xyz"]
        size = state["obj_size_xyz"]
        dist = state["obj_distance"]
        if np.allclose(center, 0):
            bb = getattr(det, "bbox_xyxy", ())
            logger.warning(
                "[perceive] drop %r: no depth in mask/bbox (bbox=%s) — check alignment / stereo holes",
                det.label,
                bb,
            )
            continue
        objects.append(
            SceneObject(
                index=out_idx,
                label=det.label,
                center_xyz=(float(center[0]), float(center[1]), float(center[2])),
                size_xyz=(float(size[0]), float(size[1]), float(size[2])),
                distance_m=float(dist[0]),
            )
        )
        detections_with_state.append((det, state))
        out_idx += 1

    scene = SceneObservation(objects=objects, task=query)
    if objects:
        parts = []
        for det, st in detections_with_state:
            d_m = float(st["obj_distance"][0])
            bb = getattr(det, "bbox_xyxy", ())
            parts.append(f"{getattr(det, 'label', '?')!r} d={d_m:.2f}m bbox={bb}")
        logger.info("[perceive] VLM %d box(es) → %d with 3D │ %s", n_vlm, len(objects), " │ ".join(parts))
    elif n_vlm > 0:
        logger.warning(
            "[perceive] VLM returned %d box(es) but none got valid depth (see DEBUG for skips)",
            n_vlm,
        )
    else:
        logger.info("[perceive] VLM: no boxes")

    return scene, detections_with_state
