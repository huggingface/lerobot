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

"""Heuristic filters on VLM detections (e.g. color consistency with the text query)."""

from __future__ import annotations

import logging
import re

import cv2
import numpy as np

from lerobot.perception.vlm_detector import Detection

logger = logging.getLogger(__name__)

# OpenCV H: 0–179. Multiple intervals for hues that wrap (red) or broad names.
_COLOR_HSV_INTERVALS: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 70, 50), (12, 255, 255)), ((165, 70, 50), (180, 255, 255))],
    "orange": [((8, 100, 80), (22, 255, 255))],
    "yellow": [((20, 80, 80), (38, 255, 255))],
    "green": [((38, 50, 50), (88, 255, 255))],
    "cyan": [((80, 50, 50), (100, 255, 255))],
    "blue": [((100, 50, 50), (128, 255, 255))],
    "purple": [((128, 50, 50), (152, 255, 255))],
    "pink": [((145, 40, 80), (175, 255, 255))],
    "magenta": [((140, 50, 50), (165, 255, 255))],
    "brown": [((8, 80, 40), (25, 255, 180))],
    "black": [((0, 0, 0), (179, 255, 90))],
    "white": [((0, 0, 180), (179, 60, 255))],
    "gray": [((0, 0, 80), (179, 80, 200)), ((0, 0, 80), (179, 40, 200))],
    "grey": [((0, 0, 80), (179, 80, 200)), ((0, 0, 80), (179, 40, 200))],
}


def color_names_in_query(query: str) -> list[str]:
    """Return color names from ``_COLOR_HSV_INTERVALS`` that appear as whole words in ``query``."""
    if not query or not query.strip():
        return []
    q = query.lower()
    found: list[str] = []
    for name in _COLOR_HSV_INTERVALS:
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", q):
            if name not in found:
                found.append(name)
    return found


def bbox_color_match_fraction(rgb: np.ndarray, bbox_xyxy: tuple[int, int, int, int], color_names: list[str]) -> float:
    """Fraction of ROI pixels falling in any HSV interval for ``color_names`` (0–1)."""
    if not color_names:
        return 1.0
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    combined = np.zeros(roi.shape[:2], dtype=bool)
    for name in color_names:
        for lo, hi in _COLOR_HSV_INTERVALS.get(name, []):
            m = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            combined |= m > 0
    n_pix = roi.shape[0] * roi.shape[1]
    return float(np.count_nonzero(combined)) / float(n_pix) if n_pix else 0.0


def filter_detections_by_query_color(
    rgb: np.ndarray,
    detections: list[Detection],
    query: str,
    min_fraction: float,
) -> list[Detection]:
    """Keep detections whose bbox contains enough pixels matching query color words.

    If the query names no known color, returns ``detections`` unchanged.
    If every detection fails the threshold, returns an empty list (caller may fall back).
    """
    if min_fraction <= 0 or not detections:
        return detections
    names = color_names_in_query(query)
    if not names:
        return detections

    kept: list[Detection] = []
    scores: list[tuple[Detection, float]] = []
    for det in detections:
        frac = bbox_color_match_fraction(rgb, det.bbox_xyxy, names)
        scores.append((det, frac))
        if frac >= min_fraction:
            kept.append(det)

    if kept:
        logger.info(
            "Color filter (%s ≥ %.3f): kept %d/%d detections.",
            names,
            min_fraction,
            len(kept),
            len(detections),
        )
        return kept

    logger.warning(
        "Color filter (%s): no detection reached min_fraction=%.3f (scores=%s).",
        names,
        min_fraction,
        [round(s, 3) for _, s in scores],
    )
    return []
