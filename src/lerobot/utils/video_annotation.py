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

"""Best-effort text overlays shared by evaluation and interactive rollouts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def annotate_frame(frame: np.ndarray, fields: Iterable[tuple[str, str | None]]) -> np.ndarray:
    """Return an RGB frame annotated with the non-empty labeled ``fields``."""
    if frame.ndim != 3 or frame.shape[-1] != 3:
        return frame
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return frame

    text_rows = [f"{label}: {value}" for label, value in fields if value]
    if not text_rows:
        return frame

    image = np.ascontiguousarray(frame).copy()
    font, scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 6
    max_width = image.shape[1] - 2 * margin
    lines: list[str] = []
    for text in text_rows:
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
            if width > max_width and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)

    line_height = 20
    header_height = min(image.shape[0], len(lines) * line_height + 6)
    backdrop = image.copy()
    cv2.rectangle(backdrop, (0, 0), (image.shape[1], header_height), (0, 0, 0), -1)
    cv2.addWeighted(backdrop, 0.55, image, 0.45, 0, dst=image)

    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (margin, 18 + index * line_height),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return image
