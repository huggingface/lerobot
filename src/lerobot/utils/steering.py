# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Shared train/runtime wording and coordinate validation for visual steering commands."""

import math


def render_steering_command(command: dict) -> str:
    if "points_by_frame" in command:
        raise ValueError("Resolve points_by_frame to the sampled frame before rendering")
    text = command["text"]
    if not isinstance(text, str) or not text.strip() or len(text) > 1000:
        raise ValueError("Steering text must contain 1–1000 characters")
    points = command.get("points", [])
    if command["style"] in {"point", "trace"} and not points:
        raise ValueError("Visual steering requires grounded points")
    if not points:
        return text.strip()
    width, height = command.get("image_size", [0, 0])
    camera = (command.get("camera") or "").removeprefix("observation.images.")
    if not camera or width <= 0 or height <= 0:
        raise ValueError("Invalid steering camera dimensions")
    if command["style"] == "trace" and len(points) < 2:
        raise ValueError("A trace requires at least two ordered points")
    for point in points:
        if (
            len(point) != 2
            or any(type(v) is not int for v in point)
            or not all(math.isfinite(v) for v in point)
        ):
            raise ValueError("Points must be integer pixel pairs")
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError("Point lies outside its source camera")
    coordinates = ", ".join(f"[{x}, {y}]" for x, y in points)
    return f"In {camera} view ({width}x{height} pixels), {text.strip()}: {coordinates}."
