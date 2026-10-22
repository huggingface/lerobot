# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Pinned research recipe and HF Jobs planning information."""

from __future__ import annotations

import math
from dataclasses import dataclass

RUNTIME_REVISION = "d61017dd3a95c59a51d836aab6f7bac1baee34eb"
RECIPE = "calibrated-droid-hawor-bounded-v1"
COMPUTE = {
    "a100-large": "A100 80 GB",
    "rtx-pro-6000": "RTX PRO 6000 96 GB",
    "h200": "H200 141 GB",
}


@dataclass
class HandBackendConfig:
    """Operator configuration inside a dependency-complete HF Jobs image.

    The image must contain no MANO files. Users supply those per Job.
    module names the separately installed runtime package, never user input.
    """

    root: str = ""
    module: str = ""
    python: str = ""
    macrodata_python: str = ""
    droid_root: str = ""
    droid_python: str = ""
    droid_moge_checkpoint: str = ""
    droid_model_cache: str = ""
    assets_manifest: str = ""
    assets_sha256: str = ""
    timeout_s: float = 7200


def estimate(frames: int, planning_fps: float | None = None) -> dict:
    if planning_fps is not None and (not math.isfinite(planning_fps) or planning_fps <= 0):
        raise ValueError("Planning FPS must be finite and positive")
    return {
        "frames": frames,
        "planning_fps": planning_fps,
        "seconds": frames / planning_fps if planning_fps else None,
        "basis": "user planning assumption" if planning_fps else "HF hardware has not been benchmarked",
        "reference_fps": 53.9,
        "reference_compute": "RTX 5090",
        "reference_scope": "Historical ten-clip calibrated HOT3D check, 1500 source frames; "
        "includes model startup, camera/hands and export; excludes installation and transfers.",
        "caveat": "Different HF GPUs have no measured FPS yet. Queue, installation, downloads and dataset copying are additional.",
    }


def public_catalog() -> dict:
    return {
        "recipe": RECIPE,
        "runtime_revision": RUNTIME_REVISION,
        "profiles": [{"key": key, "name": value, "reference_fps": None} for key, value in COMPUTE.items()],
        "reference": estimate(1500),
        "licenses": {
            "mano": "https://mano.is.tue.mpg.de/license.html",
            "models": "https://huggingface.co/macrodata/egovision-safetensors",
        },
    }
