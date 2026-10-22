# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Pinned EgoHand/SOMA recipe and HF Jobs planning information."""

from __future__ import annotations

import math
from dataclasses import dataclass

RUNTIME_REVISION = "d61017dd3a95c59a51d836aab6f7bac1baee34eb"
RECIPE = "egohand-soma-v3-checkpoint-overlap-v4"
GEM_REVISION = "20555befe2fbbe5b2015e1fff7ff214dc18ea9d6"
CHECKPOINT_SHA256 = "26b0249bf77c820fcecfa23bf2a5f9c2bb64c75eb6fa5756caa8789f7dd9f745"
ACTION_SCHEMA = "ee12-soma24x3-bimanual-156-v1"
COMPUTE = {
    "a100-large": "A100 80 GB",
    "rtx-pro-6000": "RTX PRO 6000 96 GB",
    "h200": "H200 141 GB",
}


@dataclass
class HandBackendConfig:
    """Operator paths inside the pinned HF Jobs image; never end-user machines."""

    root: str = ""
    runtime_source_sha256: str = ""
    module: str = ""
    python: str = ""
    egohand_python: str = ""
    gem_root: str = ""
    checkpoint: str = ""
    detector: str = ""
    soma_assets: str = ""
    trt_cache: str = ""
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
        "reference_fps": [39.41450219151414, 41.03790957211554],
        "reference_compute": "RTX 5090",
        "reference_scope": "Two fresh ten-clip development samples, 1500 input frames and 1120 accepted "
        "frames each. Includes startup, decode, calibration, fresh DROID-W+MoGe, hands and native156D "
        "export. Excludes installation and model downloads; not a cold filesystem-cache benchmark.",
        "caveat": "Different HF GPUs have no measured FPS yet. Queue, installation, downloads and dataset copying are additional.",
    }


def public_catalog() -> dict:
    return {
        "recipe": RECIPE,
        "runtime_revision": RUNTIME_REVISION,
        "gem_revision": GEM_REVISION,
        "action_schema": ACTION_SCHEMA,
        "action_dimension": 156,
        "camera": "DROID-W + MoGe Small (required)",
        "hand_model": "nvidia/EgoHand-v1",
        "hand_representation": "SOMA/MHR",
        "motion_infilling": False,
        "profiles": [{"key": key, "name": value, "reference_fps": None} for key, value in COMPUTE.items()],
        "reference": estimate(1500),
        "licenses": {
            "egohand": "https://huggingface.co/nvidia/EgoHand-v1",
            "detector": "https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1",
            "soma": "https://huggingface.co/nvidia/SOMA-X",
            "runtime": f"https://github.com/NVlabs/GEM-X/blob/{GEM_REVISION}/ATTRIBUTIONS.md",
        },
    }
