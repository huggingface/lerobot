# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Pinned EgoHand/SOMA recipe and native runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass

RUNTIME_REVISION = "d61017dd3a95c59a51d836aab6f7bac1baee34eb"
RECIPE = "egohand-soma-v3-checkpoint-overlap-v4"
GEM_REVISION = "20555befe2fbbe5b2015e1fff7ff214dc18ea9d6"
CHECKPOINT_SHA256 = "26b0249bf77c820fcecfa23bf2a5f9c2bb64c75eb6fa5756caa8789f7dd9f745"
ACTION_SCHEMA = "ee12-soma24x3-bimanual-156-v1"


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
