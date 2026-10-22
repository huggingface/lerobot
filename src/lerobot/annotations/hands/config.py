# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Research recipe and compute estimates; model dependencies remain external."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypedDict

from ..runtime.configs import RuntimeConfig

GROUNDED_REVISION = "d61017dd3a95c59a51d836aab6f7bac1baee34eb"
RECIPE = "calibrated-droid-hawor-bounded-v1"
BENCHMARK_URL = (
    "https://github.com/pkooij/grounded-data-pipe/blob/"
    f"{GROUNDED_REVISION}/docs/egocentric_bounded_plane.md#fresh-integrated-ten-clip-check"
)


class ComputeProfile(TypedDict):
    name: str
    hf_flavor: str | None
    reference_fps: float | None


COMPUTE: dict[str, ComputeProfile] = {
    "rtx-5090": {"name": "RTX 5090", "hf_flavor": None, "reference_fps": 53.9},
    "a100-80gb": {"name": "A100 80 GB", "hf_flavor": "a100-large", "reference_fps": None},
    "h100": {"name": "H100 (local / Slurm)", "hf_flavor": None, "reference_fps": None},
    "rtx-pro-6000": {"name": "RTX PRO 6000", "hf_flavor": "rtx-pro-6000", "reference_fps": None},
    "h200": {"name": "H200", "hf_flavor": "h200", "reference_fps": None},
}


@dataclass
class HandBackendConfig:
    # These paths exist in each worker's provisioned runtime/image. No installer
    # or public model mirror is invoked by LeRobot.
    root: str = ""
    python: str = ""
    macrodata_python: str = ""
    droid_root: str = ""
    droid_python: str = ""
    droid_moge_checkpoint: str = ""
    droid_model_cache: str = ""
    assets_manifest: str = ""
    assets_sha256: str = ""
    research_terms_accepted: bool = False
    timeout_s: float = 7200


@dataclass
class HandRuntimeConfig(RuntimeConfig):
    batch_size: int = 1
    shard_size: int = 1
    max_retries: int = 0


@dataclass
class HandPipelineConfig:
    input_root: Path = Path("raw-egocentric")
    output_root: Path = Path("hand-annotations")
    repo_id: str = "local/human-pov-hands"
    compute: str = "rtx-5090"
    planning_fps: float | None = None
    # One warm external invocation per group. The shared runtime checkpoints
    # groups, while the backend reports every individual video and segment.
    videos_per_group: int = 32
    minimum_segment_frames: int = 30
    backend: HandBackendConfig = field(default_factory=HandBackendConfig)
    runtime: HandRuntimeConfig = field(default_factory=HandRuntimeConfig)

    def validate(self) -> None:
        if self.compute not in COMPUTE:
            raise ValueError(f"Choose compute from {', '.join(COMPUTE)}")
        if self.videos_per_group < 1 or self.minimum_segment_frames < 2:
            raise ValueError("Groups must be positive and segments must contain at least two frames")
        if self.planning_fps is not None and (not math.isfinite(self.planning_fps) or self.planning_fps <= 0):
            raise ValueError("planning_fps must be finite and positive")
        if not math.isfinite(self.backend.timeout_s) or self.backend.timeout_s <= 0:
            raise ValueError("Backend timeout must be finite and positive")
        if self.runtime.batch_size != 1 or self.runtime.shard_size != 1:
            raise ValueError("Hand video groups require runtime batch_size=shard_size=1")
        if self.runtime.max_retries:
            raise ValueError(
                "Automatic hand inference retries are disabled; inspect failures before resuming"
            )
        if self.runtime.hf_jobs.detach:
            raise ValueError("Detached hand execution cannot materialize the final dataset")


def estimate(config: HandPipelineConfig, groups: list[list[dict]]) -> dict:
    """Planning information, never a throughput benchmark or final compute bill."""
    config.validate()
    reference = COMPUTE[config.compute]["reference_fps"]
    fps = config.planning_fps or reference
    workers = min(config.runtime.workers, len(groups))
    loads = (
        [
            sum(video["frames"] for group in groups[index::workers] for video in group)
            for index in range(workers)
        ]
        if workers
        else []
    )
    return {
        "compute": config.compute,
        "workers": workers,
        "source_frames": sum(loads),
        "estimated_fps_per_gpu": fps,
        "estimated_seconds": max(loads) / fps if loads and fps else None,
        "basis": "user planning assumption"
        if config.planning_fps
        else (
            "historical RTX 5090, ten short calibrated HOT3D clips, 1500 input frames"
            if reference
            else "unmeasured hardware; supply planning_fps to estimate"
        ),
        "scope": "source frames including startup, camera/hands and LeRobot export; excludes transfer and queue",
        "reference_fps": 53.9,
        "reference_compute": "rtx-5090",
        "reference_url": BENCHMARK_URL,
        "caveat": "Small development benchmark; resolution, clip length, visibility, CPU and storage affect speed. "
        "Multi-GPU timing assumes independent groups without shared-I/O contention.",
    }


def public_catalog() -> dict:
    template = json.loads(json.dumps(asdict(HandPipelineConfig()), default=str))
    template["runtime"]["slurm"]["python_executable"] = "python"
    return {
        "recipe": RECIPE,
        "runtime_revision": GROUNDED_REVISION,
        "reference_url": BENCHMARK_URL,
        "profiles": [{"key": key, **value} for key, value in COMPUTE.items()],
        "config_template": template,
        "licenses": {
            "mano": "https://mano.is.tue.mpg.de/license.html",
            "models": "https://huggingface.co/macrodata/egovision-safetensors",
        },
    }
