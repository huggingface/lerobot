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

"""Load a SuperMimic actor export (``actor.ts`` + ``manifest.json``) and check its contract.

The manifest is the deployment contract: input layout, history/goal lengths, joint order,
default pose, gains and action scale. Loading fails unless every field the controller relies
on matches this implementation and the robot dynamics spec.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .observation import OBS_DIM
from .reference import GOAL_FRAME_DIM
from .robot_spec import RobotDynamicsSpec, g1_29dof_spec

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
ACTOR_FILE = "actor.ts"
GOLDEN_FILE = "golden_vectors.npz"
SUPPORTED_SCHEMA_VERSIONS = (1,)

OBSERVATION_FIELDS = (
    ("joint_position", 0, 29),
    ("joint_velocity", 29, 58),
    ("base_angular_velocity", 58, 61),
    ("projected_gravity", 61, 64),
    ("previous_action", 64, 93),
    ("reference_relative_orientation", 93, 99),
)
GOAL_FIELDS = (
    ("reference_root_displacement", 0, 3),
    ("reference_root_orientation", 3, 9),
    ("reference_linear_velocity", 9, 12),
    ("reference_angular_velocity", 12, 15),
    ("reference_joint_position", 15, 44),
    ("reference_joint_velocity", 44, 73),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_export_dir(export: str | Path, revision: str | None = None) -> Path:
    """A local directory, or a Hugging Face Hub model repo id downloaded to the cache."""
    path = Path(export).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(export), revision=revision))


def _check_fields(
    declared: list[dict[str, Any]], expected: tuple[tuple[str, int, int], ...], what: str
) -> None:
    found = tuple((field["name"], *field["slice"]) for field in declared)
    if found != expected:
        raise ValueError(f"Export {what} layout {found} differs from the supported layout {expected}")


def validate_manifest(manifest: dict[str, Any], spec: RobotDynamicsSpec) -> None:
    if manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported SuperMimic manifest schema {manifest.get('schema_version')!r}")
    arch = manifest["architecture"]
    expected_arch = {
        "observation_dim": OBS_DIM,
        "goal_frame_dim": GOAL_FRAME_DIM,
        "action_dim": spec.num_joints,
        "goal_dim": GOAL_FRAME_DIM * int(arch["goal_frames"]),
    }
    for key, value in expected_arch.items():
        if int(arch[key]) != value:
            raise ValueError(f"Export architecture {key}={arch[key]} but this controller needs {value}")
    if not arch.get("normalization_included", False):
        raise ValueError("Export must carry its observation/goal normalizers inside the actor")
    inputs = manifest["inputs"]
    if inputs["obs_hist"]["history_order"] != "oldest first, newest last":
        raise ValueError("Export history order must be oldest first")
    _check_fields(inputs["observation_fields"], OBSERVATION_FIELDS, "observation")
    _check_fields(inputs["goal_fields_per_frame"], GOAL_FIELDS, "goal")
    rotation_6d = tuple(manifest["frames"]["rotation_6d_order"])
    if rotation_6d != ("R00", "R01", "R10", "R11", "R20", "R21"):
        raise ValueError(f"Export 6D rotation order {rotation_6d} is not supported")
    spec.check_manifest(manifest)


@dataclass
class SuperMimicExport:
    """A validated actor export: TorchScript actor plus the constants it was trained with."""

    actor: torch.jit.ScriptModule
    manifest: dict[str, Any]
    spec: RobotDynamicsSpec
    directory: Path

    @property
    def history_length(self) -> int:
        return int(self.manifest["architecture"]["history_length"])

    @property
    def goal_frames(self) -> int:
        return int(self.manifest["architecture"]["goal_frames"])

    @property
    def joint_offset(self) -> np.ndarray:
        return np.asarray(self.manifest["robot"]["nominal_calibration_offset_rad"], dtype=np.float32)

    @torch.inference_mode()
    def check_golden_vectors(self, atol: float = 2e-5, rtol: float = 1e-5) -> float:
        """Re-run the export's golden inputs; return the max abs error or raise if out of tolerance."""
        path = self.directory / GOLDEN_FILE
        if not path.is_file():
            raise FileNotFoundError(f"No {GOLDEN_FILE} in {self.directory}")
        expected_sha = self.manifest["artifacts"]["golden_vectors"]["sha256"]
        if _sha256(path) != expected_sha:
            raise ValueError(f"{GOLDEN_FILE} does not match the manifest checksum")
        golden = np.load(path)
        worst = 0.0
        cases = sorted(k.removeprefix("obs_hist_") for k in golden.files if k.startswith("obs_hist_"))
        if not cases:
            raise ValueError(f"{GOLDEN_FILE} contains no obs_hist_<n> cases")
        for case in cases:
            output = self.actor(
                torch.from_numpy(golden[f"obs_hist_{case}"]), torch.from_numpy(golden[f"goal_{case}"])
            ).numpy()
            reference = golden[f"action_{case}"]
            error = float(np.abs(output - reference).max())
            if not np.allclose(output, reference, atol=atol, rtol=rtol):
                raise ValueError(f"Golden case {case} differs from the export by {error}")
            worst = max(worst, error)
        return worst


def load_export(
    export: str | Path,
    *,
    spec: RobotDynamicsSpec | None = None,
    revision: str | None = None,
    num_threads: int | None = 1,
) -> SuperMimicExport:
    directory = resolve_export_dir(export, revision)
    manifest_path = directory / MANIFEST_FILE
    actor_path = directory / ACTOR_FILE
    if not manifest_path.is_file() or not actor_path.is_file():
        raise FileNotFoundError(f"{directory} must contain {MANIFEST_FILE} and {ACTOR_FILE}")
    manifest = json.loads(manifest_path.read_text())
    spec = g1_29dof_spec() if spec is None else spec
    validate_manifest(manifest, spec)
    expected_sha = manifest["artifacts"]["actor"]["sha256"]
    if _sha256(actor_path) != expected_sha:
        raise ValueError(f"{ACTOR_FILE} does not match the manifest checksum")
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    # TorchScript archives have no weights-only mode; only load exports from sources you trust.
    actor = torch.jit.load(str(actor_path), map_location="cpu")  # nosec B614  # nosec B614
    actor.eval()
    logger.info(
        "Loaded SuperMimic actor from %s (checkpoint iteration %s, %s parameters)",
        directory,
        manifest.get("checkpoint", {}).get("iteration"),
        manifest["architecture"].get("parameter_count"),
    )
    return SuperMimicExport(actor=actor, manifest=manifest, spec=spec, directory=directory)
