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
"""Sidecar metadata proving which W&B Artifact a materialized dataset directory holds.

`download_artifact` (see ``store.py``) makes the on-disk *structure* self-describing: it's a
loadable LeRobot dataset. It says nothing about on-disk *identity* though — a structurally valid
directory doesn't reveal which artifact ref produced it, so a directory left over from a previous
run (different `--dataset.artifact_ref`, or a moved `--output_dir`) looks just as valid as the
right one. `write_sidecar`/`read_sidecar` close that gap with a small JSON file next to the
dataset, so a resumed run can verify identity without a network call.

Pure stdlib (no `wandb`/`datasets` import): this must stay importable at the base install tier,
where `lerobot.policies.pretrained` also reads it (best-effort) for model-card generation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

SIDECAR_FILENAME = ".wandb_artifact.json"


@dataclass(frozen=True, slots=True)
class ArtifactSidecar:
    """Identity of the W&B Artifact materialized into a dataset directory."""

    requested_ref: str
    resolved_ref: str
    version: str
    digest: str


def write_sidecar(directory: Path | str, sidecar: ArtifactSidecar) -> None:
    """Write `sidecar` as `directory/{SIDECAR_FILENAME}`, overwriting any existing sidecar."""
    (Path(directory) / SIDECAR_FILENAME).write_text(json.dumps(asdict(sidecar), indent=2) + "\n")


def read_sidecar(directory: Path | str) -> ArtifactSidecar | None:
    """Return the sidecar in `directory`, or `None` if it's absent or unparsable.

    Never raises: callers decide what an absent/broken sidecar means. `_materialize_dataset_artifact`
    treats it as an identity-verification failure (fail fast); model-card generation treats it as
    "no resolved ref available", falling back to the requested ref.
    """
    path = Path(directory) / SIDECAR_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        return ArtifactSidecar(
            requested_ref=data["requested_ref"],
            resolved_ref=data["resolved_ref"],
            version=data["version"],
            digest=data["digest"],
        )
    except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as e:
        logging.warning(f"Ignoring unparsable {path}: {e}")
        return None
