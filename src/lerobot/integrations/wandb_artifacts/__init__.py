# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""W&B Artifacts integration: reference parsing, upload/download primitives, dataset inspection.

New W&B-specific surface lives here (see ``docs/adr/0001-wandb-integration-package-boundary.md``).
Per that ADR, the pre-existing, training-loop-embedded ``WandBLogger``
(``lerobot/common/wandb_utils.py``) stays in place rather than being migrated into this package —
a later ticket extends it in-line to import from this package's ``store`` module for dataset-artifact
resolution during training, instead of duplicating the upload/download primitives defined here.
"""

from importlib.metadata import version

from packaging.version import Version

from lerobot.utils.import_utils import require_package

# Every module in this package assumes wandb is installed; guard once here so importing any of
# them fails with an actionable message instead of a bare ModuleNotFoundError deep in a submodule.
require_package("wandb", extra="training")

# W&B 0.24.0 was withdrawn because runs could silently fail to upload data. Keep this safety gate
# local to the new durable-transfer integration rather than changing LeRobot's existing training
# dependency and lockfile for unrelated W&B logging users.
if Version(version("wandb")) < Version("0.24.1"):
    raise RuntimeError(
        "lerobot-wandb requires wandb>=0.24.1 because wandb 0.24.0 can silently fail to upload "
        "run data. Upgrade the training extra before using artifact transfers."
    )

from .inspect import (  # noqa: E402
    DatasetDirectoryError,
    DatasetDirectoryMetadata,
    inspect_dataset_directory,
    validate_dataset_directory,
)
from .refs import ArtifactRef, parse_artifact_ref  # noqa: E402
from .store import (  # noqa: E402
    ArtifactTypeMismatchError,
    DownloadDestinationNotEmptyError,
    MaterializedArtifact,
    download_artifact,
    upload_directory,
)

__all__ = [
    "ArtifactRef",
    "ArtifactTypeMismatchError",
    "DatasetDirectoryError",
    "DatasetDirectoryMetadata",
    "DownloadDestinationNotEmptyError",
    "MaterializedArtifact",
    "download_artifact",
    "inspect_dataset_directory",
    "parse_artifact_ref",
    "upload_directory",
    "validate_dataset_directory",
]
