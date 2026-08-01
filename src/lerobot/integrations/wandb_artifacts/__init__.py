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
"""W&B Artifact references plus lazily loaded inspection and transfer helpers.

Reference parsing has no optional dependencies. Dataset inspection is loaded only when requested,
and SDK-backed store helpers are loaded only when requested, so each public surface retains its own
dependency boundary.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .refs import ArtifactRef, parse_artifact_ref

_LAZY_EXPORTS = {
    "DatasetDirectoryError": ".inspect",
    "DatasetDirectoryMetadata": ".inspect",
    "ModelDirectoryError": ".inspect",
    "ModelDirectoryMetadata": ".inspect",
    "inspect_dataset_directory": ".inspect",
    "inspect_model_directory": ".inspect",
    "registry_link_refusal": ".inspect",
    "validate_dataset_directory": ".inspect",
    "validate_model_directory": ".inspect",
    "ArtifactTypeMismatchError": ".store",
    "DownloadDestinationNotEmptyError": ".store",
    "MaterializedArtifact": ".store",
    "declare_input": ".store",
    "download_artifact": ".store",
    "link_to_registry": ".store",
    "upload_directory": ".store",
}

if TYPE_CHECKING:
    from .inspect import (
        DatasetDirectoryError,
        DatasetDirectoryMetadata,
        ModelDirectoryError,
        ModelDirectoryMetadata,
        inspect_dataset_directory,
        inspect_model_directory,
        registry_link_refusal,
        validate_dataset_directory,
        validate_model_directory,
    )
    from .store import (
        ArtifactTypeMismatchError,
        DownloadDestinationNotEmptyError,
        MaterializedArtifact,
        declare_input,
        download_artifact,
        link_to_registry,
        upload_directory,
    )


def __getattr__(name: str) -> Any:
    """Load optional-dependency-backed symbols only when callers request them."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(f"{__name__}{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "ArtifactRef",
    "ArtifactTypeMismatchError",
    "DatasetDirectoryError",
    "DatasetDirectoryMetadata",
    "DownloadDestinationNotEmptyError",
    "MaterializedArtifact",
    "ModelDirectoryError",
    "ModelDirectoryMetadata",
    "declare_input",
    "download_artifact",
    "inspect_dataset_directory",
    "inspect_model_directory",
    "link_to_registry",
    "parse_artifact_ref",
    "registry_link_refusal",
    "upload_directory",
    "validate_dataset_directory",
    "validate_model_directory",
]
