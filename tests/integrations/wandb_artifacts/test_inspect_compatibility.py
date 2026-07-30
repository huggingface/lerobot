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

from pathlib import Path

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.io_utils import write_info
from lerobot.datasets.utils import DATA_DIR, STATS_PATH, DatasetInfo
from lerobot.integrations.wandb_artifacts.inspect import DatasetDirectoryError, validate_dataset_directory
from lerobot.utils.constants import DEFAULT_FEATURES


def _write_empty_dataset(root: Path, codebase_version: str) -> Path:
    root.mkdir()
    write_info(
        DatasetInfo(
            codebase_version=codebase_version,
            fps=30,
            features=dict(DEFAULT_FEATURES),
        ),
        root,
    )
    (root / STATS_PATH).write_text("{}")
    (root / DATA_DIR).mkdir()
    return root


@pytest.mark.parametrize("codebase_version", ["v2.1", "not-a-version"])
def test_validate_enforces_reader_version_compatibility(tmp_path, codebase_version):
    root = _write_empty_dataset(tmp_path / "dataset", codebase_version)

    with pytest.raises(DatasetDirectoryError, match="compatible dataset info"):
        validate_dataset_directory(root)
