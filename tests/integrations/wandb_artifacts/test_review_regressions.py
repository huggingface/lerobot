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

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from datasets import Dataset

from lerobot.datasets.io_utils import write_episodes, write_info
from lerobot.datasets.utils import STATS_PATH, DatasetInfo
from lerobot.integrations.wandb_artifacts.inspect import DatasetDirectoryError, validate_dataset_directory


def test_validate_requires_every_referenced_video_file(tmp_path):
    video_key = "observation.image.front"
    write_info(
        DatasetInfo(
            codebase_version="v3.0",
            fps=30,
            features={video_key: {"dtype": "video", "shape": (3, 224, 224), "names": None}},
            total_episodes=2,
        ),
        tmp_path,
    )
    (tmp_path / STATS_PATH).write_text("{}")
    (tmp_path / "data").mkdir()
    write_episodes(
        Dataset.from_dict(
            {
                "episode_index": [0, 1],
                f"videos/{video_key}/chunk_index": [0, 0],
                f"videos/{video_key}/file_index": [0, 1],
            }
        ),
        tmp_path,
    )

    with pytest.raises(DatasetDirectoryError, match=r"missing 2 video file"):
        validate_dataset_directory(tmp_path)

    video_dir = tmp_path / "videos" / video_key / "chunk-000"
    video_dir.mkdir(parents=True)
    (video_dir / "file-000.mp4").write_bytes(b"video")
    with pytest.raises(DatasetDirectoryError, match=r"missing 1 video file"):
        validate_dataset_directory(tmp_path)

    (video_dir / "file-001.mp4").write_bytes(b"video")
    validate_dataset_directory(tmp_path)


def _run_import(body: str, *blocked_packages: str) -> subprocess.CompletedProcess[str]:
    preamble = textwrap.dedent(
        f"""
        import builtins
        blocked = {blocked_packages!r}
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == package or name.startswith(package + ".") for package in blocked):
                raise ModuleNotFoundError(name + " deliberately unavailable")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        """
    )
    return subprocess.run(
        [sys.executable, "-c", preamble + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_reference_parser_imports_without_dataset_or_wandb():
    result = _run_import(
        """
        from lerobot.integrations.wandb_artifacts.refs import ArtifactRef, parse_artifact_ref
        assert parse_artifact_ref("entity/project/name:v0") == ArtifactRef(
            entity="entity", project="project", name="name", version_or_alias="v0"
        )
        """,
        "datasets",
        "wandb",
    )
    assert result.returncode == 0, result.stderr


def test_dataset_inspection_imports_without_wandb():
    result = _run_import(
        """
        from lerobot.integrations.wandb_artifacts import (
            inspect_dataset_directory,
            validate_dataset_directory,
        )
        assert callable(inspect_dataset_directory)
        assert callable(validate_dataset_directory)
        """,
        "wandb",
    )
    assert result.returncode == 0, result.stderr
