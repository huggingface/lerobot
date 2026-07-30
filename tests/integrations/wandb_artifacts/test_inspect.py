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
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.io_utils import write_info
from lerobot.datasets.utils import DEFAULT_TASKS_PATH, EPISODES_DIR, STATS_PATH, DatasetInfo
from lerobot.integrations.wandb_artifacts import inspect as inspect_module
from lerobot.integrations.wandb_artifacts.inspect import (
    DatasetDirectoryError,
    inspect_dataset_directory,
    validate_dataset_directory,
)


def _write_minimal_dataset(
    root: Path, *, features=None, total_episodes=0, total_frames=0, total_tasks=0
) -> None:
    info = DatasetInfo(
        codebase_version="v3.0",
        fps=30,
        features=features
        or {
            "action": {"dtype": "float32", "shape": (6,), "names": None},
        },
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=total_tasks,
        robot_type="so101",
    )
    write_info(info, root)
    (root / STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / STATS_PATH).write_text("{}")
    (root / "data").mkdir(parents=True, exist_ok=True)


def test_validate_accepts_minimal_dataset(tmp_path):
    _write_minimal_dataset(tmp_path)
    validate_dataset_directory(tmp_path)


def test_validate_rejects_missing_info_json(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / STATS_PATH).write_text("{}")
    (tmp_path / "data").mkdir()
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)


def test_validate_rejects_missing_stats_json(tmp_path):
    _write_minimal_dataset(tmp_path)
    (tmp_path / STATS_PATH).unlink()
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)


def test_validate_rejects_missing_data_dir(tmp_path):
    _write_minimal_dataset(tmp_path)
    (tmp_path / "data").rmdir()
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)


def test_validate_rejects_nonexistent_directory(tmp_path):
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path / "does-not-exist")


def test_validate_requires_tasks_file_when_total_tasks_nonzero(tmp_path):
    _write_minimal_dataset(tmp_path, total_tasks=2)
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)

    (tmp_path / DEFAULT_TASKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / DEFAULT_TASKS_PATH).write_bytes(b"")
    validate_dataset_directory(tmp_path)


def test_validate_requires_loader_visible_episode_shards(tmp_path):
    _write_minimal_dataset(tmp_path, total_episodes=3)
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)

    too_deep = tmp_path / EPISODES_DIR / "chunk-000" / "nested"
    too_deep.mkdir(parents=True, exist_ok=True)
    (too_deep / "file-000.parquet").write_bytes(b"")
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)

    (too_deep.parent / "file-000.parquet").write_bytes(b"")
    validate_dataset_directory(tmp_path)


def test_validate_requires_loader_visible_data_shards(tmp_path):
    _write_minimal_dataset(tmp_path, total_frames=100)
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)

    too_deep = tmp_path / "data" / "chunk-000" / "nested"
    too_deep.mkdir(parents=True, exist_ok=True)
    (too_deep / "file-000.parquet").write_bytes(b"")
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)

    (too_deep.parent / "file-000.parquet").write_bytes(b"")
    validate_dataset_directory(tmp_path)


def test_validate_cross_checks_episode_and_data_counts_independently(tmp_path):
    _write_minimal_dataset(tmp_path, total_episodes=3, total_frames=100)

    data_chunk = tmp_path / "data" / "chunk-000"
    data_chunk.mkdir(parents=True, exist_ok=True)
    (data_chunk / "file-000.parquet").write_bytes(b"")
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)

    episodes_chunk = tmp_path / EPISODES_DIR / "chunk-000"
    episodes_chunk.mkdir(parents=True, exist_ok=True)
    (episodes_chunk / "file-000.parquet").write_bytes(b"")
    validate_dataset_directory(tmp_path)


def test_validate_top_level_check_passes_but_metadata_loading_would_still_fail_without_episodes(tmp_path):
    _write_minimal_dataset(tmp_path, total_episodes=1, total_frames=0)
    assert (tmp_path / STATS_PATH).is_file()
    assert (tmp_path / "data").is_dir()
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)


def test_inspect_extracts_metadata(tmp_path):
    _write_minimal_dataset(
        tmp_path,
        features={
            "action": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.image.front": {"dtype": "video", "shape": (3, 224, 224), "names": None},
            "observation.image.wrist": {"dtype": "image", "shape": (3, 224, 224), "names": None},
        },
    )
    metadata = inspect_dataset_directory(tmp_path)

    assert metadata.schema_version == "v3.0"
    assert metadata.robot_type == "so101"
    assert metadata.fps == 30
    assert metadata.total_episodes == 0
    assert metadata.total_frames == 0
    assert metadata.total_tasks == 0
    assert set(metadata.video_keys) == {"observation.image.front"}
    assert set(metadata.camera_keys) == {"observation.image.front", "observation.image.wrist"}
    assert metadata.source_path == tmp_path.resolve()


def test_inspect_git_commit_matches_lerobot_checkout_head(tmp_path):
    _write_minimal_dataset(tmp_path)
    metadata = inspect_dataset_directory(tmp_path)

    repo_root = Path(__file__).resolve().parents[3]
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root
    )
    if expected.returncode == 0 and (repo_root / "src" / "lerobot").is_dir():
        assert metadata.git_commit == expected.stdout.strip()
    else:
        assert metadata.git_commit is None


def test_git_commit_ignores_an_enclosing_unrelated_repository(monkeypatch):
    unrelated_root = Path(inspect_module.__file__).resolve().parents[2]
    calls = []

    def _fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=f"{unrelated_root}\n")

    monkeypatch.setattr(inspect_module.subprocess, "run", _fake_run)

    assert inspect_module._current_git_commit() is None
    assert len(calls) == 1


def test_inspect_raises_on_invalid_directory(tmp_path):
    with pytest.raises(DatasetDirectoryError):
        inspect_dataset_directory(tmp_path / "missing")


def test_to_wandb_metadata_is_json_safe(tmp_path):
    import json

    _write_minimal_dataset(tmp_path)
    metadata = inspect_dataset_directory(tmp_path)
    payload = metadata.to_wandb_metadata()
    json.dumps(payload)
    assert payload["source_path"] == str(tmp_path.resolve())
    assert isinstance(payload["camera_keys"], list)
