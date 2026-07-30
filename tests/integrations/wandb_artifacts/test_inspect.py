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
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from datasets import Dataset

from lerobot.datasets.io_utils import write_episodes, write_info, write_tasks
from lerobot.datasets.utils import DATA_DIR, DEFAULT_EPISODES_PATH, STATS_PATH, DatasetInfo
from lerobot.integrations.wandb_artifacts import inspect as inspect_module
from lerobot.integrations.wandb_artifacts.inspect import (
    DatasetDirectoryError,
    inspect_dataset_directory,
    validate_dataset_directory,
)


def _write_info_and_stats(
    root: Path,
    *,
    features=None,
    total_episodes: int = 0,
    total_frames: int = 0,
    total_tasks: int = 0,
) -> None:
    write_info(
        DatasetInfo(
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
        ),
        root,
    )
    (root / STATS_PATH).write_text("{}")
    (root / DATA_DIR).mkdir(parents=True, exist_ok=True)


def _write_data_shard(
    root: Path,
    episode_indices: list[int],
    *,
    chunk_index: int = 0,
    file_index: int = 0,
) -> Path:
    path = root / DATA_DIR / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict(
        {
            "index": list(range(len(episode_indices))),
            "episode_index": episode_indices,
        }
    ).to_parquet(path)
    return path


def _write_episode_metadata(root: Path, rows: list[dict]) -> Path:
    write_episodes(Dataset.from_list(rows), root)
    return root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)


def _episode_row(
    episode_index: int,
    *,
    data_file_index: int = 0,
    video_key: str | None = None,
    video_file_index: int | None = None,
) -> dict:
    row = {
        "episode_index": episode_index,
        "data/chunk_index": 0,
        "data/file_index": data_file_index,
    }
    if video_key is not None:
        row[f"videos/{video_key}/chunk_index"] = 0
        row[f"videos/{video_key}/file_index"] = (
            episode_index if video_file_index is None else video_file_index
        )
    return row


def test_validate_accepts_minimal_empty_dataset(tmp_path):
    _write_info_and_stats(tmp_path)
    validate_dataset_directory(tmp_path)


@pytest.mark.parametrize("missing", ["info", "stats", "data"])
def test_validate_rejects_missing_required_structure(tmp_path, missing):
    _write_info_and_stats(tmp_path)
    if missing == "info":
        (tmp_path / "meta" / "info.json").unlink()
    elif missing == "stats":
        (tmp_path / STATS_PATH).unlink()
    else:
        (tmp_path / DATA_DIR).rmdir()

    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)


def test_validate_parses_stats_metadata(tmp_path):
    _write_info_and_stats(tmp_path)
    (tmp_path / STATS_PATH).write_text("")

    with pytest.raises(DatasetDirectoryError, match="could not be read as dataset stats"):
        validate_dataset_directory(tmp_path)


def test_validate_parses_and_counts_task_metadata(tmp_path):
    _write_info_and_stats(tmp_path, total_tasks=2)

    with pytest.raises(DatasetDirectoryError, match="task metadata"):
        validate_dataset_directory(tmp_path)

    write_tasks(pd.DataFrame({"task_index": [0]}), tmp_path)
    with pytest.raises(DatasetDirectoryError, match=r"contains 1 row"):
        validate_dataset_directory(tmp_path)

    write_tasks(pd.DataFrame({"task_index": [0, 1]}), tmp_path)
    validate_dataset_directory(tmp_path)


def test_validate_reads_episode_metadata_for_non_video_dataset(tmp_path):
    _write_info_and_stats(tmp_path, total_episodes=1, total_frames=1)
    _write_data_shard(tmp_path, [0])

    episode_path = tmp_path / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path.write_bytes(b"")

    with pytest.raises(DatasetDirectoryError, match="could not be read as episode metadata"):
        validate_dataset_directory(tmp_path)

    episode_path.unlink()
    _write_episode_metadata(tmp_path, [_episode_row(0)])
    validate_dataset_directory(tmp_path)


def test_validate_requires_each_episode_index_exactly_once(tmp_path):
    _write_info_and_stats(tmp_path, total_episodes=2, total_frames=2)
    _write_data_shard(tmp_path, [0, 1])
    _write_episode_metadata(tmp_path, [_episode_row(0), _episode_row(0)])

    with pytest.raises(DatasetDirectoryError, match="each episode_index"):
        validate_dataset_directory(tmp_path)


def test_validate_requires_every_episode_referenced_data_shard(tmp_path):
    _write_info_and_stats(tmp_path, total_episodes=2, total_frames=2)
    _write_episode_metadata(
        tmp_path,
        [_episode_row(0, data_file_index=0), _episode_row(1, data_file_index=1)],
    )
    _write_data_shard(tmp_path, [0], file_index=0)

    with pytest.raises(DatasetDirectoryError, match=r"missing 1 data file"):
        validate_dataset_directory(tmp_path)

    _write_data_shard(tmp_path, [1], file_index=1)
    validate_dataset_directory(tmp_path)


def test_validate_uses_loader_visible_data_layout(tmp_path):
    _write_info_and_stats(tmp_path, total_episodes=1, total_frames=1)
    _write_episode_metadata(tmp_path, [_episode_row(0)])

    too_deep = tmp_path / DATA_DIR / "chunk-000" / "nested" / "file-000.parquet"
    too_deep.parent.mkdir(parents=True)
    Dataset.from_dict({"index": [0], "episode_index": [0]}).to_parquet(too_deep)

    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(tmp_path)


def test_validate_cross_checks_total_frame_rows(tmp_path):
    _write_info_and_stats(tmp_path, total_episodes=1, total_frames=2)
    _write_episode_metadata(tmp_path, [_episode_row(0)])
    _write_data_shard(tmp_path, [0])

    with pytest.raises(DatasetDirectoryError, match=r"contains 1 row"):
        validate_dataset_directory(tmp_path)


def test_validate_cross_checks_data_episode_coverage(tmp_path):
    _write_info_and_stats(tmp_path, total_episodes=2, total_frames=2)
    _write_episode_metadata(tmp_path, [_episode_row(0), _episode_row(1)])
    _write_data_shard(tmp_path, [0, 0])

    with pytest.raises(DatasetDirectoryError, match="covers episode_index"):
        validate_dataset_directory(tmp_path)


def test_validate_requires_every_referenced_video_file(tmp_path):
    video_key = "observation.image.front"
    _write_info_and_stats(
        tmp_path,
        features={
            "action": {"dtype": "float32", "shape": (6,), "names": None},
            video_key: {"dtype": "video", "shape": (3, 224, 224), "names": None},
        },
        total_episodes=2,
        total_frames=2,
    )
    _write_episode_metadata(
        tmp_path,
        [_episode_row(0, video_key=video_key), _episode_row(1, video_key=video_key)],
    )
    _write_data_shard(tmp_path, [0, 1])

    with pytest.raises(DatasetDirectoryError, match=r"missing 2 video file"):
        validate_dataset_directory(tmp_path)

    video_dir = tmp_path / "videos" / video_key / "chunk-000"
    video_dir.mkdir(parents=True)
    (video_dir / "file-000.mp4").write_bytes(b"video")
    with pytest.raises(DatasetDirectoryError, match=r"missing 1 video file"):
        validate_dataset_directory(tmp_path)

    (video_dir / "file-001.mp4").write_bytes(b"video")
    validate_dataset_directory(tmp_path)


def test_inspect_extracts_metadata(tmp_path):
    _write_info_and_stats(
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
    _write_info_and_stats(tmp_path)
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


def test_inspection_imports_without_wandb():
    preamble = textwrap.dedent(
        """
        import builtins
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "wandb" or name.startswith("wandb."):
                raise ModuleNotFoundError(name + " deliberately unavailable")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            preamble
            + textwrap.dedent(
                """
                from lerobot.integrations.wandb_artifacts import (
                    inspect_dataset_directory,
                    validate_dataset_directory,
                )
                assert callable(inspect_dataset_directory)
                assert callable(validate_dataset_directory)
                """
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_inspect_raises_on_invalid_directory(tmp_path):
    with pytest.raises(DatasetDirectoryError):
        inspect_dataset_directory(tmp_path / "missing")


def test_to_wandb_metadata_is_json_safe(tmp_path):
    import json

    _write_info_and_stats(tmp_path)
    metadata = inspect_dataset_directory(tmp_path)
    payload = metadata.to_wandb_metadata()
    json.dumps(payload)
    assert payload["source_path"] == str(tmp_path.resolve())
    assert isinstance(payload["camera_keys"], list)
