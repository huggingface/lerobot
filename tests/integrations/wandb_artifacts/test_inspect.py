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

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from datasets import Dataset
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

from lerobot.datasets.io_utils import load_episodes, load_info, write_info, write_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_PATH,
    STATS_PATH,
    DatasetInfo,
)
from lerobot.integrations.wandb_artifacts import inspect as inspect_module
from lerobot.integrations.wandb_artifacts.inspect import (
    PEFT_ADAPTER_CONFIG_NAME,
    PEFT_ADAPTER_WEIGHTS_NAME,
    DatasetDirectoryError,
    ModelDirectoryError,
    inspect_dataset_directory,
    inspect_model_directory,
    validate_dataset_directory,
    validate_model_directory,
)
from lerobot.utils.constants import DEFAULT_FEATURES

_ACTION_FEATURE = {"dtype": "float32", "shape": (6,), "names": None}


def _write_empty_dataset(root: Path, *, features=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_info(
        DatasetInfo(
            codebase_version="v3.0",
            fps=30,
            features=features or {"action": _ACTION_FEATURE, **DEFAULT_FEATURES},
            robot_type="so101",
        ),
        root,
    )
    (root / STATS_PATH).write_text("{}")
    (root / DATA_DIR).mkdir()
    return root


def _write_dataset(root: Path, episode_lengths: tuple[int, ...] = (1,)) -> Path:
    dataset = LeRobotDataset.create(
        repo_id="tests/materialized",
        fps=30,
        features={"action": _ACTION_FEATURE},
        root=root,
        robot_type="so101",
        use_videos=False,
        video_backend="pyav",
        metadata_buffer_size=1,
    )
    for episode_index, length in enumerate(episode_lengths):
        for frame_index in range(length):
            dataset.add_frame(
                {
                    "action": np.full(6, episode_index + frame_index, dtype=np.float32),
                    "task": f"task-{episode_index}",
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return root


def _data_path(root: Path, chunk_index: int = 0, file_index: int = 0) -> Path:
    return root / DEFAULT_DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)


def _episode_path(root: Path) -> Path:
    return root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)


def _read_frame_payload(root: Path) -> tuple[dict[str, list], object]:
    frames = Dataset.from_parquet(str(_data_path(root)))
    return {column: frames[column] for column in frames.column_names}, frames.features


def _write_frame_payload(path: Path, payload: dict[str, list], features=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    Dataset.from_dict(payload, features=features).to_parquet(temp)
    temp.replace(path)


def _read_episode_rows(root: Path) -> list[dict]:
    episodes = load_episodes(root)
    return [episodes[index] for index in range(len(episodes))]


def _write_episode_rows(root: Path, rows: list[dict]) -> None:
    path = _episode_path(root)
    temp = path.with_name(f".{path.name}.tmp")
    Dataset.from_list(rows).to_parquet(temp)
    temp.replace(path)


def test_validate_accepts_writer_materialized_dataset(tmp_path):
    root = _write_dataset(tmp_path / "dataset", (2, 1))
    validate_dataset_directory(root)


def test_validate_accepts_canonical_empty_dataset(tmp_path):
    validate_dataset_directory(_write_empty_dataset(tmp_path / "dataset"))


@pytest.mark.parametrize("missing", ["info", "stats", "data"])
def test_validate_rejects_missing_required_structure(tmp_path, missing):
    root = _write_empty_dataset(tmp_path / "dataset")
    if missing == "info":
        (root / "meta" / "info.json").unlink()
    elif missing == "stats":
        (root / STATS_PATH).unlink()
    else:
        (root / DATA_DIR).rmdir()
    with pytest.raises(DatasetDirectoryError):
        validate_dataset_directory(root)


def test_validate_requires_canonical_default_frame_features(tmp_path):
    root = _write_empty_dataset(tmp_path / "dataset", features={"action": _ACTION_FEATURE})
    with pytest.raises(DatasetDirectoryError, match="required frame features"):
        validate_dataset_directory(root)


def test_validate_parses_stats_metadata(tmp_path):
    root = _write_empty_dataset(tmp_path / "dataset")
    (root / STATS_PATH).write_text("")
    with pytest.raises(DatasetDirectoryError, match="dataset stats"):
        validate_dataset_directory(root)


def test_validate_requires_task_metadata_for_nonempty_dataset(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    info = load_info(root)
    info.total_tasks = 0
    write_info(info, root)
    (root / "meta" / "tasks.parquet").unlink()
    with pytest.raises(DatasetDirectoryError, match="total_tasks=0"):
        validate_dataset_directory(root)


def test_validate_rejects_unmapped_frame_task_index(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    payload, features = _read_frame_payload(root)
    payload["task_index"] = [7]
    _write_frame_payload(_data_path(root), payload, features)
    with pytest.raises(DatasetDirectoryError, match="no matching task row"):
        validate_dataset_directory(root)


def test_validate_parses_and_counts_task_metadata(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    write_tasks(pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["a", "b"], name="task")), root)
    with pytest.raises(DatasetDirectoryError, match="exactly 1 unique tasks"):
        validate_dataset_directory(root)


def test_validate_reads_episode_metadata(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    _episode_path(root).write_bytes(b"")
    with pytest.raises(DatasetDirectoryError, match="episode metadata"):
        validate_dataset_directory(root)


def test_validate_requires_ordered_episode_rows(tmp_path):
    root = _write_dataset(tmp_path / "dataset", (1, 1))
    rows = _read_episode_rows(root)
    _write_episode_rows(root, list(reversed(rows)))
    with pytest.raises(DatasetDirectoryError, match="ordered episode rows"):
        validate_dataset_directory(root)


def test_validate_cross_checks_episode_ranges_and_lengths(tmp_path):
    root = _write_dataset(tmp_path / "dataset", (2, 1))
    rows = _read_episode_rows(root)
    rows[1]["dataset_from_index"] = 1
    _write_episode_rows(root, rows)
    with pytest.raises(DatasetDirectoryError, match="frame range/length"):
        validate_dataset_directory(root)


def test_validate_requires_every_episode_referenced_data_shard(tmp_path):
    root = _write_dataset(tmp_path / "dataset", (1, 1))
    payload, features = _read_frame_payload(root)
    first = {key: [values[0]] for key, values in payload.items()}
    second = {key: [values[1]] for key, values in payload.items()}
    _write_frame_payload(_data_path(root), first, features)
    second_path = _data_path(root, file_index=1)
    _write_frame_payload(second_path, second, features)

    rows = _read_episode_rows(root)
    rows[1]["data/file_index"] = 1
    _write_episode_rows(root, rows)
    second_path.unlink()
    with pytest.raises(DatasetDirectoryError, match="missing 1 data file"):
        validate_dataset_directory(root)
    _write_frame_payload(second_path, second, features)
    validate_dataset_directory(root)


@pytest.mark.parametrize(
    "data_path",
    [
        "../outside/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "/tmp/file-{file_index:03d}.parquet",
    ],
)
def test_validate_keeps_payload_paths_inside_dataset_root(tmp_path, data_path):
    root = _write_dataset(tmp_path / "dataset")
    info = load_info(root)
    info.data_path = data_path
    write_info(info, root)
    with pytest.raises(DatasetDirectoryError, match="outside the dataset root"):
        validate_dataset_directory(root)


def test_validate_rejects_symlinked_payload_outside_root(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    outside = tmp_path / "outside.parquet"
    shutil.copy2(_data_path(root), outside)
    _data_path(root).unlink()
    _data_path(root).symlink_to(outside)
    with pytest.raises(DatasetDirectoryError, match="outside the dataset root"):
        validate_dataset_directory(root)


def test_validate_loads_frames_with_declared_schema(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    payload, _ = _read_frame_payload(root)
    payload.pop("action")
    _write_frame_payload(_data_path(root), payload)
    with pytest.raises(DatasetDirectoryError, match="frame schema"):
        validate_dataset_directory(root)


def test_validate_cross_checks_total_frame_rows(tmp_path):
    root = _write_dataset(tmp_path / "dataset", (2,))
    payload, features = _read_frame_payload(root)
    payload = {key: values[:1] for key, values in payload.items()}
    _write_frame_payload(_data_path(root), payload, features)
    with pytest.raises(DatasetDirectoryError, match="exactly 2 rows"):
        validate_dataset_directory(root)


def test_validate_cross_checks_frame_episode_membership(tmp_path):
    root = _write_dataset(tmp_path / "dataset", (1, 1))
    payload, features = _read_frame_payload(root)
    payload["episode_index"] = [0, 0]
    _write_frame_payload(_data_path(root), payload, features)
    with pytest.raises(DatasetDirectoryError, match="do not match episode 1"):
        validate_dataset_directory(root)


def test_validate_cross_checks_core_frame_indices_and_timestamps(tmp_path):
    root = _write_dataset(tmp_path / "dataset", (2,))
    payload, features = _read_frame_payload(root)
    payload["frame_index"] = [0, 0]
    _write_frame_payload(_data_path(root), payload, features)
    with pytest.raises(DatasetDirectoryError, match="frame_index"):
        validate_dataset_directory(root)


def test_validate_requires_every_referenced_video_file(tmp_path):
    root = _write_dataset(tmp_path / "dataset", (1, 1))
    video_key = "observation.image.front"
    info = load_info(root)
    info.features[video_key] = {"dtype": "video", "shape": (3, 224, 224), "names": None}
    info.video_path = DEFAULT_VIDEO_PATH
    write_info(info, root)

    rows = _read_episode_rows(root)
    for episode_index, row in enumerate(rows):
        row[f"videos/{video_key}/chunk_index"] = 0
        row[f"videos/{video_key}/file_index"] = episode_index
        row[f"videos/{video_key}/from_timestamp"] = 0.0
        row[f"videos/{video_key}/to_timestamp"] = 1 / 30
    _write_episode_rows(root, rows)

    with pytest.raises(DatasetDirectoryError, match="missing 2 video file"):
        validate_dataset_directory(root)
    video_dir = root / "videos" / video_key / "chunk-000"
    video_dir.mkdir(parents=True)
    (video_dir / "file-000.mp4").write_bytes(b"video")
    with pytest.raises(DatasetDirectoryError, match="missing 1 video file"):
        validate_dataset_directory(root)
    (video_dir / "file-001.mp4").write_bytes(b"video")
    validate_dataset_directory(root)


def test_inspect_extracts_metadata(tmp_path):
    video_key = "observation.image.front"
    image_key = "observation.image.wrist"
    root = _write_empty_dataset(
        tmp_path / "dataset",
        features={
            "action": _ACTION_FEATURE,
            video_key: {"dtype": "video", "shape": (3, 224, 224), "names": None},
            image_key: {"dtype": "image", "shape": (3, 224, 224), "names": None},
            **DEFAULT_FEATURES,
        },
    )
    metadata = inspect_dataset_directory(root)
    assert metadata.schema_version == "v3.0"
    assert metadata.robot_type == "so101"
    assert metadata.fps == 30
    assert metadata.total_episodes == metadata.total_frames == metadata.total_tasks == 0
    assert set(metadata.video_keys) == {video_key}
    assert set(metadata.camera_keys) == {video_key, image_key}
    assert metadata.source_path == root.resolve()


def test_inspect_git_commit_matches_lerobot_checkout_head(tmp_path):
    root = _write_empty_dataset(tmp_path / "dataset")
    metadata = inspect_dataset_directory(root)
    repo_root = Path(__file__).resolve().parents[3]
    expected = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root)
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

    root = _write_empty_dataset(tmp_path / "dataset")
    payload = inspect_dataset_directory(root).to_wandb_metadata()
    json.dumps(payload)
    assert payload["source_path"] == str(root.resolve())
    assert isinstance(payload["camera_keys"], list)


# ---------------------------------------------------------------------------
# model directory validation/inspection
# ---------------------------------------------------------------------------


def _write_model_config(root: Path, *, policy_type: str | None = "act") -> None:
    import json

    root.mkdir(parents=True, exist_ok=True)
    payload = {"type": policy_type} if policy_type is not None else {}
    (root / CONFIG_NAME).write_text(json.dumps(payload))


def test_validate_model_directory_accepts_full_weights(tmp_path):
    root = tmp_path / "model"
    _write_model_config(root)
    (root / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")
    validate_model_directory(root)


def test_validate_model_directory_accepts_peft_adapter_weights(tmp_path):
    root = tmp_path / "model"
    _write_model_config(root)
    (root / PEFT_ADAPTER_CONFIG_NAME).write_text("{}")
    (root / PEFT_ADAPTER_WEIGHTS_NAME).write_bytes(b"adapter")
    validate_model_directory(root)


def test_validate_model_directory_tolerates_extra_files(tmp_path):
    root = tmp_path / "model"
    _write_model_config(root)
    (root / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")
    (root / "train_config.json").write_text("{}")
    (root / "policy_preprocessor").mkdir()
    (root / "policy_preprocessor" / "preprocessor.json").write_text("{}")
    (root / "README.md").write_text("hello")
    validate_model_directory(root)


def test_validate_model_directory_rejects_missing_config(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")
    with pytest.raises(ModelDirectoryError, match=CONFIG_NAME):
        validate_model_directory(root)


def test_validate_model_directory_rejects_config_without_weights(tmp_path):
    root = tmp_path / "model"
    _write_model_config(root)
    with pytest.raises(ModelDirectoryError, match="no model weights"):
        validate_model_directory(root)


def test_validate_model_directory_rejects_adapter_config_without_adapter_weights(tmp_path):
    root = tmp_path / "model"
    _write_model_config(root)
    (root / PEFT_ADAPTER_CONFIG_NAME).write_text("{}")
    with pytest.raises(ModelDirectoryError, match="no model weights"):
        validate_model_directory(root)


def test_validate_model_directory_rejects_non_directory(tmp_path):
    root = tmp_path / "not-a-dir"
    root.write_text("nope")
    with pytest.raises(ModelDirectoryError, match="not a directory"):
        validate_model_directory(root)


def test_validate_model_directory_never_opens_weights_file(tmp_path, monkeypatch):
    root = tmp_path / "model"
    _write_model_config(root)
    weights_path = root / SAFETENSORS_SINGLE_FILE
    weights_path.write_bytes(b"weights")

    real_open = Path.open

    def _guarded_open(self, *args, **kwargs):
        if self == weights_path:
            raise AssertionError("validate_model_directory must not open weight files")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _guarded_open)
    validate_model_directory(root)


def test_inspect_model_directory_extracts_metadata(tmp_path):
    root = tmp_path / "model"
    _write_model_config(root, policy_type="diffusion")
    (root / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")

    metadata = inspect_model_directory(root)
    assert metadata.has_full_weights is True
    assert metadata.has_adapter_weights is False
    assert metadata.policy_type == "diffusion"
    assert metadata.source_path == root.resolve()


def test_inspect_model_directory_reports_adapter_weights(tmp_path):
    root = tmp_path / "model"
    _write_model_config(root, policy_type="act")
    (root / PEFT_ADAPTER_CONFIG_NAME).write_text("{}")
    (root / PEFT_ADAPTER_WEIGHTS_NAME).write_bytes(b"adapter")

    metadata = inspect_model_directory(root)
    assert metadata.has_full_weights is False
    assert metadata.has_adapter_weights is True
    assert metadata.policy_type == "act"


def test_inspect_model_directory_tolerates_corrupt_config(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / CONFIG_NAME).write_text("{not valid json")
    (root / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")

    metadata = inspect_model_directory(root)
    assert metadata.policy_type is None


def test_inspect_model_directory_tolerates_missing_type_key(tmp_path):
    root = tmp_path / "model"
    _write_model_config(root, policy_type=None)
    (root / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")

    metadata = inspect_model_directory(root)
    assert metadata.policy_type is None


def test_inspect_model_directory_raises_on_invalid_directory(tmp_path):
    with pytest.raises(ModelDirectoryError):
        inspect_model_directory(tmp_path / "missing")


def test_model_to_wandb_metadata_is_json_safe(tmp_path):
    import json

    root = tmp_path / "model"
    _write_model_config(root)
    (root / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")
    payload = inspect_model_directory(root).to_wandb_metadata()
    json.dumps(payload)
    assert payload["source_path"] == str(root.resolve())
    assert payload["policy_type"] == "act"


def _write_adapter_config(root: Path, *, base_model_name_or_path=None) -> None:
    import json

    payload = {} if base_model_name_or_path is None else {"base_model_name_or_path": base_model_name_or_path}
    (root / PEFT_ADAPTER_CONFIG_NAME).write_text(json.dumps(payload))
    (root / PEFT_ADAPTER_WEIGHTS_NAME).write_bytes(b"adapter")


def test_inspect_model_directory_warns_when_adapter_base_model_is_not_local(tmp_path, caplog):
    root = tmp_path / "model"
    _write_model_config(root, policy_type="act")
    _write_adapter_config(root, base_model_name_or_path="my-team/some-base-policy")

    metadata = inspect_model_directory(root)

    assert metadata.is_self_contained is False
    assert metadata.base_model_name_or_path == "my-team/some-base-policy"
    assert metadata.to_wandb_metadata()["is_self_contained"] is False
    assert metadata.to_wandb_metadata()["base_model_name_or_path"] == "my-team/some-base-policy"
    assert any(
        "my-team/some-base-policy" in record.message and record.levelname == "WARNING"
        for record in caplog.records
    )


def test_inspect_model_directory_no_warning_when_base_model_present_locally(tmp_path, caplog):
    root = tmp_path / "model"
    base_model_dir = tmp_path / "local-base-model"
    base_model_dir.mkdir()
    _write_model_config(root, policy_type="act")
    _write_adapter_config(root, base_model_name_or_path=str(base_model_dir))

    metadata = inspect_model_directory(root)

    assert metadata.is_self_contained is True
    assert metadata.base_model_name_or_path == str(base_model_dir)
    assert not any(record.levelname == "WARNING" for record in caplog.records)


def test_inspect_model_directory_no_warning_with_full_weights(tmp_path, caplog):
    root = tmp_path / "model"
    _write_model_config(root, policy_type="act")
    (root / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")

    metadata = inspect_model_directory(root)

    assert metadata.is_self_contained is True
    assert metadata.base_model_name_or_path is None
    assert not any(record.levelname == "WARNING" for record in caplog.records)


def test_inspect_model_directory_tolerates_corrupt_adapter_config(tmp_path):
    root = tmp_path / "model"
    _write_model_config(root, policy_type="act")
    (root / PEFT_ADAPTER_CONFIG_NAME).write_text("{not valid json")
    (root / PEFT_ADAPTER_WEIGHTS_NAME).write_bytes(b"adapter")

    metadata = inspect_model_directory(root)

    assert metadata.is_self_contained is False
    assert metadata.base_model_name_or_path is None
