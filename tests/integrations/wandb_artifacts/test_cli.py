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
from unittest.mock import MagicMock

import pytest

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.io_utils import write_info
from lerobot.datasets.utils import STATS_PATH, DatasetInfo
from lerobot.integrations.wandb_artifacts import cli
from lerobot.integrations.wandb_artifacts.inspect import DatasetDirectoryError
from lerobot.integrations.wandb_artifacts.store import MaterializedArtifact


def _write_minimal_dataset(root: Path) -> None:
    write_info(
        DatasetInfo(
            codebase_version="v3.0", fps=30, features={"action": {"dtype": "float32", "shape": (6,)}}
        ),
        root,
    )
    (root / STATS_PATH).write_text("{}")
    (root / "data").mkdir(parents=True, exist_ok=True)


def _fake_run():
    run = MagicMock()
    run.entity = "my-team"
    run.project = "my-project"
    return run


def test_dataset_upload_validates_before_touching_wandb(tmp_path, monkeypatch):
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())
    upload_calls = []
    monkeypatch.setattr(
        cli,
        "upload_directory",
        lambda *a, **kw: upload_calls.append((a, kw)) or _materialized_upload_result(),
    )

    empty_root = tmp_path / "not-a-dataset"
    empty_root.mkdir()

    with pytest.raises(DatasetDirectoryError):
        cli.main(["dataset", "upload", "--root", str(empty_root), "--project", "p", "--name", "n"])

    assert init_calls == []
    assert upload_calls == []


def _materialized_upload_result():
    return MaterializedArtifact(
        requested_ref="my-team/my-project/pick-cube",
        resolved_ref="my-team/my-project/pick-cube:v0",
        local_path=Path("/tmp/does-not-matter"),
        version="v0",
        digest="digest",
        metadata={},
    )


def test_dataset_upload_happy_path(tmp_path, monkeypatch, capsys):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _write_minimal_dataset(dataset_root)

    run = _fake_run()
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or run)

    upload_calls = []

    def _fake_upload(passed_run, directory, *, name, artifact_type, aliases=(), metadata=None):
        upload_calls.append(
            {
                "run": passed_run,
                "directory": Path(directory),
                "name": name,
                "artifact_type": artifact_type,
                "aliases": list(aliases),
                "metadata": metadata,
            }
        )
        return _materialized_upload_result()

    monkeypatch.setattr(cli, "upload_directory", _fake_upload)

    cli.main(
        [
            "dataset",
            "upload",
            "--root",
            str(dataset_root),
            "--project",
            "my-project",
            "--entity",
            "my-team",
            "--name",
            "pick-cube",
            "--alias",
            "raw",
            "--alias",
            "clean",
        ]
    )

    assert init_calls[0]["project"] == "my-project"
    assert init_calls[0]["entity"] == "my-team"
    run.finish.assert_called_once()

    assert len(upload_calls) == 1
    call = upload_calls[0]
    assert call["directory"] == dataset_root
    assert call["name"] == "pick-cube"
    assert call["artifact_type"] == "dataset"
    assert call["aliases"] == ["raw", "clean"]
    assert call["metadata"]["schema_version"] == "v3.0"

    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube:v0" in out
    assert "raw" in out and "clean" in out


def test_dataset_upload_finishes_run_even_on_upload_failure(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _write_minimal_dataset(dataset_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    def _boom(*a, **kw):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(cli, "upload_directory", _boom)

    with pytest.raises(RuntimeError):
        cli.main(["dataset", "upload", "--root", str(dataset_root), "--project", "p", "--name", "n"])

    run.finish.assert_called_once()


def test_dataset_download_rejects_malformed_ref_before_touching_wandb(tmp_path, monkeypatch):
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())

    with pytest.raises(ValueError):
        cli.main(["dataset", "download", "--ref", "not-a-valid-ref", "--root", str(tmp_path)])

    assert init_calls == []


def test_dataset_download_happy_path(tmp_path, monkeypatch, capsys):
    run = _fake_run()
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or run)

    dest = tmp_path / "materialized"

    def _fake_download(passed_run, ref, *, expected_type, download_root):
        _write_minimal_dataset(Path(download_root))
        return MaterializedArtifact(
            requested_ref=str(ref),
            resolved_ref="my-team/my-project/pick-cube:v3",
            local_path=Path(download_root),
            version="v3",
            digest="digest",
            metadata={},
        )

    monkeypatch.setattr(cli, "download_artifact", _fake_download)

    cli.main(["dataset", "download", "--ref", "my-team/my-project/pick-cube:latest", "--root", str(dest)])

    assert init_calls[0]["entity"] == "my-team"
    assert init_calls[0]["project"] == "my-project"
    run.finish.assert_called_once()

    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube:v3" in out
    assert str(dest) in out


def test_dataset_download_allows_logging_the_run_in_a_different_project(tmp_path, monkeypatch):
    """A caller with only read access to the artifact's own project must still be able to log
    the lineage run somewhere they can write to, without changing which artifact gets fetched."""
    run = _fake_run()
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or run)

    dest = tmp_path / "materialized"
    download_calls = []

    def _fake_download(passed_run, ref, *, expected_type, download_root):
        download_calls.append(str(ref))
        _write_minimal_dataset(Path(download_root))
        return MaterializedArtifact(
            requested_ref=str(ref),
            resolved_ref="source-team/source-project/pick-cube:v3",
            local_path=Path(download_root),
            version="v3",
            digest="digest",
            metadata={},
        )

    monkeypatch.setattr(cli, "download_artifact", _fake_download)

    cli.main(
        [
            "dataset",
            "download",
            "--ref",
            "source-team/source-project/pick-cube:latest",
            "--root",
            str(dest),
            "--entity",
            "my-own-team",
            "--project",
            "my-own-project",
        ]
    )

    assert init_calls[0]["entity"] == "my-own-team"
    assert init_calls[0]["project"] == "my-own-project"
    # The fully qualified source ref must reach download_artifact unchanged.
    assert download_calls == ["source-team/source-project/pick-cube:latest"]


def test_dataset_download_rejects_result_missing_required_files(tmp_path, monkeypatch):
    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    dest = tmp_path / "materialized"

    def _fake_download_incomplete(passed_run, ref, *, expected_type, download_root):
        Path(download_root).mkdir(parents=True, exist_ok=True)  # no meta/info.json etc.
        return MaterializedArtifact(
            requested_ref=str(ref),
            resolved_ref="my-team/my-project/pick-cube:v3",
            local_path=Path(download_root),
            version="v3",
            digest="digest",
            metadata={},
        )

    monkeypatch.setattr(cli, "download_artifact", _fake_download_incomplete)

    with pytest.raises(DatasetDirectoryError):
        cli.main(["dataset", "download", "--ref", "my-team/my-project/pick-cube:latest", "--root", str(dest)])
