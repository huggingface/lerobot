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

import numpy as np
import pytest

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.integrations.wandb_artifacts import cli
from lerobot.integrations.wandb_artifacts.inspect import DatasetDirectoryError, ModelDirectoryError
from lerobot.integrations.wandb_artifacts.store import MaterializedArtifact

_ACTION_FEATURE = {"dtype": "float32", "shape": (6,), "names": None}


def _write_minimal_dataset(root: Path) -> None:
    """A tiny, genuinely valid local LeRobot dataset (`root` must not already exist).

    Built with the real dataset writer rather than hand-assembled JSON so it stays valid as
    `inspect.validate_dataset_directory`'s requirements evolve.
    """
    dataset = LeRobotDataset.create(
        repo_id="tests/wandb-artifacts-cli",
        fps=30,
        features={"action": _ACTION_FEATURE},
        root=root,
        robot_type="so101",
        use_videos=False,
        video_backend="pyav",
        metadata_buffer_size=1,
    )
    dataset.add_frame({"action": np.zeros(6, dtype=np.float32), "task": "task-0"})
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()


def _write_minimal_model(root: Path) -> None:
    import json

    root.mkdir(parents=True, exist_ok=True)
    (root / CONFIG_NAME).write_text(json.dumps({"type": "act"}))
    (root / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")


def _fake_run():
    run = MagicMock()
    run.entity = "my-team"
    run.project = "my-project"
    return run


def _materialized_upload_result():
    return MaterializedArtifact(
        requested_ref="my-team/my-project/pick-cube",
        resolved_ref="my-team/my-project/pick-cube:v0",
        local_path=Path("/tmp/does-not-matter"),
        version="v0",
        digest="digest",
        metadata={},
    )


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


def test_dataset_upload_happy_path(tmp_path, monkeypatch, capsys):
    dataset_root = tmp_path / "dataset"
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
    assert init_calls[0]["mode"] == "online"
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
    _write_minimal_dataset(dataset_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    def _boom(*a, **kw):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(cli, "upload_directory", _boom)

    with pytest.raises(RuntimeError):
        cli.main(["dataset", "upload", "--root", str(dataset_root), "--project", "p", "--name", "n"])

    run.finish.assert_called_once()


def test_transfer_commands_do_not_expose_offline_or_disabled_modes(tmp_path):
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "dataset",
                "upload",
                "--root",
                str(tmp_path),
                "--project",
                "p",
                "--name",
                "n",
                "--mode",
                "offline",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "dataset",
                "download",
                "--ref",
                "e/p/n:v0",
                "--root",
                str(tmp_path / "dataset"),
                "--mode",
                "disabled",
            ]
        )


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
    validator_calls = []

    def _fake_download(passed_run, ref, *, expected_type, download_root, validator=None):
        _write_minimal_dataset(Path(download_root))
        validator_calls.append(validator)
        validator(Path(download_root))
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
    assert init_calls[0]["mode"] == "online"
    assert validator_calls == [cli.validate_dataset_directory]
    run.finish.assert_called_once()

    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube:v3" in out
    assert str(dest) in out


def test_dataset_download_allows_logging_the_run_in_a_different_project(tmp_path, monkeypatch):
    """A read-only source project must not also be the mandatory lineage-run destination."""
    run = _fake_run()
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or run)

    dest = tmp_path / "materialized"
    download_calls = []

    def _fake_download(passed_run, ref, *, expected_type, download_root, validator=None):
        download_calls.append(str(ref))
        _write_minimal_dataset(Path(download_root))
        validator(Path(download_root))
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
    assert init_calls[0]["mode"] == "online"
    assert download_calls == ["source-team/source-project/pick-cube:latest"]


def test_dataset_download_rejects_result_missing_required_files(tmp_path, monkeypatch):
    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    dest = tmp_path / "materialized"

    def _fake_download_incomplete(passed_run, ref, *, expected_type, download_root, validator=None):
        Path(download_root).mkdir(parents=True, exist_ok=True)
        validator(Path(download_root))
        raise AssertionError("validator should have rejected the incomplete directory")

    monkeypatch.setattr(cli, "download_artifact", _fake_download_incomplete)

    with pytest.raises(DatasetDirectoryError):
        cli.main(["dataset", "download", "--ref", "my-team/my-project/pick-cube:latest", "--root", str(dest)])

    run.finish.assert_called_once()


# ---------------------------------------------------------------------------
# model upload / download
# ---------------------------------------------------------------------------


def _materialized_model_upload_result(registry_collection=None):
    return MaterializedArtifact(
        requested_ref="my-team/my-project/pick-cube-policy",
        resolved_ref="my-team/my-project/pick-cube-policy:v0",
        local_path=Path("/tmp/does-not-matter"),
        version="v0",
        digest="digest",
        metadata={},
        registry_collection=registry_collection,
    )


def test_model_upload_validates_before_touching_wandb(tmp_path, monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("wandb.init should not be called before validation")

    monkeypatch.setattr(cli.wandb, "init", _boom)

    not_a_model = tmp_path / "not-a-model"
    not_a_model.mkdir()

    with pytest.raises(ModelDirectoryError):
        cli.main(["model", "upload", "--root", str(not_a_model), "--project", "p", "--name", "n"])


def test_model_upload_happy_path_without_registry(tmp_path, monkeypatch, capsys):
    model_root = tmp_path / "model"
    _write_minimal_model(model_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    upload_calls = []

    def _fake_upload(passed_run, directory, *, name, artifact_type, aliases=(), metadata=None, **kwargs):
        upload_calls.append(
            {"artifact_type": artifact_type, "registry_collection": kwargs.get("registry_collection")}
        )
        return _materialized_model_upload_result()

    monkeypatch.setattr(cli, "upload_directory", _fake_upload)

    cli.main(["model", "upload", "--root", str(model_root), "--project", "p", "--entity", "e", "--name", "n"])

    run.finish.assert_called_once()
    assert upload_calls[0]["artifact_type"] == "model"
    assert upload_calls[0]["registry_collection"] is None

    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube-policy:v0" in out
    assert "Linked into registry collection" not in out


def test_model_upload_happy_path_with_registry(tmp_path, monkeypatch, capsys):
    model_root = tmp_path / "model"
    _write_minimal_model(model_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    upload_calls = []

    def _fake_upload(passed_run, directory, *, name, artifact_type, aliases=(), metadata=None, **kwargs):
        upload_calls.append(kwargs.get("registry_collection"))
        return _materialized_model_upload_result(registry_collection="pick-cube-policy")

    monkeypatch.setattr(cli, "upload_directory", _fake_upload)

    cli.main(
        [
            "model",
            "upload",
            "--root",
            str(model_root),
            "--project",
            "p",
            "--name",
            "n",
            "--registry-collection",
            "pick-cube-policy",
        ]
    )

    assert upload_calls == ["pick-cube-policy"]
    out = capsys.readouterr().out
    assert "Linked into registry collection: pick-cube-policy" in out


def test_model_upload_finishes_run_even_on_upload_failure(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    _write_minimal_model(model_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    def _boom(*a, **kw):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(cli, "upload_directory", _boom)

    with pytest.raises(RuntimeError):
        cli.main(["model", "upload", "--root", str(model_root), "--project", "p", "--name", "n"])

    run.finish.assert_called_once()


def test_model_download_rejects_malformed_ref_before_touching_wandb(tmp_path, monkeypatch):
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())

    with pytest.raises(ValueError):
        cli.main(["model", "download", "--ref", "not-a-valid-ref", "--root", str(tmp_path)])

    assert init_calls == []


def test_model_download_happy_path(tmp_path, monkeypatch, capsys):
    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    dest = tmp_path / "materialized"
    download_calls = []
    validator_calls = []

    def _fake_download(passed_run, ref, *, expected_type, download_root, validator=None):
        download_calls.append(expected_type)
        _write_minimal_model(Path(download_root))
        validator_calls.append(validator)
        validator(Path(download_root))
        return MaterializedArtifact(
            requested_ref=str(ref),
            resolved_ref="my-team/my-project/pick-cube-policy:v3",
            local_path=Path(download_root),
            version="v3",
            digest="digest",
            metadata={},
        )

    monkeypatch.setattr(cli, "download_artifact", _fake_download)

    cli.main(
        ["model", "download", "--ref", "my-team/my-project/pick-cube-policy:latest", "--root", str(dest)]
    )

    assert download_calls == ["model"]
    assert validator_calls == [cli.validate_model_directory]
    run.finish.assert_called_once()
    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube-policy:v3" in out
    assert str(dest) in out


def test_model_download_rejects_result_missing_required_files(tmp_path, monkeypatch):
    """The store must validate the staged download before promoting it to ``--root``.

    Mirrors ``test_dataset_download_rejects_result_missing_required_files``: the validator is
    invoked by ``download_artifact`` itself, while the download is still staged, not by the CLI
    after the fact — so a rejecting validator must stop promotion before it ever happens.
    """
    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    dest = tmp_path / "materialized"

    def _fake_download_incomplete(passed_run, ref, *, expected_type, download_root, validator=None):
        Path(download_root).mkdir(parents=True, exist_ok=True)
        validator(Path(download_root))
        raise AssertionError("validator should have rejected the incomplete directory")

    monkeypatch.setattr(cli, "download_artifact", _fake_download_incomplete)

    with pytest.raises(ModelDirectoryError):
        cli.main(
            ["model", "download", "--ref", "my-team/my-project/pick-cube-policy:latest", "--root", str(dest)]
        )

    run.finish.assert_called_once()


@pytest.mark.parametrize("destination_exists", [False, True])
def test_model_download_leaves_destination_untouched_when_staged_model_is_invalid(
    tmp_path, monkeypatch, destination_exists
):
    """Regression test for staging an invalid model: it must never be promoted to ``--root``.

    Exercises the real ``download_artifact`` (not a mock of it) end to end through
    ``cmd_model_download``, so it proves the CLI actually wires ``validate_model_directory`` in as
    the store's ``validator`` — a mocked ``download_artifact`` would hide a missing wire-up.
    """
    run = _fake_run()

    class _InvalidStagedArtifact:
        type = "model"
        version = "v0"
        digest = "digest"
        metadata = {}
        entity = "my-team"
        project = "my-project"
        name = "pick-cube-policy:v0"

        @property
        def qualified_name(self):
            return f"{self.entity}/{self.project}/{self.name}"

        def download(self, root=None, **_kwargs):
            # Staged content has config.json but no weights: fails validate_model_directory.
            root_path = Path(root)
            root_path.mkdir(parents=True, exist_ok=True)
            (root_path / CONFIG_NAME).write_text("{}")
            return root

    run.use_artifact = lambda ref: _InvalidStagedArtifact()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    dest = tmp_path / "materialized"
    if destination_exists:
        dest.mkdir()

    with pytest.raises(ModelDirectoryError):
        cli.main(
            ["model", "download", "--ref", "my-team/my-project/pick-cube-policy:latest", "--root", str(dest)]
        )

    if destination_exists:
        assert dest.is_dir()
        assert list(dest.iterdir()) == []
    else:
        assert not dest.exists()
    run.finish.assert_called_once()
