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

import wandb

from lerobot.integrations.wandb_artifacts.refs import parse_artifact_ref
from lerobot.integrations.wandb_artifacts.store import (
    ArtifactTypeMismatchError,
    DownloadDestinationNotEmptyError,
    MaterializedArtifact,
    declare_input,
    download_artifact,
    link_to_registry,
    upload_directory,
)


class _FakeArtifact:
    """Small stand-in for ``wandb.Artifact`` and ``Run.use_artifact`` results."""

    def __init__(self, name=None, type=None, metadata=None, **_kwargs):  # noqa: A002
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.added_dirs = []
        self.entity = "my-team"
        self.project = "my-project"
        self.version = "v7"
        self.digest = "abc123digest"
        self._download_root = None

    def add_dir(self, local_path, **_kwargs):
        self.added_dirs.append(local_path)

    def wait(self, timeout=None):
        if ":" not in self.name:
            self.name = f"{self.name}:{self.version}"
        return self

    @property
    def qualified_name(self):
        return f"{self.entity}/{self.project}/{self.name}"

    def download(self, root=None, **_kwargs):
        self._download_root = root
        return root


# ---------------------------------------------------------------------------
# upload_directory
# ---------------------------------------------------------------------------


def test_upload_directory_logs_artifact_with_expected_shape(tmp_path, monkeypatch):
    created = {}

    def _fake_artifact_ctor(name, type, metadata=None, **kwargs):  # noqa: A002
        artifact = _FakeArtifact(name=name, type=type, metadata=metadata)
        created["artifact"] = artifact
        return artifact

    monkeypatch.setattr(wandb, "Artifact", _fake_artifact_ctor)

    run = MagicMock()
    run.entity = "my-team"
    run.project = "my-project"
    run.log_artifact.side_effect = lambda artifact, aliases=None: artifact

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    result = upload_directory(
        run,
        dataset_dir,
        name="pick-cube",
        artifact_type="dataset",
        aliases=["latest", "raw"],
        metadata={"fps": 30},
    )

    artifact = created["artifact"]
    assert artifact.type == "dataset"
    assert artifact.metadata == {"fps": 30}
    assert artifact.added_dirs == [str(dataset_dir)]

    run.log_artifact.assert_called_once()
    call_args = run.log_artifact.call_args
    assert call_args.args[0] is artifact
    assert call_args.kwargs["aliases"] == ["latest", "raw"]

    assert isinstance(result, MaterializedArtifact)
    assert result.requested_ref == "my-team/my-project/pick-cube"
    assert result.resolved_ref == "my-team/my-project/pick-cube:v7"
    assert result.local_path == dataset_dir
    assert result.version == "v7"
    assert result.digest == "abc123digest"
    assert result.metadata == {"fps": 30}


def test_upload_directory_waits_for_commit(tmp_path, monkeypatch):
    waited = []

    class _WaitTrackingArtifact(_FakeArtifact):
        def wait(self, timeout=None):
            waited.append(True)
            return super().wait(timeout=timeout)

    monkeypatch.setattr(
        wandb,
        "Artifact",
        lambda name, type, metadata=None: _WaitTrackingArtifact(name=name, type=type, metadata=metadata),
    )

    run = MagicMock()
    run.entity = "e"
    run.project = "p"
    run.log_artifact.side_effect = lambda artifact, aliases=None: artifact

    upload_directory(run, tmp_path, name="n", artifact_type="dataset")
    assert waited == [True]


# ---------------------------------------------------------------------------
# link_to_registry
# ---------------------------------------------------------------------------


def test_link_to_registry_targets_unified_registry_collection():
    run = MagicMock()
    artifact = MagicMock()

    target_path = link_to_registry(run, artifact, collection="pick-cube-policy", aliases=["candidate"])

    assert target_path == "wandb-registry-model/pick-cube-policy"
    run.link_artifact.assert_called_once_with(
        artifact, target_path="wandb-registry-model/pick-cube-policy", aliases=["candidate"]
    )


def test_link_to_registry_without_aliases_passes_none():
    run = MagicMock()
    artifact = MagicMock()

    link_to_registry(run, artifact, collection="pick-cube-policy")

    run.link_artifact.assert_called_once_with(
        artifact, target_path="wandb-registry-model/pick-cube-policy", aliases=None
    )


def test_upload_directory_without_registry_collection_never_links(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wandb,
        "Artifact",
        lambda name, type, metadata=None: _FakeArtifact(name=name, type=type, metadata=metadata),
    )
    run = MagicMock()
    run.entity = "e"
    run.project = "p"
    run.log_artifact.side_effect = lambda artifact, aliases=None: artifact

    result = upload_directory(run, tmp_path, name="n", artifact_type="model")

    run.link_artifact.assert_not_called()
    assert result.registry_collection is None


def test_upload_directory_with_registry_collection_links_after_wait(tmp_path, monkeypatch):
    call_order = []

    class _TrackingArtifact(_FakeArtifact):
        def wait(self, timeout=None):
            call_order.append("wait")
            return super().wait(timeout=timeout)

    monkeypatch.setattr(
        wandb,
        "Artifact",
        lambda name, type, metadata=None: _TrackingArtifact(name=name, type=type, metadata=metadata),
    )
    run = MagicMock()
    run.entity = "e"
    run.project = "p"
    run.log_artifact.side_effect = lambda artifact, aliases=None: artifact
    run.link_artifact.side_effect = lambda *a, **kw: call_order.append("link")

    result = upload_directory(
        run,
        tmp_path,
        name="n",
        artifact_type="model",
        aliases=["candidate"],
        registry_collection="pick-cube-policy",
    )

    assert call_order == ["wait", "link"]
    run.link_artifact.assert_called_once()
    call_kwargs = run.link_artifact.call_args.kwargs
    assert call_kwargs["target_path"] == "wandb-registry-model/pick-cube-policy"
    assert call_kwargs["aliases"] == ["candidate"]
    assert result.registry_collection == "pick-cube-policy"


# ---------------------------------------------------------------------------
# download_artifact
# ---------------------------------------------------------------------------


def _run_with(artifact):
    run = MagicMock()
    run.use_artifact.return_value = artifact
    return run


def test_download_artifact_declares_input_and_atomically_materializes(tmp_path):
    fake = _FakeArtifact(name="pick-cube:v2", type="dataset")
    fake.version = "v2"
    run = _run_with(fake)
    destination = tmp_path / "materialized"

    result = download_artifact(
        run,
        "my-team/my-project/pick-cube:latest",
        expected_type="dataset",
        download_root=destination,
    )

    run.use_artifact.assert_called_once_with("my-team/my-project/pick-cube:latest")
    assert Path(fake._download_root).parent == tmp_path
    assert Path(fake._download_root).name.startswith(".materialized.download-")
    assert not Path(fake._download_root).exists()
    assert destination.is_dir()
    assert isinstance(result, MaterializedArtifact)
    assert result.requested_ref == "my-team/my-project/pick-cube:latest"
    assert result.resolved_ref == "my-team/my-project/pick-cube:v2"
    assert result.version == "v2"
    assert result.local_path == destination


def test_download_artifact_accepts_parsed_ref(tmp_path):
    fake = _FakeArtifact(name="pick-cube:v2", type="dataset")
    fake.version = "v2"
    run = _run_with(fake)
    destination = tmp_path / "materialized"

    ref = parse_artifact_ref("my-team/my-project/pick-cube:v2")
    result = download_artifact(run, ref, expected_type="dataset", download_root=destination)

    run.use_artifact.assert_called_once_with("my-team/my-project/pick-cube:v2")
    assert result.requested_ref == "my-team/my-project/pick-cube:v2"


def test_download_artifact_rejects_type_mismatch_without_downloading(tmp_path):
    fake = _FakeArtifact(name="candidate-model:v0", type="model")
    run = _run_with(fake)

    with pytest.raises(ArtifactTypeMismatchError):
        download_artifact(
            run,
            "my-team/my-project/candidate-model:v0",
            expected_type="dataset",
            download_root=tmp_path / "materialized",
        )

    assert fake._download_root is None


def test_download_artifact_rejects_nonempty_destination_without_touching_it(tmp_path):
    destination = tmp_path / "materialized"
    destination.mkdir()
    sentinel = destination / "unrelated.txt"
    sentinel.write_text("keep me")
    run = MagicMock()

    with pytest.raises(DownloadDestinationNotEmptyError):
        download_artifact(
            run,
            "my-team/my-project/pick-cube:v0",
            expected_type="dataset",
            download_root=destination,
        )

    run.use_artifact.assert_not_called()
    assert sentinel.read_text() == "keep me"


def test_download_artifact_rejects_existing_file_destination(tmp_path):
    destination = tmp_path / "materialized"
    destination.write_text("keep me")
    run = MagicMock()

    with pytest.raises(DownloadDestinationNotEmptyError):
        download_artifact(
            run,
            "my-team/my-project/pick-cube:v0",
            expected_type="dataset",
            download_root=destination,
        )

    run.use_artifact.assert_not_called()
    assert destination.read_text() == "keep me"


def test_download_artifact_accepts_empty_existing_destination(tmp_path):
    destination = tmp_path / "materialized"
    destination.mkdir()
    fake = _FakeArtifact(name="pick-cube:v0", type="dataset")
    fake.version = "v0"

    result = download_artifact(
        _run_with(fake),
        "my-team/my-project/pick-cube:v0",
        expected_type="dataset",
        download_root=destination,
    )

    assert destination.is_dir()
    assert result.local_path == destination


def test_download_artifact_validates_staging_before_promotion(tmp_path):
    destination = tmp_path / "materialized"
    fake = _FakeArtifact(name="pick-cube:v0", type="dataset")
    fake.version = "v0"
    validated = []

    def _validate(path: Path):
        validated.append(path)
        assert path != destination
        (path / "validated.txt").write_text("ok")

    result = download_artifact(
        _run_with(fake),
        "my-team/my-project/pick-cube:v0",
        expected_type="dataset",
        download_root=destination,
        validator=_validate,
    )

    assert len(validated) == 1
    assert result.local_path == destination
    assert (destination / "validated.txt").read_text() == "ok"


def test_download_artifact_cleans_staging_after_download_failure(tmp_path):
    destination = tmp_path / "materialized"

    class _FailingArtifact(_FakeArtifact):
        def download(self, root=None, **_kwargs):
            self._download_root = root
            Path(root, "partial.txt").write_text("partial")
            raise RuntimeError("network failed")

    fake = _FailingArtifact(name="pick-cube:v0", type="dataset")

    with pytest.raises(RuntimeError, match="network failed"):
        download_artifact(
            _run_with(fake),
            "my-team/my-project/pick-cube:v0",
            expected_type="dataset",
            download_root=destination,
        )

    assert not destination.exists()
    assert not Path(fake._download_root).exists()


def test_download_artifact_cleans_staging_after_validation_failure(tmp_path):
    destination = tmp_path / "materialized"
    fake = _FakeArtifact(name="pick-cube:v0", type="dataset")

    def _reject(_path: Path):
        raise ValueError("invalid dataset")

    with pytest.raises(ValueError, match="invalid dataset"):
        download_artifact(
            _run_with(fake),
            "my-team/my-project/pick-cube:v0",
            expected_type="dataset",
            download_root=destination,
            validator=_reject,
        )

    assert not destination.exists()
    assert not Path(fake._download_root).exists()


# ---------------------------------------------------------------------------
# declare_input
# ---------------------------------------------------------------------------


def test_declare_input_resolves_the_ref_without_downloading_anything():
    """Lineage-only: the edge is drawn and the alias resolved, but no bytes are fetched, so there
    is no local path to report.
    """
    # W&B resolves the mutable alias, so `use_artifact` hands back the immutable version.
    artifact = _FakeArtifact(name="pick-cube-policy:v7", type="model", metadata={"policy": "act"})
    run = MagicMock()
    run.use_artifact.return_value = artifact

    result = declare_input(run, "my-team/my-project/pick-cube-policy:latest", expected_type="model")

    run.use_artifact.assert_called_once_with("my-team/my-project/pick-cube-policy:latest")
    assert artifact._download_root is None
    assert result.local_path is None
    assert result.requested_ref == "my-team/my-project/pick-cube-policy:latest"
    assert result.resolved_ref == "my-team/my-project/pick-cube-policy:v7"
    assert result.version == "v7"
    assert result.metadata == {"policy": "act"}


def test_declare_input_rejects_the_wrong_artifact_type():
    run = MagicMock()
    run.use_artifact.return_value = _FakeArtifact(name="pick-cube:latest", type="dataset")

    with pytest.raises(ArtifactTypeMismatchError, match="type 'dataset'"):
        declare_input(run, "my-team/my-project/pick-cube:latest", expected_type="model")


def test_declare_input_rejects_a_malformed_ref_before_calling_wandb():
    run = MagicMock()

    with pytest.raises(ValueError):
        declare_input(run, "not-a-ref", expected_type="model")

    run.use_artifact.assert_not_called()
