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

from unittest.mock import MagicMock

import pytest

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")

import wandb

from lerobot.integrations.wandb_artifacts.refs import parse_artifact_ref
from lerobot.integrations.wandb_artifacts.store import (
    ArtifactTypeMismatchError,
    DownloadDestinationNotEmptyError,
    MaterializedArtifact,
    download_artifact,
    upload_directory,
)


class _FakeArtifact:
    """Stand-in for ``wandb.Artifact`` / the object ``Run.use_artifact`` returns.

    Mirrors the real SDK's documented behavior: ``.name`` carries no alias/version until the
    artifact is logged and committed (simulated by ``wait()`` baking the version in), and
    ``.qualified_name`` is always just ``entity/project/name`` (never appends a second version).
    """

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
# download_artifact
# ---------------------------------------------------------------------------


def test_download_artifact_declares_input_and_downloads(tmp_path):
    fake = _FakeArtifact(name="pick-cube:v2", type="dataset")
    fake.version = "v2"  # already-resolved artifact: name and version agree, as the real SDK guarantees

    run = MagicMock()
    run.use_artifact.return_value = fake

    result = download_artifact(
        run,
        "my-team/my-project/pick-cube:latest",
        expected_type="dataset",
        download_root=tmp_path,
    )

    run.use_artifact.assert_called_once_with("my-team/my-project/pick-cube:latest")
    assert fake._download_root == str(tmp_path)
    assert isinstance(result, MaterializedArtifact)
    assert result.requested_ref == "my-team/my-project/pick-cube:latest"
    assert result.resolved_ref == "my-team/my-project/pick-cube:v2"
    assert result.version == "v2"
    assert result.local_path == tmp_path


def test_download_artifact_accepts_parsed_ref(tmp_path):
    fake = _FakeArtifact(name="pick-cube", type="dataset")
    run = MagicMock()
    run.use_artifact.return_value = fake

    ref = parse_artifact_ref("my-team/my-project/pick-cube:v2")
    result = download_artifact(run, ref, expected_type="dataset", download_root=tmp_path)

    run.use_artifact.assert_called_once_with("my-team/my-project/pick-cube:v2")
    assert result.requested_ref == "my-team/my-project/pick-cube:v2"


def test_download_artifact_rejects_type_mismatch_without_downloading(tmp_path):
    fake = _FakeArtifact(name="candidate-model", type="model")
    run = MagicMock()
    run.use_artifact.return_value = fake

    with pytest.raises(ArtifactTypeMismatchError):
        download_artifact(
            run, "my-team/my-project/candidate-model:v0", expected_type="dataset", download_root=tmp_path
        )

    assert fake._download_root is None  # download() must never have been called


def test_download_artifact_rejects_nonempty_destination_without_touching_it(tmp_path, monkeypatch):
    # A nonempty destination could carry a stale file from a previously downloaded, different
    # artifact version — reject it outright rather than silently downloading alongside it. The
    # guard must fire (and leave the file alone) before use_artifact/download are ever reached.
    import shutil

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("download_artifact must never delete anything itself")

    monkeypatch.setattr(shutil, "rmtree", _must_not_be_called)

    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("keep me")

    run = MagicMock()

    with pytest.raises(DownloadDestinationNotEmptyError):
        download_artifact(
            run, "my-team/my-project/pick-cube:v0", expected_type="dataset", download_root=tmp_path
        )

    run.use_artifact.assert_not_called()
    assert sentinel.read_text() == "keep me"


def test_download_artifact_accepts_empty_existing_destination(tmp_path):
    fake = _FakeArtifact(name="pick-cube", type="dataset")
    run = MagicMock()
    run.use_artifact.return_value = fake

    download_artifact(run, "my-team/my-project/pick-cube:v0", expected_type="dataset", download_root=tmp_path)

    assert fake._download_root == str(tmp_path)


def test_download_artifact_accepts_nonexistent_destination(tmp_path):
    fake = _FakeArtifact(name="pick-cube", type="dataset")
    run = MagicMock()
    run.use_artifact.return_value = fake

    destination = tmp_path / "not-created-yet"
    download_artifact(
        run, "my-team/my-project/pick-cube:v0", expected_type="dataset", download_root=destination
    )

    assert fake._download_root == str(destination)
