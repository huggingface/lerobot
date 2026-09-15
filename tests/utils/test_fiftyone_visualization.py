#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for the FiftyOne dataset-playback backend.

A fake ``fiftyone`` package is injected into ``sys.modules`` so these run without FiftyOne (or its
MongoDB backend) installed; they check how the backend drives the FiftyOne API, not FiftyOne itself.
"""

import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from lerobot.utils import fiftyone_visualization as fv


class _FakeFiftyOneDataset:
    def __init__(self):
        self.info = {"lerobot": {"skipped_episodes": []}}
        self.n = 1

    def __len__(self):
        return self.n


class _FakeSession:
    def __init__(self):
        self.url = "http://localhost:5151/"
        self.wait_calls = []
        self.closed = False

    def wait(self, wait):
        self.wait_calls.append(wait)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_fo(monkeypatch):
    """Install fake ``fiftyone`` / ``fiftyone.types`` modules and bypass ``require_package``."""
    calls = {"from_dir": None, "launch_app": None}
    session = _FakeSession()
    fo_dataset = _FakeFiftyOneDataset()

    def from_dir(**kwargs):
        calls["from_dir"] = kwargs
        return fo_dataset

    def launch_app(dataset, **kwargs):
        calls["launch_app"] = {"dataset": dataset, **kwargs}
        return session

    fo = ModuleType("fiftyone")
    fo.Dataset = SimpleNamespace(from_dir=from_dir)
    fo.launch_app = launch_app
    fot = ModuleType("fiftyone.types")
    fot.LeRobotDataset = object()
    fo.types = fot

    monkeypatch.setitem(sys.modules, "fiftyone", fo)
    monkeypatch.setitem(sys.modules, "fiftyone.types", fot)
    monkeypatch.setattr(fv, "require_package", lambda *a, **k: None)
    return SimpleNamespace(calls=calls, session=session, fo_dataset=fo_dataset, fot=fot)


_CAM = "observation.images.front"


class _FakeMeta:
    """Stand-in for ``LeRobotDatasetMetadata``: one data shard and one video per episode."""

    def __init__(self, repo_id: str, root: Path, total_episodes: int = 3, codec: str = "av1"):
        self.repo_id = repo_id
        self.root = root
        self.revision = "v3.0"
        self.total_episodes = total_episodes
        self.camera_keys = [_CAM]
        self.video_keys = [_CAM]
        self.features = {
            _CAM: {"dtype": "video", "shape": [3, 480, 640], "info": {"video.codec": codec}},
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
        }

    def get_data_file_path(self, ep: int) -> Path:
        return Path(f"data/chunk-000/file-{ep:03d}.parquet")

    def get_video_file_path(self, ep: int, key: str) -> Path:
        return Path(f"videos/{key}/chunk-000/file-{ep:03d}.mp4")

    def write_files(self, episodes) -> None:
        for ep in episodes:
            for rel in (self.get_data_file_path(ep), self.get_video_file_path(ep, _CAM)):
                (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
                (self.root / rel).touch()


@pytest.fixture
def fake_meta(monkeypatch, tmp_path):
    """Replace ``LeRobotDatasetMetadata`` with ``_FakeMeta`` (all 3 episodes on disk) and stub the Hub."""
    state = {"meta": None, "downloads": []}
    opts = {"total_episodes": 3, "codec": "av1"}

    def make_meta(repo_id, root=None):
        meta = _FakeMeta(repo_id, Path(root) if root is not None else tmp_path / "cache", **opts)
        state["meta"] = meta
        return meta

    def snapshot_download(repo_id, **kwargs):
        state["downloads"].append({"repo_id": repo_id, **kwargs})
        return str(kwargs.get("local_dir", tmp_path / "cache"))

    monkeypatch.setattr(fv, "LeRobotDatasetMetadata", make_meta)
    monkeypatch.setattr(fv, "snapshot_download", snapshot_download)
    monkeypatch.setattr(fv, "get_safe_version", lambda repo_id, rev: rev)

    root = tmp_path / "ds"
    _FakeMeta("user/my dataset", root).write_files(range(3))
    return SimpleNamespace(state=state, opts=opts, root=root, repo_id="user/my dataset")


def _serve(fake_meta, episode_index, **kwargs):
    return fv.serve_fiftyone_dataset_playback(fake_meta.repo_id, episode_index, root=fake_meta.root, **kwargs)


def test_dataset_name_sanitization():
    assert fv._fiftyone_dataset_name("lerobot/pusht", 0) == "lerobot-pusht-episode-0"
    assert fv._fiftyone_dataset_name("lerobot/pusht", None) == "lerobot-pusht"
    assert fv._fiftyone_dataset_name("user/my dataset!", 2) == "user-my-dataset-episode-2"


def test_single_episode_playback(fake_fo, fake_meta):
    _serve(fake_meta, 1, host="0.0.0.0", port=5252, remote=True)

    from_dir = fake_fo.calls["from_dir"]
    assert from_dir["dataset_dir"] == str(fake_meta.root.resolve())
    assert Path(from_dir["dataset_dir"]).is_absolute()
    assert from_dir["dataset_type"] is fake_fo.fot.LeRobotDataset
    assert from_dir["episodes"] == [1]
    assert from_dir["name"] == "user-my-dataset-episode-1"
    assert from_dir["persistent"] is True  # kept by default so App tags/views survive
    assert from_dir["overwrite"] is True  # ... but re-running replaces it rather than accumulating

    launch = fake_fo.calls["launch_app"]
    assert launch["dataset"] is fake_fo.fo_dataset
    assert launch == {"dataset": fake_fo.fo_dataset, "port": 5252, "address": "0.0.0.0", "remote": True}

    assert fake_fo.session.wait_calls == [-1]
    assert not fake_fo.session.closed


def test_defaults_defer_to_fiftyone_config(fake_fo, fake_meta):
    _serve(fake_meta, 0)
    launch = fake_fo.calls["launch_app"]
    assert launch["port"] is None and launch["address"] is None and launch["remote"] is False


def test_all_episodes_and_custom_name(fake_fo, fake_meta):
    fake_fo.fo_dataset.n = 3
    _serve(fake_meta, None, all_episodes=True, persistent=False, dataset_name="mine")
    from_dir = fake_fo.calls["from_dir"]
    assert from_dir["episodes"] is None
    assert from_dir["name"] == "mine"
    assert from_dir["persistent"] is False


def test_no_download_when_files_are_local(fake_fo, fake_meta):
    """A dataset already on disk (e.g. ``--root``) is never touched on the Hub, even with all episodes."""
    _serve(fake_meta, None, all_episodes=True)
    assert fake_meta.state["downloads"] == []


def test_downloads_only_missing_files_for_requested_episode(fake_fo, fake_meta):
    for rel in (fake_meta.root / "data/chunk-000/file-001.parquet",):
        rel.unlink()
    _serve(fake_meta, 1)

    (download,) = fake_meta.state["downloads"]
    assert download["repo_id"] == fake_meta.repo_id
    assert download["repo_type"] == "dataset"
    assert download["local_dir"] == fake_meta.root  # --root given: materialize there, like LeRobotDataset
    assert download["allow_patterns"] == ["data/chunk-000/file-001.parquet"]  # only what's missing


def test_missing_episode_files_go_to_cache_without_root(fake_fo, fake_meta, tmp_path):
    fv.serve_fiftyone_dataset_playback(fake_meta.repo_id, 0)  # no root: nothing exists under the fake cache
    (download,) = fake_meta.state["downloads"]
    assert "local_dir" not in download
    assert download["cache_dir"] == fv.HF_LEROBOT_HUB_CACHE
    assert set(download["allow_patterns"]) == {
        "data/chunk-000/file-000.parquet",
        f"videos/{_CAM}/chunk-000/file-000.mp4",
    }
    # The importer is pointed at the snapshot returned by the download.
    assert fake_fo.calls["from_dir"]["dataset_dir"] == str((tmp_path / "cache").resolve())


def test_exit_message_persistent_is_runnable_and_boxed(fake_fo, fake_meta, capsys):
    _serve(fake_meta, 0)
    out = capsys.readouterr().out
    assert "Dataset saved: user-my-dataset-episode-0" in out
    assert "fiftyone app launch user-my-dataset-episode-0" in out
    assert "fo.load_dataset('user-my-dataset-episode-0')" in out
    assert "https://docs.voxel51.com/user_guide/index.html" in out
    # Every line of the box is the same width, including the title edge.
    box = [line for line in out.splitlines() if line and line[0] in "┌│└"]
    assert len(box) >= 3
    assert len({len(line) for line in box}) == 1


def test_exit_message_quotes_dataset_name_for_shell_and_python():
    msg = fv._exit_message('my "dataset"', persistent=True)
    assert "fiftyone app launch 'my \"dataset\"'" in msg
    assert "fo.load_dataset('my \"dataset\"')" in msg  # repr: valid Python whatever the name contains


def test_exit_message_non_persistent(fake_fo, fake_meta, capsys):
    _serve(fake_meta, 0, persistent=False)
    out = capsys.readouterr().out
    assert "was temporary" in out
    assert "fiftyone app launch" not in out


def test_requires_episode_index_without_all_episodes(fake_fo, fake_meta):
    with pytest.raises(ValueError, match="episode_index is required"):
        _serve(fake_meta, None)
    assert fake_fo.calls["from_dir"] is None


def test_warns_on_skipped_or_missing_episodes(fake_fo, fake_meta, caplog):
    fake_fo.fo_dataset.n = 2
    fake_fo.fo_dataset.info["lerobot"]["skipped_episodes"] = ["episode 2: missing video"]
    with caplog.at_level(logging.WARNING):
        _serve(fake_meta, None, all_episodes=True)
    assert "imported 2 of 3 episode(s); 1 skipped" in caplog.text
    assert "missing video" in caplog.text
    # The App is still launched so the user can inspect what did import.
    assert fake_fo.calls["launch_app"] is not None


def test_warns_on_unsupported_codec(fake_fo, fake_meta, caplog):
    with caplog.at_level(logging.WARNING):
        fake_meta.opts["codec"] = "hevc"
        _serve(fake_meta, 0)
    assert "encoded with 'hevc'" in caplog.text
    assert "av1 and h264" in caplog.text


def test_no_codec_warning_for_supported_codec(fake_fo, fake_meta, caplog):
    with caplog.at_level(logging.WARNING):
        _serve(fake_meta, 0)
    assert "encoded with" not in caplog.text


def test_keyboard_interrupt_returns_without_closing_session(fake_fo, fake_meta):
    """Ctrl-C must return normally and must NOT call ``session.close()`` (it deadlocks in FiftyOne)."""

    def interrupt(wait):
        raise KeyboardInterrupt

    fake_fo.session.wait = interrupt
    _serve(fake_meta, 0)
    assert not fake_fo.session.closed
