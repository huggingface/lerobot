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


def _make_dataset(tmp_path: Path, total_episodes: int = 3, codec: str = "av1"):
    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": [3, 480, 640],
            "info": {"video.codec": codec},
        },
        "observation.state": {"dtype": "float32", "shape": [6]},
        "action": {"dtype": "float32", "shape": [6]},
    }
    meta = SimpleNamespace(
        camera_keys=["observation.images.front"],
        features=features,
        total_episodes=total_episodes,
    )
    return SimpleNamespace(repo_id="user/my dataset", root=tmp_path / "ds", meta=meta)


def test_dataset_name_sanitization():
    assert fv._fiftyone_dataset_name("lerobot/pusht", 0) == "lerobot-pusht-episode-0"
    assert fv._fiftyone_dataset_name("lerobot/pusht", None) == "lerobot-pusht"
    assert fv._fiftyone_dataset_name("user/my dataset!", 2) == "user-my-dataset-episode-2"


def test_single_episode_playback(fake_fo, tmp_path):
    dataset = _make_dataset(tmp_path)
    fv.serve_fiftyone_dataset_playback(dataset, 1, host="0.0.0.0", port=5252, remote=True)

    from_dir = fake_fo.calls["from_dir"]
    assert from_dir["dataset_dir"] == str((tmp_path / "ds").resolve())
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


def test_defaults_defer_to_fiftyone_config(fake_fo, tmp_path):
    fv.serve_fiftyone_dataset_playback(_make_dataset(tmp_path), 0)
    launch = fake_fo.calls["launch_app"]
    assert launch["port"] is None and launch["address"] is None and launch["remote"] is False


def test_all_episodes_and_custom_name(fake_fo, tmp_path):
    fake_fo.fo_dataset.n = 3
    fv.serve_fiftyone_dataset_playback(
        _make_dataset(tmp_path, total_episodes=3),
        None,
        all_episodes=True,
        persistent=False,
        dataset_name="mine",
    )
    from_dir = fake_fo.calls["from_dir"]
    assert from_dir["episodes"] is None
    assert from_dir["name"] == "mine"
    assert from_dir["persistent"] is False


def test_exit_message_persistent_is_runnable_and_boxed(fake_fo, tmp_path, capsys):
    fv.serve_fiftyone_dataset_playback(_make_dataset(tmp_path), 0)
    out = capsys.readouterr().out
    assert "Dataset saved: user-my-dataset-episode-0" in out
    assert "fiftyone app launch user-my-dataset-episode-0" in out
    assert 'fo.load_dataset("user-my-dataset-episode-0")' in out
    assert "https://docs.voxel51.com/user_guide/index.html" in out
    # Every line of the box is the same width, including the title edge.
    box = [line for line in out.splitlines() if line and line[0] in "┌│└"]
    assert len(box) >= 3
    assert len({len(line) for line in box}) == 1


def test_exit_message_shell_quotes_dataset_name():
    assert "fiftyone app launch 'my dataset'" in fv._exit_message("my dataset", persistent=True)


def test_exit_message_non_persistent(fake_fo, tmp_path, capsys):
    fv.serve_fiftyone_dataset_playback(_make_dataset(tmp_path), 0, persistent=False)
    out = capsys.readouterr().out
    assert "was temporary" in out
    assert "fiftyone app launch" not in out


def test_requires_episode_index_without_all_episodes(fake_fo, tmp_path):
    with pytest.raises(ValueError, match="episode_index is required"):
        fv.serve_fiftyone_dataset_playback(_make_dataset(tmp_path), None)
    assert fake_fo.calls["from_dir"] is None


def test_warns_on_skipped_or_missing_episodes(fake_fo, tmp_path, caplog):
    fake_fo.fo_dataset.n = 2
    fake_fo.fo_dataset.info["lerobot"]["skipped_episodes"] = ["episode 2: missing video"]
    with caplog.at_level(logging.WARNING):
        fv.serve_fiftyone_dataset_playback(_make_dataset(tmp_path, total_episodes=3), None, all_episodes=True)
    assert "imported 2 of 3 episode(s); 1 skipped" in caplog.text
    assert "missing video" in caplog.text
    # The App is still launched so the user can inspect what did import.
    assert fake_fo.calls["launch_app"] is not None


def test_warns_on_unsupported_codec(fake_fo, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        fv.serve_fiftyone_dataset_playback(_make_dataset(tmp_path, codec="hevc"), 0)
    assert "encoded with 'hevc'" in caplog.text
    assert "av1 and h264" in caplog.text


def test_no_codec_warning_for_supported_codec(fake_fo, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        fv.serve_fiftyone_dataset_playback(_make_dataset(tmp_path, codec="av1"), 0)
    assert "encoded with" not in caplog.text


def test_keyboard_interrupt_returns_without_closing_session(fake_fo, tmp_path):
    """Ctrl-C must return normally and must NOT call ``session.close()`` (it deadlocks in FiftyOne)."""

    def interrupt(wait):
        raise KeyboardInterrupt

    fake_fo.session.wait = interrupt
    fv.serve_fiftyone_dataset_playback(_make_dataset(tmp_path), 0)
    assert not fake_fo.session.closed
