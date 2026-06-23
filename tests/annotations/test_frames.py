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
"""Unit tests for :class:`VideoFrameProvider` method bindings.

These were prompted by a real regression: ``video_for_episode`` was once
indented one level too deep so it ended up nested *inside* a module-level
helper (after that function's ``return`` statement) — silently dead code
that meant production runs with ``use_video_url=False`` would
``AttributeError`` on ``self.frame_provider.video_for_episode(...)``. The
existing module tests didn't catch it because they exercise stub providers.

The tests below assert on the class itself (not on an instance), so a
future reindent regression flips them to red without needing a real
LeRobot dataset on disk.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.annotations.steerable_pipeline.frames import VideoFrameProvider  # noqa: E402


class _FakeMeta:
    """Minimal metadata stub exposing ``video_keys`` / ``camera_keys``."""

    def __init__(self, video_keys: list[str], image_keys: list[str], video_path: Path | None = None) -> None:
        self.video_keys = video_keys
        self.camera_keys = [*video_keys, *image_keys]
        self._video_path = video_path
        self.episodes = {0: {f"videos/{key}/from_timestamp": 0.0 for key in video_keys}}

    def get_video_file_path(self, episode_index: int, camera_key: str) -> Path:
        return self._video_path


def test_default_camera_key_skips_image_only_cameras(tmp_path: Path, monkeypatch) -> None:
    """The default camera must be a *video* key — image-stored cameras have no
    ``videos/<key>/from_timestamp`` and would KeyError in the clip/decode path.

    Regression: a dataset whose first ``camera_keys`` entry was an image-stored
    camera (e.g. ``observation.images.wrist``) crashed at clip extraction.
    """
    fake = _FakeMeta(
        video_keys=["observation.images.robot0_agentview_right"],
        image_keys=["observation.images.wrist"],
    )
    import lerobot.datasets.dataset_metadata as meta_mod

    monkeypatch.setattr(meta_mod, "LeRobotDatasetMetadata", lambda *a, **k: fake, raising=True)
    provider = VideoFrameProvider(root=tmp_path)
    assert provider.camera_key == "observation.images.robot0_agentview_right"
    assert "observation.images.wrist" not in provider.camera_keys


def test_video_for_episode_is_a_method_of_videoframeprovider():
    """``video_for_episode`` must be a bound method, not nested dead code."""
    assert callable(getattr(VideoFrameProvider, "video_for_episode", None))


def test_episode_clip_path_is_a_method_of_videoframeprovider():
    """``episode_clip_path`` is now a method (was a free function reaching
    into ``provider._meta`` from outside the class)."""
    assert callable(getattr(VideoFrameProvider, "episode_clip_path", None))


def test_videoframeprovider_has_a_lock_for_concurrent_use():
    """A ``ThreadPoolExecutor`` runs the plan / interjections / vqa phases
    concurrently; the cache + warn-flag accesses must be guarded.
    """
    import threading

    # Fresh-instance check via a minimal fake to avoid touching the hub.
    # The lock is declared with ``init=False`` and has a default factory,
    # so a constructed instance must own a real ``threading.Lock``.
    lock_field = next(
        (f for f in VideoFrameProvider.__dataclass_fields__.values() if f.name == "_lock"),
        None,
    )
    assert lock_field is not None
    assert lock_field.default_factory is threading.Lock


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """A 3 s 10 fps test-pattern mp4, written with ffmpeg."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    out = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=160x120:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def _provider_for_video(tmp_path: Path, video: Path, monkeypatch) -> VideoFrameProvider:
    """A provider whose single camera resolves to ``video`` via fake metadata."""
    fake = _FakeMeta(video_keys=["observation.images.cam"], image_keys=[], video_path=video)
    import lerobot.datasets.dataset_metadata as meta_mod

    monkeypatch.setattr(meta_mod, "LeRobotDatasetMetadata", lambda *a, **k: fake, raising=True)
    return VideoFrameProvider(root=tmp_path, tolerance_s=0.2)


def test_decode_returns_one_uint8_frame_per_timestamp(
    sample_video: Path, tmp_path: Path, monkeypatch
) -> None:
    """``_decode`` routes through ``decode_video_frames`` (torchcodec when
    available, PyAV otherwise) — no subprocess fallback.
    """
    provider = _provider_for_video(tmp_path, sample_video, monkeypatch)
    timestamps = [0.0, 1.0, 2.5]
    frames = provider._decode(0, timestamps, "observation.images.cam")

    assert len(frames) == len(timestamps)
    for frame in frames:
        assert isinstance(frame, torch.Tensor)
        assert frame.dtype == torch.uint8
        assert frame.shape == (3, 120, 160)


def test_frames_at_snaps_mid_frame_grid_to_real_frames(
    sample_video: Path, tmp_path: Path, monkeypatch
) -> None:
    """Uniform sampling grids land mid-frame; ``frames_at`` must snap them to
    real frame timestamps before decoding.

    Regression: ``decode_video_frames`` rejects queries farther than
    ``tolerance_s`` (default 10 ms) from a decodable frame, so un-snapped
    mid-frame queries raised ``FrameTimestampError`` wholesale and the plan
    module silently lost its contact sheets for most episodes.
    """
    from types import SimpleNamespace

    fake = _FakeMeta(video_keys=["observation.images.cam"], image_keys=[], video_path=sample_video)
    import lerobot.datasets.dataset_metadata as meta_mod

    monkeypatch.setattr(meta_mod, "LeRobotDatasetMetadata", lambda *a, **k: fake, raising=True)
    provider = VideoFrameProvider(root=tmp_path)  # default 10 ms tolerance
    # 10 fps fixture -> frames at 0.0, 0.1, ...; queries sit mid-frame.
    record = SimpleNamespace(episode_index=0, frame_timestamps=[i / 10 for i in range(30)])

    frames = provider.frames_at(record, [0.149, 1.234, 2.04], camera_key="observation.images.cam")

    assert len(frames) == 3
    for frame in frames:
        assert isinstance(frame, torch.Tensor)
        assert frame.shape == (3, 120, 160)


def test_decode_returns_empty_list_on_missing_file(tmp_path: Path, monkeypatch) -> None:
    """A missing video is a recoverable no-frames condition, never a crash."""
    provider = _provider_for_video(tmp_path, tmp_path / "does_not_exist.mp4", monkeypatch)
    assert provider._decode(0, [0.0], "observation.images.cam") == []


def test_episode_clip_path_trims_via_reencode_video(tmp_path: Path, monkeypatch) -> None:
    """Clip extraction delegates to ``video_utils.reencode_video`` with the
    episode's ``[from_timestamp, to_timestamp)`` trim window — no subprocess.
    """
    from types import SimpleNamespace

    import lerobot.annotations.steerable_pipeline.frames as frames_mod

    src = tmp_path / "src.mp4"
    src.write_bytes(b"src")
    fake = _FakeMeta(video_keys=["observation.images.cam"], image_keys=[], video_path=src)
    fake.episodes[0]["videos/observation.images.cam/from_timestamp"] = 1.5
    fake.episodes[0]["videos/observation.images.cam/to_timestamp"] = 4.0
    import lerobot.datasets.dataset_metadata as meta_mod

    monkeypatch.setattr(meta_mod, "LeRobotDatasetMetadata", lambda *a, **k: fake, raising=True)

    captured = {}

    def fake_reencode(
        input_video_path,
        output_video_path,
        camera_encoder=None,
        overwrite=False,
        start_time_s=None,
        end_time_s=None,
    ):
        captured.update(
            src=Path(input_video_path),
            encoder=camera_encoder,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        Path(output_video_path).write_bytes(b"clip")

    monkeypatch.setattr(frames_mod, "reencode_video", fake_reencode, raising=True)
    provider = VideoFrameProvider(root=tmp_path)
    record = SimpleNamespace(episode_index=0, frame_timestamps=[0.0, 1.0])

    out = provider.episode_clip_path(record, tmp_path / "clips")

    assert out == tmp_path / "clips" / "ep_000000.mp4"
    assert captured["src"] == src
    assert captured["start_time_s"] == 1.5
    assert captured["end_time_s"] == 4.0
    # H.264 so the clip is decodable by vllm's libav build (sources are often AV1).
    assert captured["encoder"].vcodec == "h264"


def test_videoframeprovider_serializes_decodes_with_a_lock() -> None:
    """torchcodec's cached per-file decoder is single-threaded; the provider
    must own a dedicated lock that ``_decode`` holds around the decoder call.
    """
    import threading

    lock_field = VideoFrameProvider.__dataclass_fields__.get("_decode_lock")
    assert lock_field is not None
    assert lock_field.default_factory is threading.Lock
