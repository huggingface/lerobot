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
"""Frame-provider sampling, real video decoding, clipping and concurrent access."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.annotations.steerable_pipeline.frames import VideoFrameProvider  # noqa: E402


def test_frame_module_defers_direct_video_imports():
    # Shared dataset/config imports may still load Torch. This narrower contract
    # keeps the frame module itself usable without eagerly importing decoders.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
import importlib
from lerobot.annotations.steerable_pipeline import reader

original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in {"torch", "lerobot.configs", "lerobot.datasets.video_utils"}:
        raise AssertionError(f"Eager frame dependency: {name}")
    return original(name, *args, **kwargs)

builtins.__import__ = guarded
frames = importlib.import_module("lerobot.annotations.steerable_pipeline.frames")
assert frames.null_provider().camera_keys == []
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_timestamp_badges_do_not_encourage_invalid_json_numbers(monkeypatch):
    from PIL import Image, ImageDraw

    from lerobot.annotations.steerable_pipeline.frames import _draw_timestamp_badge

    labels = []
    draw_text = ImageDraw.ImageDraw.text

    def capture_text(self, xy, text, *args, **kwargs):
        labels.append(text)
        return draw_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    _draw_timestamp_badge(Image.new("RGB", (192, 128)), 2.52)
    assert labels == ["2.52s"]


class _FakeMeta:
    """Minimal metadata stub exposing ``video_keys`` / ``camera_keys``."""

    def __init__(self, video_keys: list[str], image_keys: list[str], video_path: Path | None = None) -> None:
        self.video_keys = video_keys
        self.camera_keys = [*video_keys, *image_keys]
        self.depth_keys = []
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


def test_video_for_episode_samples_full_duration(tmp_path, monkeypatch):
    provider = _provider_for_video(tmp_path, tmp_path / "video.mp4", monkeypatch)
    # Timestamp sentinels make the requested sampling grid independently observable.
    monkeypatch.setattr(provider, "_decode", lambda ep, timestamps, camera: timestamps)
    record = SimpleNamespace(episode_index=0, frame_timestamps=[i / 10 for i in range(29)])
    assert provider.video_for_episode(record, 3) == [0.0, 1.4, 2.8]
    assert provider.video_for_episode(record, 1) == [0.0]
    assert provider.video_for_episode(record, 0) == []
    assert provider.video_for_episode(record, 100) == record.frame_timestamps


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
        video_encoder=None,
        overwrite=False,
        start_time_s=None,
        end_time_s=None,
    ):
        captured.update(
            src=Path(input_video_path),
            encoder=video_encoder,
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


def test_concurrent_requests_serialize_decode_and_preserve_cache(tmp_path, monkeypatch):
    from lerobot.annotations.steerable_pipeline import frames

    provider = _provider_for_video(tmp_path, tmp_path / "video.mp4", monkeypatch)
    record = SimpleNamespace(episode_index=0, frame_timestamps=[0.0, 1.0])
    started, second_ready, release, overlap = (threading.Event() for _ in range(4))
    calls = []

    def decode(path, timestamps, *args, **kwargs):
        calls.append(timestamps)
        if timestamps == [0.0]:
            started.set()
            assert release.wait(5)
        elif not release.is_set():
            overlap.set()
        return torch.stack([torch.full((3, 2, 2), int(t), dtype=torch.uint8) for t in timestamps])

    def second_request():
        second_ready.set()
        return provider.frames_at(record, [1.0])

    monkeypatch.setattr(frames, "decode_video_frames", decode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(provider.frames_at, record, [0.0])
        try:
            assert started.wait(5)
            second = pool.submit(second_request)
            assert second_ready.wait(5)
            assert not overlap.wait(0.2), "The same cached decoder was entered concurrently"
        finally:
            release.set()
        assert first.result(timeout=5)[0].eq(0).all()
        assert second.result(timeout=5)[0].eq(1).all()
    cached = provider.frames_at(record, [1.0, 0.0, 1.0])
    assert [frame[0, 0, 0].item() for frame in cached] == [1, 0, 1]
    assert calls == [[0.0], [1.0]]
