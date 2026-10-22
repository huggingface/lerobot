#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.scripts import lerobot_dataset_viz
from lerobot.scripts.lerobot_dataset_viz import _log_video_assets, visualize_dataset

CAMERA = "observation.images.camera"
IMAGE_CAMERA = "observation.images.image"
DEPTH_CAMERA = "observation.images.depth"


@pytest.mark.parametrize("flags,enabled", [([], True), (["--no-display-compressed-videos"], False)])
def test_compressed_video_cli_default_and_opt_out(monkeypatch, flags, enabled):
    calls = []
    dataset = object()
    monkeypatch.setattr(
        "sys.argv", ["lerobot-dataset-viz", "--repo-id", "test/video", "--episode-index", "0", *flags]
    )
    monkeypatch.setattr(lerobot_dataset_viz, "LeRobotDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(
        lerobot_dataset_viz, "visualize_dataset", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    lerobot_dataset_viz.main()
    assert calls[0][0] == (dataset,)
    assert calls[0][1]["display_compressed_videos"] is enabled


@pytest.fixture(scope="module", params=["h264", "libsvtav1"])
def dataset_root(tmp_path_factory, request):
    """Two episodes sharing one real MP4."""
    pytest.importorskip("av")
    root = tmp_path_factory.mktemp("rerun-video") / "dataset"
    dataset = LeRobotDataset.create(
        "test/rerun-video",
        root=root,
        fps=10,
        features={
            CAMERA: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
        },
        rgb_encoder=RGBEncoderConfig(vcodec=request.param),
        video_backend="pyav",
    )
    for episode in range(2):
        for frame in range(3):
            dataset.add_frame(
                {
                    CAMERA: np.full((64, 64, 3), episode * 100 + frame * 20, dtype=np.uint8),
                    "action": np.array([episode * 10 + frame], dtype=np.float32),
                    "observation.state": np.array([episode * 10 + frame + 0.5], dtype=np.float32),
                    "task": f"episode {episode}",
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return root


@pytest.fixture
def dataset(dataset_root):
    return LeRobotDataset("test/rerun-video", root=dataset_root, episodes=[1], video_backend="pyav")


@pytest.fixture
def rerun_logs(monkeypatch):
    """Capture real archetypes and timeline values while keeping Rerun serialization active."""
    rr = pytest.importorskip("rerun")
    calls = []
    times = {}
    original_log, original_set_time = rr.log, rr.set_time

    def log(key, entity, **kwargs):
        calls.append((key, entity, times.copy(), kwargs))
        original_log(key, entity, **kwargs)

    def set_time(timeline, **kwargs):
        times[timeline] = kwargs
        original_set_time(timeline, **kwargs)

    monkeypatch.setattr(rr, "log", log)
    monkeypatch.setattr(rr, "set_time", set_time)
    return rr, calls


@pytest.mark.parametrize("batch_size,num_workers", [(1, 0), (2, 0), (8, 1)])
def test_compressed_video_shared_file_alignment(
    dataset, tmp_path, monkeypatch, rerun_logs, batch_size, num_workers
):
    rr, calls = rerun_logs
    assert dataset.meta.get_video_file_path(0, CAMERA) == dataset.meta.get_video_file_path(1, CAMERA)
    assert dataset.meta.episodes[1][f"videos/{CAMERA}/from_timestamp"] == pytest.approx(0.3)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Video asset display must not decode RGB through the dataset reader")

    monkeypatch.setattr(dataset.reader, "get_items", unexpected_decode)
    rrd_path = visualize_dataset(
        dataset,
        1,
        batch_size=batch_size,
        num_workers=num_workers,
        save=True,
        output_dir=tmp_path,
    )
    assert rrd_path.is_file() and rrd_path.stat().st_size > 0
    assets = [c for c in calls if isinstance(c[1], rr.AssetVideo)]
    assert len(assets) == 1
    assert assets[0][0] == CAMERA and assets[0][3]["static"] is True
    references = [c for c in calls if isinstance(c[1], rr.VideoFrameReference)]
    assert [c[1].timestamp.as_arrow_array()[0].as_py() for c in references] == [
        0,
        100_000_000,
        200_000_000,
    ]
    assert [c[2]["frame_index"]["sequence"] for c in references] == [0, 1, 2]
    assert [c[2]["timestamp"]["timestamp"] for c in references] == pytest.approx([0, 0.1, 0.2])
    assert not any(isinstance(c[1], rr.Image) for c in calls)
    actions = [c for c in calls if c[0] == "action"]
    assert [c[1].scalars.as_arrow_array()[0].as_py() for c in actions] == [10, 11, 12]
    assert [c[2] for c in actions] == [c[2] for c in references]
    tasks = [c for c in calls if c[0] == "task"]
    assert len(tasks) == 1 and tasks[0][1].text.as_arrow_array()[0].as_py() == "### episode 1"


@pytest.mark.parametrize("episode_index", [0, 1])
def test_embedded_video_contains_only_selected_episode(dataset_root, tmp_path, rerun_logs, episode_index):
    av = pytest.importorskip("av")
    rr, calls = rerun_logs
    dataset = LeRobotDataset(
        "test/rerun-video", root=dataset_root, episodes=[episode_index], video_backend="pyav"
    )
    video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, CAMERA)
    original_bytes = video_path.read_bytes()
    with av.open(str(video_path)) as source:
        source_frames = [frame.to_ndarray(format="rgb24") for frame in source.decode(video=0)]

    visualize_dataset(dataset, episode_index, save=True, output_dir=tmp_path)

    asset = next(c[1] for c in calls if isinstance(c[1], rr.AssetVideo))
    clip = bytes(asset.blob.as_arrow_array()[0].as_py())
    with av.open(io.BytesIO(clip)) as video:
        frames = list(video.decode(video=0))
    assert len(frames) == 3
    assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx([0, 0.1, 0.2])
    np.testing.assert_array_equal(
        np.stack([frame.to_ndarray(format="rgb24") for frame in frames]),
        np.stack(source_frames[episode_index * 3 : (episode_index + 1) * 3]),
    )
    assert len(clip) < len(original_bytes)
    assert video_path.read_bytes() == original_bytes


@pytest.mark.parametrize("compressed", [False, True])
def test_individual_frame_display(dataset, tmp_path, rerun_logs, compressed):
    rr, calls = rerun_logs
    visualize_dataset(
        dataset,
        1,
        save=True,
        output_dir=tmp_path,
        display_compressed_videos=False,
        display_compressed_images=compressed,
    )
    assert not any(isinstance(c[1], rr.AssetVideo | rr.VideoFrameReference) for c in calls)
    images = [c for c in calls if c[0] == CAMERA]
    assert len(images) == 3
    assert all(isinstance(c[1], rr.EncodedImage if compressed else rr.Image) for c in images)


def test_unreadable_asset_falls_back_to_real_decoding(dataset, tmp_path, monkeypatch, rerun_logs):
    rr, calls = rerun_logs

    def unreadable(*args, **kwargs):
        raise RuntimeError("Unsupported video asset")

    monkeypatch.setattr(rr.AssetVideo, "read_frame_timestamps_nanos", unreadable)
    visualize_dataset(dataset, 1, save=True, output_dir=tmp_path)
    assert len([c for c in calls if c[0] == CAMERA and isinstance(c[1], rr.Image)]) == 3
    assert not any(isinstance(c[1], rr.AssetVideo | rr.VideoFrameReference) for c in calls)


def test_missing_asset_uses_frame_fallback(dataset, monkeypatch, rerun_logs, caplog):
    monkeypatch.setattr(dataset.meta, "get_video_file_path", lambda *args: Path("missing.mp4"))
    assert _log_video_assets(dataset, 1) == {}
    assert "No local MP4" in caplog.text


def test_image_transforms_are_not_bypassed(dataset, tmp_path, rerun_logs):
    rr, calls = rerun_logs
    dataset.set_image_transforms(lambda image: image * 0)
    visualize_dataset(dataset, 1, save=True, output_dir=tmp_path)
    images = [c[1] for c in calls if c[0] == CAMERA]
    assert len(images) == 3 and all(isinstance(image, rr.Image) for image in images)
    assert all(not any(image.buffer.as_arrow_array()[0].as_py()) for image in images)


@pytest.mark.parametrize("unsupported", ["storage", "delta_timestamps", "episodes"])
def test_unsupported_dataset_uses_reader(dataset, rerun_logs, unsupported, caplog):
    if unsupported == "storage":
        dataset.meta.info.storage_format = "lance"
    elif unsupported == "delta_timestamps":
        dataset.delta_timestamps = {CAMERA: [0.0]}
    else:
        dataset.episodes = [0, 1]
    assert _log_video_assets(dataset, 1) == {}
    assert "Falling back to individual frames" in caplog.text


def test_compressed_video_default_preserves_foxglove(dataset, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "lerobot.utils.foxglove_visualization.serve_foxglove_dataset_playback",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert visualize_dataset(dataset, 1, display_mode="foxglove") is None
    assert calls == [
        ((dataset, 1), {"host": "127.0.0.1", "port": 8765, "compress_images": False, "autoplay": True})
    ]


@pytest.mark.parametrize("include_video", [False, True])
def test_image_and_depth_cameras_keep_frame_rendering(tmp_path, rerun_logs, include_video):
    rr, calls = rerun_logs
    features = {
        IMAGE_CAMERA: {"dtype": "image", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
        DEPTH_CAMERA: {
            "dtype": "image",
            "shape": (32, 32, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True, "depth_unit": "mm"},
        },
    }
    if include_video:
        features[CAMERA] = {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}
    root = tmp_path / "dataset"
    writer = LeRobotDataset.create(
        "test/mixed",
        root=root,
        fps=10,
        features=features,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )
    for _ in range(3):
        frame = {
            IMAGE_CAMERA: np.full((32, 32, 3), 100, dtype=np.uint8),
            DEPTH_CAMERA: np.full((32, 32, 1), 1500, dtype=np.uint16),
            "task": "mixed cameras",
        }
        if include_video:
            frame[CAMERA] = frame[IMAGE_CAMERA].copy()
        writer.add_frame(frame)
    writer.save_episode(parallel_encoding=False)
    writer.finalize()
    dataset = LeRobotDataset(
        "test/mixed", root=root, episodes=[0], video_backend="pyav", depth_output_unit="m"
    )
    visualize_dataset(dataset, 0, save=True, output_dir=tmp_path)
    assert len([c for c in calls if c[0] == IMAGE_CAMERA and isinstance(c[1], rr.Image)]) == 3
    depths = [c[1] for c in calls if c[0] == DEPTH_CAMERA]
    assert len(depths) == 3 and all(isinstance(depth, rr.DepthImage) for depth in depths)
    assert all(depth.meter.as_arrow_array()[0].as_py() == 1 for depth in depths)
    assert all(
        np.all(np.frombuffer(bytes(depth.buffer.as_arrow_array()[0].as_py()), dtype=np.float32) == 1.5)
        for depth in depths
    )
    assert len([c for c in calls if isinstance(c[1], rr.VideoFrameReference)]) == (3 if include_video else 0)
