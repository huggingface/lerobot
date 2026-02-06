"""Tests for StreamingVideoEncoder, _CameraEncoder, and parallel speedup helpers."""

from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
from PIL import Image

from lerobot.datasets.compute_stats import (
    _load_single_image,
    auto_downsample_height_width,
    compute_episode_stats,
    sample_images,
)
from lerobot.datasets.video_utils import StreamingVideoEncoder, _CameraEncoder


def _make_random_frames(n=10, h=64, w=64):
    """Return list of HWC uint8 numpy frames."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _count_video_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


# ---------- _CameraEncoder ----------


class TestCameraEncoder:
    def test_encodes_numpy_hwc(self, tmp_path):
        path = tmp_path / "cam.mp4"
        enc = _CameraEncoder(path, fps=10, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30)
        enc.start()
        for frame in _make_random_frames(5):
            enc.queue.put(frame)
        enc.finish()

        assert path.exists()
        assert _count_video_frames(path) == 5

    def test_encodes_numpy_chw(self, tmp_path):
        path = tmp_path / "cam.mp4"
        enc = _CameraEncoder(path, fps=10, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30)
        enc.start()
        for frame in _make_random_frames(5):
            enc.queue.put(frame.transpose(2, 0, 1))  # CHW
        enc.finish()

        assert path.exists()
        assert _count_video_frames(path) == 5

    def test_encodes_pil_images(self, tmp_path):
        path = tmp_path / "cam.mp4"
        enc = _CameraEncoder(path, fps=10, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30)
        enc.start()
        for frame in _make_random_frames(5):
            enc.queue.put(Image.fromarray(frame))
        enc.finish()

        assert path.exists()
        assert _count_video_frames(path) == 5

    def test_cancel_removes_artifacts(self, tmp_path):
        d = tmp_path / "sub"
        d.mkdir()
        path = d / "cam.mp4"
        enc = _CameraEncoder(path, fps=10, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30)
        enc.start()
        for frame in _make_random_frames(3):
            enc.queue.put(frame)
        enc.cancel()

        assert not d.exists()


# ---------- StreamingVideoEncoder ----------


class TestStreamingVideoEncoder:
    def test_single_camera_episode(self, tmp_path):
        encoder = StreamingVideoEncoder(fps=10)
        encoder.start_episode(["cam0"], tmp_path)
        for frame in _make_random_frames(8):
            encoder.feed_frame("cam0", frame)
        paths = encoder.finish_episode()

        assert "cam0" in paths
        assert paths["cam0"].exists()
        assert _count_video_frames(paths["cam0"]) == 8

    def test_multi_camera_episode(self, tmp_path):
        encoder = StreamingVideoEncoder(fps=10)
        keys = ["cam_left", "cam_right"]
        encoder.start_episode(keys, tmp_path)
        for frame in _make_random_frames(6):
            for key in keys:
                encoder.feed_frame(key, frame)
        paths = encoder.finish_episode()

        for key in keys:
            assert key in paths
            assert _count_video_frames(paths[key]) == 6

    def test_stop_episode_cancels_encoding(self, tmp_path):
        encoder = StreamingVideoEncoder(fps=10)
        encoder.start_episode(["cam0"], tmp_path)
        for frame in _make_random_frames(3):
            encoder.feed_frame("cam0", frame)
        encoder.stop_episode()

        assert len(encoder._encoders) == 0

    def test_feed_frame_ignores_unknown_key(self, tmp_path):
        encoder = StreamingVideoEncoder(fps=10)
        encoder.start_episode(["cam0"], tmp_path)
        encoder.feed_frame("nonexistent", _make_random_frames(1)[0])
        paths = encoder.finish_episode()
        assert "nonexistent" not in paths

    def test_close_cleans_up(self, tmp_path):
        encoder = StreamingVideoEncoder(fps=10)
        encoder.start_episode(["cam0"], tmp_path)
        encoder.feed_frame("cam0", _make_random_frames(1)[0])
        encoder.close()
        assert len(encoder._encoders) == 0

    def test_sequential_episodes(self, tmp_path):
        encoder = StreamingVideoEncoder(fps=10)
        for _ep in range(3):
            encoder.start_episode(["cam0"], tmp_path)
            for frame in _make_random_frames(4):
                encoder.feed_frame("cam0", frame)
            paths = encoder.finish_episode()
            assert _count_video_frames(paths["cam0"]) == 4
        encoder.close()


# ---------- auto_downsample_height_width ----------


class TestAutoDownsample:
    def test_small_image_unchanged(self):
        img = np.zeros((3, 100, 100), dtype=np.uint8)
        result = auto_downsample_height_width(img)
        assert result.shape == (3, 100, 100)

    def test_large_image_downsampled(self):
        img = np.zeros((3, 480, 640), dtype=np.uint8)
        result = auto_downsample_height_width(img)
        assert result.shape[1] < 480
        assert result.shape[2] < 640


# ---------- _load_single_image (parallel helper) ----------


def test_load_single_image(tmp_path):
    img_path = tmp_path / "test.png"
    Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(img_path)
    result = _load_single_image(str(img_path))
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[0] == 3  # channel first
    assert result.shape[1] < 480  # downsampled


def _mock_load_image_as_numpy(path, dtype, channel_first):
    return np.ones((3, 32, 32), dtype=dtype) if channel_first else np.ones((32, 32, 3), dtype=dtype)


def test_sample_images_parallel():
    paths = [f"img_{i}.jpg" for i in range(50)]
    with patch("lerobot.datasets.compute_stats.load_image_as_numpy", side_effect=_mock_load_image_as_numpy):
        images = sample_images(paths)
    assert isinstance(images, np.ndarray)
    assert images.shape[1:] == (3, 32, 32)
    assert images.dtype == np.uint8


# ---------- parallel compute_episode_stats ----------


def test_compute_episode_stats_parallel_image_features():
    """Image features should be computed in parallel via ThreadPoolExecutor."""
    episode_data = {
        "observation.cam0": [f"img_{i}.jpg" for i in range(30)],
        "observation.cam1": [f"img2_{i}.jpg" for i in range(30)],
        "action": np.random.normal(0, 1, (30, 6)),
    }
    features = {
        "observation.cam0": {"dtype": "image"},
        "observation.cam1": {"dtype": "image"},
        "action": {"dtype": "float32", "shape": (6,)},
    }

    with patch("lerobot.datasets.compute_stats.load_image_as_numpy", side_effect=_mock_load_image_as_numpy):
        stats = compute_episode_stats(episode_data, features)

    for cam in ["observation.cam0", "observation.cam1"]:
        assert cam in stats
        assert stats[cam]["mean"].shape == (3, 1, 1)
    assert "action" in stats
    assert stats["action"]["mean"].shape == (6,)
