"""Tests for StreamingVideoEncoder, _CameraEncoder, and parallel speedup helpers."""

from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
from PIL import Image

from lerobot.datasets.compute_stats import (
    RunningQuantileStats,
    _load_single_image,
    auto_downsample_height_width,
    compute_episode_stats,
    get_feature_stats,
    sample_images,
)
from lerobot.datasets.video_utils import StreamingVideoEncoder, _CameraEncoder

RNG = np.random.default_rng(42)


def _make_random_frames(n=10, h=64, w=64):
    """Return list of HWC uint8 numpy frames."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _count_video_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


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


# test: streaming vs batch stats (equivalence)

STAT_KEYS = ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]


def _streaming_image_stats(frames_chw: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Mimic the streaming path: _feed_streaming_frame → save_episode."""
    rs = RunningQuantileStats()
    for img_chw in frames_chw:
        img_ds = auto_downsample_height_width(img_chw)
        c = img_ds.shape[0]
        rs.update(img_ds.transpose(1, 2, 0).reshape(-1, c).astype(np.float64))
    stats = rs.get_statistics()
    return {k: v if k == "count" else (v.reshape(-1, 1, 1) / 255.0) for k, v in stats.items()}


def _batch_image_stats(frames_chw: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Mimic the batch path: sample_images → get_feature_stats."""
    ds = [auto_downsample_height_width(f) for f in frames_chw]
    stacked = np.stack(ds)  # (N, C, H, W)
    stats = get_feature_stats(stacked, axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def _assert_image_stats_close(streaming, batch, atol=1e-6):
    for key in STAT_KEYS:
        np.testing.assert_allclose(streaming[key], batch[key], atol=atol, err_msg=f"Mismatch on '{key}'")


class TestStreamingVsBatchStats:
    """Verify streaming (per-frame) and batch (stacked) image stats match."""

    def test_small_images_no_downsample(self):
        frames = [RNG.integers(0, 255, (3, 64, 64), dtype=np.uint8) for _ in range(30)]
        _assert_image_stats_close(_streaming_image_stats(frames), _batch_image_stats(frames))

    def test_large_images_with_downsample(self):
        frames = [RNG.integers(0, 255, (3, 480, 640), dtype=np.uint8) for _ in range(20)]
        _assert_image_stats_close(_streaming_image_stats(frames), _batch_image_stats(frames))

    def test_single_color_images(self):
        frames = [np.full((3, 64, 64), c, dtype=np.uint8) for c in [0, 128, 255]]
        _assert_image_stats_close(_streaming_image_stats(frames), _batch_image_stats(frames))

    def test_gradient_images(self):
        grad = np.tile(np.arange(256, dtype=np.uint8), (3, 1, 1))  # (3, 1, 256)
        _assert_image_stats_close(_streaming_image_stats([grad] * 10), _batch_image_stats([grad] * 10))

    def test_output_shape_and_keys(self):
        frames = [RNG.integers(0, 255, (3, 64, 64), dtype=np.uint8) for _ in range(5)]
        for stats in [_streaming_image_stats(frames), _batch_image_stats(frames)]:
            for key in STAT_KEYS:
                assert stats[key].shape == (3, 1, 1), f"{key} shape {stats[key].shape}"
            assert stats["count"].shape == (1,)

    def test_nonzero_std_for_random_images(self):
        frames = [RNG.integers(0, 255, (3, 64, 64), dtype=np.uint8) for _ in range(10)]
        for stats in [_streaming_image_stats(frames), _batch_image_stats(frames)]:
            assert np.all(stats["std"] > 0.1), "std should be non-trivial for random uint8 images"

    def test_count_is_total_pixels(self):
        h, w, n = 64, 64, 5
        frames = [RNG.integers(0, 255, (3, h, w), dtype=np.uint8) for _ in range(n)]
        assert _streaming_image_stats(frames)["count"].item() == n * h * w

    def test_min_max_range(self):
        frames = [np.full((3, 8, 8), 100, dtype=np.uint8), np.full((3, 8, 8), 200, dtype=np.uint8)]
        stats = _streaming_image_stats(frames)
        np.testing.assert_allclose(stats["min"], np.full((3, 1, 1), 100.0 / 255))
        np.testing.assert_allclose(stats["max"], np.full((3, 1, 1), 200.0 / 255))

    def test_quantile_ordering(self):
        frames = [RNG.integers(0, 255, (3, 64, 64), dtype=np.uint8) for _ in range(20)]
        stats = _streaming_image_stats(frames)
        assert np.all(stats["q01"] <= stats["q10"])
        assert np.all(stats["q10"] <= stats["q50"])
        assert np.all(stats["q50"] <= stats["q90"])
        assert np.all(stats["q90"] <= stats["q99"])
        assert np.all(stats["min"] <= stats["q01"])
        assert np.all(stats["q99"] <= stats["max"])
