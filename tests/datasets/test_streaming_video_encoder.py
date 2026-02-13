"""Tests for streaming video encoding and hardware-accelerated encoding."""

import multiprocessing
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest

from lerobot.datasets.video_utils import (
    StreamingVideoEncoder,
    _CameraEncoderProcess,
    _get_codec_options,
    _set_low_priority,
    detect_available_hw_encoders,
    resolve_vcodec,
)

# ─── _get_codec_options tests ───


class TestGetCodecOptions:
    def test_libsvtav1_defaults(self):
        opts = _get_codec_options("libsvtav1")
        assert opts["g"] == "2"
        assert opts["crf"] == "30"
        assert opts["preset"] == "13"  # Changed from 12

    def test_libsvtav1_custom_preset(self):
        opts = _get_codec_options("libsvtav1", preset=8)
        assert opts["preset"] == "8"

    def test_h264_options(self):
        opts = _get_codec_options("h264", g=10, crf=23)
        assert opts["g"] == "10"
        assert opts["crf"] == "23"
        assert "preset" not in opts

    def test_videotoolbox_options(self):
        opts = _get_codec_options("h264_videotoolbox", g=2, crf=30)
        assert opts["g"] == "2"
        # CRF 30 maps to quality = max(1, min(100, 100 - 30*2)) = 40
        assert opts["q:v"] == "40"
        assert "crf" not in opts

    def test_nvenc_options(self):
        opts = _get_codec_options("h264_nvenc", g=2, crf=25)
        assert opts["rc"] == "constqp"
        assert opts["qp"] == "25"
        assert "crf" not in opts
        # NVENC doesn't support g
        assert "g" not in opts

    def test_vaapi_options(self):
        opts = _get_codec_options("h264_vaapi", crf=28)
        assert opts["qp"] == "28"

    def test_qsv_options(self):
        opts = _get_codec_options("h264_qsv", crf=25)
        assert opts["global_quality"] == "25"

    def test_no_g_no_crf(self):
        opts = _get_codec_options("h264", g=None, crf=None)
        assert "g" not in opts
        assert "crf" not in opts


# ─── HW encoder detection tests ───


class TestHWEncoderDetection:
    def test_detect_available_hw_encoders_returns_list(self):
        result = detect_available_hw_encoders()
        assert isinstance(result, list)

    def test_detect_available_hw_encoders_only_valid(self):
        from lerobot.datasets.video_utils import HW_ENCODERS

        result = detect_available_hw_encoders()
        for encoder in result:
            assert encoder in HW_ENCODERS

    def test_resolve_vcodec_passthrough(self):
        assert resolve_vcodec("libsvtav1") == "libsvtav1"
        assert resolve_vcodec("h264") == "h264"

    def test_resolve_vcodec_auto_fallback(self):
        """When no HW encoders are available, auto should fall back to libsvtav1."""
        with patch("lerobot.datasets.video_utils.av.codec.Codec", side_effect=Exception("not found")):
            assert resolve_vcodec("auto") == "libsvtav1"

    def test_resolve_vcodec_auto_picks_hw(self):
        """When a HW encoder is available, auto should pick it."""
        _ = av.codec.Codec

        def mock_codec(name, mode):
            if name == "h264_videotoolbox":
                return MagicMock()
            raise Exception("not found")

        with patch("lerobot.datasets.video_utils.av.codec.Codec", side_effect=mock_codec):
            assert resolve_vcodec("auto") == "h264_videotoolbox"


# ─── _CameraEncoderProcess tests ───


class TestCameraEncoderProcess:
    def test_encodes_valid_mp4(self, tmp_path):
        """Test that the encoder process creates a valid MP4 file with correct frame count."""
        num_frames = 30
        height, width = 64, 96
        fps = 30
        video_path = tmp_path / "test_output" / "test.mp4"

        frame_queue = multiprocessing.Queue(maxsize=60)
        result_queue = multiprocessing.Queue(maxsize=1)

        process = _CameraEncoderProcess(
            video_path=video_path,
            fps=fps,
            vcodec="libsvtav1",
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            preset=13,
            frame_queue=frame_queue,
            result_queue=result_queue,
        )
        process.start()

        # Feed frames (HWC uint8)
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            frame_queue.put(frame)

        # Send sentinel
        frame_queue.put(None)
        process.join(timeout=60)
        assert not process.is_alive()

        # Check result
        status, data = result_queue.get(timeout=5)
        assert status == "ok"
        assert data is not None  # Stats should be returned
        assert "mean" in data
        assert "std" in data
        assert "min" in data
        assert "max" in data
        assert "count" in data

        # Verify the MP4 file is valid
        assert video_path.exists()
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            # The frame count should match
            total_frames = sum(1 for _ in container.decode(stream))
        assert total_frames == num_frames

    def test_handles_chw_input(self, tmp_path):
        """Test that CHW format input is handled correctly."""
        num_frames = 5
        fps = 30
        video_path = tmp_path / "test_chw" / "test.mp4"

        frame_queue = multiprocessing.Queue(maxsize=60)
        result_queue = multiprocessing.Queue(maxsize=1)

        process = _CameraEncoderProcess(
            video_path=video_path,
            fps=fps,
            vcodec="libsvtav1",
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            preset=13,
            frame_queue=frame_queue,
            result_queue=result_queue,
        )
        process.start()

        # Feed CHW frames
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (3, 64, 96), dtype=np.uint8)
            frame_queue.put(frame)

        frame_queue.put(None)
        process.join(timeout=60)

        status, _ = result_queue.get(timeout=5)
        assert status == "ok"
        assert video_path.exists()


# ─── StreamingVideoEncoder tests ───


class TestStreamingVideoEncoder:
    def test_single_camera_episode(self, tmp_path):
        """Test encoding a single camera episode."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13)

        video_keys = ["observation.images.laptop"]
        encoder.start_episode(video_keys, tmp_path)

        num_frames = 20
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame("observation.images.laptop", frame)

        results = encoder.finish_episode()
        assert "observation.images.laptop" in results

        mp4_path, stats = results["observation.images.laptop"]
        assert mp4_path.exists()
        assert stats is not None

        # Verify frame count
        with av.open(str(mp4_path)) as container:
            stream = container.streams.video[0]
            total_frames = sum(1 for _ in container.decode(stream))
        assert total_frames == num_frames

        encoder.close()

    def test_multi_camera_episode(self, tmp_path):
        """Test encoding multiple cameras simultaneously."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30)

        video_keys = ["observation.images.laptop", "observation.images.phone"]
        encoder.start_episode(video_keys, tmp_path)

        num_frames = 15
        for _ in range(num_frames):
            for key in video_keys:
                frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
                encoder.feed_frame(key, frame)

        results = encoder.finish_episode()

        for key in video_keys:
            assert key in results
            mp4_path, stats = results[key]
            assert mp4_path.exists()
            assert stats is not None

        encoder.close()

    def test_sequential_episodes(self, tmp_path):
        """Test that multiple sequential episodes work correctly."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30)
        video_keys = ["observation.images.cam"]

        for ep in range(3):
            encoder.start_episode(video_keys, tmp_path)
            num_frames = 10 + ep * 5
            for _ in range(num_frames):
                frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
                encoder.feed_frame("observation.images.cam", frame)
            results = encoder.finish_episode()

            mp4_path, stats = results["observation.images.cam"]
            assert mp4_path.exists()

            with av.open(str(mp4_path)) as container:
                stream = container.streams.video[0]
                total_frames = sum(1 for _ in container.decode(stream))
            assert total_frames == num_frames

        encoder.close()

    def test_cancel_episode(self, tmp_path):
        """Test that canceling an episode cleans up properly."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30)
        video_keys = ["observation.images.cam"]

        encoder.start_episode(video_keys, tmp_path)

        for _ in range(5):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame("observation.images.cam", frame)

        encoder.cancel_episode()

        # Should be able to start a new episode after cancel
        encoder.start_episode(video_keys, tmp_path)
        for _ in range(5):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame("observation.images.cam", frame)
        results = encoder.finish_episode()

        assert "observation.images.cam" in results
        encoder.close()

    def test_feed_without_start_raises(self, tmp_path):
        """Test that feeding frames without starting an episode raises."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p")
        with pytest.raises(RuntimeError, match="No active episode"):
            encoder.feed_frame("cam", np.zeros((64, 96, 3), dtype=np.uint8))
        encoder.close()

    def test_finish_without_start_raises(self, tmp_path):
        """Test that finishing without starting raises."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p")
        with pytest.raises(RuntimeError, match="No active episode"):
            encoder.finish_episode()
        encoder.close()

    def test_close_is_idempotent(self, tmp_path):
        """Test that close() can be called multiple times safely."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p")
        encoder.close()
        encoder.close()  # Should not raise


# ─── Integration tests with LeRobotDataset ───


class TestStreamingEncoderIntegration:
    def test_add_frame_save_episode_streaming(self, tmp_path):
        """Full integration test: add_frame -> save_episode with streaming encoding."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            "observation.images.cam": {
                "dtype": "video",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": (6,), "names": ["j1", "j2", "j3", "j4", "j5", "j6"]},
        }

        dataset = LeRobotDataset.create(
            repo_id="test/streaming",
            fps=30,
            features=features,
            root=tmp_path / "streaming_test",
            use_videos=True,
            streaming_encoding=True,
        )

        assert dataset._streaming_encoder is not None

        num_frames = 20
        for _ in range(num_frames):
            frame = {
                "observation.images.cam": np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8),
                "action": np.random.randn(6).astype(np.float32),
                "task": "test task",
            }
            dataset.add_frame(frame)

        dataset.save_episode()

        # Verify dataset metadata
        assert dataset.meta.total_episodes == 1
        assert dataset.meta.total_frames == num_frames

        # Verify stats exist for the video key
        assert dataset.meta.stats is not None
        assert "observation.images.cam" in dataset.meta.stats
        assert "action" in dataset.meta.stats

        dataset.finalize()

    def test_streaming_disabled_creates_pngs(self, tmp_path):
        """Test that disabling streaming encoding falls back to PNG path."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            "observation.images.cam": {
                "dtype": "video",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": (6,), "names": ["j1", "j2", "j3", "j4", "j5", "j6"]},
        }

        dataset = LeRobotDataset.create(
            repo_id="test/no_streaming",
            fps=30,
            features=features,
            root=tmp_path / "no_streaming_test",
            use_videos=True,
            streaming_encoding=False,
        )

        assert dataset._streaming_encoder is None

        num_frames = 5
        for _ in range(num_frames):
            frame = {
                "observation.images.cam": np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8),
                "action": np.random.randn(6).astype(np.float32),
                "task": "test task",
            }
            dataset.add_frame(frame)

        # With streaming disabled, PNG files should be written
        images_dir = dataset.root / "images"
        assert images_dir.exists()

        dataset.save_episode()
        dataset.finalize()

    def test_multi_episode_streaming(self, tmp_path):
        """Test recording multiple episodes with streaming encoding."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            "observation.images.cam": {
                "dtype": "video",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["j1", "j2"]},
        }

        dataset = LeRobotDataset.create(
            repo_id="test/multi_ep",
            fps=30,
            features=features,
            root=tmp_path / "multi_ep_test",
            use_videos=True,
            streaming_encoding=True,
        )

        for ep in range(3):
            num_frames = 10 + ep * 5
            for _ in range(num_frames):
                frame = {
                    "observation.images.cam": np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8),
                    "action": np.random.randn(2).astype(np.float32),
                    "task": f"task_{ep}",
                }
                dataset.add_frame(frame)
            dataset.save_episode()

        assert dataset.meta.total_episodes == 3
        assert dataset.meta.total_frames == 10 + 15 + 20

        dataset.finalize()

    def test_clear_episode_buffer_cancels_streaming(self, tmp_path):
        """Test that clearing episode buffer cancels streaming encoding."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            "observation.images.cam": {
                "dtype": "video",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["j1", "j2"]},
        }

        dataset = LeRobotDataset.create(
            repo_id="test/cancel",
            fps=30,
            features=features,
            root=tmp_path / "cancel_test",
            use_videos=True,
            streaming_encoding=True,
        )

        # Add some frames
        for _ in range(5):
            frame = {
                "observation.images.cam": np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8),
                "action": np.random.randn(2).astype(np.float32),
                "task": "task",
            }
            dataset.add_frame(frame)

        # Cancel and re-record
        dataset.clear_episode_buffer()

        # Record a new episode
        for _ in range(10):
            frame = {
                "observation.images.cam": np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8),
                "action": np.random.randn(2).astype(np.float32),
                "task": "task",
            }
            dataset.add_frame(frame)
        dataset.save_episode()

        assert dataset.meta.total_episodes == 1
        assert dataset.meta.total_frames == 10

        dataset.finalize()


# ─── _set_low_priority tests ───


class TestSetLowPriority:
    def test_does_not_raise(self):
        """_set_low_priority should never raise, even if individual calls fail."""
        _set_low_priority()
