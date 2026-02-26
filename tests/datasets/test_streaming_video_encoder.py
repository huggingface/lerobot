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

"""Tests for streaming video encoding and hardware-accelerated encoding."""

import queue
import threading
from unittest.mock import patch

import av
import numpy as np
import pytest

from lerobot.datasets.video_utils import (
    VALID_VIDEO_CODECS,
    StreamingVideoEncoder,
    _CameraEncoderThread,
    _get_codec_options,
    detect_available_hw_encoders,
    resolve_vcodec,
)
from lerobot.utils.constants import OBS_IMAGES

# ─── _get_codec_options tests ───


class TestGetCodecOptions:
    def test_libsvtav1_defaults(self):
        opts = _get_codec_options("libsvtav1")
        assert opts["g"] == "2"
        assert opts["crf"] == "30"
        assert opts["preset"] == "12"

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
        with patch("lerobot.datasets.video_utils.detect_available_hw_encoders", return_value=[]):
            assert resolve_vcodec("auto") == "libsvtav1"

    def test_resolve_vcodec_auto_picks_hw(self):
        """When a HW encoder is available, auto should pick it."""
        with patch(
            "lerobot.datasets.video_utils.detect_available_hw_encoders",
            return_value=["h264_videotoolbox"],
        ):
            assert resolve_vcodec("auto") == "h264_videotoolbox"

    def test_resolve_vcodec_auto_returns_valid(self):
        """Test that resolve_vcodec('auto') returns a known valid codec."""
        result = resolve_vcodec("auto")
        assert result in VALID_VIDEO_CODECS

    def test_hw_encoder_names_accepted_in_validation(self):
        """Test that HW encoder names pass validation in VALID_VIDEO_CODECS."""
        assert "auto" in VALID_VIDEO_CODECS
        assert "h264_videotoolbox" in VALID_VIDEO_CODECS
        assert "h264_nvenc" in VALID_VIDEO_CODECS

    def test_resolve_vcodec_invalid_raises(self):
        """Test that resolve_vcodec raises ValueError for invalid codecs."""
        with pytest.raises(ValueError, match="Invalid vcodec"):
            resolve_vcodec("not_a_real_codec")


# ─── _CameraEncoderThread tests ───


class TestCameraEncoderThread:
    def test_encodes_valid_mp4(self, tmp_path):
        """Test that the encoder thread creates a valid MP4 file with correct frame count."""
        num_frames = 30
        height, width = 64, 96
        fps = 30
        video_path = tmp_path / "test_output" / "test.mp4"

        frame_queue: queue.Queue = queue.Queue(maxsize=60)
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        encoder_thread = _CameraEncoderThread(
            video_path=video_path,
            fps=fps,
            vcodec="libsvtav1",
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            preset=13,
            frame_queue=frame_queue,
            result_queue=result_queue,
            stop_event=stop_event,
        )
        encoder_thread.start()

        # Feed frames (HWC uint8)
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            frame_queue.put(frame)

        # Send sentinel
        frame_queue.put(None)
        encoder_thread.join(timeout=60)
        assert not encoder_thread.is_alive()

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

        frame_queue: queue.Queue = queue.Queue(maxsize=60)
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        encoder_thread = _CameraEncoderThread(
            video_path=video_path,
            fps=fps,
            vcodec="libsvtav1",
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            preset=13,
            frame_queue=frame_queue,
            result_queue=result_queue,
            stop_event=stop_event,
        )
        encoder_thread.start()

        # Feed CHW frames
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (3, 64, 96), dtype=np.uint8)
            frame_queue.put(frame)

        frame_queue.put(None)
        encoder_thread.join(timeout=60)

        status, _ = result_queue.get(timeout=5)
        assert status == "ok"
        assert video_path.exists()

    def test_stop_event_cancellation(self, tmp_path):
        """Test that setting the stop event causes the thread to exit."""
        fps = 30
        video_path = tmp_path / "test_cancel" / "test.mp4"

        frame_queue: queue.Queue = queue.Queue(maxsize=60)
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        encoder_thread = _CameraEncoderThread(
            video_path=video_path,
            fps=fps,
            vcodec="libsvtav1",
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            preset=13,
            frame_queue=frame_queue,
            result_queue=result_queue,
            stop_event=stop_event,
        )
        encoder_thread.start()

        # Feed a few frames
        for _ in range(3):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            frame_queue.put(frame)

        # Signal stop instead of sending sentinel
        stop_event.set()
        encoder_thread.join(timeout=10)
        assert not encoder_thread.is_alive()


# ─── StreamingVideoEncoder tests ───


class TestStreamingVideoEncoder:
    def test_single_camera_episode(self, tmp_path):
        """Test encoding a single camera episode."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13)

        video_keys = [f"{OBS_IMAGES}.laptop"]
        encoder.start_episode(video_keys, tmp_path)

        num_frames = 20
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame(f"{OBS_IMAGES}.laptop", frame)

        results = encoder.finish_episode()
        assert f"{OBS_IMAGES}.laptop" in results

        mp4_path, stats = results[f"{OBS_IMAGES}.laptop"]
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

        video_keys = [f"{OBS_IMAGES}.laptop", f"{OBS_IMAGES}.phone"]
        encoder.start_episode(video_keys, tmp_path)

        num_frames = 15
        for _ in range(num_frames):
            frame0 = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            frame1 = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame(video_keys[0], frame0)
            encoder.feed_frame(video_keys[1], frame1)

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
        video_keys = [f"{OBS_IMAGES}.cam"]

        for ep in range(3):
            encoder.start_episode(video_keys, tmp_path)
            num_frames = 10 + ep * 5
            for _ in range(num_frames):
                frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
                encoder.feed_frame(f"{OBS_IMAGES}.cam", frame)
            results = encoder.finish_episode()

            mp4_path, stats = results[f"{OBS_IMAGES}.cam"]
            assert mp4_path.exists()

            with av.open(str(mp4_path)) as container:
                stream = container.streams.video[0]
                total_frames = sum(1 for _ in container.decode(stream))
            assert total_frames == num_frames

        encoder.close()

    def test_cancel_episode(self, tmp_path):
        """Test that canceling an episode cleans up properly."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30)
        video_keys = [f"{OBS_IMAGES}.cam"]

        encoder.start_episode(video_keys, tmp_path)

        for _ in range(5):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame(f"{OBS_IMAGES}.cam", frame)

        encoder.cancel_episode()

        # Should be able to start a new episode after cancel
        encoder.start_episode(video_keys, tmp_path)
        for _ in range(5):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame(f"{OBS_IMAGES}.cam", frame)
        results = encoder.finish_episode()

        assert f"{OBS_IMAGES}.cam" in results
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

    def test_video_duration_matches_frame_count(self, tmp_path):
        """Test that encoded video duration matches num_frames / fps."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13)
        video_keys = [f"{OBS_IMAGES}.cam"]
        encoder.start_episode(video_keys, tmp_path)

        num_frames = 90  # 3 seconds at 30fps
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame(f"{OBS_IMAGES}.cam", frame)

        results = encoder.finish_episode()
        mp4_path, _ = results[f"{OBS_IMAGES}.cam"]

        expected_duration = num_frames / 30.0  # 3.0 seconds

        with av.open(str(mp4_path)) as container:
            stream = container.streams.video[0]
            total_frames = sum(1 for _ in container.decode(stream))
            if stream.duration is not None:
                actual_duration = float(stream.duration * stream.time_base)
            else:
                actual_duration = float(container.duration / av.time_base)

        assert total_frames == num_frames
        # Allow small tolerance for duration due to codec framing
        assert abs(actual_duration - expected_duration) < 0.5, (
            f"Video duration {actual_duration:.2f}s != expected {expected_duration:.2f}s"
        )

        encoder.close()

    def test_multi_camera_start_episode_called_once(self, tmp_path):
        """Test that with multiple cameras, no frames are lost due to double start_episode."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30)

        video_keys = [f"{OBS_IMAGES}.cam1", f"{OBS_IMAGES}.cam2"]
        encoder.start_episode(video_keys, tmp_path)

        num_frames = 30
        for _ in range(num_frames):
            frame0 = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            frame1 = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame(video_keys[0], frame0)
            encoder.feed_frame(video_keys[1], frame1)

        results = encoder.finish_episode()

        # Both cameras should have all frames
        for key in video_keys:
            mp4_path, stats = results[key]
            assert mp4_path.exists()
            with av.open(str(mp4_path)) as container:
                stream = container.streams.video[0]
                total_frames = sum(1 for _ in container.decode(stream))
            assert total_frames == num_frames, (
                f"Camera {key}: expected {num_frames} frames, got {total_frames}"
            )

        encoder.close()

    def test_encoder_threads_passed_to_thread(self, tmp_path):
        """Test that encoder_threads is stored and passed through to encoder threads."""
        encoder = StreamingVideoEncoder(
            fps=30, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, encoder_threads=2
        )
        assert encoder.encoder_threads == 2

        video_keys = [f"{OBS_IMAGES}.cam"]
        encoder.start_episode(video_keys, tmp_path)

        # Verify the thread received the encoder_threads value
        thread = encoder._threads[f"{OBS_IMAGES}.cam"]
        assert thread.encoder_threads == 2

        # Feed some frames and finish to ensure it works end-to-end
        num_frames = 10
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame(f"{OBS_IMAGES}.cam", frame)

        results = encoder.finish_episode()
        mp4_path, stats = results[f"{OBS_IMAGES}.cam"]
        assert mp4_path.exists()
        assert stats is not None

        with av.open(str(mp4_path)) as container:
            stream = container.streams.video[0]
            total_frames = sum(1 for _ in container.decode(stream))
        assert total_frames == num_frames

        encoder.close()

    def test_encoder_threads_none_by_default(self, tmp_path):
        """Test that encoder_threads defaults to None (codec auto-detect)."""
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1", pix_fmt="yuv420p")
        assert encoder.encoder_threads is None
        encoder.close()

    def test_graceful_frame_dropping(self, tmp_path):
        """Test that full queue drops frames instead of crashing."""
        encoder = StreamingVideoEncoder(
            fps=30, vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13, queue_maxsize=1
        )
        video_keys = [f"{OBS_IMAGES}.cam"]
        encoder.start_episode(video_keys, tmp_path)

        # Feed many frames quickly - with queue_maxsize=1, some will be dropped
        num_frames = 50
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            encoder.feed_frame(f"{OBS_IMAGES}.cam", frame)

        # Should not raise - frames are dropped gracefully
        results = encoder.finish_episode()
        assert f"{OBS_IMAGES}.cam" in results

        mp4_path, _ = results[f"{OBS_IMAGES}.cam"]
        assert mp4_path.exists()

        # Some frames should have been dropped (queue was tiny)
        dropped = encoder._dropped_frames.get(f"{OBS_IMAGES}.cam", 0)
        # We can't guarantee drops but can verify no crash occurred
        assert dropped >= 0

        encoder.close()


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

    def test_multi_camera_add_frame_streaming(self, tmp_path):
        """Test that start_episode is called once with multiple video keys."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            "observation.images.cam1": {
                "dtype": "video",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam2": {
                "dtype": "video",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["j1", "j2"]},
        }

        dataset = LeRobotDataset.create(
            repo_id="test/multi_cam",
            fps=30,
            features=features,
            root=tmp_path / "multi_cam_test",
            use_videos=True,
            streaming_encoding=True,
        )

        num_frames = 15
        for _ in range(num_frames):
            frame = {
                "observation.images.cam1": np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8),
                "observation.images.cam2": np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8),
                "action": np.random.randn(2).astype(np.float32),
                "task": "test task",
            }
            dataset.add_frame(frame)

        dataset.save_episode()

        assert dataset.meta.total_episodes == 1
        assert dataset.meta.total_frames == num_frames

        dataset.finalize()
