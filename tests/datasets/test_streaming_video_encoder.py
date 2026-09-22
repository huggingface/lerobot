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

"""Tests for streaming video encoding."""

import logging
import queue
import threading

import numpy as np
import pytest

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

import av  # noqa: E402

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.pyav_utils import get_codec
from lerobot.datasets.video_utils import (
    StreamingVideoEncoder,
    _CameraEncoderThread,
)
from lerobot.utils.constants import OBS_IMAGES

# Cross-codec validation tests only fire when the target codec is present
# in the local FFmpeg build; on other platforms validate() is a no-op.
_has_videotoolbox = get_codec("h264_videotoolbox") is not None
_videotoolbox_only = pytest.mark.skipif(
    not _has_videotoolbox, reason="h264_videotoolbox not in local FFmpeg build"
)


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

        enc_cfg = RGBEncoderConfig(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13)
        encoder_thread = _CameraEncoderThread(
            video_path=video_path,
            fps=fps,
            video_encoder=enc_cfg,
            frame_queue=frame_queue,
            result_queue=result_queue,
            stop_event=stop_event,
        )
        encoder_thread.start()

        # Feed frames (HWC uint8)
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            frame_queue.put((i, frame))

        # Send sentinel
        frame_queue.put((num_frames, None))
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

        enc_cfg = RGBEncoderConfig(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13)
        encoder_thread = _CameraEncoderThread(
            video_path=video_path,
            fps=fps,
            video_encoder=enc_cfg,
            frame_queue=frame_queue,
            result_queue=result_queue,
            stop_event=stop_event,
        )
        encoder_thread.start()

        # Feed CHW frames
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (3, 64, 96), dtype=np.uint8)
            frame_queue.put((i, frame))

        frame_queue.put((num_frames, None))
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

        enc_cfg = RGBEncoderConfig(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13)
        encoder_thread = _CameraEncoderThread(
            video_path=video_path,
            fps=fps,
            video_encoder=enc_cfg,
            frame_queue=frame_queue,
            result_queue=result_queue,
            stop_event=stop_event,
        )
        encoder_thread.start()

        # Feed a few frames
        for i in range(3):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            frame_queue.put((i, frame))

        # Signal stop instead of sending sentinel
        stop_event.set()
        encoder_thread.join(timeout=10)
        assert not encoder_thread.is_alive()

    def test_fills_index_gaps_by_repeating_frames(self, tmp_path):
        """Skipped frame indices (queue back-pressure) are filled so the video stays aligned."""
        fps = 30
        video_path = tmp_path / "test_gaps" / "test.mp4"

        frame_queue: queue.Queue = queue.Queue(maxsize=60)
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        # High quality so solid-colour frames decode close to their source value.
        enc_cfg = RGBEncoderConfig(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=10, preset=13)
        encoder_thread = _CameraEncoderThread(
            video_path=video_path,
            fps=fps,
            video_encoder=enc_cfg,
            frame_queue=frame_queue,
            result_queue=result_queue,
            stop_event=stop_event,
        )
        encoder_thread.start()

        # Solid-colour frames make it trivial to identify which source frame filled each slot.
        def solid(v):
            return np.full((64, 96, 3), v, dtype=np.uint8)

        # Recorded 8 frames; indices 1, 4, 5 and the trailing 7 never reached the queue.
        frame_queue.put((0, solid(20)))
        frame_queue.put((2, solid(80)))
        frame_queue.put((3, solid(140)))
        frame_queue.put((6, solid(200)))
        frame_queue.put((8, None))
        encoder_thread.join(timeout=60)
        assert not encoder_thread.is_alive()

        status, stats = result_queue.get(timeout=5)
        assert status == "ok"

        expected = [20, 20, 80, 140, 140, 140, 200, 200]
        with av.open(str(video_path)) as container:
            decoded = [f.to_ndarray(format="rgb24").mean() for f in container.decode(video=0)]
        assert len(decoded) == len(expected)
        np.testing.assert_allclose(decoded, expected, atol=8)

    def test_repeated_frame_reuses_cached_conversion(self, tmp_path, monkeypatch):
        """Encoding the same array again reuses the cached av.VideoFrame instead of re-converting."""
        import lerobot.datasets.video_utils as vu

        thread = _CameraEncoderThread(
            video_path=tmp_path / "x.mp4",
            fps=30,
            video_encoder=RGBEncoderConfig(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13),
            frame_queue=queue.Queue(),
            result_queue=queue.Queue(),
            stop_event=threading.Event(),
        )

        conversions = 0
        real_fromarray = vu.Image.fromarray

        def counting_fromarray(arr, *args, **kwargs):
            nonlocal conversions
            conversions += 1
            return real_fromarray(arr, *args, **kwargs)

        monkeypatch.setattr(vu.Image, "fromarray", counting_fromarray)

        encoded = []

        class FakeStream:
            def encode(self, frame):
                encoded.append((frame, frame.pts))
                return []

        class FakeStats:
            def __init__(self):
                self.n = 0

            def update(self, row):
                self.n += 1

        stream, stats = FakeStream(), FakeStats()
        frame = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        other = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)

        pts = thread._encode(frame, 0, None, stream, stats)  # builds + caches
        pts = thread._encode(frame, pts, None, stream, stats)  # reuse cache
        pts = thread._encode(frame, pts, None, stream, stats)  # reuse cache
        thread._encode(other, pts, None, stream, stats)  # different array -> rebuild

        assert conversions == 2  # once for `frame`, once for `other`
        assert encoded[0][0] is encoded[1][0] is encoded[2][0]  # same cached VideoFrame reused
        assert [pts for _, pts in encoded] == [0, 1, 2, 3]  # each repeat still gets its own pts
        assert stats.n == 2  # only the two distinct frames fold into the stats; repeats are excluded

    def test_gap_before_first_frame_is_filled_forward(self, tmp_path):
        """A gap before the first delivered frame is filled with that frame."""
        fps = 30
        video_path = tmp_path / "test_leading_gap" / "test.mp4"

        frame_queue: queue.Queue = queue.Queue(maxsize=60)
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        enc_cfg = RGBEncoderConfig(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13)
        encoder_thread = _CameraEncoderThread(
            video_path=video_path,
            fps=fps,
            video_encoder=enc_cfg,
            frame_queue=frame_queue,
            result_queue=result_queue,
            stop_event=stop_event,
        )
        encoder_thread.start()

        frame_queue.put((2, np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)))
        frame_queue.put((3, np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)))
        frame_queue.put((4, None))
        encoder_thread.join(timeout=60)

        status, _ = result_queue.get(timeout=5)
        assert status == "ok"
        with av.open(str(video_path)) as container:
            total_frames = sum(1 for _ in container.decode(video=0))
        assert total_frames == 4


# ─── StreamingVideoEncoder tests ───


class TestStreamingVideoEncoder:
    def _make_encoder_config(self, **kwargs):
        """Helper to build an RGBEncoderConfig."""
        return RGBEncoderConfig(**kwargs)

    def test_single_camera_episode(self, tmp_path):
        """Test encoding a single camera episode."""
        video_keys = [f"{OBS_IMAGES}.laptop"]
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=self._make_encoder_config(
                vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13
            ),
        )

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
        video_keys = [f"{OBS_IMAGES}.laptop", f"{OBS_IMAGES}.phone"]
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=self._make_encoder_config(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30),
        )
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
        video_keys = [f"{OBS_IMAGES}.cam"]
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=self._make_encoder_config(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30),
        )

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
        video_keys = [f"{OBS_IMAGES}.cam"]
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=self._make_encoder_config(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30),
        )

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
        encoder = StreamingVideoEncoder(fps=30)
        with pytest.raises(RuntimeError, match="No active episode"):
            encoder.feed_frame("cam", np.zeros((64, 96, 3), dtype=np.uint8))
        encoder.close()

    def test_finish_without_start_raises(self, tmp_path):
        """Test that finishing without starting raises."""
        encoder = StreamingVideoEncoder(fps=30)
        with pytest.raises(RuntimeError, match="No active episode"):
            encoder.finish_episode()
        encoder.close()

    def test_close_is_idempotent(self, tmp_path):
        """Test that close() can be called multiple times safely."""
        encoder = StreamingVideoEncoder(fps=30)
        encoder.close()
        encoder.close()  # Should not raise

    def test_video_duration_matches_frame_count(self, tmp_path):
        """Test that encoded video duration matches num_frames / fps."""
        video_keys = [f"{OBS_IMAGES}.cam"]
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=self._make_encoder_config(
                vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13
            ),
        )
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
        video_keys = [f"{OBS_IMAGES}.cam1", f"{OBS_IMAGES}.cam2"]
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=self._make_encoder_config(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30),
        )
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
        video_keys = [f"{OBS_IMAGES}.cam"]
        cfg = RGBEncoderConfig(
            vcodec="libsvtav1",
            pix_fmt="yuv420p",
            g=2,
            crf=30,
        )
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=cfg,
            encoder_threads=2,
        )
        assert encoder._encoder_threads == 2
        encoder.start_episode(video_keys, tmp_path)

        # Verify codec options include thread tuning for libsvtav1 (lp=…)
        thread = encoder._threads[f"{OBS_IMAGES}.cam"]
        codec_opts = thread.video_encoder.get_codec_options(encoder_threads=thread.encoder_threads)
        assert "svtav1-params" in codec_opts or "threads" in codec_opts

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
        encoder = StreamingVideoEncoder(fps=30)
        assert encoder._encoder_threads is None
        encoder.close()

    def test_tiny_queue_never_shortens_video(self, tmp_path):
        """A full queue never crashes and never shortens the video: rejected frames are repeated."""
        key = f"{OBS_IMAGES}.cam"
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=self._make_encoder_config(
                vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13
            ),
            queue_maxsize=1,
            max_repeated_frames=0,  # disable the back-pressure guard; this test stresses the queue
        )
        encoder.start_episode([key], tmp_path)

        # Feed many frames quickly - with queue_maxsize=1, some will be rejected by the queue
        num_frames = 50
        for _ in range(num_frames):
            encoder.feed_frame(key, np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8))

        results = encoder.finish_episode()
        mp4_path, stats = results[key]
        assert mp4_path.exists()

        # Whether or not any frame was rejected, the video has exactly one frame per fed frame
        assert encoder._repeated_frames.get(key, 0) >= 0
        with av.open(str(mp4_path)) as container:
            total_frames = sum(1 for _ in container.decode(video=0))
        assert total_frames == num_frames

        encoder.close()

    def test_full_queue_repeats_frame_instead_of_dropping(self, tmp_path, caplog):
        """When the queue rejects a frame, the video still has one frame per fed frame."""
        key = f"{OBS_IMAGES}.cam"
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=self._make_encoder_config(
                vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13
            ),
        )
        encoder.start_episode([key], tmp_path)

        num_frames = 12
        rejected = {4, 5, num_frames - 1}
        frame_queue = encoder._frame_queues[key]
        real_put = frame_queue.put

        def flaky_put(item, *args, **kwargs):
            if item[0] in rejected:
                raise queue.Full
            return real_put(item, *args, **kwargs)

        frame_queue.put = flaky_put
        with caplog.at_level(logging.WARNING):
            for _ in range(num_frames):
                encoder.feed_frame(key, np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8))
        frame_queue.put = real_put

        assert "repeated the previous frame" in caplog.text

        results = encoder.finish_episode()
        mp4_path, stats = results[key]
        with av.open(str(mp4_path)) as container:
            total_frames = sum(1 for _ in container.decode(video=0))
        assert total_frames == num_frames

        encoder.close()

    def test_raises_after_max_consecutive_repeats(self, tmp_path):
        """feed_frame raises once too many frames are repeated back-to-back, and the counter resets on success."""
        key = f"{OBS_IMAGES}.cam"
        max_repeated = 5
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=self._make_encoder_config(
                vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13
            ),
            max_repeated_frames=max_repeated,
        )
        encoder.start_episode([key], tmp_path)

        frame_queue = encoder._frame_queues[key]
        real_put = frame_queue.put
        reject = False

        def flaky_put(item, *args, **kwargs):
            if reject:
                raise queue.Full
            return real_put(item, *args, **kwargs)

        frame_queue.put = flaky_put
        frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)

        # A successful frame between two short reject bursts keeps the consecutive counter from tripping.
        reject = True
        for _ in range(max_repeated - 1):
            encoder.feed_frame(key, frame)
        reject = False
        encoder.feed_frame(key, frame)  # resets the consecutive counter
        assert encoder._consecutive_repeats[key] == 0

        # A sustained burst of exactly max_repeated repeats trips the guard.
        reject = True
        with pytest.raises(RuntimeError, match="fell behind"):
            for _ in range(max_repeated):
                encoder.feed_frame(key, frame)

        frame_queue.put = real_put
        encoder.cancel_episode()
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

        assert dataset.writer._streaming_encoder is not None

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

        assert dataset.writer._streaming_encoder is None

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
