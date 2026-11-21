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

pytestmark = pytest.mark.dataset

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
        # Count is the episode length, not the number of pixels fed to the tracker.
        assert int(data["count"][0]) == num_frames

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
        for _ in range(3):
            frame = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
            frame_queue.put(frame)

        # Signal stop instead of sending sentinel
        stop_event.set()
        encoder_thread.join(timeout=10)
        assert not encoder_thread.is_alive()


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

    def test_graceful_frame_dropping(self, tmp_path):
        """Test that full queue drops frames instead of crashing."""
        video_keys = [f"{OBS_IMAGES}.cam"]
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=self._make_encoder_config(
                vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13
            ),
            queue_maxsize=1,
        )
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


def _record_constant_episode(dataset, num_frames, value):
    for _ in range(num_frames):
        dataset.add_frame(
            {
                "observation.images.cam": np.full((64, 96, 3), value, dtype=np.uint8),
                "action": np.array([value / 255], dtype=np.float32),
                "task": "test task",
            }
        )
    dataset.save_episode()


def test_streaming_aggregate_is_frame_weighted_across_resolutions(tmp_path):
    from lerobot.datasets.compute_stats import aggregate_stats

    key = "observation.images.cam"
    episodes = []
    for width, num_frames, value in [(96, 2, 0), (128, 4, 255)]:
        encoder = StreamingVideoEncoder(
            fps=30,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=18),
        )
        try:
            directory = tmp_path / str(width)
            directory.mkdir()
            encoder.start_episode([key], directory)
            for _ in range(num_frames):
                encoder.feed_frame(key, np.full((64, width, 3), value, dtype=np.uint8))
            path, stats = encoder.finish_episode()[key]
            with av.open(str(path)) as container:
                assert sum(1 for _ in container.decode(video=0)) == num_frames
            episodes.append(
                {
                    key: {
                        stat: values if stat == "count" else values.reshape(-1, 1, 1) / 255
                        for stat, values in stats.items()
                    }
                }
            )
        finally:
            encoder.close()
    combined = aggregate_stats(episodes)[key]
    np.testing.assert_array_equal(combined["count"], [6])
    np.testing.assert_allclose(combined["mean"], 2 / 3, atol=1e-6)
    np.testing.assert_allclose(combined["std"], np.sqrt(2 / 9), atol=1e-6)


@pytest.fixture
def legacy_streaming_dataset(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lerobot.datasets.io_utils import load_stats, write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "legacy"
    dataset = LeRobotDataset.create(
        repo_id="test/legacy",
        fps=30,
        features={
            "observation.images.cam": {
                "dtype": "video",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": (1,), "names": ["j1"]},
        },
        root=root,
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=18),
    )
    _record_constant_episode(dataset, 2, 0)
    dataset.finalize()

    # Reproduce the persisted files written by the old, pixel-count streaming encoder.
    path = next((root / "meta/episodes").glob("*/*.parquet"))
    table = pq.read_table(path)
    count_column = "stats/observation.images.cam/count"
    index = table.schema.get_field_index(count_column)
    table = table.set_column(index, table.schema.field(index), pa.array([[2 * 64 * 96]]))
    pq.write_table(table, path)
    stats = load_stats(root)
    stats["observation.images.cam"]["count"] = np.array([2 * 64 * 96])
    write_stats(stats, root)
    return root


@pytest.mark.parametrize("streaming_encoding", [False, True])
def test_resume_legacy_streaming_stats_use_frame_weights(legacy_streaming_dataset, streaming_encoding):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = legacy_streaming_dataset
    legacy_path = next((root / "meta/episodes").glob("*/*.parquet"))
    legacy_bytes = legacy_path.read_bytes()
    dataset = LeRobotDataset.resume(
        repo_id="test/legacy",
        root=root,
        streaming_encoding=streaming_encoding,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=18),
    )
    np.testing.assert_array_equal(dataset.meta.stats["observation.images.cam"]["count"], [2])
    _record_constant_episode(dataset, 4, 255)
    dataset.finalize()

    stats = LeRobotDatasetMetadata("test/legacy", root=root).stats
    for key in ["observation.images.cam", "action"]:
        np.testing.assert_array_equal(stats[key]["count"], [6])
        np.testing.assert_allclose(stats[key]["mean"], 2 / 3, atol=1e-6)
        np.testing.assert_allclose(stats[key]["std"], np.sqrt(2 / 9), atol=1e-6)
    assert legacy_path.read_bytes() == legacy_bytes


def test_legacy_mixed_image_aggregate_is_rebuilt(legacy_streaming_dataset):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.io_utils import load_stats, write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = legacy_streaming_dataset
    dataset = LeRobotDataset.resume(
        repo_id="test/legacy",
        root=root,
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=18),
    )
    _record_constant_episode(dataset, 4, 255)
    dataset.finalize()
    # Emulate a previously resumed dataset with both legacy pixel counts and frame counts.
    stats = load_stats(root)
    mean = 4 / (2 * 64 * 96 + 4)
    stats["observation.images.cam"]["count"] = np.array([2 * 64 * 96 + 4])
    stats["observation.images.cam"]["mean"] = np.full((3, 1, 1), mean)
    stats["observation.images.cam"]["std"] = np.full((3, 1, 1), np.sqrt(mean * (1 - mean)))
    write_stats(stats, root)
    original = (root / "meta/stats.json").read_bytes()

    restored = LeRobotDatasetMetadata("test/legacy", root=root).stats
    np.testing.assert_array_equal(restored["observation.images.cam"]["count"], [6])
    np.testing.assert_allclose(restored["observation.images.cam"]["mean"], 2 / 3, atol=1e-6)
    np.testing.assert_allclose(restored["observation.images.cam"]["std"], np.sqrt(2 / 9), atol=1e-6)
    for stat in stats["action"]:
        np.testing.assert_array_equal(restored["action"][stat], stats["action"][stat])
    assert (root / "meta/stats.json").read_bytes() == original


def test_edit_legacy_streaming_stats_preserves_frame_counts(legacy_streaming_dataset, tmp_path):
    import pyarrow.parquet as pq

    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = legacy_streaming_dataset
    dataset = LeRobotDataset.resume(
        repo_id="test/legacy",
        root=root,
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=18),
    )
    _record_constant_episode(dataset, 4, 255)
    dataset.finalize()
    source = LeRobotDataset("test/legacy", root=root, video_backend="pyav")
    edited = delete_episodes(source, [1], output_dir=tmp_path / "edited")
    np.testing.assert_array_equal(edited.meta.stats["observation.images.cam"]["count"], [2])
    np.testing.assert_allclose(edited.meta.stats["observation.images.cam"]["mean"], 0)
    for path in (edited.root / "meta/episodes").glob("*/*.parquet"):
        assert pq.read_table(path)["stats/observation.images.cam/count"].to_pylist() == [[2]]


def test_merge_legacy_streaming_stats_preserves_frame_weights(legacy_streaming_dataset, tmp_path):
    import pyarrow.parquet as pq

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = legacy_streaming_dataset
    source = LeRobotDataset("test/legacy", root=root, video_backend="pyav")
    other = LeRobotDataset.create(
        repo_id="test/current",
        fps=30,
        features=source.meta.features,
        root=tmp_path / "current",
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=18),
    )
    _record_constant_episode(other, 4, 255)
    other.finalize()
    other = LeRobotDataset("test/current", root=other.root, video_backend="pyav")
    merged = merge_datasets(
        [source, other],
        output_repo_id="test/merged",
        output_dir=tmp_path / "merged",
        concatenate_videos=False,
        concatenate_data=False,
    )
    np.testing.assert_array_equal(merged.meta.stats["observation.images.cam"]["count"], [6])
    np.testing.assert_allclose(merged.meta.stats["observation.images.cam"]["mean"], 2 / 3, atol=1e-6)
    np.testing.assert_allclose(merged.meta.stats["observation.images.cam"]["std"], np.sqrt(2 / 9), atol=1e-6)
    counts = []
    for path in sorted((merged.root / "meta/episodes").glob("*/*.parquet")):
        counts.extend(pq.read_table(path)["stats/observation.images.cam/count"].to_pylist())
    assert counts == [[2], [4]]


@pytest.fixture
def dropped_streaming_dataset(tmp_path, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id="test/dropped",
        fps=30,
        features={
            "observation.images.cam": {
                "dtype": "video",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": (1,), "names": ["j1"]},
        },
        root=tmp_path / "dropped",
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=18),
    )
    original_put = queue.Queue.put
    submitted = 0

    def put_with_drops(frame_queue, item, *args, **kwargs):
        nonlocal submitted
        if isinstance(item, np.ndarray):
            submitted += 1
            if submitted > 2:
                raise queue.Full
        return original_put(frame_queue, item, *args, **kwargs)

    # Exercise the real queue-full handling deterministically, without timing assumptions.
    with monkeypatch.context() as patch:
        patch.setattr(queue.Queue, "put", put_with_drops)
        _record_constant_episode(dataset, 4, 255)
    dataset.finalize()
    np.testing.assert_array_equal(dataset.meta.stats["observation.images.cam"]["count"], [2])
    assert dataset.meta.total_frames == 4
    with av.open(str(dataset.root / dataset.meta.get_video_file_path(0, "observation.images.cam"))) as video:
        assert sum(1 for _ in video.decode(video=0)) == 2
    return dataset.root


@pytest.mark.parametrize("operation", ["load", "resume", "edit", "merge", "mixed"])
def test_dropped_streaming_frame_counts_are_preserved(
    dropped_streaming_dataset, legacy_streaming_dataset, tmp_path, operation
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    key = "observation.images.cam"
    root = dropped_streaming_dataset
    original_path = next((root / "meta/episodes").glob("*/*.parquet"))
    original = original_path.read_bytes()
    if operation == "load":
        meta = LeRobotDatasetMetadata("test/dropped", root=root)
        np.testing.assert_array_equal(meta.stats[key]["count"], [2])
        np.testing.assert_allclose(meta.stats[key]["mean"], 1)
    elif operation in ("resume", "edit", "mixed"):
        dataset = LeRobotDataset.resume(
            repo_id="test/dropped",
            root=root,
            streaming_encoding=True,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=18),
        )
        _record_constant_episode(dataset, 4, 0)
        dataset.finalize()
        if operation == "mixed":
            # Mix a current dropped-frame count with an old pixel-count episode.
            path = sorted((root / "meta/episodes").glob("*/*.parquet"))[-1]
            table = pq.read_table(path)
            column = "stats/observation.images.cam/count"
            index = table.schema.get_field_index(column)
            table = table.set_column(index, table.schema.field(index), pa.array([[4 * 64 * 96]]))
            pq.write_table(table, path)
        if operation == "edit":
            source = LeRobotDataset("test/dropped", root=root, video_backend="pyav")
            edited = delete_episodes(source, [1], output_dir=tmp_path / "edited_dropped")
            np.testing.assert_array_equal(edited.meta.stats[key]["count"], [2])
            for path in (edited.root / "meta/episodes").glob("*/*.parquet"):
                assert pq.read_table(path)[f"stats/{key}/count"].to_pylist() == [[2]]
        else:
            meta = LeRobotDatasetMetadata("test/dropped", root=root)
            np.testing.assert_array_equal(meta.stats[key]["count"], [6])
            np.testing.assert_allclose(meta.stats[key]["mean"], 1 / 3, atol=1e-6)
    else:
        source = LeRobotDataset("test/dropped", root=root, video_backend="pyav")
        legacy = LeRobotDataset("test/legacy", root=legacy_streaming_dataset, video_backend="pyav")
        merged = merge_datasets(
            [source, legacy],
            output_repo_id="test/merged_dropped",
            output_dir=tmp_path / "merged_dropped",
            concatenate_videos=False,
            concatenate_data=False,
        )
        np.testing.assert_array_equal(merged.meta.stats[key]["count"], [4])
        np.testing.assert_allclose(merged.meta.stats[key]["mean"], 0.5, atol=1e-6)
        counts = []
        for path in sorted((merged.root / "meta/episodes").glob("*/*.parquet")):
            counts.extend(pq.read_table(path)[f"stats/{key}/count"].to_pylist())
        assert counts == [[2], [2]]
    assert original_path.read_bytes() == original


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
