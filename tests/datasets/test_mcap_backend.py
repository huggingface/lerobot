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
"""mcap storage backend: LeRobotDataset over an mcap root must return the same
items as over the default parquet/mp4 layout, through the same public class."""

import io
import json
import pickle
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("mcap")

import av
from mcap.writer import Writer
from PIL import Image as PILImage

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mcap_backend import McapDatasetReader, _default_gop_frame_cache_bytes
from lerobot.datasets.mcap_convert import ChannelSpec, convert_lerobot_dataset_to_mcap, ingest_mcap_episodes
from lerobot.datasets.mcap_utils import (
    EPISODE_START_TIME_COLUMN,
    _recover_frames_by_position,
    _strip_sei_nal_units,
    build_gop_time_and_keyframe_index,
    keyframe_index_column,
    log_time_column,
    mux_and_decode_by_position,
)
from lerobot.datasets.video_utils import FrameTimestampError

REPO_ID = "dummy/mcap-test"
FPS = 10


def _build_ground_truth(root: Path, total_episodes: int = 3, use_videos: bool = True) -> LeRobotDataset:
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    if use_videos:
        features["observation.images.cam"] = {
            "dtype": "video",
            "shape": (3, 32, 32),
            "names": ["channels", "height", "width"],
        }
    ds = LeRobotDataset.create(repo_id=REPO_ID, fps=FPS, features=features, root=root, use_videos=use_videos)
    rng = np.random.default_rng(0)
    for ep in range(total_episodes):
        n_frames = 12 + ep
        for _ in range(n_frames):
            frame = {
                "observation.state": torch.from_numpy(rng.random(6, dtype=np.float32)),
                "action": torch.from_numpy(rng.random(6, dtype=np.float32)),
                "task": f"task_{ep % 2}",
            }
            if use_videos:
                frame["observation.images.cam"] = torch.from_numpy(
                    rng.integers(0, 255, size=(3, 32, 32), dtype=np.uint8)
                )
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()
    return LeRobotDataset(REPO_ID, root=root)


@pytest.fixture
def dataset_roots(tmp_path) -> tuple[Path, Path]:
    src_root = tmp_path / "src"
    upstream = _build_ground_truth(src_root, use_videos=False)
    mcap_root = tmp_path / "mcap"
    convert_lerobot_dataset_to_mcap(upstream, mcap_root)
    return src_root, mcap_root


@pytest.fixture
def video_dataset_roots(tmp_path) -> tuple[Path, Path]:
    src_root = tmp_path / "src_video"
    upstream = _build_ground_truth(src_root, total_episodes=2, use_videos=True)
    mcap_root = tmp_path / "mcap_video"
    convert_lerobot_dataset_to_mcap(upstream, mcap_root)
    return src_root, mcap_root


def assert_items_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key, expected_val in expected.items():
        if isinstance(expected_val, str):
            assert actual[key] == expected_val, key
        else:
            assert actual[key].dtype == expected_val.dtype, key
            assert actual[key].shape == expected_val.shape, key
            torch.testing.assert_close(actual[key], expected_val, rtol=0, atol=0, msg=key)


def test_tabular_parity(dataset_roots):
    src_root, mcap_root = dataset_roots
    upstream = LeRobotDataset(REPO_ID, root=src_root)
    mcap_ds = LeRobotDataset(REPO_ID, root=mcap_root)
    assert isinstance(mcap_ds.reader, McapDatasetReader)
    assert mcap_ds.meta.storage_format == "mcap"
    assert len(mcap_ds) == len(upstream)
    for idx in [0, len(upstream) // 2, len(upstream) - 1]:
        assert_items_equal(mcap_ds[idx], upstream[idx])

    # delta windows: first/last frames of an episode exercise clamping + pads
    fps = upstream.meta.fps
    delta_timestamps = {
        "observation.state": [-2 / fps, -1 / fps, 0.0],
        "action": [0.0, 1 / fps, 2 / fps, 3 / fps],
    }
    upstream_d = LeRobotDataset(REPO_ID, root=src_root, delta_timestamps=delta_timestamps)
    mcap_d = LeRobotDataset(REPO_ID, root=mcap_root, delta_timestamps=delta_timestamps)
    ep_start = int(upstream_d.meta.episodes[1]["dataset_from_index"])
    ep_end = int(upstream_d.meta.episodes[1]["dataset_to_index"])
    for idx in [ep_start, ep_start + 5, ep_end - 1]:
        expected = upstream_d[idx]
        actual = mcap_d[idx]
        assert actual["observation.state_is_pad"].tolist() == expected["observation.state_is_pad"].tolist()
        assert_items_equal(actual, expected)

    # batched fetch must equal singles (duplicates and overlapping windows included)
    indices = [3, 3, 17, ep_start, len(upstream) - 1]
    for idx, item in zip(indices, mcap_d.__getitems__(indices), strict=True):
        assert_items_equal(item, mcap_d[idx])

    # negative indexing mirrors upstream; out-of-range raises
    assert_items_equal(mcap_ds[-1], upstream[-1])
    with pytest.raises(IndexError):
        mcap_ds[len(mcap_ds)]

    # episode subset: relative/absolute mapping mirrors upstream
    upstream_s = LeRobotDataset(REPO_ID, root=src_root, episodes=[1])
    mcap_s = LeRobotDataset(REPO_ID, root=mcap_root, episodes=[1])
    assert len(mcap_s) == len(upstream_s)
    assert mcap_s.num_episodes == upstream_s.num_episodes
    assert mcap_s.absolute_to_relative_idx == upstream_s.reader._absolute_to_relative_idx
    for idx in [0, len(upstream_s) - 1]:
        assert_items_equal(mcap_s[idx], upstream_s[idx])


def test_video_parity(video_dataset_roots):
    src_root, mcap_root = video_dataset_roots
    upstream = LeRobotDataset(REPO_ID, root=src_root, video_backend="torchcodec")
    mcap_ds = LeRobotDataset(REPO_ID, root=mcap_root)
    for idx in [0, len(upstream) // 2, len(upstream) - 1]:
        assert_items_equal(mcap_ds[idx], upstream[idx])

    fps = upstream.meta.fps
    video_key = upstream.meta.video_keys[0]
    delta_timestamps = {video_key: [-1 / fps, 0.0, 1 / fps], "action": [0.0, 1 / fps]}
    upstream_d = LeRobotDataset(
        REPO_ID, root=src_root, delta_timestamps=delta_timestamps, video_backend="torchcodec"
    )
    mcap_d = LeRobotDataset(REPO_ID, root=mcap_root, delta_timestamps=delta_timestamps)
    ep_start = int(upstream_d.meta.episodes[1]["dataset_from_index"])
    for idx in [0, ep_start - 1, ep_start]:
        expected = upstream_d[idx]
        actual = mcap_d[idx]
        assert actual[f"{video_key}_is_pad"].tolist() == expected[f"{video_key}_is_pad"].tolist()
        assert_items_equal(actual, expected)

    # one-element window: squeeze semantics must match upstream's shape
    single = {video_key: [0.0]}
    upstream_1 = LeRobotDataset(REPO_ID, root=src_root, delta_timestamps=single, video_backend="torchcodec")
    mcap_1 = LeRobotDataset(REPO_ID, root=mcap_root, delta_timestamps=single)
    for idx in [0, len(upstream_1) - 1]:
        assert_items_equal(mcap_1[idx], upstream_1[idx])

    # return_uint8: raw frames, no normalization
    mcap_u8 = LeRobotDataset(REPO_ID, root=mcap_root, return_uint8=True)
    item = mcap_u8[0]
    assert item[video_key].dtype == torch.uint8


def test_tolerance_violation(video_dataset_roots):
    """Shifting an episode's sync-grid origin away from its real message times
    must raise, not silently return a mismatched frame."""
    _, mcap_root = video_dataset_roots
    info_path = mcap_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    assert info["storage_format"] == "mcap"

    episodes_path = next((mcap_root / "meta" / "episodes").rglob("*.parquet"))
    import pyarrow.parquet as pq

    table = pq.read_table(episodes_path)
    col = table.column(EPISODE_START_TIME_COLUMN).to_pylist()
    # +37ms: off-grid (not a multiple of 1/fps=100ms) so it can't coincidentally
    # land back on a real message a whole number of frames away.
    col[0] += 37_000_000
    table = table.set_column(
        table.schema.get_field_index(EPISODE_START_TIME_COLUMN),
        EPISODE_START_TIME_COLUMN,
        [col],
    )
    pq.write_table(table, episodes_path)

    mcap_ds = LeRobotDataset(REPO_ID, root=mcap_root)
    with pytest.raises(FrameTimestampError):
        mcap_ds[0]


def _drop_column(episodes_path: Path, column: str) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(episodes_path)
    table = table.drop([column])
    pq.write_table(table, episodes_path)


def test_missing_required_column_raises(dataset_roots):
    """A dataset missing one of the base mcap columns (not just a sync-index column)
    must fail with a clear, named error -- not a raw pyarrow KeyError."""
    _, mcap_root = dataset_roots
    episodes_path = next((mcap_root / "meta" / "episodes").rglob("*.parquet"))
    _drop_column(episodes_path, EPISODE_START_TIME_COLUMN)

    with pytest.raises(ValueError, match=EPISODE_START_TIME_COLUMN):
        LeRobotDataset(REPO_ID, root=mcap_root)


def test_missing_sync_index_column_raises(dataset_roots):
    """A dataset missing one feature's mcap/{key}/log_time_ns column (the mcap
    equivalent of lance's VIDEO_INDEX_COLUMNS) must fail with a clear, named error."""
    _, mcap_root = dataset_roots
    episodes_path = next((mcap_root / "meta" / "episodes").rglob("*.parquet"))
    _drop_column(episodes_path, log_time_column("action"))

    with pytest.raises(ValueError, match="sync-index column"):
        LeRobotDataset(REPO_ID, root=mcap_root)


def test_missing_episode_file_raises(dataset_roots):
    """A dataset whose episode metadata references an .mcap file that isn't on disk
    (incomplete download, wrong root, ...) must fail clearly, not with a raw open() error."""
    _, mcap_root = dataset_roots
    mcap_file = next((mcap_root / "data").rglob("*.mcap"))
    mcap_file.unlink()

    with pytest.raises(FileNotFoundError, match="missing on disk"):
        LeRobotDataset(REPO_ID, root=mcap_root)


def test_pickle_and_dataloader(dataset_roots):
    _, mcap_root = dataset_roots
    mcap_ds = LeRobotDataset(REPO_ID, root=mcap_root)
    restored = pickle.loads(pickle.dumps(mcap_ds))
    assert_items_equal(restored[7], mcap_ds[7])

    loader = torch.utils.data.DataLoader(mcap_ds, batch_size=8, num_workers=0, shuffle=True)
    batch = next(iter(loader))
    assert batch["observation.state"].shape[0] == 8
    assert len(batch["task"]) == 8


# ── ingest_mcap_episodes: onboarding raw, independently-recorded mcap files ──────────
#
# Unlike every other fixture in this file, these datasets never pass through a
# LeRobotDataset at all before becoming mcap -- they're written directly with
# mcap.writer.Writer on arbitrary topic names (no topic_for_key convention), with
# channels at genuinely different native rates, irregular jitter, a channel that
# starts late and one that stops early, and a deliberate multi-second dropout. This
# is the actual "any mcap dataset" onboarding path ingest_mcap_episodes exists for.

CHALLENGE_FPS = 20  # below /cam/wrist's ~30Hz and /robot/joint_state's ~100Hz, above /cam/top's ~12Hz
CHALLENGE_TOLERANCE_S = 0.06  # covers normal jitter + up to half of /cam/top's ~83ms period
CHALLENGE_IMG_SIZE = 24
CHALLENGE_DURATIONS = (7.0, 11.0, 9.0)
CHALLENGE_GRIP_LATE_START_S = 0.4  # gripper_cmd comes online 0.4s after the sensors
CHALLENGE_DROPOUT_WINDOW = (  # raw-local seconds into episode 1 where /cam/top is silent
    CHALLENGE_DURATIONS[1] * 0.4,
    CHALLENGE_DURATIONS[1] * 0.4 + 1.2,
)


def _challenge_png(rng: np.random.Generator) -> bytes:
    frame = rng.integers(0, 255, size=(CHALLENGE_IMG_SIZE, CHALLENGE_IMG_SIZE, 3), dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()


def _irregular_jitter(rng: np.random.Generator, n: int, base_jitter_s: float) -> np.ndarray:
    """Mix of small continuous jitter and occasional larger delay spikes -- more
    realistic than uniform jitter (real sensors have bursty scheduling delays)."""
    jitter = rng.normal(0.0, base_jitter_s / 2, n)
    spike_mask = rng.random(n) < 0.05
    jitter[spike_mask] += rng.uniform(base_jitter_s, base_jitter_s * 3, spike_mask.sum())
    return jitter


def _write_raw_episode(path: Path, duration_s: float, ep_idx: int, seed: int) -> None:
    """Write one raw, independently-timestamped mcap recording -- no LeRobot or
    topic_for_key convention involved, exactly what a real logging pipeline produces."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()
        channels = {
            "/cam/wrist": writer.register_channel(topic="/cam/wrist", message_encoding="png", schema_id=0),
            "/cam/top": writer.register_channel(topic="/cam/top", message_encoding="png", schema_id=0),
            "/robot/joint_state": writer.register_channel(
                topic="/robot/joint_state",
                message_encoding="raw",
                schema_id=writer.register_schema(name="joint_state", encoding="jsonschema", data=b"{}"),
            ),
            "/robot/gripper_cmd": writer.register_channel(
                topic="/robot/gripper_cmd",
                message_encoding="raw",
                schema_id=writer.register_schema(name="gripper_cmd", encoding="jsonschema", data=b"{}"),
            ),
        }

        epoch_s = 1.0  # keeps every jittered timestamp comfortably non-negative (mcap log_time is uint64)

        def emit(topic: str, t_s: float, data: bytes) -> None:
            t_ns = round((t_s + epoch_s) * 1e9)
            writer.add_message(channel_id=channels[topic], log_time=t_ns, data=data, publish_time=t_ns)

        # /robot/joint_state: ~100Hz, small jitter, spans the full episode.
        n = int(duration_s * 100)
        times = np.arange(n) / 100 + _irregular_jitter(rng, n, 0.002)
        for t in np.sort(times):
            emit("/robot/joint_state", float(t), rng.random(7).astype(np.float32).tobytes())

        # /robot/gripper_cmd: ~15Hz, starts late (actuator online after the sensors).
        n = int((duration_s - CHALLENGE_GRIP_LATE_START_S) * 15)
        times = CHALLENGE_GRIP_LATE_START_S + np.arange(n) / 15 + _irregular_jitter(rng, n, 0.01)
        for t in np.sort(times):
            emit("/robot/gripper_cmd", float(t), rng.random(1).astype(np.float32).tobytes())

        # /cam/wrist: ~30Hz, small jitter, spans the full episode.
        n = int(duration_s * 30)
        times = np.arange(n) / 30 + _irregular_jitter(rng, n, 0.005)
        for t in np.sort(times):
            emit("/cam/wrist", float(t), _challenge_png(rng))

        # /cam/top: ~12Hz. Episode 1 has a dropout mid-recording (dropped connection);
        # episode 2 stops early (camera disconnects before the rest).
        end = duration_s - 0.3 if ep_idx == 2 else duration_s
        n = int(end * 12)
        times = np.arange(n) / 12 + _irregular_jitter(rng, n, 0.01)
        if ep_idx == 1:
            gap_start, gap_end = CHALLENGE_DROPOUT_WINDOW
            times = times[(times < gap_start) | (times > gap_end)]
        for t in np.sort(times):
            if 0 <= t <= duration_s:
                emit("/cam/top", float(t), _challenge_png(rng))

        writer.finish()


@pytest.fixture
def challenging_ingest_root(tmp_path) -> Path:
    raw_dir = tmp_path / "raw"
    for ep_idx, duration in enumerate(CHALLENGE_DURATIONS):
        _write_raw_episode(raw_dir / f"session_{ep_idx}.mcap", duration, ep_idx, seed=ep_idx)

    channels = [
        ChannelSpec(key="observation.images.wrist", topic="/cam/wrist", kind="video"),
        ChannelSpec(key="observation.images.top", topic="/cam/top", kind="video"),
        ChannelSpec(
            key="observation.state", topic="/robot/joint_state", kind="tabular", dtype="float32", shape=(7,)
        ),
        ChannelSpec(key="action", topic="/robot/gripper_cmd", kind="tabular", dtype="float32", shape=(1,)),
    ]
    dst_root = tmp_path / "ingested_mcap"
    ingest_mcap_episodes(
        mcap_paths=[raw_dir / f"session_{ep_idx}.mcap" for ep_idx in range(len(CHALLENGE_DURATIONS))],
        channels=channels,
        dst_root=dst_root,
        fps=CHALLENGE_FPS,
        robot_type="bench-arm",
        tasks=["pick up the cube", "place the cube", "pick up the cube"],
    )
    return dst_root


def test_ingest_raw_mcap_episodes_dropout_raises(challenging_ingest_root):
    """A real dropout (not matched to any grid frame within tolerance) must raise for
    every grid frame genuinely inside it, and must not raise anywhere else -- the core
    correctness property of onboarding genuinely async, multi-rate, unrelated-topic data."""
    ds = LeRobotDataset("challenge/mcap", root=challenging_ingest_root, tolerance_s=CHALLENGE_TOLERANCE_S)
    assert ds.meta.total_episodes == 3

    ep1 = ds.meta.episodes[1]
    ep1_start, ep1_end = int(ep1["dataset_from_index"]), int(ep1["dataset_to_index"])
    gap_window = CHALLENGE_DROPOUT_WINDOW
    # /cam/top only samples every ~1/12s, so the first/last usable frame after/before the
    # gap can itself be up to one period further from the edge than the dropout's exact
    # bounds -- margin accounts for that natural edge effect, not a reader bug.
    edge_margin_s = 1 / 12
    # Grid frame 0 sits at the INTERSECTION start, i.e. raw-local t=0.4s (gripper_cmd is
    # the last channel to come online) -- shift back to raw-local time to compare.
    grid_origin_raw_local_s = CHALLENGE_GRIP_LATE_START_S

    n_ok = n_gap_raised = n_edge = 0
    for abs_idx in range(ep1_start, ep1_end):
        t_s = (abs_idx - ep1_start) / CHALLENGE_FPS + grid_origin_raw_local_s
        in_gap = gap_window[0] < t_s < gap_window[1]
        near_edge = (
            gap_window[0] - edge_margin_s < t_s <= gap_window[0]
            or gap_window[1] <= t_s < gap_window[1] + edge_margin_s
        )
        if near_edge:
            # Either outcome is acceptable right at the boundary; just tally it.
            try:
                ds[abs_idx]
                n_ok += 1
            except FrameTimestampError:
                n_edge += 1
            continue
        try:
            ds[abs_idx]
            assert not in_gap, f"frame at t={t_s:.2f}s inside the dropout did NOT raise"
            n_ok += 1
        except FrameTimestampError:
            assert in_gap, f"frame at t={t_s:.2f}s outside the dropout unexpectedly raised"
            n_gap_raised += 1

    assert n_ok > 0
    assert n_gap_raised > 0


def test_ingest_raw_mcap_episodes_unaffected_episodes_readable(challenging_ingest_root):
    """Episodes with no dropout (just late-starting/early-stopping channels) must read
    every frame without error."""
    ds = LeRobotDataset("challenge/mcap", root=challenging_ingest_root, tolerance_s=CHALLENGE_TOLERANCE_S)
    for ep_idx in (0, 2):
        ep = ds.meta.episodes[ep_idx]
        for abs_idx in range(int(ep["dataset_from_index"]), int(ep["dataset_to_index"])):
            ds[abs_idx]


def test_ingest_raw_mcap_episodes_batched_and_delta_timestamps(challenging_ingest_root):
    """Batched get_items and cross-episode-boundary delta-timestamp windows must work
    on genuinely independent, multi-rate, unrelated-topic-name source data."""
    ds = LeRobotDataset("challenge/mcap", root=challenging_ingest_root, tolerance_s=CHALLENGE_TOLERANCE_S)
    ep1_end = int(ds.meta.episodes[1]["dataset_to_index"])
    delta_timestamps = {
        "observation.state": [-2 / CHALLENGE_FPS, -1 / CHALLENGE_FPS, 0.0],
        "observation.images.wrist": [-1 / CHALLENGE_FPS, 0.0],
    }
    ds_d = LeRobotDataset(
        "challenge/mcap",
        root=challenging_ingest_root,
        delta_timestamps=delta_timestamps,
        tolerance_s=CHALLENGE_TOLERANCE_S,
    )
    batch = ds_d.__getitems__([0, 5, ep1_end - 1, len(ds_d) - 1])
    for item in batch:
        assert item["observation.state"].shape == (3, 7)
        assert item["observation.images.wrist"].shape[1:] == (3, CHALLENGE_IMG_SIZE, CHALLENGE_IMG_SIZE)


# ── GOP-encoded video channels: a continuous H.264 bitstream, not per-message images ──
#
# Real mcap recordings (e.g. foxglove.CompressedVideo) often store video as one
# continuous, GOP-compressed bitstream rather than standalone-decodable PNG/JPEG per
# message: each message is a raw H.264 packet that depends on the preceding keyframe.
# These fixtures encode a genuinely GOP-structured clip (fixed keyframe interval, no
# B-frames) and feed its raw packets into an mcap channel exactly as a real recorder
# would, to exercise ingest_mcap_episodes's and McapDatasetReader's native support for
# that layout -- as opposed to every other fixture in this file, which only ever sees
# independently-decodable image bytes.

GOP_FPS = 20
GOP_SIZE = 8
GOP_FRAME_SIZE = 48
GOP_EPISODE_FRAMES = (40, 24)
GOP_VIDEO_KEY = "observation.images.cam"
GOP_TOPIC = "/cam/gop"


def _encode_gop_h264(
    n_frames: int, fps: int, gop_size: int, seed: int
) -> tuple[list[bytes], list[np.ndarray]]:
    """Encode a synthetic smoothly-varying clip as H.264 with a fixed keyframe interval
    (no B-frames, no scene-cut re-keyframing -- a real closed-GOP stream). Returns the
    raw per-access-unit packet bytes (one per message, in order) and the frames
    (H, W, C uint8) a decoder actually produces for them -- the ground truth the reader
    must reproduce bit-for-bit, since H.264 decode is deterministic given the same bytes.
    """
    enc_ctx = av.CodecContext.create("h264", "w")
    enc_ctx.width = GOP_FRAME_SIZE
    enc_ctx.height = GOP_FRAME_SIZE
    enc_ctx.pix_fmt = "yuv420p"
    enc_ctx.gop_size = gop_size
    enc_ctx.time_base = Fraction(1, fps)
    enc_ctx.options = {"bf": "0", "x264-params": f"scenecut=0:keyint={gop_size}:min-keyint={gop_size}"}

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(GOP_FRAME_SIZE, GOP_FRAME_SIZE, 3), dtype=np.uint8)
    packets: list[bytes] = []
    for i in range(n_frames):
        source = np.roll(base, (i * 3) % GOP_FRAME_SIZE, axis=1)
        frame = av.VideoFrame.from_ndarray(source, format="rgb24").reformat(format="yuv420p")
        frame.pts = i
        for packet in enc_ctx.encode(frame):
            packets.append(bytes(packet))
    for packet in enc_ctx.encode(None):
        packets.append(bytes(packet))

    dec_ctx = av.CodecContext.create("h264", "r")
    frames = [
        decoded.to_ndarray(format="rgb24") for data in packets for decoded in dec_ctx.decode(av.Packet(data))
    ]
    assert len(frames) == n_frames, "one message must decode to exactly one frame in this test's scope"
    return packets, frames


def _write_raw_gop_mcap(path: Path, packets: list[bytes], fps: int) -> None:
    """Write one raw mcap episode whose only channel is a continuous H.264 bitstream --
    message_encoding names the codec, same convention a real foxglove.CompressedVideo
    recorder's channel would use, but without the protobuf wrapper (out of scope)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()
        channel_id = writer.register_channel(topic=GOP_TOPIC, message_encoding="h264", schema_id=0)
        for i, data in enumerate(packets):
            t_ns = round(i / fps * 1e9) + 1_000_000_000  # comfortably positive, arbitrary origin
            writer.add_message(channel_id=channel_id, log_time=t_ns, data=data, publish_time=t_ns)
        writer.finish()


@pytest.fixture
def gop_ingest_root(tmp_path) -> tuple[Path, list[list[np.ndarray]]]:
    raw_dir = tmp_path / "raw_gop"
    ground_truth: list[list[np.ndarray]] = []
    paths = []
    for ep_idx, n_frames in enumerate(GOP_EPISODE_FRAMES):
        packets, frames = _encode_gop_h264(n_frames, GOP_FPS, GOP_SIZE, seed=ep_idx)
        path = raw_dir / f"ep_{ep_idx}.mcap"
        _write_raw_gop_mcap(path, packets, GOP_FPS)
        paths.append(path)
        ground_truth.append(frames)

    dst_root = tmp_path / "gop_mcap"
    ingest_mcap_episodes(
        mcap_paths=paths,
        channels=[ChannelSpec(key=GOP_VIDEO_KEY, topic=GOP_TOPIC, kind="video", codec="h264")],
        dst_root=dst_root,
        fps=GOP_FPS,
        robot_type="gop-bench",
        tasks="encode test",
    )
    return dst_root, ground_truth


def test_ingest_gop_video_metadata(gop_ingest_root):
    """A GOP-encoded channel must round-trip its codec and keyframe index through
    meta/info.json and meta/episodes -- the signal the reader keys its decode path on."""
    dst_root, _ = gop_ingest_root
    ds = LeRobotDataset("gop/mcap", root=dst_root)
    assert ds.meta.features[GOP_VIDEO_KEY]["video_codec"] == "h264"
    assert keyframe_index_column(GOP_VIDEO_KEY) in ds.meta.episodes.column_names

    for ep_idx, n_frames in enumerate(GOP_EPISODE_FRAMES):
        kf_positions = ds.meta.episodes[ep_idx][keyframe_index_column(GOP_VIDEO_KEY)]
        expected = list(range(0, n_frames, GOP_SIZE))
        assert kf_positions == expected


def test_ingest_gop_video_random_access_correctness(gop_ingest_root):
    """Decoded frames at scattered/shuffled indices -- not just sequential ones -- must
    exactly match what a plain, from-scratch decode of the same packets produces."""
    dst_root, ground_truth = gop_ingest_root
    ds = LeRobotDataset("gop/mcap", root=dst_root, return_uint8=True)

    rng = np.random.default_rng(0)
    for ep_idx, n_frames in enumerate(GOP_EPISODE_FRAMES):
        ep = ds.meta.episodes[ep_idx]
        start = int(ep["dataset_from_index"])
        order = rng.permutation(n_frames)[: min(12, n_frames)]
        for local_idx in order:
            item = ds[start + int(local_idx)]
            actual = item[GOP_VIDEO_KEY].numpy()
            expected = np.transpose(ground_truth[ep_idx][local_idx], (2, 0, 1))
            np.testing.assert_array_equal(actual, expected)

    # Batched + duplicated + out-of-order indices across both episodes in one call.
    ep0_start = int(ds.meta.episodes[0]["dataset_from_index"])
    ep1_start = int(ds.meta.episodes[1]["dataset_from_index"])
    indices = [ep1_start + 10, ep0_start + 2, ep1_start + 10, ep0_start + 2, ep1_start + 0]
    items = ds.__getitems__(indices)
    for idx, item in zip(indices, items, strict=True):
        assert_items_equal(item, ds[idx])


def test_ingest_gop_video_decode_uses_keyframe_index(gop_ingest_root):
    """Decoding a single late frame must only replay packets from its preceding
    keyframe onward, not the whole episode from the start -- the entire point of
    building/using the keyframe index instead of a from-scratch decode per read."""
    dst_root, _ = gop_ingest_root
    ds = LeRobotDataset("gop/mcap", root=dst_root, return_uint8=True)

    ep_idx = 0
    n_frames = GOP_EPISODE_FRAMES[ep_idx]
    start = int(ds.meta.episodes[ep_idx]["dataset_from_index"])
    target_local_idx = n_frames - 2  # deep into the last GOP, far from episode start
    expected_kf = (target_local_idx // GOP_SIZE) * GOP_SIZE
    expected_messages_decoded = target_local_idx - expected_kf + 1

    with patch("lerobot.datasets.mcap_backend.av.Packet", wraps=av.Packet) as counted_packet:
        ds[start + target_local_idx]
        assert counted_packet.call_count == expected_messages_decoded
        assert counted_packet.call_count < n_frames


def test_ingest_gop_video_decode_caches_across_reads(gop_ingest_root):
    """A repeat read of the same frame -- or of a DIFFERENT frame already covered by a
    previously-decoded window -- must not touch the codec again at all: confirmed on
    real Labelbox/robotics-datasets content that without this cache, a shuffled
    batch_size=1 DataLoader re-decodes the entire remaining episode on every single
    query (a ~390x slowdown against the default format), since a real recording's
    episode is routinely shorter than the encoder's keyframe interval."""
    dst_root, _ = gop_ingest_root
    ds = LeRobotDataset("gop/mcap", root=dst_root, return_uint8=True)

    ep_idx = 0
    n_frames = GOP_EPISODE_FRAMES[ep_idx]
    start = int(ds.meta.episodes[ep_idx]["dataset_from_index"])
    target_local_idx = n_frames - 2  # deep into the last GOP, far from episode start
    expected_kf = (target_local_idx // GOP_SIZE) * GOP_SIZE
    expected_messages_decoded = target_local_idx - expected_kf + 1

    with patch("lerobot.datasets.mcap_backend.av.Packet", wraps=av.Packet) as counted_packet:
        first = ds[start + target_local_idx]
        assert counted_packet.call_count == expected_messages_decoded

        # Exact same frame again: must be a pure cache hit, zero new decode work.
        again = ds[start + target_local_idx]
        assert counted_packet.call_count == expected_messages_decoded
        np.testing.assert_array_equal(first[GOP_VIDEO_KEY].numpy(), again[GOP_VIDEO_KEY].numpy())

        # A DIFFERENT frame, but one already decoded as a byproduct of that same
        # [preceding keyframe, target] window -- also a pure cache hit.
        neighbor = ds[start + target_local_idx - 1]
        assert counted_packet.call_count == expected_messages_decoded
        assert neighbor[GOP_VIDEO_KEY].shape == first[GOP_VIDEO_KEY].shape


def test_default_gop_frame_cache_bytes_sizes_to_largest_episode():
    """The computed default must equal the LARGEST episode's total decoded-frame bytes
    across every GOP key, times the documented margin -- not the sum across all
    episodes (that would multiply badly across DataLoader workers for no correctness
    benefit, see the function's docstring) and not any single key alone (a multi-camera
    episode needs every camera's frames live at once, not just one)."""
    # 2 keys (cameras), 2 episodes; episode 1 is the larger one for cam_a, episode 0 for
    # cam_b -- the combined per-episode total must still pick whichever episode's SUM
    # across both keys is largest, not whichever key's own largest episode is largest.
    log_time_ns = {
        "cam_a": [np.zeros(10, dtype=np.int64), np.zeros(20, dtype=np.int64)],
        "cam_b": [np.zeros(15, dtype=np.int64), np.zeros(5, dtype=np.int64)],
    }
    frame_bytes = {"cam_a": 100, "cam_b": 1000}
    # episode 0: 10*100 + 15*1000 = 16000; episode 1: 20*100 + 5*1000 = 7000 -> max=16000
    result = _default_gop_frame_cache_bytes(log_time_ns, frame_bytes, episode_indices=[0, 1])
    assert result == 16000 * 2  # _GOP_FRAME_CACHE_MARGIN

    # A caller-restricted episode subset (e.g. LeRobotDataset(episodes=[1])) must only
    # size against episode 1, not the whole dataset's largest.
    restricted = _default_gop_frame_cache_bytes(log_time_ns, frame_bytes, episode_indices=[1])
    assert restricted == 7000 * 2

    # No GOP-encoded keys at all -> nothing to size for.
    assert _default_gop_frame_cache_bytes({}, {}, episode_indices=[0, 1]) == 0


# ── Leading parameter-set-only packets (real encoders routinely emit these) ───────────
#
# Real streams (e.g. Labelbox/robotics-datasets' HEVC channels) commonly front-load one
# or more VPS/SPS/PPS-only packets before any picture data: decode correctly produces
# zero frames for those, not an error. build_gop_time_and_keyframe_index must drop such
# messages from its index instead of counting them against "must start with a keyframe".
#
# The two isolated tests below script mux_and_decode_by_position directly -- the single
# seam both build_gop_time_and_keyframe_index and the reader's own GOP decode path call
# through -- to pin down this exact index-building logic deterministically. The
# end-to-end test uses genuinely valid Annex-B parameter-set bytes (SPS/PPS sliced off a
# real encoded keyframe, no slice data) with no mocking at all, since a real container
# decode (confirmed empirically) tolerates a standalone parameter-set packet correctly as
# long as it's well-formed -- unlike a bare one-packet-at-a-time CodecContext.decode(),
# which can raise even on genuinely valid input (see mux_and_decode_by_position's
# docstring).


class _FakeFrame:
    def __init__(self, key_frame: bool, pict_type: object = None, height: int = 48, width: int = 48) -> None:
        self.key_frame = key_frame
        self.pict_type = pict_type if pict_type is not None else av.video.frame.PictureType.I
        self.height = height
        self.width = width


def _write_scripted_messages(path: Path, topic: str, n_messages: int) -> None:
    """Raw message bytes are irrelevant here -- mux_and_decode_by_position is mocked
    out entirely -- so each message is a single placeholder byte, just enough for mcap
    to store it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()
        channel_id = writer.register_channel(topic=topic, message_encoding="h264", schema_id=0)
        for i in range(n_messages):
            t_ns = i * 1_000_000 + 1_000_000_000
            writer.add_message(channel_id=channel_id, log_time=t_ns, data=bytes([i]), publish_time=t_ns)
        writer.finish()


def test_build_gop_index_tolerates_leading_parameter_set_only_messages(tmp_path):
    """Messages that decode to zero frames (parameter-set-only packets) must be
    dropped from the index, not treated as a violation of the keyframe-first
    invariant -- the exact pattern a real Labelbox/robotics-datasets HEVC channel
    produces: messages 0-2 decode to nothing, message 3 is the first real keyframe."""
    path = tmp_path / "leading_params.mcap"
    topic = "/cam/scripted"
    _write_scripted_messages(path, topic, n_messages=6)

    # Positions 0-2 are absent entirely (zero-output messages); 3, 4, 5 decoded.
    scripted = {3: _FakeFrame(True), 4: _FakeFrame(False), 5: _FakeFrame(False)}
    with patch("lerobot.datasets.mcap_utils.mux_and_decode_by_position", return_value=scripted):
        times, kf_positions, has_bframes, _shape = build_gop_time_and_keyframe_index(path, topic, "h264")

    # The three zero-output messages (indices 0-2) never make it into the index at all.
    assert times.tolist() == [1_003_000_000, 1_004_000_000, 1_005_000_000]
    assert kf_positions.tolist() == [0]
    assert has_bframes is False


def test_build_gop_index_rejects_real_open_gop(tmp_path):
    """A genuine non-keyframe frame appearing before any keyframe -- not merely a
    zero-output parameter-set message -- must still be rejected."""
    path = tmp_path / "open_gop.mcap"
    topic = "/cam/scripted"
    _write_scripted_messages(path, topic, n_messages=3)

    scripted = {1: _FakeFrame(False), 2: _FakeFrame(True)}  # position 0 absent (zero-output)
    with (
        patch("lerobot.datasets.mcap_utils.mux_and_decode_by_position", return_value=scripted),
        pytest.raises(ValueError, match="must start with a keyframe"),
    ):
        build_gop_time_and_keyframe_index(path, topic, "h264")


def _sps_pps_only(keyframe_packet: bytes) -> bytes:
    """Slice the Annex-B SPS/PPS/SEI prefix off a real encoded keyframe packet, dropping
    the IDR slice NAL -- a genuine, valid parameter-set-only packet with no picture data,
    the same shape a real encoder emits once up front rather than repeating per GOP."""
    starts = [i for i in range(len(keyframe_packet) - 3) if keyframe_packet[i : i + 3] == b"\x00\x00\x01"]
    idr_start = next(s for s in starts if keyframe_packet[s + 3] & 0x1F == 5)
    return keyframe_packet[:idr_start]


def test_ingest_gop_video_with_leading_parameter_set_only_messages(tmp_path):
    """End-to-end: ingesting a channel whose first message is a real, standalone
    parameter-set-only packet must succeed, and the resulting dataset must read
    correctly -- not just the lower-level index builder in isolation.

    A real container's own Annex-B parser (confirmed empirically) merges a
    parameter-set-only packet with whatever NEXT packet supplies its first slice NAL,
    rather than keeping them as separate decode positions -- so the merged picture gets
    attributed to the EARLIER (leading) message's position, not the LATER message that
    actually supplied it. Naively using that position's own timestamp desyncs every
    later grid index by one slot (confirmed empirically -- indices past the merge
    decoded the wrong frame entirely, not just with a timing wobble); the fix in
    build_gop_time_and_keyframe_index attributes the merged frame to the LAST message in
    the consumed run instead, which is exactly this test's regression check.
    """
    packets, frames = _encode_gop_h264(16, GOP_FPS, GOP_SIZE, seed=0)
    all_packets = [_sps_pps_only(packets[0])] + packets
    path = tmp_path / "raw_leading" / "ep_0.mcap"
    _write_raw_gop_mcap(path, all_packets, GOP_FPS)

    dst_root = tmp_path / "ingested_leading"
    ingest_mcap_episodes(
        mcap_paths=[path],
        channels=[ChannelSpec(key=GOP_VIDEO_KEY, topic=GOP_TOPIC, kind="video", codec="h264")],
        dst_root=dst_root,
        fps=GOP_FPS,
        robot_type="gop-bench",
        tasks="encode test",
    )

    ds = LeRobotDataset("gop/mcap", root=dst_root, return_uint8=True)
    assert len(ds) == 16
    for local_idx in range(16):
        actual = ds[local_idx][GOP_VIDEO_KEY].numpy()
        expected = np.transpose(frames[local_idx], (2, 0, 1))
        np.testing.assert_array_equal(actual, expected)


# ── B-frame GOP streams: decoder-side reordering, not just independent packets ────────
#
# Real streams (e.g. Labelbox/robotics-datasets' HEVC channels) use B-frames: decode
# order (== message/log_time order, confirmed on real data) no longer matches
# presentation order. Feeding packets one at a time to a bare CodecContext -- correct
# for the no-B-frame fixtures above -- silently drops frames and misorders the rest
# (confirmed empirically while designing this fix). McapDatasetReader instead mux-decodes
# such channels through a real throwaway container (_decode_gop_window_muxed).

BFRAME_FPS = 20
BFRAME_GOP_SIZE = 12
BFRAME_FRAME_SIZE = 48
BFRAME_N_FRAMES = 48  # 4 GOPs -- enough to exercise scattered access across GOP boundaries
BFRAME_VIDEO_KEY = "observation.images.cam"
BFRAME_TOPIC = "/cam/bframe"


def _encode_gop_h264_bframes(
    n_frames: int, fps: int, gop_size: int, seed: int
) -> tuple[list[bytes], list[np.ndarray]]:
    """Encode a synthetic clip as H.264 WITH B-frames (confirmed via pict_type below),
    using the same bare, container-less ``av.CodecContext.create(..., "w")`` encoder as
    ``_encode_gop_h264`` -- each keyframe packet carries its own inline SPS/PPS (unlike
    the high-level ``container.add_stream()``/mp4-muxing API, which was observed, while
    building this fixture, to push parameter sets into the stream's ``extradata``
    instead and strip them from the bitstream -- not wrong, just a different real-world
    shape than this fixture targets, and not what this fixture needs).

    Returns the raw per-access-unit packet bytes (one per message, in encode/arrival
    order -- decode order, not presentation order, since B-frames reorder) and the
    ground-truth frames (one per message, in that SAME arrival-order indexing, H, W, C
    uint8) a correct decoder actually produces for them. Ground truth is built with the
    identical mux-then-recover-by-pts technique ``_decode_gop_window_muxed`` uses at read
    time (see its docstring) -- but decoding the WHOLE clip in one pass, with no early
    exit once some requested range is covered. That distinction matters: a tail-of-stream
    B-frame reorder can place an early-index frame's decode output AFTER a later-index
    one even within the final GOP (observed empirically), so stopping early -- as
    `video_utils.decode_video_frames_pyav`'s seek-forward-and-break strategy does, and
    which was tried here first -- can go to the wrong GOP's last-keyframe on a short,
    hand-built test file and/or exit before a late frame actually arrives. Decoding the
    whole clip to exhaustion sidesteps both: it is still an INDEPENDENT check of the
    reader's output, since the reader never decodes a whole episode in one un-windowed
    pass -- its own correctness rests on how it splits, merges and bounds per-GOP windows
    (_decode_gop_episode), which this ground truth does not replicate.
    """
    enc_ctx = av.CodecContext.create("h264", "w")
    enc_ctx.width = BFRAME_FRAME_SIZE
    enc_ctx.height = BFRAME_FRAME_SIZE
    enc_ctx.pix_fmt = "yuv420p"
    enc_ctx.gop_size = gop_size
    enc_ctx.time_base = Fraction(1, fps)
    enc_ctx.options = {
        "bf": "3",
        "x264-params": f"scenecut=0:keyint={gop_size}:min-keyint={gop_size}:bframes=3:b-adapt=0",
    }

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(BFRAME_FRAME_SIZE, BFRAME_FRAME_SIZE, 3), dtype=np.uint8)
    packets: list[bytes] = []
    for i in range(n_frames):
        source = np.roll(base, (i * 3) % BFRAME_FRAME_SIZE, axis=1)
        frame = av.VideoFrame.from_ndarray(source, format="rgb24").reformat(format="yuv420p")
        frame.pts = i
        for packet in enc_ctx.encode(frame):
            packets.append(bytes(packet))
    for packet in enc_ctx.encode(None):
        packets.append(bytes(packet))
    assert len(packets) == n_frames

    buf = io.BytesIO()
    gt_container = av.open(buf, mode="w", format="mpegts")
    gt_stream = gt_container.add_stream("h264")
    for i, data in enumerate(packets):
        packet = av.Packet(data)
        packet.pts = i
        packet.dts = i
        packet.time_base = Fraction(1, 1)
        packet.stream = gt_stream
        gt_container.mux(packet)
    gt_container.close()

    buf.seek(0)
    frames: list[np.ndarray | None] = [None] * n_frames
    has_bframes = False
    with av.open(buf) as in_container:
        for frame in in_container.decode(in_container.streams.video[0]):
            frames[round(frame.pts * frame.time_base)] = frame.to_ndarray(format="rgb24")
            if frame.pict_type == av.video.frame.PictureType.B:
                has_bframes = True
    assert has_bframes, "test fixture must actually contain B-frames"
    assert all(f is not None for f in frames), "ground truth must cover every encoded frame"
    return packets, frames  # type: ignore[return-value]


def _write_raw_bframe_mcap(path: Path, packets: list[bytes], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()
        channel_id = writer.register_channel(topic=BFRAME_TOPIC, message_encoding="h264", schema_id=0)
        for i, data in enumerate(packets):
            t_ns = round(i / fps * 1e9) + 1_000_000_000
            writer.add_message(channel_id=channel_id, log_time=t_ns, data=data, publish_time=t_ns)
        writer.finish()


@pytest.fixture
def bframe_ingest_root(tmp_path) -> tuple[Path, list[np.ndarray]]:
    packets, frames = _encode_gop_h264_bframes(BFRAME_N_FRAMES, BFRAME_FPS, BFRAME_GOP_SIZE, seed=0)
    raw_path = tmp_path / "raw_bframe" / "ep_0.mcap"
    _write_raw_bframe_mcap(raw_path, packets, BFRAME_FPS)

    dst_root = tmp_path / "ingested_bframe"
    ingest_mcap_episodes(
        mcap_paths=[raw_path],
        channels=[ChannelSpec(key=BFRAME_VIDEO_KEY, topic=BFRAME_TOPIC, kind="video", codec="h264")],
        dst_root=dst_root,
        fps=BFRAME_FPS,
        robot_type="bframe-bench",
        tasks="encode test",
    )
    return dst_root, frames


def test_ingest_bframe_video_metadata(bframe_ingest_root):
    """The B-frame flag the reader dispatches on must be detected and persisted."""
    dst_root, _ = bframe_ingest_root
    ds = LeRobotDataset("bframe/mcap", root=dst_root)
    assert ds.meta.features[BFRAME_VIDEO_KEY]["video_codec"] == "h264"
    assert ds.meta.features[BFRAME_VIDEO_KEY]["video_has_bframes"] is True
    assert ds.reader._gop_has_bframes[BFRAME_VIDEO_KEY] is True


def test_ingest_bframe_video_scattered_random_access_correctness(bframe_ingest_root):
    """Scattered/shuffled reads of a real B-frame stream must decode correctly -- the
    bar already met for Lance's mp4 B-frame path: compare against an independent
    ground-truth decode (a full, unwindowed decode of the exact same compressed bytes;
    see `_encode_gop_h264_bframes`'s docstring for why it's still a meaningful check),
    not merely 'doesn't crash'."""
    dst_root, frames = bframe_ingest_root
    ds = LeRobotDataset("bframe/mcap", root=dst_root, return_uint8=True)
    assert len(ds) == BFRAME_N_FRAMES

    rng = np.random.default_rng(1)
    order = rng.permutation(BFRAME_N_FRAMES)[:14]  # scattered across all 4 GOPs
    for local_idx in order:
        local_idx = int(local_idx)
        actual = ds[local_idx][BFRAME_VIDEO_KEY].numpy()
        expected = np.transpose(frames[local_idx], (2, 0, 1))
        np.testing.assert_array_equal(actual, expected)

    # Batched + duplicated + out-of-order indices, exactly like the no-B-frame test.
    indices = [40, 3, 40, 3, 25, 11]
    items = ds.__getitems__(indices)
    for idx, item in zip(indices, items, strict=True):
        assert_items_equal(item, ds[idx])


def test_ingest_bframe_video_range_spans_full_gop(bframe_ingest_root):
    """Unlike the no-B-frame path (keyframe -> target), a B-frame channel's fetch range
    must span the WHOLE GOP (keyframe -> next keyframe), mirroring the lance backend's
    `_window_byte_range` -- verified here directly against the reader's own computed
    keyframe index rather than indirectly through decode-call counting."""
    dst_root, _ = bframe_ingest_root
    ds = LeRobotDataset("bframe/mcap", root=dst_root)
    kf_positions = ds.reader._kf_positions[BFRAME_VIDEO_KEY][0]
    assert kf_positions.tolist() == list(range(0, BFRAME_N_FRAMES, BFRAME_GOP_SIZE))

    # Reproduce _decode_gop_episode's own range computation for a target deep in GOP 2
    # (indices [24, 35]) and confirm it reaches all the way to the start of GOP 3 (36),
    # not just to the target -- a B-frame at the target can reference a later frame in
    # the same GOP, which a target-only range would cut off.
    target_idx = 30
    kf_pos = max(int(np.searchsorted(kf_positions, target_idx, side="right")) - 1, 0)
    lo_idx = int(kf_positions[kf_pos])
    next_kf_pos = kf_pos + 1
    hi_idx = int(kf_positions[next_kf_pos]) - 1 if next_kf_pos < len(kf_positions) else len(kf_positions) - 1
    assert (lo_idx, hi_idx) == (24, 35)


# ── pts/dts identity recovery robustness ──────────────────────────────────────────────
#
# Real streams from a single recording can disagree on which of pts/dts survives the
# mux/demux round trip intact: confirmed on two independent, reproducible failure modes
# -- (a) frames flushed at the very end of a stream (no new packet feeds them) reliably
# lose dts while keeping pts (real bytes, below); (b) a real multi-camera recording
# (Labelbox/robotics-datasets) was found where most non-keyframe frames of two of four
# camera streams lost pts specifically while keeping dts, for a structural reason in
# those streams' own encoding that further investigation didn't pin down. Since neither
# field is universally reliable, mux_and_decode_by_position uses whichever one survives
# per frame rather than assuming either always does.


def test_mux_and_decode_by_position_survives_dts_loss_at_flush_tail():
    """Real, reproducible failure mode: a B-frame stream's flush-tail frames lose dts
    (confirmed empirically -- no new packet is fed for them, so they have no decode
    time to report) while keeping pts. Every position must still be recovered."""
    packets, _frames = _encode_gop_h264_bframes(48, 30, 12, seed=0)
    frames_by_position = mux_and_decode_by_position(packets, "h264")
    assert sorted(frames_by_position.keys()) == list(range(48))
    # Confirm this genuinely exercises the degenerate case, not vacuously.
    assert any(f.dts is None for f in frames_by_position.values())


class _TimestampFrame:
    """Minimal duck-typed stand-in for a decoded av.VideoFrame -- _recover_frames_by_position
    only ever looks at .pts/.dts/.time_base, so a real mux/demux round trip isn't needed
    to test its identity-recovery logic directly."""

    def __init__(self, pts: int | None, dts: int | None, time_base: Fraction) -> None:
        self.pts = pts
        self.dts = dts
        self.time_base = time_base


def test_recover_frames_by_position_pts_fallback_to_dts():
    """A frame with pts=None but a valid dts must still be recovered -- the exact
    pattern the real Labelbox streams hit (most non-keyframe frames there lose pts but
    keep dts)."""
    tb = Fraction(1, 1)
    scripted = [
        _TimestampFrame(pts=0, dts=0, time_base=tb),
        _TimestampFrame(pts=None, dts=1, time_base=tb),
        _TimestampFrame(pts=2, dts=2, time_base=tb),
    ]
    frames_by_position = _recover_frames_by_position(scripted, "h264", n_messages=3)
    assert sorted(frames_by_position.keys()) == [0, 1, 2]


def test_recover_frames_by_position_raises_when_both_timestamps_missing():
    """A frame with neither pts nor dts must raise clearly -- not silently associate
    with the wrong message, and not crash with a raw TypeError."""
    tb = Fraction(1, 1)
    scripted = [
        _TimestampFrame(pts=0, dts=0, time_base=tb),
        _TimestampFrame(pts=None, dts=None, time_base=tb),
    ]
    with pytest.raises(ValueError, match="lost both its pts and dts"):
        _recover_frames_by_position(scripted, "h264", n_messages=2)


def test_recover_frames_by_position_raises_on_duplicate_position():
    """Two frames recovering the SAME position must raise clearly rather than let one
    silently overwrite the other in the result dict."""
    tb = Fraction(1, 1)
    scripted = [
        _TimestampFrame(pts=0, dts=0, time_base=tb),
        _TimestampFrame(pts=0, dts=1, time_base=tb),
    ]
    with pytest.raises(ValueError, match="both recovered position"):
        _recover_frames_by_position(scripted, "h264", n_messages=2)


# ── Oversized SEI bundled with picture data (real Labelbox wrist cameras) ─────────────
#
# A real recording was found where every message of two (of four) camera streams was
# ~1MB, almost entirely a NAL type 39 (HEVC PREFIX_SEI_NUT) bundled alongside the actual
# picture NAL -- versus hundreds of bytes to ~34KB for a comparable stream without this.
# SEI carries no information that affects decoded picture content by spec, so
# mux_and_decode_by_position strips it (_strip_sei_nal_units) before muxing. These tests
# can't reproduce the exact real-world mux/demux misbehavior synthetically (confirmed:
# injecting an oversized SEI NAL of various sizes into synthetic H.264/HEVC streams,
# including the very first keyframe message, at both small and 144-message/~1MB-per-
# message scale, never actually lost pts/dts in this environment) -- but the fix is
# correct independent of that (SEI remove-ability is a spec guarantee, not an empirical
# one), so these verify what's actually checkable: the stripped bytes are byte-identical
# to never having carried the SEI, decoded content is unaffected, and the full ingest
# pipeline doesn't regress when every message (including the first) carries one.


def _make_sei_nal(codec: str, nal_type: int, size: int) -> bytes:
    """A structurally-valid (not semantically meaningful) oversized Annex-B SEI NAL --
    payload bytes avoid 0x00 0x00 sequences so they can never be mistaken for a start
    code by a real NAL splitter."""
    start_code = b"\x00\x00\x00\x01"
    header = bytes([(nal_type << 1) & 0xFE, 0x01]) if codec == "hevc" else bytes([nal_type & 0x1F])
    return start_code + header + bytes([0xFF]) * size


def _inject_sei(data: bytes, sei: bytes) -> bytes:
    return sei + data


def test_strip_sei_nal_units_h264():
    sei = _make_sei_nal("h264", 6, 1_000_000)
    packet = b"\x00\x00\x00\x01\x67SPS_BYTES\x00\x00\x00\x01\x68PPS_BYTES\x00\x00\x00\x01\x65IDR_SLICE"
    bloated = _inject_sei(packet, sei)
    assert len(bloated) > 1_000_000
    assert _strip_sei_nal_units(bloated, "h264") == packet


def test_strip_sei_nal_units_hevc_prefix_and_suffix():
    sei_prefix = _make_sei_nal("hevc", 39, 500_000)
    sei_suffix = _make_sei_nal("hevc", 40, 500_000)
    vps = b"\x00\x00\x00\x01\x40\x01VPS"
    idr = b"\x00\x00\x00\x01\x26\x01IDR_SLICE"
    bloated = sei_prefix + vps + idr + sei_suffix
    assert len(bloated) > 1_000_000
    assert _strip_sei_nal_units(bloated, "hevc") == vps + idr


def test_strip_sei_nal_units_sei_only_message_kept_unstripped():
    """A message that's genuinely SEI-only (no slice/parameter-set NAL) must be left
    as-is, not stripped down to an empty packet (muxing an empty packet reliably
    raises, confirmed empirically, and such a message never decodes to a frame either
    way, so stripping it would gain nothing)."""
    sei_only = _make_sei_nal("hevc", 39, 100)
    assert _strip_sei_nal_units(sei_only, "hevc") == sei_only


def test_strip_sei_nal_units_passthrough_for_unknown_codec():
    data = b"\x00\x00\x00\x01\x06some_sei_payload\x00\x00\x00\x01\x65slice"
    assert _strip_sei_nal_units(data, "av1") == data


def test_ingest_gop_video_with_oversized_sei_on_every_message(tmp_path):
    """End-to-end: every message (including the episode's first, the keyframe) carries
    an oversized SEI NAL bundled with its picture data -- the real-world shape found on
    Labelbox's wrist cameras. Ingest and scattered reads must still produce bit-exact
    correct content; stripping SEI must not touch decoded pixels."""
    packets, frames = _encode_gop_h264(16, GOP_FPS, GOP_SIZE, seed=0)
    sei = _make_sei_nal("h264", 6, 1_000_000)
    bloated_packets = [_inject_sei(p, sei) for p in packets]
    assert all(len(p) > 1_000_000 for p in bloated_packets)

    raw_path = tmp_path / "raw_sei" / "ep_0.mcap"
    _write_raw_gop_mcap(raw_path, bloated_packets, GOP_FPS)
    dst_root = tmp_path / "ingested_sei"
    ingest_mcap_episodes(
        mcap_paths=[raw_path],
        channels=[ChannelSpec(key=GOP_VIDEO_KEY, topic=GOP_TOPIC, kind="video", codec="h264")],
        dst_root=dst_root,
        fps=GOP_FPS,
        robot_type="sei-bench",
        tasks="encode test",
    )

    ds = LeRobotDataset("sei/mcap", root=dst_root, return_uint8=True)
    assert len(ds) == 16
    rng = np.random.default_rng(2)
    for local_idx in rng.permutation(16):
        local_idx = int(local_idx)
        actual = ds[local_idx][GOP_VIDEO_KEY].numpy()
        expected = np.transpose(frames[local_idx], (2, 0, 1))
        np.testing.assert_array_equal(actual, expected)
