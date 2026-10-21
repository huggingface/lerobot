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
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mcap")

from mcap.writer import Writer
from PIL import Image as PILImage

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mcap_backend import McapDatasetReader
from lerobot.datasets.mcap_convert import ChannelSpec, convert_lerobot_dataset_to_mcap, ingest_mcap_episodes
from lerobot.datasets.mcap_utils import EPISODE_START_TIME_COLUMN, log_time_column
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
