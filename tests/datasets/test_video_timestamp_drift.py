"""Regression tests for video timestamp drift during episode concatenation.

See: https://github.com/huggingface/lerobot/issues/3177

When multiple episodes are concatenated into a single video file, the
`from_timestamp` metadata must stay consistent with the actual video PTS.
Previously, `from_timestamp` was computed by accumulating float values from
parquet metadata (`to_timestamp`), which diverged from the actual video PTS
that PyAV/ffmpeg produces via rational arithmetic.

The fix reads the actual video duration via `get_video_duration_in_s()` after
each concatenation instead of accumulating floats from parquet.

To verify old code fails and new code passes:
    # 1. Run with fix applied (should PASS):
    PYTHONPATH=src:$PYTHONPATH python -m pytest tests/datasets/test_video_timestamp_drift.py -v -s -c /dev/null

    # 2. Revert fix, run again (should FAIL):
    git stash push -m "fix" -- src/lerobot/datasets/lerobot_dataset.py src/lerobot/datasets/aggregate.py
    PYTHONPATH=src:$PYTHONPATH python -m pytest tests/datasets/test_video_timestamp_drift.py -v -s -c /dev/null
    git stash pop
"""

import av
import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import get_video_duration_in_s


def _get_actual_frame_pts(video_path) -> list[float]:
    """Read all frame PTS from a video file (ground truth)."""
    pts_list = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            pts_list.append(float(frame.pts * stream.time_base))
    return pts_list


# Use varying frame counts per episode to exercise different duration arithmetic.
# Non-uniform counts produce "ugly" float durations (e.g., 7/30, 13/30) that
# amplify accumulation error across different rounding paths.
FRAME_COUNTS = [7, 13, 11, 17, 9, 15, 8, 12, 14, 10] * 5  # 50 episodes


@pytest.mark.parametrize("num_episodes", [50])
def test_from_timestamp_matches_actual_video_pts(tmp_path, num_episodes):
    """Verify from_timestamp for each episode EXACTLY matches actual video PTS.

    This test exercises the real code path: LeRobotDataset.add_frame() →
    save_episode() → _save_episode_video() → concatenate_video_files().

    With the fix (get_video_duration_in_s): from_timestamp == actual PTS (0.0 drift).
    Without the fix (accumulated float): from_timestamp drifts from actual PTS.
    """
    fps = 30
    vid_key = "observation.image"
    frame_counts = FRAME_COUNTS[:num_episodes]

    # Create dataset with video feature, batch_encoding_size=1 to trigger
    # per-episode concatenation (same path as real recording)
    dataset = LeRobotDataset.create(
        repo_id="test/timestamp_drift",
        fps=fps,
        features={
            vid_key: {
                "dtype": "video",
                "shape": (3, 64, 64),
                "names": ["channels", "height", "width"],
            }
        },
        root=tmp_path / "dataset",
    )
    dataset.batch_encoding_size = 1

    # Record episodes with varying frame counts, collecting metadata after each save
    episode_metadata = []
    for ep_idx in range(num_episodes):
        for frame_idx in range(frame_counts[ep_idx]):
            dataset.add_frame({
                vid_key: np.full((3, 64, 64), fill_value=(frame_idx * 5) % 256, dtype=np.uint8),
                "task": "test",
            })
        dataset.save_episode()
        # Capture metadata from memory right after save (latest_episode is always set)
        episode_metadata.append(dict(dataset.meta.latest_episode))

    # Read actual PTS from the concatenated video file (ground truth)
    ep0 = episode_metadata[0]
    video_path = (
        dataset.root / dataset.meta.video_path.format(
            video_key=vid_key,
            chunk_index=ep0[f"videos/{vid_key}/chunk_index"][0],
            file_index=ep0[f"videos/{vid_key}/file_index"][0],
        )
    )
    actual_pts = _get_actual_frame_pts(video_path)

    # Verify: from_timestamp for each episode matches actual PTS at boundary
    boundary_frame = 0
    for ep_idx in range(num_episodes):
        ep_meta = episode_metadata[ep_idx]
        computed_from_ts = ep_meta[f"videos/{vid_key}/from_timestamp"][0]
        actual_boundary_pts = actual_pts[boundary_frame]

        drift = abs(computed_from_ts - actual_boundary_pts)
        assert drift == 0.0, (
            f"Episode {ep_idx}: from_timestamp ({computed_from_ts:.15f}) "
            f"does not EXACTLY match actual video PTS ({actual_boundary_pts:.15f}). "
            f"Drift: {drift:.2e}. "
            f"This indicates from_timestamp is computed via float accumulation "
            f"instead of reading the actual video duration (see issue #3177)."
        )

        boundary_frame += frame_counts[ep_idx]

    print(f"\nAll {num_episodes} episode boundaries: from_timestamp == actual PTS (drift = 0.0)")
