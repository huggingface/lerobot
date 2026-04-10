"""Tests for floating-point drift in video timestamps during episode concatenation.

Regression test for https://github.com/huggingface/lerobot/issues/3177

When episodes are concatenated into a single video file, `from_timestamp` must match
the actual video PTS at episode boundaries. Accumulating float values (from parquet
metadata or repeated addition) causes progressive drift that eventually exceeds the
decode tolerance, raising FrameTimestampError in later episodes.
"""

import av
import numpy as np
import pytest

from lerobot.datasets.io_utils import load_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import get_video_duration_in_s
from tests.fixtures.constants import DUMMY_REPO_ID


def _get_pts_at_frame(video_path, frame_index):
    """Read the actual PTS (in seconds) of a specific frame from the video file."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_index:
                return float(frame.pts * stream.time_base)
    raise ValueError(f"Frame {frame_index} not found in {video_path}")


NUM_EPISODES = 50
FPS = 30


@pytest.mark.parametrize("num_episodes", [NUM_EPISODES])
def test_from_timestamp_matches_actual_video_pts(tmp_path, num_episodes):
    """from_timestamp must match actual video PTS at every episode boundary.

    Records many episodes with varying frame counts into a dataset that concatenates
    them into a single video file, then verifies that each episode's from_timestamp
    in the parquet metadata exactly matches the actual PTS of the boundary frame
    in the concatenated video.
    """
    rng = np.random.RandomState(42)
    frames_per_episode = rng.randint(7, 18, size=num_episodes).tolist()

    features = {
        "observation.images.laptop": {
            "dtype": "video",
            "shape": (3, 64, 96),
            "names": ["channels", "height", "width"],
        },
        "state": {"dtype": "float32", "shape": (2,), "names": None},
    }

    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=FPS,
        features=features,
        root=tmp_path / "drift_test",
    )
    # Force all episodes into one video file by setting a large size limit
    dataset.meta.update_chunk_settings(video_files_size_in_mb=10000)

    for ep_idx in range(num_episodes):
        for _ in range(frames_per_episode[ep_idx]):
            dataset.add_frame(
                {
                    "observation.images.laptop": np.random.randint(
                        0, 256, (64, 96, 3), dtype=np.uint8
                    ),
                    "state": np.random.randn(2).astype(np.float32),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()

    dataset.finalize()

    # Verify from_timestamp matches actual video PTS at each episode boundary
    video_key = "observation.images.laptop"
    episodes = load_episodes(dataset.root)
    cumulative_frames = 0
    max_drift = 0.0

    for ep_idx in range(num_episodes):
        from_ts = episodes[ep_idx][f"videos/{video_key}/from_timestamp"]
        chunk_idx = episodes[ep_idx][f"videos/{video_key}/chunk_index"]
        file_idx = episodes[ep_idx][f"videos/{video_key}/file_index"]

        video_path = dataset.root / dataset.meta.video_path.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )

        if cumulative_frames > 0:
            actual_pts = _get_pts_at_frame(video_path, cumulative_frames)
            drift = abs(from_ts - actual_pts)
            max_drift = max(max_drift, drift)

            # Drift must be less than half a frame (strict correctness)
            half_frame = 0.5 / FPS
            assert drift < half_frame, (
                f"Episode {ep_idx}: from_timestamp drift {drift:.10f}s exceeds "
                f"half-frame threshold {half_frame:.6f}s. "
                f"from_timestamp={from_ts}, actual_pts={actual_pts}"
            )

        cumulative_frames += frames_per_episode[ep_idx]


@pytest.mark.parametrize(
    "num_episodes,container_tb_denom",
    [
        (50, 15360),    # typical AV1/H.264 container
        (50, 90000),    # MPEG-TS style container
        (500, 15360),   # long recording session
        (500, 90000),   # long recording + MPEG-TS
    ],
)
def test_round6_accumulates_drift_but_actual_duration_does_not(num_episodes, container_tb_denom):
    """round(..., 6) accumulates non-zero drift; reading actual duration does not.

    Simulates the two timestamp computation paths:
    - PR #3239 approach: accumulate episode durations with round(x, 6) at each step
    - This PR's approach: read total duration from the concatenated video file
      (equivalent to a single rational→float conversion, no accumulation)

    With realistic container time bases (1/15360, 1/90000), round(6) accumulates
    measurable drift because each rounding step introduces up to 5e-7 error.
    Reading the actual duration from the video avoids accumulation entirely.
    """
    from fractions import Fraction

    fps = 30
    rng = np.random.RandomState(42)
    frames_per_episode = rng.randint(7, 18, size=num_episodes).tolist()
    container_time_base = Fraction(1, container_tb_denom)

    # Ground truth: rational arithmetic (what ffmpeg internally computes)
    rational_boundary = Fraction(0)

    # PR #3239: accumulate with round(..., 6)
    round6_boundary = 0.0
    round6_max_drift = 0.0
    round6_nonzero_count = 0

    # This PR: read actual total duration (single conversion per step)
    cumulative_container_pts = 0
    actual_max_drift = 0.0

    for n_frames in frames_per_episode[:-1]:
        exact_duration = Fraction(n_frames, fps)
        container_pts = int(exact_duration / container_time_base + Fraction(1, 2))

        # Ground truth
        rational_boundary += container_pts * container_time_base
        ground_truth = float(rational_boundary)

        # round(6) path: accumulated sum with rounding
        ep_duration = float(container_pts * container_time_base)
        round6_boundary = round(round6_boundary + ep_duration, 6)
        r6_drift = abs(round6_boundary - ground_truth)
        round6_max_drift = max(round6_max_drift, r6_drift)
        if r6_drift > 0:
            round6_nonzero_count += 1

        # actual-duration path: single conversion of cumulative integer PTS
        cumulative_container_pts += container_pts
        actual_boundary = float(cumulative_container_pts * container_time_base)
        actual_drift = abs(actual_boundary - ground_truth)
        actual_max_drift = max(actual_max_drift, actual_drift)

    # round(6) accumulates non-zero drift in most episodes
    assert round6_max_drift > 0, "round(6) should have non-zero drift"
    assert round6_nonzero_count > 0, "round(6) should drift in at least some episodes"

    # actual-duration approach has zero drift (single conversion, no accumulation)
    assert actual_max_drift == 0.0, (
        f"Actual-duration approach should have exactly zero drift, got {actual_max_drift}"
    )

    # actual-duration is strictly better
    assert actual_max_drift < round6_max_drift, (
        f"Actual-duration drift ({actual_max_drift}) should be less than "
        f"round(6) drift ({round6_max_drift})"
    )
