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
"""Rollout summarization: success metrics and representative-video selection."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.integrations.wandb_artifacts.inspect import (
    DatasetDirectoryMetadata,
    inspect_dataset_directory,
)
from lerobot.integrations.wandb_artifacts.rollout import (
    RolloutSummary,
    select_representative_video,
    validate_success_count,
)

_ACTION_FEATURE = {"dtype": "float32", "shape": (2,), "names": None}


def _camera_feature() -> dict:
    return {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}


def _write_rollout_dataset(root: Path, *, episodes: int, camera_keys: tuple[str, ...] = ()) -> None:
    """A genuinely valid rollout dataset, written by the real dataset writer."""
    features = {"action": _ACTION_FEATURE} | {key: _camera_feature() for key in camera_keys}
    dataset = LeRobotDataset.create(
        repo_id="tests/rollout_wandb",
        fps=10,
        features=features,
        root=root,
        robot_type="so101",
        use_videos=bool(camera_keys),
        video_backend="pyav",
        metadata_buffer_size=1,
    )
    for _ in range(episodes):
        for _ in range(4):
            frame = {"action": np.zeros(2, dtype=np.float32), "task": "pick the cube"}
            for key in camera_keys:
                frame[key] = np.zeros((32, 32, 3), dtype=np.uint8)
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()


def _metadata(*, episodes: int, frames: int, fps: int = 10) -> DatasetDirectoryMetadata:
    return DatasetDirectoryMetadata(
        schema_version="v3.0",
        robot_type="so101",
        fps=fps,
        total_episodes=episodes,
        total_frames=frames,
        total_tasks=1,
        camera_keys=(),
        video_keys=(),
        source_path=Path("/tmp/rollout"),
        git_commit=None,
    )


def _summary(metadata: DatasetDirectoryMetadata, successes: int) -> RolloutSummary:
    return RolloutSummary.build(
        metadata,
        successes=successes,
        model_requested_ref="team/proj/policy:latest",
        model_resolved_ref="team/proj/policy:v3",
        video=None,
    )


def test_success_rate_and_duration_are_derived_from_the_dataset():
    summary = _summary(_metadata(episodes=8, frames=400), successes=6)

    assert summary.success_rate == 0.75
    assert summary.duration_s == 40.0
    assert summary.to_wandb_metadata()["model_artifact_resolved_ref"] == "team/proj/policy:v3"


def test_empty_rollout_reports_a_zero_success_rate_instead_of_dividing_by_zero():
    """An aborted session that recorded nothing is a legitimate thing to upload."""
    summary = _summary(_metadata(episodes=0, frames=0, fps=0), successes=0)

    assert summary.success_rate == 0.0
    assert summary.duration_s == 0.0


@pytest.mark.parametrize("successes", [-1, 9])
def test_impossible_success_counts_are_rejected(successes):
    with pytest.raises(ValueError, match="between 0 and the rollout's 8 episode"):
        validate_success_count(successes, 8)

    # No construction path can skip the check, whichever way the summary is built.
    with pytest.raises(ValueError, match="between 0 and the rollout's 8 episode"):
        _summary(_metadata(episodes=8, frames=400), successes=successes)


def test_representative_video_covers_every_episode_stored_in_the_chosen_file(tmp_path):
    """Dataset v3 concatenates episodes into one file until a size target is hit, so the chosen
    video is a span. The reported episode list must say so rather than imply episode 0 alone.
    """
    root = tmp_path / "rollout"
    _write_rollout_dataset(root, episodes=3, camera_keys=("observation.images.cam",))

    video = select_representative_video(root)

    assert video is not None
    assert video.video_key == "observation.images.cam"
    assert video.episodes == (0, 1, 2)
    assert video.path.is_file()
    # Every video file in this dataset — exactly one — is the one we picked.
    assert sorted(root.rglob("*.mp4")) == [video.path]


def test_representative_video_picks_one_camera_deterministically(tmp_path):
    """Multiple cameras produce multiple files; exactly one is chosen, by stable sort, and the
    others stay in the artifact only.
    """
    root = tmp_path / "rollout"
    _write_rollout_dataset(
        root, episodes=2, camera_keys=("observation.images.wrist", "observation.images.front")
    )

    video = select_representative_video(root)

    assert video is not None
    assert video.video_key == "observation.images.front"  # sorts before "wrist"
    assert len(sorted(root.rglob("*.mp4"))) == 2
    assert select_representative_video(root) == video  # deterministic across calls


def test_no_video_in_a_state_only_rollout(tmp_path):
    root = tmp_path / "rollout"
    _write_rollout_dataset(root, episodes=2)

    assert select_representative_video(root) is None
    # The directory is still a perfectly valid rollout dataset.
    assert inspect_dataset_directory(root).total_episodes == 2
