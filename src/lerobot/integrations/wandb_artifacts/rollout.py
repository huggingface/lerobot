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
"""Rollout-specific summarization on top of a validated LeRobot dataset directory.

A rollout dataset *is* a LeRobot dataset on disk (see
``docs/adr/0004-rollout-artifacts-are-their-own-type.md``), so everything structural here is
delegated to ``inspect``. What this module adds is the part a directory can't tell you: how many
episodes the operator judged successful, which model produced them, and which single video is worth
putting in front of a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.datasets.io_utils import load_episodes, load_info

from .inspect import DatasetDirectoryMetadata

ROLLOUT_ARTIFACT_TYPE = "rollout"


def validate_success_count(successes: int, episodes: int) -> None:
    """Reject a success count that can't describe this rollout.

    Called before the W&B run is created (a bad count must not leave a junk run behind) and again
    from :meth:`RolloutSummary.__post_init__`, so no construction path can skip it.
    """
    if successes < 0 or successes > episodes:
        raise ValueError(
            f"--episodes-succeeded must be between 0 and the rollout's {episodes} episode(s), "
            f"got {successes}."
        )


@dataclass(frozen=True, slots=True)
class RepresentativeVideo:
    """The one rollout video logged as run media, and the episodes it actually shows.

    In Dataset v3 a single ``.mp4`` holds however many episodes fit under the writer's file-size
    target (``dataset_writer`` concatenates each new episode onto the current file), so this is a
    span, not one episode. ``episodes`` is every episode index stored in ``path``.
    """

    path: Path
    video_key: str
    episodes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    """Operator- and model-facing facts about one rollout session.

    Written to two sinks — the Artifact's metadata (so the artifact stays self-describing wherever
    it is found later) and the run summary (so the numbers are visible in the run UI) — but built
    exactly once, here.
    """

    episodes: int
    successes: int
    success_rate: float
    frames: int
    duration_s: float
    fps: int
    robot_type: str | None
    model_artifact_requested_ref: str
    model_artifact_resolved_ref: str
    representative_video_path: str | None
    representative_video_episodes: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_success_count(self.successes, self.episodes)

    @classmethod
    def build(
        cls,
        metadata: DatasetDirectoryMetadata,
        *,
        successes: int,
        model_requested_ref: str,
        model_resolved_ref: str,
        video: RepresentativeVideo | None,
    ) -> RolloutSummary:
        episodes = metadata.total_episodes
        return cls(
            episodes=episodes,
            successes=successes,
            # An aborted session that recorded nothing is a legitimate thing to upload; it has no
            # success rate, and must not raise on the way to saying so.
            success_rate=successes / episodes if episodes else 0.0,
            frames=metadata.total_frames,
            duration_s=metadata.total_frames / metadata.fps if metadata.fps else 0.0,
            fps=metadata.fps,
            robot_type=metadata.robot_type,
            model_artifact_requested_ref=model_requested_ref,
            model_artifact_resolved_ref=model_resolved_ref,
            representative_video_path=str(video.path) if video is not None else None,
            representative_video_episodes=video.episodes if video is not None else (),
        )

    def to_wandb_metadata(self) -> dict[str, Any]:
        """JSON-safe dict form, for both an Artifact's ``metadata`` and a run's ``summary``."""
        return {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "frames": self.frames,
            "duration_s": self.duration_s,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "model_artifact_requested_ref": self.model_artifact_requested_ref,
            "model_artifact_resolved_ref": self.model_artifact_resolved_ref,
            "representative_video_path": self.representative_video_path,
            "representative_video_episodes": list(self.representative_video_episodes),
        }


def select_representative_video(root: Path | str) -> RepresentativeVideo | None:
    """Pick one video file deterministically, or ``None`` if the rollout has no video.

    Chosen by a stable sort over ``(video_key, chunk_index, file_index)`` taken from the episode
    metadata rather than by globbing the directory, so the choice is driven by what the dataset
    *declares* and stays correct under a non-default ``video_path`` template. Every other video file
    stays in the Artifact and is never logged as run media.

    Only valid on a directory that already passed ``validate_dataset_directory``, which is what
    proves the metadata read here is consistent with the files on disk.
    """
    root = Path(root)
    info = load_info(root)
    video_keys = sorted(key for key, feature in info.features.items() if feature["dtype"] == "video")
    if not video_keys or info.video_path is None or info.total_episodes == 0:
        return None

    episodes = load_episodes(root)
    located = [
        (
            key,
            int(row[f"videos/{key}/chunk_index"]),
            int(row[f"videos/{key}/file_index"]),
            int(row["episode_index"]),
        )
        for key in video_keys
        for row in episodes
    ]
    if not located:
        return None

    key, chunk_index, file_index, _ = min(located)
    return RepresentativeVideo(
        path=root / info.video_path.format(video_key=key, chunk_index=chunk_index, file_index=file_index),
        video_key=key,
        episodes=tuple(
            sorted(
                episode_index
                for k, c, f, episode_index in located
                if (k, c, f) == (key, chunk_index, file_index)
            )
        ),
    )
