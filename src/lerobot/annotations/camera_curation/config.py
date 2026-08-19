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
"""Config for ``lerobot-curate-cameras`` (VLM camera-view curation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.annotations.steerable_pipeline.config import AnnotationJobConfig, VlmConfig

# The closed vocabulary of canonical camera-view labels. Combos are formed by
# joining two of these with ``_`` (e.g. ``left_wrist``).
# Base positions relative to the robot + the left/right qualifiers used to
# disambiguate side and wrist views (e.g. left_side, right_wrist).
DEFAULT_VIEW_VOCABULARY: tuple[str, ...] = (
    "front",
    "rear",
    "side",
    "top",
    "bottom",
    "wrist",
    "left",
    "right",
)


@dataclass
class CameraCurationConfig:
    """Top-level config for ``lerobot-curate-cameras``.

    The VLM decision only ever reads the first episode (a cheap partial
    download). ``mode="report"`` writes the labels + quality verdicts into
    ``meta/`` and moves nothing (works for any dataset). ``mode="rename"``
    additionally renames the camera keys to ``observation.images.<label>`` —
    for video datasets this is a server-side, download-free Hub commit.
    """

    # Hub dataset id (downloaded when ``root`` is unset) — also the rename target.
    repo_id: str | None = None
    # Local dataset directory (skips the Hub download).
    root: Path | None = None

    # Nested collections: a repo that holds many independent LeRobotDatasets under
    # per-dataset subfolders (e.g. ``lerobot/community_dataset_v3`` → ``<user>/<task>/``)
    # has no root ``meta/info.json``. List the sub-dataset prefixes to curate each in
    # turn — each is materialized (meta + first episode) and, in rename mode, renamed
    # in place under its own prefix on the Hub. Leave unset on a nested repo to
    # auto-discover and process EVERY sub-dataset.
    subpaths: tuple[str, ...] | None = None

    # Cap the number of sub-datasets processed (after discovery/sorting). Handy for
    # a trial run over a large collection; None processes all.
    limit: int | None = None

    # Sub-datasets processed concurrently in a nested sweep (one shared VLM server
    # feeds all workers). The work is download/latency-bound, so >1 overlaps I/O
    # for a large speedup on a single GPU. Hub rename commits are serialized
    # internally to avoid branch-ref races.
    parallelism: int = 4

    # "report": write mapping + verdicts into meta/, no file moves.
    # "rename": physically rename camera keys to observation.images.<label>.
    mode: str = "report"

    # Commit target branch for the Hub rename; keeps ``main`` intact when set.
    # None commits to the default branch.
    branch: str | None = None

    # Episode inspected by the VLM (first episode by default).
    episode_index: int = 0
    # Frames sampled from that episode per camera and shown to the VLM.
    n_frames: int = 4

    # Closed label vocabulary and whether two-token combos (left_wrist) are allowed.
    view_vocabulary: tuple[str, ...] = DEFAULT_VIEW_VOCABULARY
    allow_combos: bool = True

    # When two cameras of a dataset get the same view label, run a second joint
    # VLM pass over just those cameras (all shown together) asking for DISTINCT
    # labels, before falling back to on_collision. Labels-only, so it can't leak
    # a quality verdict across cameras.
    relabel_on_conflict: bool = True

    # What to do when cameras still collide on a label after the relabel pass:
    #   "skip"   : rename only the unambiguous cameras; leave the colliding ones
    #              as-is and record them (partial rename — the default).
    #   "suffix" : rename all, disambiguating with a numeric suffix (top, top_2).
    #   "error"  : skip the whole dataset's rename and report the conflict.
    on_collision: str = "skip"
    # Remove cameras judged unusable (default: only flag them, still rename).
    drop_unusable: bool = False

    # Where to write the machine-readable report (default <root>/meta/camera_curation.json).
    report_path: Path | None = None

    vlm: VlmConfig = field(default_factory=VlmConfig)
    job: AnnotationJobConfig = field(default_factory=AnnotationJobConfig)

    seed: int = 1729
    # Keyframe decode backend forwarded to ``decode_video_frames`` (None = default).
    video_backend: str | None = None

    # Upload the result (rename mode). Kept off by default so runs are dry.
    push_to_hub: bool = False
    push_commit_message: str | None = None
