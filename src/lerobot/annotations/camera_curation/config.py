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
# Base positions (side / top / bottom / wrist) plus the direction qualifiers used
# as suffixes to say WHERE around the robot a side/wrist camera sits
# (front_side, rear_side, left_side, right_side, left_wrist, right_wrist).
DEFAULT_VIEW_VOCABULARY: tuple[str, ...] = (
    "side",
    "top",
    "bottom",
    "wrist",
    "left",
    "right",
    "front",
    "rear",
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

    # Fraction of the episode to sample frames from, as (lo, hi) in [0, 1].
    # Defaults to the middle 10-90% — this skips the unrepresentative setup/teardown
    # frames (arm at rest, scene being reset) and favours the active-manipulation
    # part, where arm motion is visible (helps tell a moving wrist camera from a
    # fixed one). Widen to (0.0, 1.0) for the whole episode.
    sample_window: tuple[float, float] = (0.1, 0.9)

    # Episode inspected by the VLM (first episode by default).
    episode_index: int = 0
    # Frames sampled from that episode per camera and shown to the VLM.
    n_frames: int = 4

    # Closed label vocabulary and whether two-token combos (left_wrist) are allowed.
    view_vocabulary: tuple[str, ...] = DEFAULT_VIEW_VOCABULARY
    allow_combos: bool = True

    # Ignore the existing camera key names when choosing labels — force relabeling
    # purely from the VLM's view of the frames. Turns off the name-based
    # disambiguation that borrows words from the current key to break a collision
    # (e.g. "..._left" + wrist -> left_wrist). Use it when the existing names are
    # unreliable/misleading (a camera keyed "top" that is really a side view);
    # collisions then resolve by confidence only.
    ignore_key_names: bool = False

    # Second labeling pass: after the per-camera pass, show ALL of a dataset's
    # cameras together in one call and re-decide each mount type + label by
    # comparing them (better at telling top/side apart than judging cameras in
    # isolation). Mount-type + label only — quality stays from the per-camera
    # pass, so no cross-camera leak.
    joint_labeling: bool = False

    # What to do when cameras still collide on a label after classification:
    #   "skip"   : rename the unambiguous cameras; for a contested label keep the
    #              highest-confidence camera and skip the rest (partial rename —
    #              the default). A label already taken by an existing feature is
    #              always skipped.
    #   "suffix" : rename all, disambiguating with a numeric suffix (top, top_2).
    #   "error"  : skip the whole dataset's rename and report the conflict.
    on_collision: str = "skip"

    # Candidate fallback: when two cameras collide on a label, let a losing camera
    # (the lower-confidence one) take its next-best ALTERNATIVE from the VLM's
    # ranked candidates instead of being skipped/suffixed — but only if that
    # alternative is a valid vocabulary label, is still free, and its confidence is
    # at least candidate_fallback_min_confidence. Off by default; applied before the
    # on_collision policy, which still handles whatever remains contested.
    candidate_fallback: bool = False
    candidate_fallback_min_confidence: float = 0.4

    # Promote a hedged label to a directional candidate: when the primary view
    # label is a plain base (``side``/``wrist``, no direction) but the VLM's own
    # ranked candidates include a directional refinement of that SAME position
    # (e.g. ``front_side`` for ``side``, ``left_wrist`` for ``wrist``) with
    # confidence >= promote_direction_min_confidence, adopt it. Unlike
    # candidate_fallback this needs no collision — it just recovers the direction
    # the model hedged out of on the primary label. The position never changes
    # (``side`` never becomes ``top``). Off by default.
    promote_direction_candidate: bool = False
    promote_direction_min_confidence: float = 0.4

    # Borrow a direction from the existing key name: when the VLM labels a camera a
    # plain base that accepts a direction (``side``/``wrist``, no direction) and the
    # ORIGINAL key name carries a direction qualifier (``front``/``rear``/``left``/
    # ``right`` — e.g. ``observation.images.front_side``), adopt it so the label
    # becomes ``front_side``. ONLY the direction qualifier is borrowed, never a
    # position word, so a mislabeled position in the key (a "top" that is really a
    # side) cannot leak in. This is the last-resort fix when the model refuses to
    # emit the direction itself (not even as a candidate). Off by default; unlike
    # the label decision it deliberately still reads the key even under
    # ignore_key_names, since a bare direction qualifier is usually trustworthy.
    direction_from_key_name: bool = False

    # Trust the key name's direction OVER the VLM's. Like direction_from_key_name
    # but stronger: when the ORIGINAL key carries a direction qualifier, it REPLACES
    # whatever direction the VLM assigned, correcting a wrong left/right/front/rear
    # (e.g. a camera keyed ``left_side`` the model called ``right_side`` becomes
    # ``left_side``). Only the direction qualifier is overridden; the VLM's position
    # (``side``/``wrist``) is kept, so a wrong position in the key cannot leak in.
    # Use it when the model is unreliable on direction but the owner's key naming is
    # trustworthy — the reliable fix for left/right, which VLMs read poorly even
    # from a clear frame. Implies direction_from_key_name. Off by default.
    trust_key_direction: bool = False

    # Remove cameras judged unusable (default: only flag them, still rename).
    drop_unusable: bool = False

    # Where to write the machine-readable report (default <root>/meta/camera_curation.json).
    # For a nested collection the aggregated report doubles as the resume log: it is
    # rewritten after EACH sub-dataset finishes (default camera_curation_collection.json).
    report_path: Path | None = None

    # Resume a nested sweep: skip sub-datasets already recorded (without an error)
    # in the report/progress file, and process only the rest.
    resume: bool = False

    # Persist the nested progress/resume log to the Hub (the collection repo) after
    # each sub-dataset, and fetch it back on --resume. This is what makes --resume
    # work on an ephemeral HF-Jobs pod, whose local filesystem is wiped on restart:
    # the checkpoint lives on the Hub instead of on disk. Uploaded to --branch when
    # set. Adds one small metadata commit per sub-dataset.
    progress_to_hub: bool = False

    vlm: VlmConfig = field(default_factory=VlmConfig)
    job: AnnotationJobConfig = field(default_factory=AnnotationJobConfig)

    seed: int = 1729
    # Keyframe decode backend forwarded to ``decode_video_frames`` (None = default).
    video_backend: str | None = None

    # Upload the result (rename mode). Kept off by default so runs are dry.
    push_to_hub: bool = False
    push_commit_message: str | None = None
