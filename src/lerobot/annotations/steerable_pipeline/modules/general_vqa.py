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
"""``vqa`` module: general VQA at a timed cadence.

Every ``1/hz`` seconds an emission tick fires; each tick anchors ``K``
consecutive frames, and every anchored frame gets its own VQA pair. Each
pair is grounded on that single anchor frame — there is no per-pair frame
window. For datasets with multiple cameras, every anchored frame produces
one ``(vqa, user)`` + ``(vqa, assistant)`` pair *per camera*: each pair is
generated against that camera's frame and stamped with the matching
``camera`` field on the emitted rows. The resolver disambiguates via
``camera=...``; recipes that consume VQA do so through one sub-recipe
per camera (see ``recipes/pi05_hirobot.yaml``).

Within a single (frame, camera) we still emit at most one ``(vqa, user)``
and one ``(vqa, assistant)`` row, so the resolver contract stays scalar.

Question types covered (per the plan's ``vqa`` table): bbox, keypoint,
count, attribute, spatial. The assistant's ``content`` is a JSON string
whose schema depends on the question type. Malformed JSON triggers one
retry inside :meth:`VlmClient.generate_json`.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import VqaConfig
from ..frames import FrameProvider, null_provider, to_image_blocks
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord
from ..staging import EpisodeStaging
from ..validator import classify_vqa_answer
from ..vlm_client import VlmClient


def _emission_anchor_indices(frame_timestamps: Sequence[float], hz: float, k: int) -> list[int]:
    """Return the relative frame indices to anchor VQA emissions to.

    For each emission tick (every ``1/hz`` seconds), we anchor ``k``
    consecutive frames starting at the tick. Ticks fall on the nearest
    available source frame timestamp.
    """
    if hz <= 0 or k <= 0 or not frame_timestamps:
        return []
    t0 = frame_timestamps[0]
    t_last = frame_timestamps[-1]
    period = 1.0 / hz
    indices: list[int] = []
    t = t0
    while t <= t_last + 1e-9:
        # find the index of the nearest frame to t
        nearest_i = min(range(len(frame_timestamps)), key=lambda i: abs(frame_timestamps[i] - t))
        for offset in range(k):
            j = nearest_i + offset
            if j >= len(frame_timestamps):
                break
            if not indices or indices[-1] != j:
                indices.append(j)
        t += period
    # dedupe while preserving order
    seen: set[int] = set()
    deduped: list[int] = []
    for i in indices:
        if i in seen:
            continue
        seen.add(i)
        deduped.append(i)
    return deduped


@dataclass
class GeneralVqaModule:
    """Emit grounded VQA pairs at a timed cadence."""

    vlm: VlmClient
    config: VqaConfig
    seed: int = 1729
    frame_provider: FrameProvider = field(default_factory=null_provider)
    _warned_no_camera: bool = field(default=False, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        if not record.frame_timestamps:
            staging.write("vqa", [])
            return
        rng = random.Random(f"{self.seed}:{record.episode_index}:vqa")
        anchor_idx = _emission_anchor_indices(
            record.frame_timestamps, self.config.vqa_emission_hz, self.config.K
        )
        cameras = self._target_cameras()
        if not cameras:
            # No camera available — emit nothing rather than producing
            # untagged rows that would fail validation. Surface a loud one-
            # time warning so this is never silently a no-op.
            if not self._warned_no_camera:
                logging.getLogger(__name__).warning(
                    "vqa module found no cameras on the frame provider — "
                    "every episode will emit zero VQA rows. Check that the "
                    "dataset declares observation.images.* features in "
                    "meta/info.json; passing --vlm.camera_key=<key> at the "
                    "CLI now also seeds the cameras list as a fallback."
                )
                self._warned_no_camera = True
            staging.write("vqa", [])
            return

        # Build all messages first (one per (frame, camera)), then issue them
        # as a single batched generate_json call so the client can fan them
        # out concurrently.
        per_call: list[tuple[float, str, str, list[dict[str, Any]]]] = []
        for idx in anchor_idx:
            ts = float(record.frame_timestamps[idx])
            qtype = rng.choice(self.config.question_types)
            for camera in cameras:
                messages = self._build_messages(record, qtype, ts, camera)
                # Skip cameras that decoded to zero frames at this ts: no point
                # asking the VLM to ground a bbox without an image.
                if not _has_image_block(messages):
                    continue
                per_call.append((ts, camera, qtype, messages))

        if not per_call:
            staging.write("vqa", [])
            return

        results = self.vlm.generate_json([m for _, _, _, m in per_call])

        rows: list[dict[str, Any]] = []
        for (ts, camera, _qtype, _messages), result in zip(per_call, results, strict=True):
            qa = self._postprocess(result)
            if qa is None:
                continue
            question, answer = qa
            rows.append(
                {
                    "role": "user",
                    "content": question,
                    "style": "vqa",
                    "timestamp": ts,
                    "camera": camera,
                    "tool_calls": None,
                }
            )
            rows.append(
                {
                    "role": "assistant",
                    "content": json.dumps(answer, sort_keys=True),
                    "style": "vqa",
                    "timestamp": ts,
                    "camera": camera,
                    "tool_calls": None,
                }
            )
        staging.write("vqa", rows)

    def _target_cameras(self) -> list[str]:
        """Return the cameras the ``vqa`` module should iterate per anchored frame.

        Defaults to every camera the provider exposes. Datasets with no
        cameras (or test/null providers) yield an empty list, which makes
        ``run_episode`` a no-op.

        When ``config.restrict_to_default_camera`` is set, VQA grounds on
        only the provider's default camera (the single ``--vlm.camera_key``
        stream), matching the plan / interjection modules so the whole
        pipeline focuses on one view.
        """
        all_cameras = list(getattr(self.frame_provider, "camera_keys", []) or [])
        if getattr(self.config, "restrict_to_default_camera", False):
            default = getattr(self.frame_provider, "camera_key", None)
            if default and default in all_cameras:
                return [default]
            # ``restrict_to_default_camera`` is set but the configured default
            # isn't one the provider exposes. Returning it anyway would make
            # ``_decode`` raise a KeyError deep in frame extraction, so warn and
            # fall through to every available camera instead.
            if default:
                logging.getLogger(__name__).warning(
                    "restrict_to_default_camera is set but camera_key=%r is not in the "
                    "provider's cameras %s; grounding VQA on all available cameras instead.",
                    default,
                    all_cameras,
                )
        return all_cameras

    def _build_messages(
        self,
        record: EpisodeRecord,
        question_type: str,
        frame_timestamp: float,
        camera_key: str,
    ) -> list[dict[str, Any]]:
        prompt = load_prompt("vqa").format(
            episode_task=record.episode_task,
            question_type=question_type,
        )
        images = self.frame_provider.frames_at(record, [frame_timestamp], camera_key=camera_key)
        content = [*to_image_blocks(images), {"type": "text", "text": prompt}]
        return [{"role": "user", "content": content}]

    def _postprocess(self, result: Any) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(result, dict):
            return None
        question = result.get("question")
        answer = result.get("answer")
        if not isinstance(question, str) or not question.strip():
            return None
        if not isinstance(answer, dict):
            return None
        # The validator will enforce shape; here we just sanity-check that the
        # answer matches *some* known shape so we can drop garbage early.
        if classify_vqa_answer(answer) is None:
            return None
        return question.strip(), answer


def _has_image_block(messages: list[dict[str, Any]]) -> bool:
    """Return True if any user content block is a populated image block."""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                return True
    return False
