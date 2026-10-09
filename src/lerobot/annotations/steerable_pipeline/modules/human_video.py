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
"""``human_video`` module: robot → human demonstration videos.

For every subtask segment, following the HumanGen pipeline of Zero-WAM
(https://arxiv.org/abs/2608.26103):

1. the shared VLM reads frames of the robot segment and writes the human task, the object states
   and an image-edit prompt;
2. an image-editing model turns the segment's first frame into a first-person human view (robot
   removed, two hands at rest);
3. the VLM writes a step-by-step video prompt from the edited frame and the robot frames;
4. an image-to-video model animates the edited frame.

Steps 2 and 4 run on Hugging Face Inference Providers. Videos and first frames are written under
``<root>/<output_dir>/episode_XXXXXX/`` and described in the staged ``human_video.json``, which the
executor merges into ``meta/human_videos.jsonl``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import PIL.Image

from ..config import HumanVideoConfig
from ..frames import FrameProvider, null_provider, to_image_blocks
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "human_video.json"


class HumanVideoGenerator(Protocol):
    """Image editing + image-to-video backend."""

    def edit_image(self, image: PIL.Image.Image, prompt: str) -> PIL.Image.Image: ...

    def image_to_video(
        self, image: PIL.Image.Image, prompt: str, end_image: PIL.Image.Image | None = None
    ) -> bytes: ...


def _png_bytes(image: PIL.Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _png_data_url(image: PIL.Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(image)).decode()


def motion_instruction(hand: str) -> str:
    """How the video prompt describes the acting hand(s); a single acting hand keeps the other still."""
    if hand == "both":
        return (
            "how both hands work together: say what the left hand and what the right hand does as they reach the "
            "objects, grasp, move and release them at the target, then both return to rest without touching "
            "anything."
        )
    other = "left" if hand == "right" else "right"
    return (
        f"how the {hand} hand reaches the object, grasps it, moves it and releases it at the target, then returns "
        f"to rest without touching anything. The {other} hand stays completely still, resting on the near edge of "
        "the table for the whole video."
    )


@dataclass
class InferenceProvidersGenerator:
    """:class:`HumanVideoGenerator` on Hugging Face Inference Providers.

    ``api_key`` defaults to the logged-in Hugging Face token, so requests are routed through
    Hugging Face and billed to that account.
    """

    config: HumanVideoConfig
    api_key: str | None = None

    def __post_init__(self) -> None:
        from huggingface_hub import InferenceClient  # noqa: PLC0415

        self._client = InferenceClient(provider=self.config.provider, api_key=self.api_key)

    def edit_image(self, image: PIL.Image.Image, prompt: str) -> PIL.Image.Image:
        edited = self._client.image_to_image(
            _png_bytes(image),
            prompt=prompt,
            model=self.config.edit_model,
            guidance_scale=self.config.edit_guidance_scale,
        )
        # Editors may return a different (often square) size; keep the camera's aspect ratio.
        return edited.convert("RGB").resize(image.size, PIL.Image.LANCZOS)

    def _video_parameters(self, prompt: str, end_image: PIL.Image.Image | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"prompt": prompt, "resolution": self.config.resolution}
        params["duration"] = self.config.duration_s
        if self.config.seed is not None:
            params["seed"] = self.config.seed
        if end_image is not None:
            params["end_image_url"] = _png_data_url(end_image)
        return params

    def image_to_video(
        self, image: PIL.Image.Image, prompt: str, end_image: PIL.Image.Image | None = None
    ) -> bytes:
        params = self._video_parameters(prompt, end_image)
        try:
            return self._client.image_to_video(_png_bytes(image), model=self.config.video_model, **params)
        except ValueError as err:
            # Models such as MiniMax-H3 are mapped under the newer ``image-text-to-video`` task,
            # which InferenceClient has no method for yet; its fal image-to-video helper speaks
            # the same protocol.
            if "image-text-to-video" not in str(err) or self.config.provider != "fal-ai":
                raise
        return self._fal_image_text_to_video(image, params)

    def _fal_image_text_to_video(self, image: PIL.Image.Image, params: dict[str, Any]) -> bytes:
        from huggingface_hub import get_token  # noqa: PLC0415
        from huggingface_hub.inference._providers.fal_ai import FalAIImageToVideoTask  # noqa: PLC0415
        from huggingface_hub.utils import get_session, hf_raise_for_status  # noqa: PLC0415

        helper = FalAIImageToVideoTask()
        helper.task = "image-text-to-video"
        request = helper.prepare_request(
            inputs=_png_bytes(image),
            parameters=params,
            headers={},
            model=self.config.video_model,
            api_key=self.api_key or get_token(),
        )
        response = get_session().post(request.url, json=request.json, headers=request.headers)
        hf_raise_for_status(response)
        return helper.get_response(response.json(), request)


def subtask_spans(record: EpisodeRecord, staging: EpisodeStaging) -> list[dict[str, Any]]:
    """Subtask spans ``{text, start, end}``: from the ``plan`` staging, else from the dataset."""
    rows = [r for r in staging.read("plan") if r.get("style") == "subtask"] if staging.has("plan") else []
    if not rows:
        frames = record.frames_df()
        if "language_persistent" in frames.columns and len(frames):
            stored = frames["language_persistent"].iloc[0]  # the episode's list, repeated on every frame
            rows = [r for r in (list(stored) if stored is not None else []) if r.get("style") == "subtask"]
    rows = sorted(rows, key=lambda r: float(r["timestamp"]))
    end_of_episode = float(record.frame_timestamps[-1])
    spans = []
    for i, row in enumerate(rows):
        start = float(row["timestamp"])
        end = float(rows[i + 1]["timestamp"]) if i + 1 < len(rows) else end_of_episode
        if end > start and row.get("content"):
            spans.append({"text": str(row["content"]), "start": start, "end": end})
    return spans


def _nearest(timestamps: Sequence[float], t: float) -> float:
    return min(timestamps, key=lambda ts: abs(ts - t))


def _to_pil(frame: Any) -> PIL.Image.Image:
    return to_image_blocks([frame])[0]["image"].convert("RGB")


@dataclass
class HumanVideoModule:
    """Generates one human demonstration video per subtask segment."""

    vlm: VlmClient
    config: HumanVideoConfig
    root: Path
    generator: HumanVideoGenerator | None = None
    frame_provider: FrameProvider = field(default_factory=null_provider)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _camera(self) -> str | None:
        if self.config.camera_key:
            return self.config.camera_key
        return getattr(self.frame_provider, "camera_key", None) or next(
            iter(self.frame_provider.camera_keys), None
        )

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        spans = subtask_spans(record, staging)
        if self.config.max_segments_per_episode is not None:
            spans = spans[: self.config.max_segments_per_episode]
        if not spans:
            logger.warning("human_video: episode %d has no subtask spans; skipped", record.episode_index)
            self._write_manifest(staging, [])
            return
        camera = self._camera()
        out_dir = self.root / self.config.output_dir / f"episode_{record.episode_index:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        contexts = [self._context(record, span, camera) for span in spans]
        plans = self.vlm.generate_json(
            [self._plan_messages(record, span, ctx) for span, ctx in zip(spans, contexts, strict=True)]
        )

        def generate(k: int) -> dict[str, Any]:
            return self._generate_segment(record, k, spans[k], contexts[k], plans[k], camera, out_dir)

        with ThreadPoolExecutor(max_workers=max(1, self.config.max_concurrency)) as pool:
            entries = list(pool.map(generate, range(len(spans))))
        self._write_manifest(staging, entries)

    def _context(self, record: EpisodeRecord, span: dict[str, Any], camera: str | None) -> list[Any]:
        """First frame (at ``start + frame_offset_s``) followed by ``context_frames`` robot frames."""
        ts = record.frame_timestamps
        first = _nearest(ts, min(span["start"] + self.config.frame_offset_s, span["end"]))
        n = max(1, self.config.context_frames)
        samples = [span["start"] + (span["end"] - span["start"]) * i / max(1, n - 1) for i in range(n)]
        return self.frame_provider.frames_at(
            record, [first, *[_nearest(ts, t) for t in samples]], camera_key=camera
        )

    def _plan_messages(
        self, record: EpisodeRecord, span: dict[str, Any], frames: list[Any]
    ) -> list[dict[str, Any]]:
        robot_frames = frames[1:] or frames
        prompt = load_prompt("human_video_plan").format(
            n=len(robot_frames),
            duration=span["end"] - span["start"],
            subtask=span["text"],
            task=record.episode_task,
        )
        return [
            {"role": "user", "content": [*to_image_blocks(robot_frames), {"type": "text", "text": prompt}]}
        ]

    def _video_messages(
        self, edited: PIL.Image.Image, frames: list[Any], plan: dict[str, Any]
    ) -> list[dict[str, Any]]:
        prompt = load_prompt("human_video_prompt").format(
            human_task=plan.get("human_task", ""),
            object=plan.get("object", ""),
            target=plan.get("target", ""),
            steps="; ".join(map(str, plan.get("steps") or [])),
            final_state=plan.get("final_state", "the task completed"),
            motion=motion_instruction(plan.get("hand", "right")),
            duration_s=self.config.duration_s,
        )
        images = [{"type": "image", "image": edited}, *to_image_blocks(frames[1:])]
        return [{"role": "user", "content": [*images, {"type": "text", "text": prompt}]}]

    def _generate_segment(
        self,
        record: EpisodeRecord,
        k: int,
        span: dict[str, Any],
        frames: list[Any],
        plan: Any,
        camera: str | None,
        out_dir: Path,
    ) -> dict[str, Any]:
        stem = f"segment_{k:03d}"
        entry: dict[str, Any] = {
            "episode_index": record.episode_index,
            "segment_index": k,
            "subtask": span["text"],
            "start_timestamp": span["start"],
            "end_timestamp": span["end"],
            "camera": camera,
            "provider": self.config.provider,
            "edit_model": self.config.edit_model,
            "video_model": self.config.video_model,
            "resolution": self.config.resolution,
        }
        try:
            if not frames or not isinstance(plan, dict):
                raise ValueError("missing frames or VLM plan")
            first = _to_pil(frames[0])
            entry.update(
                {
                    "human_task": plan.get("human_task"),
                    "edit_prompt": plan.get("edit_prompt"),
                    "end_edit_prompt": plan.get("end_edit_prompt"),
                }
            )
            edited = first
            if self.config.edit_model and self.generator is not None:
                edited = self.generator.edit_image(first, str(plan.get("edit_prompt") or ""))
            edited.save(out_dir / f"{stem}_first_frame.png")
            end = None
            if (
                self.config.end_frame
                and self.config.edit_model
                and self.generator is not None
                and len(frames) > 1
            ):
                end = self.generator.edit_image(_to_pil(frames[-1]), str(plan.get("end_edit_prompt") or ""))
                end.save(out_dir / f"{stem}_last_frame.png")
            video_plan = self.vlm.generate_json([self._video_messages(edited, frames, plan)])[0]
            if not isinstance(video_plan, dict) or not video_plan.get("video_prompt"):
                raise ValueError(f"VLM returned no video prompt: {video_plan!r}")
            entry.update({"video_prompt": video_plan["video_prompt"], "caption": video_plan.get("caption")})
            if self.generator is None:
                raise RuntimeError("no generator configured")
            video = self.generator.image_to_video(edited, str(video_plan["video_prompt"]), end_image=end)
            (out_dir / f"{stem}.mp4").write_bytes(video)
            entry.update(
                {
                    "status": "ok",
                    "video_path": str((out_dir / f"{stem}.mp4").relative_to(self.root)),
                    "first_frame_path": str((out_dir / f"{stem}_first_frame.png").relative_to(self.root)),
                    "hand": plan.get("hand"),
                }
            )
            if end is not None:
                entry["last_frame_path"] = str((out_dir / f"{stem}_last_frame.png").relative_to(self.root))
        except Exception as err:  # noqa: BLE001  - one failed segment must not stop the episode
            logger.warning("human_video: episode %d segment %d failed: %s", record.episode_index, k, err)
            entry.update({"status": "failed", "error": str(err)[:500]})
        return entry

    @staticmethod
    def _write_manifest(staging: EpisodeStaging, entries: list[dict[str, Any]]) -> None:
        staging.episode_dir.mkdir(parents=True, exist_ok=True)
        (staging.episode_dir / MANIFEST_FILENAME).write_text(json.dumps(entries, indent=2))
