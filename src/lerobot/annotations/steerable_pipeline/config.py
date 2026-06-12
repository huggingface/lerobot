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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PlanConfig:
    """``plan`` module: subtasks + plan + memory + task augmentation."""

    enabled: bool = True

    # ``task_aug`` rephrasings at t=0 (renderer rotates ${task} among them); 0 disables.
    n_task_rephrasings: int = 10

    # Derive the task from video instead of episode_task: off / if_short / always.
    # Affects prompts only; ``meta/tasks.parquet`` is untouched.
    derive_task_from_video: str = "if_short"
    derive_task_min_words: int = 3

    # --- Frame input: timestamped contact sheets (always on) ---------------
    # The subtask describe/segment passes ALWAYS render the episode as
    # macrodata/refiner-style contact sheets: sampled frames packed into JPEG
    # grids with each frame's timestamp burned into its corner, so the VLM
    # cites the exact source time of a boundary directly. This is far cheaper
    # in vision tokens than one image per frame (≈2× faster subtask generation
    # in practice), which is why the sampling is dense by default.
    #
    # ``frames_per_second`` is the sampling rate: 2.0 = one frame every 0.5s.
    frames_per_second: float = 2.0
    # Frame budget per VLM call (= columns × rows × sheets). When a whole
    # episode sampled at ``frames_per_second`` exceeds this, the episode is
    # AUTOMATICALLY split into consecutive windows of
    # ``max_frames_per_prompt`` frames each (one describe→segment call per
    # window, still at the full ``frames_per_second`` density), and the
    # per-window spans are merged + stitched into one contiguous cover. So an
    # episode of any length is always covered at the full sampling density.
    max_frames_per_prompt: int = 60
    contact_sheet_columns: int = 5
    contact_sheet_frames_per_sheet: int = 20
    contact_sheet_frame_width: int = 224
    contact_sheet_quality: int = 84

    min_subtask_seconds: float = 1.5
    plan_max_steps: int = 8

    # Narrate-only grounding pass before segmenting — best defense against subtasks
    # invented from the task text (+1 VLM call/episode).
    subtask_describe_first: bool = True

    # Emit ``style="plan"`` rows at each boundary; False = subtasks + memory only.
    emit_plan: bool = True

    # Emit ``style="memory"`` rows at each boundary; False = subtasks (+ plan) only.
    # Symmetric counterpart of ``emit_plan``.
    emit_memory: bool = True

    # (subtask spans are always stitched to a contiguous full-episode cover; not configurable.)

    # Optional EgoMimic-style 5-axis task augmentation; replaces n_task_rephrasings.
    task_aug_axes: TaskAugAxesConfig = field(default_factory=lambda: TaskAugAxesConfig())


@dataclass
class TaskAugAxesConfig:
    """5-axis t=0 task augmentation (EgoMimic-style): synonym / omit_arm /
    omit_orientation / omit_grasp_method / combined. Replaces n_task_rephrasings
    when enabled; each variant becomes a ``task_aug`` row. Axes with nothing to
    omit emit fewer entries. Defaults (3+3+2+2+2) match EgoMimic."""

    enabled: bool = False

    synonym_paraphrase: int = 3
    omit_arm: int = 3
    omit_orientation: int = 2
    omit_grasp_method: int = 2
    combined_omissions: int = 2


@dataclass
class InterjectionsConfig:
    """``interjections`` module: interjections + paired speech."""

    enabled: bool = True

    # Each emits a paired (interjection, speech) row + a plan refresh at that ts.
    max_interjections_per_episode: int = 3
    interjection_min_t: float = 2.0

    # Frame window centered on the timestamp so the VLM sees motion, not one frame.
    interjection_window_seconds: float = 2.0
    interjection_window_frames: int = 4


@dataclass
class VqaConfig:
    """``vqa`` module: general VQA."""

    enabled: bool = True
    vqa_emission_hz: float = 1.0
    K: int = 1
    """Consecutive frames per emission tick. The VLM grounds on the FIRST frame,
    so K>1 smears stale labels onto moved frames. Default 1 (no smear)."""
    question_types: tuple[str, ...] = ("bbox", "keypoint", "count", "attribute", "spatial")

    # True: ground VQA only on --vlm.camera_key (default: every camera).
    restrict_to_default_camera: bool = False


@dataclass
class VlmConfig:
    """Shared Qwen-VL client configuration."""

    # Only ``openai`` (OpenAI-compatible vLLM server, auto-spawned when
    # auto_serve=True); ``stub`` is for tests.
    backend: str = "openai"
    model_id: str = "Qwen/Qwen3.6-27B"

    # OpenAI-compatible endpoint; ``EMPTY`` key works for local servers.
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"

    # Spawn a server if none answers api_base; False = fail fast on a remote.
    auto_serve: bool = True
    serve_port: int = 8000
    # Override the auto-serve command; ``{port}`` substituted per replica.
    serve_command: str | None = None

    # Independent servers for round-robin routing (one per GPU). num_gpus=0 = one each.
    parallel_servers: int = 1
    num_gpus: int = 0
    client_concurrency: int = 16
    serve_ready_timeout_s: float = 600.0

    max_new_tokens: int = 512
    temperature: float = 0.2

    # Auto-serve context length (None → 32768); other vLLM flags go in serve_command.
    max_model_len: int | None = None

    # Camera for keyframes; None → first ``observation.images.*`` key.
    camera_key: str | None = None
    # Forwarded as extra_body.chat_template_kwargs (e.g. {"enable_thinking": false}).
    chat_template_kwargs: dict[str, Any] | None = None


@dataclass
class ExecutorConfig:
    """Executor settings (intra-process episode concurrency; distribution via HF Jobs)."""

    # Episodes processed concurrently per phase; main knob for saturating the servers.
    episode_parallelism: int = 16


@dataclass
class AnnotationPipelineConfig:
    """Top-level config for ``lerobot-annotate`` (rewrites data shards in place)."""

    # Hub dataset: download source when ``root`` unset; push target when push_to_hub
    # is on and ``new_repo_id`` unset.
    repo_id: str | None = None

    # Separate push target (matches the LeRobot edit tools). Unset → push in place.
    new_repo_id: str | None = None

    root: Path | None = None

    # Defaults to ``<root>/.annotate_staging/``.
    staging_dir: Path | None = None

    seed: int = 1729

    plan: PlanConfig = field(default_factory=PlanConfig)
    interjections: InterjectionsConfig = field(default_factory=InterjectionsConfig)
    vqa: VqaConfig = field(default_factory=VqaConfig)

    vlm: VlmConfig = field(default_factory=VlmConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)

    skip_validation: bool = False
    only_episodes: tuple[int, ...] | None = None

    # Keyframe decode backend forwarded to ``decode_video_frames``. None →
    # library default (torchcodec when available, else PyAV). Or pin
    # ``"torchcodec"`` / ``"pyav"`` explicitly.
    video_backend: str | None = None

    # Upload to the Hub (new_repo_id if set, else repo_id; one must be set).
    push_to_hub: bool = False
    push_private: bool = False
    push_commit_message: str | None = None

    def resolved_staging_dir(self, root: Path) -> Path:
        return self.staging_dir if self.staging_dir is not None else root / ".annotate_staging"
