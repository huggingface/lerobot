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
    """``plan`` module: plan + subtasks + memory + task augmentation.

    The ``plan`` module attaches the whole episode as one Qwen-VL video
    block; ``max_video_frames`` only caps the frames packed in (a
    model-capacity bound, not an annotation-logic knob).
    """

    enabled: bool = True

    # Number of ``task_aug`` rephrasings emitted at ``t=0``. The renderer's
    # ``${task}`` binding rotates among them per ``sample_idx``. ``0`` disables.
    n_task_rephrasings: int = 10

    # When to derive the task from the video instead of using
    # ``record.episode_task``: ``off``, ``if_short`` (short / placeholder /
    # missing canonical task), or ``always``. The derived task replaces the
    # canonical one for every ``plan``-module prompt; ``meta/tasks.parquet``
    # is never modified.
    derive_task_from_video: str = "if_short"
    derive_task_min_words: int = 3

    # Frame sampling for the subtask-decomposition prompt.
    frames_per_second: float = 1.0
    max_video_frames: int = 128

    min_subtask_seconds: float = 1.5
    plan_max_steps: int = 8

    # When True (and backend supports it, e.g. ``openai``), the ``plan``
    # module sends a ``video_url`` block pointing at a per-episode mp4
    # subclip and lets the server sample frames at ``use_video_url_fps``.
    use_video_url: bool = False
    use_video_url_fps: float = 1.0


@dataclass
class InterjectionsConfig:
    """``interjections`` module: interjections + paired speech."""

    enabled: bool = True

    # Each interjection emits a paired ``(interjection, speech)`` event row
    # and triggers a ``plan`` refresh at the same timestamp via the
    # ``plan`` module.
    max_interjections_per_episode: int = 3
    interjection_min_t: float = 2.0

    # Visual context attached to the interjection prompt: a short window
    # of frames centered on the chosen timestamp so the VLM sees the
    # ongoing motion rather than a single frozen frame.
    interjection_window_seconds: float = 2.0
    interjection_window_frames: int = 4


@dataclass
class VqaConfig:
    """``vqa`` module: general VQA."""

    enabled: bool = True
    vqa_emission_hz: float = 1.0
    K: int = 3
    question_types: tuple[str, ...] = ("bbox", "keypoint", "count", "attribute", "spatial")


@dataclass
class VlmConfig:
    """Shared Qwen-VL client configuration."""

    # One of ``vllm``, ``transformers``, ``openai``, or ``stub`` (tests).
    # ``openai`` talks to a local OpenAI-compatible server; the CLI
    # auto-spawns one when ``auto_serve=True``.
    backend: str = "openai"
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # OpenAI-compatible server endpoint; ``EMPTY`` works for local servers.
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"

    # When True with ``backend=openai``, the CLI probes ``api_base`` and
    # spawns a server if none answers (default: ``transformers serve``).
    # Set to False to fail fast when pointing at a remote endpoint.
    auto_serve: bool = True
    serve_port: int = 8000
    # Override the auto-serve command. ``{port}`` is substituted per replica
    # when ``parallel_servers > 1``.
    serve_command: str | None = None

    # Run multiple independent inference servers for round-robin client
    # routing (each pinned to a GPU via ``CUDA_VISIBLE_DEVICES`` and bound
    # to ``serve_port + i``). ``num_gpus=0`` means one GPU per replica.
    parallel_servers: int = 1
    num_gpus: int = 0
    client_concurrency: int = 16
    serve_ready_timeout_s: float = 600.0

    max_new_tokens: int = 512
    temperature: float = 0.2
    json_mode: bool = True
    batch_size: int = 4
    tensor_parallel_size: int = 1

    # Fraction of GPU memory vllm allocates for weights + KV cache.
    gpu_memory_utilization: float = 0.9
    # Cap context length (None = model default). On 80 GB H100 a 30B BF16
    # model often needs <= 8192 to leave KV-cache headroom.
    max_model_len: int | None = None
    trust_remote_code: bool = False

    # Override the camera stream used for keyframe attachment. None picks
    # the first ``observation.images.*`` key the dataset declares.
    camera_key: str | None = None
    # Forwarded as ``extra_body.chat_template_kwargs`` on every chat call;
    # use to pass model-specific flags such as ``{"enable_thinking": false}``.
    chat_template_kwargs: dict[str, Any] | None = None


@dataclass
class ExecutorConfig:
    """Executor settings.

    Distributed execution is provided by Hugging Face Jobs (see
    ``examples/annotation/run_hf_job.py``); this config only controls
    intra-process episode concurrency.
    """

    # Episodes processed concurrently within each module phase. Each
    # in-flight episode dispatches 3-5 dependent VLM calls, so this is the
    # main knob for saturating ``parallel_servers`` and ``client_concurrency``.
    episode_parallelism: int = 16


@dataclass
class AnnotationPipelineConfig:
    """Top-level config for ``lerobot-annotate``.

    The writer rewrites ``data/chunk-*/file-*.parquet`` in place. Multiple
    revisions of the same dataset live in separate copies.
    """

    # Hub dataset id. Used as the download source when ``root`` is unset,
    # and as the destination repo when ``push_to_hub`` is enabled and
    # ``dest_repo_id`` is unset.
    repo_id: str | None = None

    # Optional separate Hub dataset id to push the annotated result to. When
    # unset, ``push_to_hub`` uploads back to ``repo_id`` (annotate in place);
    # when set, the source ``repo_id`` is left untouched.
    dest_repo_id: str | None = None

    root: Path | None = None

    # Defaults to ``<root>/.annotate_staging/`` when unset.
    staging_dir: Path | None = None

    seed: int = 1729

    plan: PlanConfig = field(default_factory=PlanConfig)
    interjections: InterjectionsConfig = field(default_factory=InterjectionsConfig)
    vqa: VqaConfig = field(default_factory=VqaConfig)

    vlm: VlmConfig = field(default_factory=VlmConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)

    skip_validation: bool = False
    only_episodes: tuple[int, ...] | None = None

    # When True, upload the annotated dataset to the Hugging Face Hub:
    # to ``dest_repo_id`` if set, otherwise back to ``repo_id``. One of
    # the two must be set for this to take effect.
    push_to_hub: bool = False
    push_private: bool = False
    push_commit_message: str | None = None

    def resolved_staging_dir(self, root: Path) -> Path:
        return self.staging_dir if self.staging_dir is not None else root / ".annotate_staging"
