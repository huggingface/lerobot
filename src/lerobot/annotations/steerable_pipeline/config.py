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
class Module1Config:
    """Module 1 hyperparameters: plan + subtasks + memory + task augmentation.

    Subtask decomposition sees the **whole episode** as one Qwen-VL video
    block — no keyframe stride or count: the model handles temporal pooling
    itself and decides where to cut. ``max_video_frames`` only caps the
    number of frames packed into the video block (a model-capacity bound,
    not an annotation-logic knob).
    """

    enabled: bool = True
    n_task_rephrasings: int = 10
    """Number of task rephrasings to generate at ``t=0`` as ``task_aug``
    persistent rows (PR 1 ``CORE_STYLES``). The renderer's ``${task}``
    binding rotates among them deterministically per ``sample_idx``,
    realizing Xiao 2022 / CAST-style task-prompt diversity without
    touching ``meta/tasks.parquet``. Set to 0 to disable."""
    derive_task_from_video: str = "if_short"
    """When to bypass the user-provided ``record.episode_task`` and
    derive a fresh task description from the episode video alone:

    - ``off``       never; always use the canonical task as the basis.
    - ``if_short``  derive when the canonical task is empty, has fewer
                    than ``derive_task_min_words`` words, or matches a
                    placeholder string (``debug``, ``unnamed``, ``tbd``,
                    ...). Default — fixes noisy / placeholder tasks
                    without forcing derivation everywhere.
    - ``always``    ignore the canonical task entirely; always derive
                    from the video. Useful when the dataset's task
                    labels are uniformly bad.

    The video-derived task replaces the canonical task as the basis for
    subtask decomposition, plan, memory, AND the ``task_aug`` rephrasings,
    so every downstream annotation is grounded in what's actually visible.
    ``meta/tasks.parquet`` is NOT modified — the Module-1-derived task
    only lives in ``language_persistent`` rows."""
    derive_task_min_words: int = 3
    """Word-count threshold for ``derive_task_from_video=if_short``."""
    frames_per_second: float = 1.0
    """Sample one image-frame per ``1/fps`` seconds across the episode for
    Module 1's subtask-decomposition prompt. ``1.0`` = 1 fps. Capped by
    ``max_video_frames`` to avoid blowing up the request payload."""
    max_video_frames: int = 128
    """Hard cap on the number of frames Module 1 sends. With ``fps=1`` and
    a 30 s episode this yields 30 frames. Bumped from 32 since each frame
    is small (~30-100 KB PNG when base64'd)."""
    min_subtask_seconds: float = 1.5
    plan_max_steps: int = 8
    use_video_url: bool = False
    """When True (and backend supports it, e.g. ``openai``), Module 1
    sends a ``video_url`` content block pointing at the episode's mp4
    file instead of pre-decoded frames. Lets the server sample frames at
    its own ``fps`` — no in-process conv3d cost. The video file is
    extracted as a per-episode subclip to ``staging/.video_clips/`` so
    the model sees only this episode's frames."""
    use_video_url_fps: float = 1.0
    """Frame-rate hint to send to the server (mm_processor_kwargs.fps).
    Only used when ``use_video_url=True``. ``1.0`` = sample 1 frame per
    second, which is plenty for subtask-boundary detection on most
    manipulation episodes."""


@dataclass
class Module2Config:
    """Module 2 hyperparameters: interjections + paired speech."""

    enabled: bool = True
    max_interjections_per_episode: int = 3
    """Number of mid-episode interjections to generate per episode. Each
    creates a paired ``(interjection, speech)`` event row plus triggers a
    ``plan`` refresh at the same timestamp via Module 1. Bumped from the
    original ``1`` after qwen36moe-10 showed plan/interjection coverage
    was too sparse for Hi Robot-style training."""
    interjection_min_t: float = 2.0
    interjection_window_seconds: float = 2.0
    """How many seconds of video to attach to the interjection prompt as
    visual context. Without this the VLM only sees a single frozen frame
    and writes generic interjections that aren't grounded in the actual
    motion happening at the chosen timestamp."""
    interjection_window_frames: int = 4
    """How many frames to sample over ``interjection_window_seconds``.
    Default 4 ⇒ ~0.5 fps over the leading 2 seconds — enough for the
    model to read the ongoing motion, cheap enough to keep prompt size
    bounded for the 32k context."""


@dataclass
class Module3Config:
    """Module 3 hyperparameters: general VQA."""

    enabled: bool = True
    vqa_emission_hz: float = 1.0
    K: int = 3
    question_types: tuple[str, ...] = ("bbox", "keypoint", "count", "attribute", "spatial")


@dataclass
class VlmConfig:
    """Shared Qwen-VL client configuration."""

    backend: str = "openai"
    """One of ``vllm``, ``transformers``, ``openai``, or ``stub`` (tests only).

    Default ``openai`` talks to a local OpenAI-compatible server (vllm /
    transformers) which the CLI auto-spawns when ``auto_serve=True``."""
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    api_base: str = "http://localhost:8000/v1"
    """Base URL for the ``openai`` backend."""
    api_key: str = "EMPTY"
    """API key for the ``openai`` backend; ``EMPTY`` works for local servers."""
    auto_serve: bool = True
    """When True with ``backend=openai``, the CLI probes ``api_base``
    first; if no server answers, it spawns one (default:
    ``transformers serve``), waits for it to be ready, runs the
    pipeline, and tears it down on exit. Default ``True`` so a single
    ``lerobot-annotate`` call can drive the whole flow. Set to ``False``
    if you want to fail fast when no server is reachable (e.g. you're
    pointing at a remote endpoint that should already be up)."""
    serve_port: int = 8000
    """Port the auto-spawned server binds to. Sets ``api_base`` automatically."""
    serve_command: str | None = None
    """Override the auto-serve command (full shell command). When ``None``,
    we run ``transformers serve <model_id> --port <serve_port> --continuous-batching``.

    When ``parallel_servers > 1``, the literal ``{port}`` placeholder in
    this command (if present) is substituted per-replica."""
    parallel_servers: int = 1
    """When >1, spawn this many independent inference servers (each pinned
    to a GPU via ``CUDA_VISIBLE_DEVICES`` and listening on
    ``serve_port + i``) and round-robin client requests across them.
    Useful when DP/TP NCCL setup is broken on the node — single-GPU
    replicas don't need cross-GPU communication. When
    ``parallel_servers > num_gpus``, replicas are round-robin-assigned
    to GPUs (e.g. 4 replicas on 2 GPUs → 0,1,0,1)."""
    num_gpus: int = 0
    """How many physical GPUs are available for round-robin replica
    placement. ``0`` means ``parallel_servers`` (one GPU per replica,
    backward-compatible default). Set this to ``2`` with
    ``parallel_servers=4`` to pack 2 replicas per GPU."""
    client_concurrency: int = 16
    """Maximum number of in-flight chat requests the client issues in
    parallel. vllm batches them internally for free, so bumping this
    typically gives big throughput wins on a single TP=1 server. Set to
    ``1`` for strict serial calls."""
    serve_ready_timeout_s: float = 600.0
    """Max seconds to wait for the server to start serving requests."""
    max_new_tokens: int = 512
    temperature: float = 0.2
    json_mode: bool = True
    batch_size: int = 4
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    """Fraction of GPU memory vllm allocates for weights + KV cache.
    Lower (e.g. 0.7) when the vision encoder needs cuDNN workspace, or to
    avoid CUDNN_STATUS_NOT_INITIALIZED on tight VRAM (30B BF16 on 80 GB)."""
    max_model_len: int | None = None
    """Cap context length. ``None`` keeps the model's default; on H100 80 GB
    a 30B BF16 model often needs ``max_model_len=8192`` or smaller to leave
    room for KV cache."""
    trust_remote_code: bool = False
    """Pass ``trust_remote_code`` to HF auto-classes. Default ``False`` —
    only enable for models that actually ship custom code in their repo
    (rare for first-class VL releases). On Qwen3-VL it triggers an
    std::bad_alloc post-load even though the official transformers class
    is sufficient, so leaving this off is safest."""
    camera_key: str | None = None
    """Override the camera stream used for keyframe attachment. ``None`` picks
    the first ``observation.images.*`` key the dataset declares."""
    chat_template_kwargs: dict[str, Any] | None = None
    """Forwarded as ``extra_body.chat_template_kwargs`` on every chat call.
    Use this to pass model-specific template flags such as
    ``{"enable_thinking": false}`` for Qwen3.5/Qwen3.6 to suppress the
    reasoning preamble that otherwise eats the entire ``max_new_tokens``
    budget before any JSON is emitted."""


@dataclass
class ExecutorConfig:
    """Executor selection and SLURM hyperparameters."""

    auto_threshold: int = 32
    force_local: bool = False
    slurm_partition: str | None = None
    slurm_gpus: int = 1
    slurm_time: str = "06:00:00"
    workers: int = 1
    episode_parallelism: int = 16
    """Number of episodes processed concurrently within each module phase.
    Each in-flight episode sends 3–5 dependent VLM calls; bumping this is
    how you actually saturate ``parallel_servers`` and ``client_concurrency``
    — without it, the executor loops one episode at a time and the
    inference servers sit ~90% idle. Set to ``1`` for strict serial
    execution."""


@dataclass
class AnnotationPipelineConfig:
    """Top-level config for ``lerobot-annotate``.

    Mirrors the structure of :class:`lerobot.configs.train.TrainPipelineConfig`:
    a draccus-parsed dataclass that contains nested per-module sub-configs and
    leaves the dataset, executor, and VLM choices independently knobbable.

    Output is always in-place: the writer rewrites ``data/chunk-*/file-*.parquet``
    in place. Multiple revisions of the same dataset live in separate copies.
    """

    repo_id: str | None = None
    root: Path | None = None

    staging_dir: Path | None = None
    """If unset, defaults to ``<root>/.annotate_staging/``."""

    seed: int = 1729

    module_1: Module1Config = field(default_factory=Module1Config)
    module_2: Module2Config = field(default_factory=Module2Config)
    module_3: Module3Config = field(default_factory=Module3Config)

    vlm: VlmConfig = field(default_factory=VlmConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)

    skip_validation: bool = False
    only_episodes: tuple[int, ...] | None = None

    push_to_hub: str | None = None
    """If set, after the pipeline completes, upload the annotated dataset
    root to the Hugging Face Hub as a dataset repo with this id (e.g.
    ``pepijn/super_poulain_steerable``). Creates the repo if missing."""
    push_private: bool = False
    """When ``push_to_hub`` is set, create the repo as private."""
    push_commit_message: str | None = None
    """Override the commit message used for the hub upload."""

    def resolved_staging_dir(self, root: Path) -> Path:
        return self.staging_dir if self.staging_dir is not None else root / ".annotate_staging"
