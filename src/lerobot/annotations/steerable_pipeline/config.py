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

    # Derive the task from video instead of ``record.episode_task``: ``off``,
    # ``if_short`` (canonical task short/placeholder/missing), or ``always``.
    # Affects prompts only; ``meta/tasks.parquet`` is untouched.
    derive_task_from_video: str = "if_short"
    derive_task_min_words: int = 3

    # Frames sampled uniformly, capped at ``max_video_frames`` — a HARD context
    # cap (~250-320 tokens/frame, so 32 fit a 32k VLM; 128 overflow). Lower it
    # if you hit "Input length exceeds maximum context length".
    frames_per_second: float = 1.0
    max_video_frames: int = 32

    # Windowed subtask generation for constant temporal density: when > 0 and
    # the episode is longer, process it in windows of this length (each at
    # ``frames_per_second``) instead of subsampling the whole episode; spans are
    # merged + stitched. ~max_video_frames / frames_per_second. 0 disables.
    subtask_window_seconds: float = 0.0

    min_subtask_seconds: float = 1.5
    plan_max_steps: int = 8

    # Grounding pass that narrates ONLY what's visible before segmenting — the
    # strongest lever against subtasks invented from the task text. ON by
    # default (+1 VLM call/episode); False trades quality for fewer calls.
    subtask_describe_first: bool = True

    # Emit ``style="plan"`` rows (the numbered still-todo list, re-emitted at
    # every subtask boundary). False keeps only subtasks + memory and skips
    # the per-boundary ``_generate_plan`` call.
    emit_plan: bool = True

    # NOTE: subtask spans are ALWAYS stitched into a contiguous full-episode
    # cover (see ``_stitch_full_coverage``) — not configurable.

    # When True, send a server-side ``video_url`` clip (sampled at
    # ``use_video_url_fps``) instead of embedded frames.
    use_video_url: bool = False
    use_video_url_fps: float = 1.0

    # Optional structured per-subtask action records (EgoMimic-style). When
    # enabled, the VLM extracts a typed record per subtask span; see
    # ``ActionRecordsConfig``. Purely additive — off by default.
    action_records: ActionRecordsConfig = field(default_factory=lambda: ActionRecordsConfig())

    # Optional 5-axis task-augmentation taxonomy for the t=0 variants
    # (EgoMimic-style: synonym / omit_arm / omit_orientation /
    # omit_grasp_method / combined). Replaces the free-form
    # ``n_task_rephrasings`` flow when enabled; see ``TaskAugAxesConfig``.
    task_aug_axes: TaskAugAxesConfig = field(default_factory=lambda: TaskAugAxesConfig())


@dataclass
class ActionRecordsConfig:
    """Structured per-subtask action record extraction.

    When ``enabled=True``, after subtask-span generation the module makes
    one extra VLM call per subtask to extract a typed record::

        {
            "verb": "pick" | "place" | "press" | ...,  # closed vocabulary
            "object": "<canonical_object_name>",
            "arm": "left" | "right" | "both" | null,
            "grasp_type": "pinch" | "wrap" | "hook" | ... | null,
            "destination": "<canonical_destination>" | null,
            "mistake": "<short text>" | null,
        }

    Emitted as a separate ``style="action_record"`` row at the subtask's
    start timestamp. PURELY ADDITIVE — it never touches the subtask text,
    so downstream training can use the typed schema (e.g. auxiliary
    verb/arm/grasp heads) while the conditioning string stays unchanged.

    Cost: one extra VLM call per subtask (~8x plan-module calls on an
    8-subtask episode).
    """

    enabled: bool = False

    # Emit the ``style="action_record"`` row (JSON content) at the subtask
    # start — the only output of the feature. ``enabled=False`` skips it.
    emit_record_row: bool = True

    # Frames sampled from the subtask span for the per-subtask VLM call.
    frames_per_subtask: int = 4

    # Closed verb vocabulary; the prompt picks exactly one. Override
    # per-dataset (e.g. door-only manipulation) for a tighter constraint.
    verb_vocabulary: tuple[str, ...] = (
        "pick",
        "place",
        "push",
        "pull",
        "open",
        "close",
        "turn",
        "press",
        "lift",
        "insert",
        "pour",
        "move",
        "reach",
        "grasp",
        "release",
        "wipe",
        "dump",
    )

    # Closed grasp-type vocabulary (``null`` always allowed). Adjust
    # per-hardware (e.g. drop ``hook`` / ``key`` for parallel-jaw grippers).
    grasp_vocabulary: tuple[str, ...] = (
        "pinch",
        "wrap",
        "hook",
        "key",
        "lateral",
    )


@dataclass
class TaskAugAxesConfig:
    """Structured 5-axis augmentation taxonomy for t=0 task variants.

    When ``enabled=True``, replaces the free-form ``n_task_rephrasings``
    flow with variants along five named axes (EgoMimic-style):
    ``synonym_paraphrase`` (reword, keep all info), ``omit_arm``,
    ``omit_orientation``, ``omit_grasp_method``, and ``combined_omissions``
    (drop two at once).

    Default counts (3+3+2+2+2 = 12) match EgoMimic. Axes with nothing to
    omit emit fewer entries rather than pad. Each variant becomes a
    ``task_aug`` row at ``t=0``, identical in style to the free-form ones.
    """

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

    # Each interjection emits a paired (interjection, speech) event row and
    # triggers a ``plan`` refresh at the same timestamp.
    max_interjections_per_episode: int = 3
    interjection_min_t: float = 2.0

    # A short frame window centered on the timestamp so the VLM sees the
    # motion, not one frozen frame.
    interjection_window_seconds: float = 2.0
    interjection_window_frames: int = 4


@dataclass
class VqaConfig:
    """``vqa`` module: general VQA."""

    enabled: bool = True
    vqa_emission_hz: float = 1.0
    K: int = 1
    """Consecutive frames each emission tick anchors a VQA pair to. The VLM
    grounds its answer on the FIRST anchored frame, so K>1 copies that answer
    onto later (moved) frames — stale labels. Default 1 (no smear)."""
    question_types: tuple[str, ...] = ("bbox", "keypoint", "count", "attribute", "spatial")

    # By default VQA iterates every camera (one pair per camera per tick). Set
    # True to ground VQA only on ``--vlm.camera_key`` — the single view the
    # plan / interjection modules use.
    restrict_to_default_camera: bool = False


@dataclass
class VlmConfig:
    """Shared Qwen-VL client configuration."""

    # Only ``openai`` is supported (in-process vllm/transformers were removed;
    # the shipped workflow is HF Jobs). Talks to an OpenAI-compatible vLLM
    # server, auto-spawned in-job when ``auto_serve=True``. ``stub`` is for tests.
    backend: str = "openai"
    model_id: str = "Qwen/Qwen3.6-27B"

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

    # Independent servers for round-robin routing (each pinned to a GPU,
    # bound to ``serve_port + i``). ``num_gpus=0`` = one GPU per replica.
    parallel_servers: int = 1
    num_gpus: int = 0
    client_concurrency: int = 16
    serve_ready_timeout_s: float = 600.0

    max_new_tokens: int = 512
    temperature: float = 0.2

    # Context length for the auto-spawned vLLM server (None → 32768). vLLM
    # tuning flags (tensor-parallel size, GPU memory fraction, ...) go in
    # ``serve_command`` directly, not here.
    max_model_len: int | None = None

    # Override the camera stream used for keyframe attachment. None picks
    # the first ``observation.images.*`` key the dataset declares.
    camera_key: str | None = None
    # Forwarded as ``extra_body.chat_template_kwargs`` on every chat call;
    # use to pass model-specific flags such as ``{"enable_thinking": false}``.
    chat_template_kwargs: dict[str, Any] | None = None


@dataclass
class ExecutorConfig:
    """Executor settings — intra-process episode concurrency only
    (distributed execution is delegated to Hugging Face Jobs)."""

    # Episodes processed concurrently per module phase. Each dispatches 3-5 VLM
    # calls, so this is the main knob for saturating ``parallel_servers`` /
    # ``client_concurrency``.
    episode_parallelism: int = 16


@dataclass
class AnnotationPipelineConfig:
    """Top-level config for ``lerobot-annotate``.

    The writer rewrites ``data/chunk-*/file-*.parquet`` in place. Multiple
    revisions of the same dataset live in separate copies.
    """

    # Hub dataset id: download source when ``root`` is unset, and push target
    # when ``push_to_hub`` is on and ``new_repo_id`` is unset.
    repo_id: str | None = None

    # Optional separate push target (named to match the LeRobot dataset edit
    # tools). Unset → push back to ``repo_id`` in place; set → source untouched.
    new_repo_id: str | None = None

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

    # Keyframe decode backend. Unset → ffmpeg CLI: decodes AV1 in an isolated
    # child process, so it's crash- and thread-safe under concurrent decode
    # (torchcodec SIGSEGVs there). Set ``"torchcodec"`` / ``"pyav"`` to pin one.
    video_backend: str | None = None

    # Upload the annotated dataset to the Hub (to ``new_repo_id`` if set, else
    # back to ``repo_id`` — one of the two must be set).
    push_to_hub: bool = False
    push_private: bool = False
    push_commit_message: str | None = None

    def resolved_staging_dir(self, root: Path) -> Path:
        return self.staging_dir if self.staging_dir is not None else root / ".annotate_staging"
