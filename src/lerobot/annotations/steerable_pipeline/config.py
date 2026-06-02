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

    # Frame sampling for the subtask-decomposition prompt. Frames are
    # sampled uniformly across the whole episode up to ``max_video_frames``
    # (so longer episodes are subsampled, not truncated).
    #
    # ``max_video_frames`` is a HARD context-budget cap. With the embedded-
    # frame path (use_video_url=false), every frame becomes ~250-320 vision
    # tokens, so 128 frames ≈ 33-39k tokens — over a 32k-context VLM. 32
    # frames (~8-10k tokens) leaves ample room for the prompt + the
    # describe / verify passes. Raise only if your serving context is
    # larger AND your episodes need finer temporal resolution; if you hit
    # "Input length exceeds maximum context length", lower this.
    frames_per_second: float = 1.0
    max_video_frames: int = 32

    # Windowed subtask generation for CONSTANT temporal density. When > 0
    # and an episode is longer than this many seconds, the plan module
    # processes the episode in consecutive windows of this length, each
    # sampled at ``frames_per_second``, instead of subsampling the whole
    # episode to ``max_video_frames`` (which makes long episodes sparse).
    # The describe -> segment -> verify chain runs per window; results are
    # offset to absolute time, merged, and stitched into a contiguous
    # whole-episode cover. Cost scales with episode length (≈ chain calls
    # × ceil(duration / window)). Set to ~max_video_frames / frames_per_
    # second (e.g. 32s at 1 fps) so each window fills — but never exceeds —
    # the per-call frame budget. 0 disables (single whole-episode call).
    subtask_window_seconds: float = 0.0

    min_subtask_seconds: float = 1.5
    plan_max_steps: int = 8

    # Multi-call subtask quality chain. ON by default — the single-call
    # 'watch video -> emit subtask JSON' pattern makes the VLM commit to
    # structured output before reasoning about the video, so it
    # pattern-matches the task text and hallucinates steps. The chain
    # costs 2 extra VLM calls/episode (3 total for subtasks) but is the
    # difference between trustworthy and fabricated labels. Set either to
    # False to trade quality for fewer calls on datasets you've verified
    # are easy.
    #
    # ``subtask_describe_first``: run a grounding pass that narrates ONLY
    # what is visible in the video (no subtask JSON yet), then inject that
    # description into the segmentation prompt. Forces the model to
    # observe before committing to structured output — the strongest
    # lever against subtasks invented from the task text. +1 VLM call/ep.
    subtask_describe_first: bool = True
    # ``subtask_verify``: after segmentation, re-watch the video and drop
    # any proposed subtask that can't be verified as visible. Prunes
    # hallucinations; can only remove subtasks, never add/rewrite them.
    # Fail-open (keeps un-verified spans if the verify call returns
    # nothing). +1 VLM call/ep.
    subtask_verify: bool = True

    # NOTE: subtask spans are ALWAYS stitched into a contiguous
    # full-episode cover (first subtask pulled back to t0, gaps closed,
    # last span extended to t_last) as a deterministic post-step in
    # ``_generate_subtasks._stitch_full_coverage``. This is not
    # configurable — a sparse / gap-ridden subtask timeline is never
    # desirable for conditioning, so it is unconditional.

    # When True (and backend supports it, e.g. ``openai``), the ``plan``
    # module sends a ``video_url`` block pointing at a per-episode mp4
    # subclip and lets the server sample frames at ``use_video_url_fps``.
    use_video_url: bool = False
    use_video_url_fps: float = 1.0

    # Structured per-subtask action records (Phase 1a + 1b, inspired by
    # EgoMimic's annotator form). For each generated subtask span, the
    # VLM extracts a typed record (verb / object / arm / grasp_type /
    # destination / mistake). A deterministic Python template renders
    # that record back to canonical subtask text — reducing the VLM's
    # "creative" surface to just the perception step. See
    # ``ActionRecordsConfig`` for details. Off by default (back-compat).
    action_records: ActionRecordsConfig = field(default_factory=lambda: ActionRecordsConfig())

    # Structured 5-axis augmentation taxonomy for the t=0 task variants
    # (replaces the free-form ``n_task_rephrasings`` flow when enabled).
    # Mirrors EgoMimic's ``augment_prompt.txt`` taxonomy: instead of N
    # free-form rephrasings, the VLM produces variants along named
    # axes (synonym / omit_arm / omit_orientation / omit_grasp_method /
    # combined). Off by default (back-compat).
    task_aug_axes: TaskAugAxesConfig = field(default_factory=lambda: TaskAugAxesConfig())


@dataclass
class ActionRecordsConfig:
    """Structured per-subtask action record extraction.

    When ``enabled=True``, after the existing subtask-span generation in
    ``plan_subtasks_memory.py``, the module makes one extra VLM call per
    subtask to extract a typed record::

        {
            "verb": "pick" | "place" | "press" | ...,  # closed vocabulary
            "object": "<canonical_object_name>",
            "arm": "left" | "right" | "both" | null,
            "grasp_type": "pinch" | "wrap" | "hook" | ... | null,
            "destination": "<canonical_destination>" | null,
            "mistake": "<short text>" | null,
        }

    The record is emitted as a separate row with ``style="action_record"``
    (``content=json.dumps(record)``) at the subtask's start timestamp.
    It is PURELY ADDITIVE — it never touches the VLM's subtask text.
    Downstream training can consume the typed schema directly (e.g.
    auxiliary supervision on verb / arm / grasp classification heads)
    while the subtask string the policy conditions on stays exactly what
    the subtask module produced. (Reconstructing subtask text from these
    fields was too easy for the VLM to hallucinate on tasks that don't
    fit the manipulation schema — navigation tasks yielded nonsense like
    ``move stove to stove`` — so that path was removed.)

    Cost: one extra VLM call per subtask. For an 8-subtask episode this
    means ~8x more VLM calls in the plan module — still cheap relative
    to the action-expert training cost, but worth knowing.
    """

    enabled: bool = False

    # When True (default), emit a separate row with ``style="action_record"``
    # and ``content=json.dumps(record)`` at the subtask's start timestamp.
    # This is the only output of the feature — set ``enabled=False`` to
    # skip the extra VLM calls entirely.
    emit_record_row: bool = True

    # Frame sampling for the per-subtask VLM call (similar to the
    # interjection module's window). Anchored to the subtask span.
    frames_per_subtask: int = 4

    # Closed verb vocabulary. The prompt instructs the VLM to pick
    # exactly one. Override per-dataset (e.g. ``["pick", "place", "open",
    # "close"]`` for door-only manipulation) for tighter constraint.
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

    # Closed grasp-type vocabulary. ``null`` is always allowed (no
    # contact / unclear). Adjust per-hardware (e.g. drop ``hook`` /
    # ``key`` for parallel-jaw grippers).
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
    flow with a structured prompt that produces variants along five
    named axes (mirroring EgoMimic's ``augment_prompt.txt``):

      * ``synonym_paraphrase`` — different wording / verbs, all
        information preserved.
      * ``omit_arm`` — drop the left/right/both arm specification.
      * ``omit_orientation`` — drop orientation cues (upright,
        sideways, ...).
      * ``omit_grasp_method`` — drop grip / grasp method specification.
      * ``combined_omissions`` — combine two of the above
        simultaneously.

    Default counts (3+3+2+2+2 = 12 variants per task) match EgoMimic.
    Axes that have nothing to omit in the source task (e.g. ``omit_arm``
    when the task doesn't mention an arm) emit fewer entries rather
    than pad — the prompt instructs the VLM accordingly.

    Each variant is emitted as a ``task_aug`` row at ``t=0`` (same
    style as the free-form variants), so the rest of the pipeline /
    training recipe doesn't need to know about the taxonomy.
    """

    enabled: bool = False

    synonym_paraphrase: int = 3
    omit_arm: int = 3
    omit_orientation: int = 2
    omit_grasp_method: int = 2
    combined_omissions: int = 2

    @property
    def total(self) -> int:
        """Sum of requested variants across all axes (upper bound)."""
        return (
            self.synonym_paraphrase
            + self.omit_arm
            + self.omit_orientation
            + self.omit_grasp_method
            + self.combined_omissions
        )


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
    K: int = 1
    """How many *consecutive* frames each emission tick anchors a VQA pair
    to. The VLM grounds its answer (bbox / keypoint coordinates, count, …)
    against the *first* anchored frame's image, so anchoring K>1 frames
    copies that same answer onto later frames where the scene has already
    moved — stale labels. Default ``1``: a VQA pair lands on exactly its
    emission frame, no temporal smear. Raise it only to trade label
    precision for more (noisier) VQA frames."""
    question_types: tuple[str, ...] = ("bbox", "keypoint", "count", "attribute", "spatial")

    # Camera restriction. By default VQA iterates EVERY camera the
    # dataset declares (one VQA pair per camera per emission tick). Set
    # ``restrict_to_default_camera=True`` to ground VQA on only the
    # single ``--vlm.camera_key`` stream — the same camera the plan /
    # interjection modules use — so the whole pipeline focuses on one
    # view. Use this when you want every annotation grounded on, e.g.,
    # ``observation.images.base`` and nothing else.
    restrict_to_default_camera: bool = False


@dataclass
class VlmConfig:
    """Shared Qwen-VL client configuration."""

    # One of ``vllm``, ``transformers``, ``openai``, or ``stub`` (tests).
    # ``openai`` talks to a local OpenAI-compatible server; the CLI
    # auto-spawns one when ``auto_serve=True``.
    backend: str = "openai"
    model_id: str = "Qwen/Qwen3.6-35B-A3B-FP8"

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

    # Keyframe decode backend. When unset, the pipeline decodes with the
    # ffmpeg CLI: it decodes AV1 and runs each decode as an isolated child
    # process, which is both crash-safe and safe under the concurrent
    # decode the executor performs (torchcodec is not thread-safe and
    # SIGSEGVs there). Set to ``"torchcodec"`` or ``"pyav"`` to pin an
    # in-process decoder when its build is known thread-safe.
    video_backend: str | None = None

    # When True, upload the annotated dataset to the Hugging Face Hub:
    # to ``dest_repo_id`` if set, otherwise back to ``repo_id``. One of
    # the two must be set for this to take effect.
    push_to_hub: bool = False
    push_private: bool = False
    push_commit_message: str | None = None

    def resolved_staging_dir(self, root: Path) -> Path:
        return self.staging_dir if self.staging_dir is not None else root / ".annotate_staging"
