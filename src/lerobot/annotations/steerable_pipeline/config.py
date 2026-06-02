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
class VocabularyConfig:
    """Phase 0 — dataset-level canonical vocabulary discovery.

    Watches the first ``sample_episodes`` episode videos and asks the VLM
    to derive a small canonical vocabulary (subtask labels + memory
    milestones) that every episode in the dataset will reuse. The VLM
    decides the count itself from what it sees in the clips — short
    pick-and-place demos get ~6 labels, longer multi-step recipes more.
    The output lands at ``meta/canonical_vocabulary.json`` and feeds
    phase 1's subtask + memory generation as both a prompt-side
    constraint and a post-VLM validation gate.

    Why this exists: free-form LLM rephrasing per episode produces near-
    unique subtask strings, which makes the downstream low-level policy's
    conditioning effectively noise — at inference the policy generates a
    *new* paraphrase the action expert has never seen and produces tiny
    cautious actions. Forcing every episode onto the same small set of
    canonical strings gives the action expert dense supervision per
    string and a small target distribution to learn against.

    Set ``enabled=False`` to fall back to free-form generation (original
    behaviour). ``reuse_existing=True`` keeps a hand-edited vocabulary
    file from being clobbered on re-runs.
    """

    enabled: bool = True
    sample_episodes: int = 3
    max_video_frames_per_episode: int = 32
    # When True (default), an existing meta/canonical_vocabulary.json is
    # loaded as-is and no VLM call is made — lets operators hand-edit the
    # file. Set False to always rediscover from the sample episodes.
    reuse_existing: bool = True


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

    # Structured per-subtask action records (Phase 1a + 1b, inspired by
    # EgoMimic's annotator form). For each generated subtask span, the
    # VLM extracts a typed record (verb / object / arm / grasp_type /
    # destination / mistake). A deterministic Python template renders
    # that record back to canonical subtask text — reducing the VLM's
    # "creative" surface to just the perception step. See
    # ``ActionRecordsConfig`` for details. Off by default (back-compat).
    action_records: "ActionRecordsConfig" = field(default_factory=lambda: ActionRecordsConfig())

    # Structured 5-axis augmentation taxonomy for the t=0 task variants
    # (replaces the free-form ``n_task_rephrasings`` flow when enabled).
    # Mirrors EgoMimic's ``augment_prompt.txt`` taxonomy: instead of N
    # free-form rephrasings, the VLM produces variants along named
    # axes (synonym / omit_arm / omit_orientation / omit_grasp_method /
    # combined). Off by default (back-compat).
    task_aug_axes: "TaskAugAxesConfig" = field(default_factory=lambda: TaskAugAxesConfig())


@dataclass
class ActionRecordsConfig:
    """Structured per-subtask action record extraction.

    When ``enabled=True``, after the existing subtask-span generation in
    ``plan_subtasks_memory.py``, the module makes one extra VLM call per
    subtask to extract a typed record::

        {
          "verb": "pick" | "place" | "press" | ...,    # closed vocabulary
          "object": "<canonical_object_name>",
          "arm": "left" | "right" | "both" | null,
          "grasp_type": "pinch" | "wrap" | "hook" | ... | null,
          "destination": "<canonical_destination>" | null,
          "mistake": "<short text>" | null,
        }

    A deterministic Python template then renders the record back to
    canonical subtask text (e.g. ``pick blue cube with left arm using
    pinch grip``). When ``replace_subtask_text=True`` (default), the
    rendered text REPLACES the VLM's free-form subtask text — eliminating
    cross-episode phrasing drift. When ``emit_record_row=True``
    (default), the structured record is also emitted as a row with
    ``style="action_record"`` so downstream consumers can train on the
    typed schema directly.

    Cost: one extra VLM call per subtask. For an 8-subtask episode this
    means ~8x more VLM calls in the plan module — still cheap relative
    to the action-expert training cost, but worth knowing.
    """

    enabled: bool = False

    # When True, replace the VLM-generated subtask text with the
    # deterministic template's rendering of the structured record.
    # Strongly recommended — it's the whole point of the structured
    # intermediate. Set False to keep both representations side by side.
    replace_subtask_text: bool = True

    # When True, emit a separate row with ``style="action_record"`` and
    # ``content=json.dumps(record)`` at the subtask's start timestamp.
    # Lets downstream training consume the typed schema directly (e.g.
    # auxiliary supervision on verb/arm/grasp classification heads).
    emit_record_row: bool = True

    # Frame sampling for the per-subtask VLM call (similar to the
    # interjection module's window). Anchored to the subtask span.
    frames_per_subtask: int = 4

    # Closed verb vocabulary. The prompt instructs the VLM to pick
    # exactly one. Override per-dataset (e.g. ``["pick", "place", "open",
    # "close"]`` for door-only manipulation) for tighter constraint.
    verb_vocabulary: tuple[str, ...] = (
        "pick", "place", "push", "pull", "open", "close", "turn",
        "press", "lift", "insert", "pour", "move", "reach", "grasp",
        "release", "wipe", "dump",
    )

    # Closed grasp-type vocabulary. ``null`` is always allowed (no
    # contact / unclear). Adjust per-hardware (e.g. drop ``hook`` /
    # ``key`` for parallel-jaw grippers).
    grasp_vocabulary: tuple[str, ...] = (
        "pinch", "wrap", "hook", "key", "lateral",
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

    vocabulary: VocabularyConfig = field(default_factory=VocabularyConfig)
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
