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

"""Language annotations from simulator ground truth, in the format ``lerobot-annotate`` writes.

Subtasks come from the script, so their boundaries are exact. VQA answers (boxes, points, counts,
colors, spatial relations) come from the scene state and a segmentation render of the very frame.
Rows are built against frame indices; ``write_language_columns`` stamps them with the dataset's own
frame timestamps once it is finalized, then stages, validates and writes them with the annotation
pipeline's own classes.
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.annotations.steerable_pipeline.executor import Executor
from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.validator import StagingValidator
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter

from .config import AnnotationConfig
from .motion import Plan
from .render import Visible
from .scene import SceneParams
from .tasks import TaskInstance

BBOX_QUESTIONS = [
    "Where is the {label}?",
    "Find the {label}.",
    "Locate the {label} in the image.",
    "Detect the {label}.",
]
ALL_BBOX_QUESTIONS = [
    "Detect all the objects.",
    "Locate every object in the scene.",
    "What objects do you see?",
]
POINT_QUESTIONS = ["Point to the {label}.", "Where is the center of the {label}?", "Mark the {label}."]
COUNT_QUESTIONS = ["How many objects are on the table?", "Count the objects you can see."]
COLOR_QUESTIONS = ["What color is the {noun}?", "Which color is the {noun}?"]
SPATIAL_QUESTIONS = ["Where is the {a} relative to the {b}?", "How is the {a} placed relative to the {b}?"]


@dataclass
class EpisodeRows:
    """One episode's rows, keyed by frame index until the dataset gives their timestamps."""

    persistent: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    events: list[tuple[int, dict[str, Any]]] = field(default_factory=list)


def row(role: str, content: str | None, style: str | None, camera: str | None = None) -> dict[str, Any]:
    return {"role": role, "content": content, "style": style, "camera": camera, "tool_calls": None}


def persistent_rows(instance: TaskInstance, plan: Plan, cfg: AnnotationConfig) -> list[tuple[int, dict]]:
    """``task_aug``, ``subtask``, ``plan`` and ``memory`` rows, at the exact subtask boundaries."""
    rows = [
        (0, row("user", text, "task_aug"))
        for text in [instance.instruction, *instance.phrasings][: cfg.task_aug + 1]
    ]
    changes = np.flatnonzero(np.diff(plan.subtask_of_frame)) + 1
    starts = [0, *changes.tolist()]
    labels = [plan.subtasks[plan.subtask_of_frame[s]] for s in starts]
    for i, (start, label) in enumerate(zip(starts, labels, strict=True)):
        rows.append((start, row("assistant", label, "subtask")))
        remaining = "\n".join(f"{k + 1}. {text}" for k, text in enumerate(labels[i:]))
        rows.append((start, row("assistant", remaining, "plan")))
        if cfg.memory and i > 0 and instance.memories:
            rows.append(
                (start, row("assistant", instance.memories[min(i, len(instance.memories)) - 1], "memory"))
            )
    return rows


def vqa_frames(n_frames: int, fps: int, cfg: AnnotationConfig, rng: np.random.Generator) -> list[int]:
    """Frames that get VQA pairs: about ``vqa_hz`` per second, with a random phase."""
    if cfg.vqa_hz <= 0:
        return []
    period = fps / cfg.vqa_hz
    first = rng.uniform(0, period)
    return sorted({int(f) for f in np.arange(first, n_frames, period)})


def vqa_pair(
    camera_key: str,
    visible: dict[str, Visible],
    scene: SceneParams,
    positions: dict[str, np.ndarray],
    cfg: AnnotationConfig,
    rng: np.random.Generator,
) -> tuple[dict, dict] | None:
    """A question about what ``camera_key`` sees and its ground-truth answer, or None if nothing fits."""
    if not visible:
        return None
    kinds = list(cfg.question_weights)
    weights = np.asarray([cfg.question_weights[k] for k in kinds], dtype=np.float64)
    for kind in rng.choice(kinds, size=len(kinds), replace=False, p=weights / weights.sum()):
        qa = _QUESTIONS[kind](visible, scene, positions, rng)
        if qa is not None:
            question, answer = qa
            return (
                row("user", question, "vqa", camera_key),
                row("assistant", json.dumps(answer, sort_keys=True), "vqa", camera_key),
            )
    return None


def _pick(rng: np.random.Generator, items: list):
    return items[rng.integers(len(items))]


def _bbox(visible, scene, positions, rng):
    items = sorted(visible.values(), key=lambda v: v.name)
    if len(items) > 1 and rng.random() < 0.4:
        detections = [{"label": v.label, "bbox_format": "xyxy", "bbox": v.bbox} for v in items]
        return _pick(rng, ALL_BBOX_QUESTIONS), {"detections": detections}
    v = _pick(rng, items)
    return _pick(rng, BBOX_QUESTIONS).format(label=v.label), {
        "detections": [{"label": v.label, "bbox_format": "xyxy", "bbox": v.bbox}]
    }


def _keypoint(visible, scene, positions, rng):
    v = _pick(rng, sorted(visible.values(), key=lambda v: v.name))
    return _pick(rng, POINT_QUESTIONS).format(label=v.label), {
        "label": v.label,
        "point_format": "xy",
        "point": v.center,
    }


def _count(visible, scene, positions, rng):
    objects = [o for o in scene.objects if o.name in visible]
    return _pick(rng, COUNT_QUESTIONS), {"label": "object", "count": len(objects)}


def _attribute(visible, scene, positions, rng):
    """The color of an item whose noun is unique among what the camera sees."""
    items = [i for i in [*scene.objects, *scene.containers] if i.name in visible and i.color]
    nouns = [i.label.split(" ", 1)[1] for i in items]
    unique = [(i, n) for i, n in zip(items, nouns, strict=True) if nouns.count(n) == 1]
    if not unique:
        return None
    item, noun = _pick(rng, unique)
    return _pick(rng, COLOR_QUESTIONS).format(noun=noun), {
        "label": item.label,
        "attribute": "color",
        "value": item.color,
    }


def _spatial(visible, scene, positions, rng):
    """Left/right as the camera sees it, or "in" for an object inside a container."""
    if len(visible) < 2:
        return None
    a, b = (visible[n] for n in rng.choice(sorted(visible), size=2, replace=False))
    container = next((c for c in scene.containers if c.name == b.name), None)
    if container is not None and a.name in positions:
        rel = positions[a.name][:2] - np.asarray(container.pos)
        if np.all(np.abs(rel) < container.inner_half) and positions[a.name][2] < container.wall_height + 0.03:
            relation = "in"
        else:
            relation = "left_of" if a.center[0] < b.center[0] else "right_of"
    else:
        relation = "left_of" if a.center[0] < b.center[0] else "right_of"
    # Too close to call left or right from the image.
    if relation != "in" and abs(a.center[0] - b.center[0]) < 0.05:
        return None
    return _pick(rng, SPATIAL_QUESTIONS).format(a=a.label, b=b.label), {
        "subject": a.label,
        "relation": relation,
        "object": b.label,
    }


_QUESTIONS = {
    "bbox": _bbox,
    "keypoint": _keypoint,
    "count": _count,
    "attribute": _attribute,
    "spatial": _spatial,
}


def write_language_columns(root: Path, rows: dict[int, EpisodeRows], camera_keys: tuple[str, ...]) -> None:
    """Stamp, stage, validate and write every episode's rows into the finalized dataset at ``root``."""
    records = list(iter_episodes(root))
    staging_dir = root / ".annotate_staging"
    for record in records:
        episode = rows.get(record.episode_index)
        if episode is None:
            continue
        stamps = record.frame_timestamps
        staging = EpisodeStaging(staging_dir, record.episode_index)
        staging.write("plan", [{**r, "timestamp": stamps[f]} for f, r in episode.persistent])
        staging.write("vqa", [{**r, "timestamp": stamps[f]} for f, r in episode.events])
    report = StagingValidator(dataset_camera_keys=camera_keys).validate(records, staging_dir)
    if not report.ok:
        raise ValueError(f"Invalid annotations in {root}: {report.errors[:5]}")
    LanguageColumnsWriter().write_all(records, staging_dir, root)
    Executor._ensure_annotation_metadata_in_info(root)
    shutil.rmtree(staging_dir)
