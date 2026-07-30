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
"""Camera-view curation: per-camera VLM quality + label judgments and the
lightweight (download-free) Hub rename that applies the chosen labels.

The decision (:func:`curate_cameras`) is a pure function of a
``{camera_key: [frames]}`` map and a VLM client, so it unit-tests with a stub
VLM and no dataset. The orchestrating CLI (``lerobot-curate-cameras``) samples
those frames from the dataset's first episode.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lerobot.annotations.steerable_pipeline.frames import to_image_blocks
from lerobot.datasets.dataset_tools import _remap_camera_key_in_meta, _resolve_rename_collisions
from lerobot.datasets.io_utils import load_info, write_info
from lerobot.utils.io_utils import write_json

from .config import CameraCurationConfig

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "camera_curation.txt"

# The canonical prefix every curated camera key gets.
OBS_IMAGE_PREFIX = "observation.images."


@dataclass
class CameraVerdict:
    """One camera's VLM verdict."""

    camera_key: str
    usable: bool
    view_label: str | None
    blur_reason: str | None = None
    confidence: float | None = None
    # Populated by ``build_name_mapping`` once collisions are resolved.
    proposed_new_key: str | None = None


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def is_valid_view_label(label: str, vocabulary: tuple[str, ...], allow_combos: bool) -> bool:
    """True if ``label`` is a single vocab word, or (when allowed) an underscore
    combo of at most two distinct vocab words."""
    if not label:
        return False
    tokens = label.split("_")
    if not allow_combos:
        return len(tokens) == 1 and tokens[0] in vocabulary
    if not (1 <= len(tokens) <= 2):
        return False
    return all(tok in vocabulary for tok in tokens) and len(set(tokens)) == len(tokens)


def _build_messages(frames: list[Any], cfg: CameraCurationConfig) -> list[dict[str, Any]]:
    if cfg.allow_combos:
        combo_rule = (
            "You may combine at most two of these words with an underscore when "
            "one word is not precise enough (e.g. \"left_wrist\"). "
        )
    else:
        combo_rule = "Use exactly one of these words (no combinations). "
    prompt = _load_prompt().format(
        vocabulary=", ".join(cfg.view_vocabulary),
        combo_rule=combo_rule,
    )
    content = [*to_image_blocks(frames), {"type": "text", "text": prompt}]
    return [{"role": "user", "content": content}]


def _parse_verdict(camera_key: str, result: Any, cfg: CameraCurationConfig) -> CameraVerdict:
    """Turn a parsed VLM JSON object into a :class:`CameraVerdict` (defensively)."""
    if not isinstance(result, dict):
        return CameraVerdict(camera_key=camera_key, usable=True, view_label=None, blur_reason=None)

    usable = bool(result.get("usable", True))
    blur_reason = result.get("blur_reason")
    blur_reason = str(blur_reason) if blur_reason else None

    raw_label = result.get("view_label")
    label = str(raw_label).strip().lower().replace(" ", "_") if raw_label else ""
    view_label = label if is_valid_view_label(label, cfg.view_vocabulary, cfg.allow_combos) else None
    if raw_label and view_label is None:
        logger.warning(
            "camera %s: VLM returned view_label=%r which is not in the vocabulary %s; leaving unlabeled",
            camera_key,
            raw_label,
            cfg.view_vocabulary,
        )

    confidence = result.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    return CameraVerdict(
        camera_key=camera_key,
        usable=usable,
        view_label=view_label,
        blur_reason=blur_reason,
        confidence=confidence,
    )


def curate_cameras(
    frames_by_camera: dict[str, list[Any]],
    cfg: CameraCurationConfig,
    vlm: Any,
) -> list[CameraVerdict]:
    """Judge each camera's quality + view label from a few sampled frames.

    ``frames_by_camera`` maps a camera key to a list of decoded frames (torch
    tensors or PIL images). One batched ``generate_json`` call is issued across
    all cameras. Cameras with no frames are still reported (usable, unlabeled)
    so the caller sees the full camera set.
    """
    ordered_keys = list(frames_by_camera)
    callable_keys = [k for k in ordered_keys if frames_by_camera[k]]

    verdicts: dict[str, CameraVerdict] = {
        k: CameraVerdict(camera_key=k, usable=True, view_label=None) for k in ordered_keys
    }

    if callable_keys:
        messages_batch = [_build_messages(frames_by_camera[k], cfg) for k in callable_keys]
        results = vlm.generate_json(messages_batch)
        for key, result in zip(callable_keys, results, strict=True):
            verdicts[key] = _parse_verdict(key, result, cfg)

    return [verdicts[k] for k in ordered_keys]


def build_name_mapping(
    verdicts: list[CameraVerdict],
    existing_features: dict[str, dict],
    cfg: CameraCurationConfig,
) -> dict[str, str]:
    """Compute ``{old_key: observation.images.<label>}`` for labeled cameras.

    Cameras without a valid label (or already at their canonical name) are
    skipped. Collisions are resolved with ``cfg.on_collision`` and the resolved
    target is written back onto each verdict's ``proposed_new_key``.
    """
    desired: dict[str, str] = {}
    for v in verdicts:
        if v.view_label is None:
            continue
        target = f"{OBS_IMAGE_PREFIX}{v.view_label}"
        if target != v.camera_key:
            desired[v.camera_key] = target

    if not desired:
        return {}

    resolved = _resolve_rename_collisions(desired, existing_features, cfg.on_collision)
    by_key = {v.camera_key: v for v in verdicts}
    for old, new in resolved.items():
        by_key[old].proposed_new_key = new
    return resolved


def build_report(
    verdicts: list[CameraVerdict],
    mapping: dict[str, str],
    cfg: CameraCurationConfig,
) -> dict[str, Any]:
    """Assemble the machine-readable curation report."""
    return {
        "repo_id": cfg.repo_id,
        "episode_index": cfg.episode_index,
        "view_vocabulary": list(cfg.view_vocabulary),
        "cameras": {
            v.camera_key: {
                "view_label": v.view_label,
                "usable": v.usable,
                "blur_reason": v.blur_reason,
                "confidence": v.confidence,
                "proposed_new_key": mapping.get(v.camera_key),
            }
            for v in verdicts
        },
    }


def write_report(
    root: Path,
    verdicts: list[CameraVerdict],
    mapping: dict[str, str],
    cfg: CameraCurationConfig,
) -> Path:
    """Write ``meta/camera_curation.json`` and stamp verdicts into ``info.json``.

    Stamping goes into each camera's ``features[key]["info"]["curation"]`` so the
    verdict travels with the dataset. Returns the report path.
    """
    report = build_report(verdicts, mapping, cfg)
    default_report_path = root / "meta" / "camera_curation.json"
    report_path = Path(cfg.report_path) if cfg.report_path is not None else default_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, report_path)

    info = load_info(root)
    changed = False
    for v in verdicts:
        feature = info.features.get(v.camera_key)
        if feature is None:
            continue
        feature.setdefault("info", {})
        if feature["info"] is None:
            feature["info"] = {}
        feature["info"]["curation"] = {
            "view_label": v.view_label,
            "usable": v.usable,
            "blur_reason": v.blur_reason,
            "confidence": v.confidence,
        }
        changed = True
    if changed:
        write_info(info, root)

    return report_path


def _swap_key_in_path(path: str, old_key: str, new_key: str) -> str:
    """Rewrite the ``<old_key>`` path segment of a ``videos/<key>/...`` repo path."""
    prefix = f"videos/{old_key}/"
    return f"videos/{new_key}/{path[len(prefix):]}" if path.startswith(prefix) else path


def rename_camera_keys_on_hub(
    repo_id: str,
    name_mapping: dict[str, str],
    local_root: Path,
    *,
    revision: str | None = None,
    branch: str | None = None,
    token: str | None = None,
    commit_message: str | None = None,
) -> Any:
    """Rename camera keys on the Hub without downloading video data.

    Edits the small ``meta/`` files locally (under ``local_root``, which must be
    a writable dataset root whose ``meta/`` is already present), then commits, in
    one atomic ``create_commit``: ``CommitOperationCopy`` + ``CommitOperationDelete``
    to move each ``videos/<old>/*`` LFS file server-side, and ``CommitOperationAdd``
    for the edited meta files. Renames in place on ``repo_id`` (cross-repo copies
    are unsupported); pass ``branch`` to commit to a branch and keep ``main`` intact.

    Only video keys can be moved this way — reject swaps/cycles and image keys
    (handled by the local ``rename_features`` path instead).
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi

    # A swap/cycle (a target that is also a source) cannot be expressed in a
    # single base-revision commit; defer to the local rename path.
    swaps = set(name_mapping.values()) & set(name_mapping)
    if swaps:
        raise NotImplementedError(
            f"Hub rename cannot swap keys in one commit (offending: {sorted(swaps)}); "
            "use the local rename_features path for swaps/cycles."
        )

    # Determine which OLD keys are video-stored (only those have a videos/ tree)
    # BEFORE remapping the metadata.
    info = load_info(local_root)
    video_old_keys = {
        old for old in name_mapping if info.features.get(old, {}).get("dtype") == "video"
    }
    image_old_keys = {
        old for old in name_mapping if info.features.get(old, {}).get("dtype") == "image"
    }
    if image_old_keys:
        raise NotImplementedError(
            f"Hub rename cannot move image data stored in the data parquet (keys: {sorted(image_old_keys)}); "
            "use --mode report (metadata mapping) or the local rename_features path for image datasets."
        )

    # 1. Rewrite meta/ locally (info features, episodes columns, stats keys).
    _remap_camera_key_in_meta(local_root, name_mapping)

    api = HfApi(token=token)
    operations: list[Any] = []

    # 2. Add the (small) meta files we just edited.
    meta_dir = local_root / "meta"
    meta_files = [meta_dir / "info.json"]
    stats_file = meta_dir / "stats.json"
    if stats_file.exists():
        meta_files.append(stats_file)
    meta_files.extend(sorted((meta_dir / "episodes").glob("*/*.parquet")))
    for fpath in meta_files:
        rel = fpath.relative_to(local_root).as_posix()
        operations.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(fpath)))

    # 3. Move video LFS files server-side (copy + delete), no download.
    repo_files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    n_moved = 0
    for old in video_old_keys:
        new = name_mapping[old]
        prefix = f"videos/{old}/"
        for f in repo_files:
            if f.startswith(prefix):
                operations.append(
                    CommitOperationCopy(src_path_in_repo=f, path_in_repo=_swap_key_in_path(f, old, new))
                )
                operations.append(CommitOperationDelete(path_in_repo=f))
                n_moved += 1
    logger.info(
        "hub rename: moving %d video file(s) server-side across %d camera(s)",
        n_moved,
        len(video_old_keys),
    )

    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        revision=branch or revision,
        commit_message=commit_message or "curate: rename camera views (lerobot-curate-cameras)",
    )
    return commit_info


def as_report_dict(verdicts: list[CameraVerdict]) -> list[dict[str, Any]]:
    """Convenience: verdicts as plain dicts (for logging/JSON)."""
    return [asdict(v) for v in verdicts]
