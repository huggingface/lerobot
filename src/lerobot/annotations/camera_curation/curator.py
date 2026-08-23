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
import re
from collections import Counter
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
_JOINT_LABEL_PROMPT_PATH = Path(__file__).parent / "prompts" / "camera_joint_label.txt"

# The canonical prefix every curated camera key gets.
OBS_IMAGE_PREFIX = "observation.images."

# The mount-type categories the VLM may return.
MOUNT_ROBOT = "robot_mounted"
MOUNT_FIXED = "fixed"
MOUNT_UNKNOWN = "unknown"
_MOUNT_TYPES = (MOUNT_FIXED, MOUNT_ROBOT, MOUNT_UNKNOWN)

# The single POSITION word of a wrist label (robot-mounted cameras only).
_WRIST_POSITION = "wrist"


@dataclass
class CameraVerdict:
    """One camera's VLM verdict."""

    camera_key: str
    usable: bool
    view_label: str | None
    blur_reason: str | None = None
    confidence: float | None = None
    # How the camera is mounted ("fixed" / "robot_mounted" / "unknown" / None).
    mount_type: str | None = None
    # Populated by ``build_name_mapping`` once collisions are resolved (pure
    # Python, deterministic — the VLM never sees or proposes a dataset key).
    proposed_new_key: str | None = None


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _load_joint_label_prompt() -> str:
    return _JOINT_LABEL_PROMPT_PATH.read_text(encoding="utf-8")


# Direction qualifiers: suffix-only words that say where a side/wrist camera sits.
# They never stand alone as a label — always "<qualifier>_<position>".
_QUALIFIERS = ("left", "right", "front", "rear")


def _position_token(label: str) -> str | None:
    """The single non-qualifier POSITION word of a (valid) label.

    ``left_wrist`` -> ``wrist``, ``front_side`` -> ``side``, ``top`` -> ``top``.
    """
    positions = [tok for tok in label.split("_") if tok not in _QUALIFIERS]
    return positions[0] if positions else None


def _normalize_mount_type(raw: Any) -> str | None:
    """Coerce a raw VLM ``mount_type`` string to a known category or None."""
    if not raw:
        return None
    value = str(raw).strip().lower()
    return value if value in _MOUNT_TYPES else None


def _reconcile_label_with_mount(view_label: str | None, mount_type: str | None) -> str | None:
    """Deterministically reconcile a view label against the mount type.

    ``mount_type`` is the more reliable signal (it is judged from temporal
    motion), so it is authoritative over the finer view label:

    - ``robot_mounted``  -> the camera is a wrist camera. Keep an existing wrist
      label (preserving handedness); otherwise force plain ``wrist`` (even when
      the label was missing/``unknown``).
    - ``fixed``          -> a wrist label contradicts a fixed mount, so drop it
      (we cannot infer the fixed position) and leave the camera unlabeled; any
      non-wrist label is kept.
    - ``unknown``/None   -> no mount signal, so trust the view label as-is.

    This replaces the old VLM "relabel" pass with a pure, deterministic check.
    """
    if mount_type == MOUNT_ROBOT:
        if view_label is not None and _position_token(view_label) == _WRIST_POSITION:
            return view_label
        return _WRIST_POSITION
    if mount_type == MOUNT_FIXED:
        if view_label is not None and _position_token(view_label) == _WRIST_POSITION:
            return None
        return view_label
    return view_label


def is_valid_view_label(label: str, vocabulary: tuple[str, ...], allow_combos: bool) -> bool:
    """Validate a view label against the vocabulary.

    A single token must be a POSITION (a vocab word that is not a direction
    qualifier) — e.g. ``side``/``top``/``wrist``, never a bare ``front``/``left``.
    A combo (when allowed) must be exactly one qualifier + one position, e.g.
    ``front_side`` or ``left_wrist``; nonsense like ``front_rear`` or ``side_top``
    is rejected.
    """
    if not label:
        return False
    tokens = label.split("_")
    if not all(tok in vocabulary for tok in tokens):
        return False
    if len(tokens) == 1:
        return tokens[0] not in _QUALIFIERS
    if not allow_combos or len(tokens) != 2 or tokens[0] == tokens[1]:
        return False
    qualifiers = [t for t in tokens if t in _QUALIFIERS]
    positions = [t for t in tokens if t not in _QUALIFIERS]
    return len(qualifiers) == 1 and len(positions) == 1


def _combo_rule(cfg: CameraCurationConfig) -> str:
    if cfg.allow_combos:
        return (
            "You may combine at most two of these words with an underscore when "
            'one word is not precise enough (e.g. "left_wrist"). '
        )
    return "Use exactly one of these words (no combinations). "


def _build_messages(frames: list[Any], cfg: CameraCurationConfig) -> list[dict[str, Any]]:
    prompt = _load_prompt().format(
        vocabulary=", ".join(cfg.view_vocabulary),
        combo_rule=_combo_rule(cfg),
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

    mount_type = _normalize_mount_type(result.get("mount_type"))
    view_label = _parse_view_label(camera_key, result.get("view_label"), cfg)
    # Mount type is the authoritative signal: reconcile the finer label against it.
    view_label = _reconcile_label_with_mount(view_label, mount_type)

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
        mount_type=mount_type,
    )


def _parse_view_label(camera_key: str, raw_label: Any, cfg: CameraCurationConfig) -> str | None:
    """Normalize and validate a raw VLM ``view_label`` into a canonical label or None."""
    label = str(raw_label).strip().lower().replace(" ", "_") if raw_label else ""
    # "unknown" is an explicit abstain — the model isn't sure, so leave the camera
    # unlabeled (no rename) rather than forcing a guess.
    if label == "unknown" or not label:
        return None
    if is_valid_view_label(label, cfg.view_vocabulary, cfg.allow_combos):
        # Canonicalize combo order (``wrist_left`` -> ``left_wrist``) so the set
        # of possible keys is deterministic regardless of the VLM's word order.
        return _order_combo(label.split("_"), cfg.view_vocabulary)
    logger.warning(
        "camera %s: VLM returned view_label=%r which is not in the vocabulary %s; leaving unlabeled",
        camera_key,
        raw_label,
        cfg.view_vocabulary,
    )
    return None


def curate_cameras(
    frames_by_camera: dict[str, list[Any]],
    cfg: CameraCurationConfig,
    vlm: Any,
) -> list[CameraVerdict]:
    """Judge each camera's quality + view label from a few sampled frames.

    ``frames_by_camera`` maps a camera key to a list of decoded frames (torch
    tensors or PIL images). Pass 1 judges each camera on its own (quality + mount
    type + an initial label, the label reconciled against the mount type). With
    ``cfg.joint_labeling`` a second mount-type+label pass shows all cameras
    together and re-decides by comparison. Quality is always the per-camera
    verdict. Any remaining label collisions are resolved deterministically later
    in :func:`build_name_mapping`. Cameras with no frames are reported unlabeled.
    """
    ordered_keys = list(frames_by_camera)
    callable_keys = [k for k in ordered_keys if frames_by_camera[k]]

    verdicts: dict[str, CameraVerdict] = {
        k: CameraVerdict(camera_key=k, usable=True, view_label=None) for k in ordered_keys
    }

    if callable_keys:
        # Pass 1: per-camera (quality + mount type + initial label).
        messages_batch = [_build_messages(frames_by_camera[k], cfg) for k in callable_keys]
        results = vlm.generate_json(messages_batch)
        for key, result in zip(callable_keys, results, strict=True):
            verdicts[key] = _parse_verdict(key, result, cfg)
        # Pass 2: joint labeling (mount type + label only; quality untouched).
        if cfg.joint_labeling and len(callable_keys) >= 2:
            relabeled = _joint_label(frames_by_camera, callable_keys, cfg, vlm)
            for key, (mount_type, label) in relabeled.items():
                verdicts[key].mount_type = mount_type
                verdicts[key].view_label = label

    return [verdicts[k] for k in ordered_keys]


def _joint_label(
    frames_by_camera: dict[str, list[Any]],
    camera_keys: list[str],
    cfg: CameraCurationConfig,
    vlm: Any,
) -> dict[str, tuple[str | None, str | None]]:
    """Show all cameras together and ask for one mount type + label each.

    Returns ``{camera_key: (mount_type, view_label)}`` for cameras present in the
    response (positionally). The label is reconciled against the mount type with
    the same deterministic rule as the per-camera pass. Quality is never touched
    here, so this can't leak a quality verdict across cameras.
    """
    prompt = _load_joint_label_prompt().format(
        n=len(camera_keys),
        vocabulary=", ".join(cfg.view_vocabulary),
        combo_rule=_combo_rule(cfg),
    )
    content: list[dict[str, Any]] = []
    for i, key in enumerate(camera_keys, 1):
        # Number the cameras neutrally — never show the key name, which would bias
        # the model toward the (often unreliable) existing label. Responses map
        # back to camera_keys by position.
        content.append({"type": "text", "text": f"Camera {i}:"})
        content.extend(to_image_blocks(frames_by_camera[key]))
    content.append({"type": "text", "text": prompt})

    results = vlm.generate_json([[{"role": "user", "content": content}]])
    result = results[0] if results else None
    items = result.get("cameras") if isinstance(result, dict) else None
    if not isinstance(items, list):
        logger.warning("joint label: response missing a 'cameras' list; keeping per-camera labels")
        return {}

    out: dict[str, tuple[str | None, str | None]] = {}
    for i, key in enumerate(camera_keys):
        item = items[i] if i < len(items) else None
        if not isinstance(item, dict):
            continue
        mount_type = _normalize_mount_type(item.get("mount_type"))
        label = _parse_view_label(key, item.get("view_label"), cfg)
        out[key] = (mount_type, _reconcile_label_with_mount(label, mount_type))
    return out


def build_name_mapping(
    verdicts: list[CameraVerdict],
    existing_features: dict[str, dict],
    cfg: CameraCurationConfig,
) -> tuple[dict[str, str], dict[str, str]]:
    """Compute the rename mapping for labeled cameras.

    Returns ``(mapping, skipped)`` where ``mapping`` is ``{old_key:
    observation.images.<label>}`` for cameras that will be renamed and ``skipped``
    is ``{old_key: reason}`` for cameras dropped because of an unresolved label
    conflict.

    Cameras without a valid label (or already at their canonical name) are
    omitted from both. Collisions are first reduced by disambiguating from the
    original key names (``..._left`` + ``wrist`` → ``left_wrist``); whatever still
    collides is handled per ``cfg.on_collision``: ``"skip"`` (default) renames only
    the unambiguous cameras and lists the colliding ones in ``skipped``;
    ``"suffix"`` renames everything with numeric suffixes; ``"error"`` raises.
    The chosen target is written back onto each renamed verdict's
    ``proposed_new_key``.
    """
    label_by_cam = {v.camera_key: v.view_label for v in verdicts if v.view_label is not None}
    if cfg.allow_combos and not cfg.ignore_key_names:
        label_by_cam = _disambiguate_from_source_names(label_by_cam, cfg.view_vocabulary)

    desired: dict[str, str] = {}
    for cam, label in label_by_cam.items():
        target = f"{OBS_IMAGE_PREFIX}{label}"
        if target != cam:
            desired[cam] = target

    if not desired:
        return {}, {}

    if cfg.on_collision == "skip":
        confidence_by_cam = {v.camera_key: v.confidence for v in verdicts}
        resolved, skipped = _skip_colliding(desired, existing_features, confidence_by_cam)
    else:
        # "error" raises; "suffix" disambiguates every entry.
        resolved = _resolve_rename_collisions(desired, existing_features, cfg.on_collision)
        skipped = {}

    by_key = {v.camera_key: v for v in verdicts}
    for old, new in resolved.items():
        by_key[old].proposed_new_key = new
    return resolved, skipped


def _skip_colliding(
    desired: dict[str, str],
    existing_features: dict[str, dict],
    confidence_by_cam: dict[str, float | None],
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve label collisions, keeping the most confident contender.

    - Unique, unoccupied target → renamed.
    - Target equal to an existing (untouched) feature → all contenders skipped
      (the name is already taken; confidence can't free it).
    - Several cameras wanting the same *new* target → the highest-confidence
      camera is renamed and the rest are skipped (ties broken by camera key for
      determinism; a missing confidence ranks lowest).

    Returns ``(kept, skipped)`` — ``skipped`` maps old_key → reason.
    """
    sources = set(desired)
    untouched = set(existing_features) - sources

    by_target: dict[str, list[str]] = {}
    for cam, target in desired.items():
        by_target.setdefault(target, []).append(cam)

    def _rank(cam: str) -> tuple[float, str]:
        conf = confidence_by_cam.get(cam)
        return (conf if conf is not None else -1.0, cam)

    kept: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for target, cams in by_target.items():
        if target in untouched:
            for cam in cams:
                skipped[cam] = f"label '{target}' is already an existing feature"
            continue
        if len(cams) == 1:
            kept[cams[0]] = target
            continue
        # Contested new label: the most confident camera wins, keeping its rename;
        # the others are skipped rather than sinking the whole dataset.
        winner = max(cams, key=_rank)
        kept[winner] = target
        for cam in cams:
            if cam != winner:
                skipped[cam] = f"label '{target}' also chosen by a more confident camera ({winner})"
    return kept, skipped


def _extract_vocab_tokens(camera_key: str, vocabulary: tuple[str, ...]) -> list[str]:
    """Vocabulary words present in a camera key, in vocabulary order.

    Splits on non-alphanumeric boundaries so ``observation.images.cam_left`` →
    ``["left"]`` and ``left_wrist_0_rgb`` → ``["wrist", "left"]``.
    """
    parts = set(re.split(r"[^a-z0-9]+", camera_key.lower()))
    return [tok for tok in vocabulary if tok in parts]


def _order_combo(tokens: list[str], vocabulary: tuple[str, ...]) -> str:
    """Join vocab tokens into a combo label with the direction first.

    A direction qualifier (left/right/front/rear) leads, the position word
    follows — ``{wrist, left}`` → ``left_wrist``, ``{side, front}`` →
    ``front_side``.
    """
    uniq = list(dict.fromkeys(tokens))

    def sort_key(tok: str) -> tuple[bool, int]:
        return (tok not in _QUALIFIERS, vocabulary.index(tok) if tok in vocabulary else len(vocabulary))

    return "_".join(sorted(uniq, key=sort_key))


def _disambiguate_from_source_names(
    label_by_cam: dict[str, str], vocabulary: tuple[str, ...]
) -> dict[str, str]:
    """When several cameras share a label, enrich each from its source-key words.

    Only labels shared by 2+ cameras are touched; a distinguishing vocab word is
    pulled from the camera's original key and combined with the label (kept only
    if the result is a valid ≤2-word combo). Anything still colliding afterwards
    is left for the caller's collision policy.
    """
    counts = Counter(label_by_cam.values())
    out = dict(label_by_cam)
    for cam, label in label_by_cam.items():
        if counts[label] < 2:
            continue  # unique label — leave the VLM's clean single word alone
        label_tokens = label.split("_")
        extra = next((tok for tok in _extract_vocab_tokens(cam, vocabulary) if tok not in label_tokens), None)
        if extra is None:
            continue
        combined = _order_combo([*label_tokens, extra], vocabulary)
        if is_valid_view_label(combined, vocabulary, allow_combos=True):
            out[cam] = combined
    return out


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
                "mount_type": v.mount_type,
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

    stamp_verdicts_into_info(root, verdicts)
    return report_path


def stamp_verdicts_into_info(root: Path, verdicts: list[CameraVerdict]) -> None:
    """Record each camera's verdict under ``features[key]["info"]["curation"]``.

    Written into ``meta/info.json`` so the verdict travels with the dataset. A
    no-op for cameras absent from ``info.json``.
    """
    info = load_info(root)
    changed = False
    for v in verdicts:
        feature = info.features.get(v.camera_key)
        if feature is None:
            continue
        if feature.get("info") is None:
            feature["info"] = {}
        feature["info"]["curation"] = {
            "view_label": v.view_label,
            "mount_type": v.mount_type,
            "usable": v.usable,
            "blur_reason": v.blur_reason,
            "confidence": v.confidence,
        }
        changed = True
    if changed:
        write_info(info, root)


def _swap_key_in_path(path: str, old_key: str, new_key: str, path_prefix: str | None = None) -> str:
    """Rewrite the ``<old_key>`` segment of a ``[<prefix>/]videos/<key>/...`` repo path."""
    repo_prefix = f"{path_prefix}/" if path_prefix else ""
    old = f"{repo_prefix}videos/{old_key}/"
    new = f"{repo_prefix}videos/{new_key}/"
    return f"{new}{path[len(old) :]}" if path.startswith(old) else path


def rename_camera_keys_on_hub(
    repo_id: str,
    name_mapping: dict[str, str],
    local_root: Path,
    *,
    path_prefix: str | None = None,
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

    ``path_prefix`` scopes every repo path to a sub-dataset within a nested
    collection (e.g. ``user/task`` in ``lerobot/community_dataset_v3``); the local
    ``meta/`` still lives directly under ``local_root``.

    Only video keys can be moved this way — reject swaps/cycles and image keys
    (handled by the local ``rename_features`` path instead).
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi

    repo_prefix = f"{path_prefix}/" if path_prefix else ""

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
    video_old_keys = {old for old in name_mapping if info.features.get(old, {}).get("dtype") == "video"}
    image_old_keys = {old for old in name_mapping if info.features.get(old, {}).get("dtype") == "image"}
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
        operations.append(CommitOperationAdd(path_in_repo=f"{repo_prefix}{rel}", path_or_fileobj=str(fpath)))

    # 3. Move video LFS files server-side (copy + delete), no download.
    repo_files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    n_moved = 0
    for old in video_old_keys:
        new = name_mapping[old]
        prefix = f"{repo_prefix}videos/{old}/"
        for f in repo_files:
            if f.startswith(prefix):
                operations.append(
                    CommitOperationCopy(
                        src_path_in_repo=f, path_in_repo=_swap_key_in_path(f, old, new, path_prefix)
                    )
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
