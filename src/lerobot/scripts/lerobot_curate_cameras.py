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
"""``lerobot-curate-cameras`` — VLM camera-view curation for a LeRobot dataset.

Downloads only the first episode, asks a VLM to (1) flag blurry/unusable views
and (2) label each view (``front``/``rear``/``left_side``/``top``/``wrist``/…), then either records the
result in ``meta/`` (``--mode=report``) or renames the camera keys to
``observation.images.<label>`` (``--mode=rename``). For video datasets the
rename is a download-free, server-side Hub commit.

Examples:

  # Cheap, mutation-free triage (writes meta/camera_curation.json):
  uv run lerobot-curate-cameras --repo_id=user/dataset --mode=report

  # Apply the labels by renaming camera keys on a new branch (video datasets):
  uv run lerobot-curate-cameras --repo_id=user/dataset --mode=rename --branch=curated

  # Run the VLM decision on a GPU via HF Jobs:
  uv run lerobot-curate-cameras --repo_id=user/dataset --mode=rename --job.target=h200

  # Nested collection (no root meta/info.json): curate specific sub-datasets:
  uv run lerobot-curate-cameras --repo_id=lerobot/community_dataset_v3 \\
      --subpaths='[00ri/so100_battery, 1g0rrr/demo2_frame_holder]' --mode=report

  # Whole nested collection (auto-discovers every sub-dataset) on a GPU job:
  uv run lerobot-curate-cameras --repo_id=lerobot/community_dataset_v3 \\
      --mode=rename --branch=curated --job.target=h200 --job.timeout=24h
"""

import json
import logging
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.annotations.camera_curation import curator
from lerobot.annotations.camera_curation.config import CameraCurationConfig
from lerobot.annotations.steerable_pipeline.frames import make_frame_provider
from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.annotations.steerable_pipeline.vlm_client import make_vlm_client
from lerobot.configs import parser
from lerobot.utils.import_utils import _datasets_available, require_package

if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def _write_and_echo_report(report: dict, cfg: CameraCurationConfig, default_name: str) -> Path:
    """Write the report to a persistent path (never a temp dir) and echo to stdout."""
    out = Path(cfg.report_path) if cfg.report_path is not None else Path(default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("curate-cameras: report written to %s", out)
    print("===== camera curation report =====", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print("===== end camera curation report =====", flush=True)
    return out


def _central_indices(n: int, k: int, window: tuple[float, float] = (0.0, 1.0)) -> list[int]:
    """Return k indices spread across ``window`` (lo, hi fractions) of n items.

    The default window is the whole episode; narrowing it skips the
    unrepresentative start/end (setup, teardown, an operator resetting the scene).
    """
    if n <= 0 or k <= 0:
        return []
    lo, hi = window
    a = int(n * lo)
    b = max(a, min(n - 1, int(round(n * hi)) - 1))
    if k == 1:
        return [(a + b) // 2]
    span = b - a
    if k >= span + 1:
        return list(range(a, b + 1))
    step = span / (k - 1)
    return sorted({a + round(i * step) for i in range(k)})


def _to_uint8_frame(frame: Any) -> Any:
    """Scale a float [0,1] image tensor to uint8; pass uint8/PIL through."""
    import torch

    if isinstance(frame, torch.Tensor) and torch.is_floating_point(frame):
        return (frame.clamp(0, 1) * 255).to(torch.uint8)
    return frame


def _sample_frames(
    root: Path,
    meta: Any,
    cfg: CameraCurationConfig,
    dataset: "LeRobotDataset | None" = None,
) -> dict[str, list[Any]]:
    """Sample ``n_frames`` from the inspected episode for each (non-depth) camera.

    Video cameras go through the annotation frame provider (uint8 frames, works
    from a bare dataset ``root``). Image cameras need decoded rows, so they are
    only sampled when a full ``LeRobotDataset`` is supplied (the single-dataset
    path); in sub-path mode they are reported unlabeled with a warning.
    """
    depth_keys = set(meta.depth_keys)
    video_keys = set(meta.video_keys)
    image_keys = set(meta.image_keys)
    cameras = [k for k in meta.camera_keys if k not in depth_keys]

    frames: dict[str, list[Any]] = {k: [] for k in cameras}

    video_cameras = [k for k in cameras if k in video_keys]
    if video_cameras:
        provider = make_frame_provider(root, video_backend=cfg.video_backend)
        records = list(iter_episodes(root, only_episodes=(cfg.episode_index,)))
        record = records[0] if records else None
        if record is not None:
            ts_all = list(record.frame_timestamps)
            timestamps = [ts_all[i] for i in _central_indices(len(ts_all), cfg.n_frames, cfg.sample_window)]
            if timestamps:
                for key in video_cameras:
                    frames[key] = provider.frames_at(record, timestamps, camera_key=key)

    image_cameras = [k for k in cameras if k in image_keys]
    if image_cameras:
        if dataset is not None:
            n = len(dataset)
            for i in _central_indices(n, cfg.n_frames, cfg.sample_window):
                item = dataset[i]
                for key in image_cameras:
                    if key in item:
                        frames[key].append(_to_uint8_frame(item[key]))
        else:
            logger.warning(
                "image cameras %s are not sampled in sub-path mode; they are reported unlabeled",
                image_cameras,
            )

    return frames


def _apply_rename(
    root: Path,
    meta: Any,
    cfg: CameraCurationConfig,
    mapping: dict[str, str],
    verdicts: list["curator.CameraVerdict"],
    *,
    path_prefix: str | None = None,
    dataset: "LeRobotDataset | None" = None,
    commit_lock: "threading.Lock | None" = None,
) -> None:
    """Apply the computed ``{old: new}`` camera-key mapping.

    Video datasets on the Hub → download-free server-side rename commit (scoped to
    ``path_prefix`` for a sub-dataset), including overlapping/cyclic renames (the
    Hub path routes those through a temp key). Otherwise (image dataset or a
    local-only root) → local ``rename_features`` over a full copy — single-dataset
    only.

    ``commit_lock`` serializes the Hub commit across concurrent workers so
    parallel renames don't race on the branch head ref.
    """
    video_keys = set(meta.video_keys)
    all_video = set(mapping) <= video_keys

    # The Hub path now handles overlapping/cyclic renames (via a temp key), so a
    # swap no longer forces the local path — only image data / local roots do.
    if cfg.repo_id is not None and all_video:
        if cfg.drop_unusable:
            logger.warning(
                "--drop_unusable is only applied via the local rename path; the Hub rename keeps "
                "flagged views (they are still recorded in meta/). Re-run with a local --root to drop."
            )
        # Edit a throwaway copy of meta/ so the local cached copy stays pristine
        # and only the intended files land in the commit.
        work = Path(tempfile.mkdtemp(prefix="lerobot_curate_"))
        try:
            shutil.copytree(root / "meta", work / "meta")
            # Serialize the commit: concurrent create_commit calls to the same
            # branch race on its head ref. The remap is local and quick, so the
            # lock barely dents parallelism.
            with commit_lock or nullcontext():
                commit = curator.rename_camera_keys_on_hub(
                    cfg.repo_id,
                    mapping,
                    work,
                    path_prefix=path_prefix,
                    branch=cfg.branch,
                    commit_message=cfg.push_commit_message,
                )
            oid = getattr(commit, "oid", None)
            ref = cfg.branch or "main"
            target = f"{cfg.repo_id}/{path_prefix}" if path_prefix else cfg.repo_id
            logger.info("Hub rename committed to %s@%s (%s)", target, ref, oid)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return

    if path_prefix is not None:
        logger.warning(
            "sub-path %s: rename needs the server-side video path (image/swap not supported "
            "in sub-path mode); skipping rename for this sub-dataset (verdicts still recorded).",
            path_prefix,
        )
        return

    # Local path: needs the full dataset, so re-load without the episode filter.
    require_package("datasets", "dataset")
    from lerobot.datasets import LeRobotDataset as _LeRobotDataset, remove_feature, rename_features

    logger.info("Local rename path (image/local/swap): loading the full dataset from %s", root)
    full = _LeRobotDataset(cfg.repo_id or "local", root=root)
    renamed = rename_features(
        full, mapping, output_dir=root, repo_id=full.repo_id, on_collision=cfg.on_collision
    )

    if cfg.drop_unusable:
        unusable_new_keys = [
            mapping[v.camera_key] for v in verdicts if not v.usable and v.camera_key in mapping
        ]
        if unusable_new_keys:
            logger.info("Dropping unusable views: %s", unusable_new_keys)
            renamed = remove_feature(renamed, unusable_new_keys, output_dir=root, repo_id=renamed.repo_id)

    if cfg.push_to_hub:
        logger.info("Pushing renamed dataset to %s", renamed.repo_id)
        renamed.push_to_hub()


def _decide(
    root: Path,
    meta: Any,
    cfg: CameraCurationConfig,
    vlm: Any,
    *,
    label: str,
    dataset: "LeRobotDataset | None" = None,
) -> tuple[list["curator.CameraVerdict"], dict[str, str], dict[str, Any]]:
    """Sample frames, run the VLM, and compute the rename mapping for one dataset.

    Returns ``(verdicts, mapping, rename_notes)``. ``mapping`` renames only the
    unambiguous cameras; cameras in an unresolved label conflict are left out and
    recorded in ``rename_notes["collisions"]`` (so those cameras are skipped while
    the rest are still renamed). ``on_collision="error"`` instead records
    ``rename_notes["rename_error"]`` and renames nothing for this dataset. Either
    way the verdicts (quality report) are always kept.
    """
    frames = _sample_frames(root, meta, cfg, dataset=dataset)
    n_with_frames = sum(1 for v in frames.values() if v)
    logger.info("curate-cameras[%s]: %d camera(s), %d with sampled frames", label, len(frames), n_with_frames)
    verdicts = curator.curate_cameras(frames, cfg, vlm)
    for v in verdicts:
        logger.info(
            "  [%s] %s -> label=%s usable=%s%s",
            label,
            v.camera_key,
            v.view_label,
            v.usable,
            "" if v.usable else f" (blur_reason={v.blur_reason!r})",
        )
    rename_notes: dict[str, Any] = {}
    try:
        mapping, collisions = curator.build_name_mapping(verdicts, meta.features, cfg)
        logger.info("  [%s] rename mapping: %s", label, mapping or "(none)")
        if collisions:
            rename_notes["collisions"] = collisions
            logger.warning(
                "  [%s] %d camera(s) skipped due to label conflicts (rest still renamed): %s",
                label,
                len(collisions),
                sorted(collisions),
            )
    except ValueError as exc:  # on_collision="error": skip the whole dataset's rename
        mapping = {}
        rename_notes["rename_error"] = str(exc)
        logger.warning("  [%s] rename skipped, verdicts kept: %s", label, exc)
    return verdicts, mapping, rename_notes


def _materialize_subdataset(repo_id: str, subpath: str, work_dir: Path, episode_index: int):
    """Download a sub-dataset's ``meta/`` + first-episode files into ``work_dir/subpath``.

    Returns ``(sub_root, meta)``. Only the first episode's data/video shards are
    fetched, matching the single-dataset partial-download behavior.
    """
    from huggingface_hub import snapshot_download

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    snapshot_download(
        repo_id, repo_type="dataset", allow_patterns=[f"{subpath}/meta/**"], local_dir=str(work_dir)
    )
    sub_root = Path(work_dir) / subpath
    meta = LeRobotDatasetMetadata("local", root=sub_root)

    rels = [meta.get_data_file_path(episode_index).as_posix()]
    rels += [meta.get_video_file_path(episode_index, vk).as_posix() for vk in meta.video_keys]
    snapshot_download(
        repo_id,
        repo_type="dataset",
        allow_patterns=[f"{subpath}/{rel}" for rel in rels],
        local_dir=str(work_dir),
    )
    return sub_root, meta


def _discover_subpaths(repo_id: str) -> list[str] | None:
    """Enumerate sub-dataset prefixes in a nested collection on the Hub.

    Returns the sorted list of ``<prefix>`` for every ``<prefix>/meta/info.json``,
    or ``None`` when the repo has a root ``meta/info.json`` (a single dataset).
    """
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    if "meta/info.json" in files:
        return None  # standard single dataset
    prefixes = sorted(f[: -len("/meta/info.json")] for f in files if f.endswith("/meta/info.json"))
    return prefixes


def _process_subdataset(
    cfg: CameraCurationConfig,
    vlm: Any,
    subpath: str,
    index: int,
    total: int,
    commit_lock: "threading.Lock",
) -> tuple[str, dict[str, Any]]:
    """Curate one sub-dataset end to end; returns ``(subpath, report_entry)``.

    Self-contained so it is safe to run concurrently: its own temp dir (removed
    on exit), its own metadata/frame provider, and the shared VLM server. A
    failure is captured as ``{"error": ...}`` so one bad sub-dataset never sinks
    the sweep.
    """
    logger.info("===== sub-dataset %d/%d: %s =====", index, total, subpath)
    sub_work = Path(tempfile.mkdtemp(prefix="lerobot_curate_sub_"))
    try:
        sub_root, meta = _materialize_subdataset(cfg.repo_id, subpath, sub_work, cfg.episode_index)
        verdicts, mapping, rename_notes = _decide(sub_root, meta, cfg, vlm, label=subpath)
        # Record verdicts first — a collision below must not lose the quality report.
        curator.stamp_verdicts_into_info(sub_root, verdicts)
        entry = curator.build_report(verdicts, mapping, cfg)
        entry.update(rename_notes)  # "collisions" (partial skip) and/or "rename_error"
        if cfg.mode == "rename" and mapping:
            # ``mapping`` is only the unambiguous cameras; conflicting ones are left out.
            _apply_rename(
                sub_root, meta, cfg, mapping, verdicts, path_prefix=subpath, commit_lock=commit_lock
            )
        return subpath, entry
    except Exception as exc:  # noqa: BLE001 - one bad sub-dataset shouldn't sink the sweep
        logger.error("sub-dataset %s failed: %s", subpath, exc)
        return subpath, {"error": str(exc)}
    finally:
        # Drop this sub-dataset's downloaded files immediately (see rename cleanup).
        shutil.rmtree(sub_work, ignore_errors=True)


_COLLECTION_REPORT_NAME = "camera_curation_collection.json"


def _collection_report_path(cfg: CameraCurationConfig) -> Path:
    """The aggregated nested report path, which doubles as the resume/progress log."""
    return Path(cfg.report_path) if cfg.report_path is not None else Path(_COLLECTION_REPORT_NAME)


# Aggregate sets tracked across a nested sweep; each becomes a sorted list in the
# summary report. ``failed`` maps subpath -> error (kept for retry + visibility).
_AGG_SET_KEYS = (
    "completed",
    "renamed",
    "with_unusable",
    "with_name_collision",
    "conflicts",
    "rename_skipped",
)


def _empty_aggregate() -> dict[str, Any]:
    agg: dict[str, Any] = {k: set() for k in _AGG_SET_KEYS}
    agg["failed"] = {}
    return agg


def _seed_aggregate_from_report(agg: dict[str, Any], report_path: Path) -> None:
    """Seed the aggregate from a prior summary file so ``--resume`` can continue.

    Reads only the summary lists (``completed``/``renamed``/…) and ``failed`` — the
    file carries no per-dataset detail. Best-effort: a missing/corrupt file just
    means a fresh start.
    """
    if not report_path.exists():
        return
    try:
        prior = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("resume: could not read %s (%s); starting fresh", report_path, exc)
        return
    if not isinstance(prior, dict):
        return
    for key in _AGG_SET_KEYS:
        vals = prior.get(key)
        if isinstance(vals, list):
            agg[key].update(vals)
    failed = prior.get("failed")
    if isinstance(failed, dict):
        agg["failed"].update(failed)


def _update_aggregate(agg: dict[str, Any], subpath: str, entry: dict[str, Any]) -> None:
    """Fold one sub-dataset's result into the aggregate (no per-dataset detail kept)."""
    if isinstance(entry, dict) and "error" in entry:
        agg["failed"][subpath] = entry["error"]  # not marked completed -> retried on resume
        return
    agg["failed"].pop(subpath, None)  # a previously-failed dataset that now succeeded
    agg["completed"].add(subpath)
    cameras = (entry.get("cameras") or {}) if isinstance(entry, dict) else {}
    if any((cam or {}).get("proposed_new_key") for cam in cameras.values()):
        agg["renamed"].add(subpath)
    if entry.get("has_unusable"):
        agg["with_unusable"].add(subpath)
    if entry.get("has_name_collision"):
        agg["with_name_collision"].add(subpath)
    if entry.get("collisions"):
        agg["conflicts"].add(subpath)
    if entry.get("rename_error"):
        agg["rename_skipped"].add(subpath)


def _fetch_progress_from_hub(cfg: CameraCurationConfig, report_path: Path) -> bool:
    """Best-effort: pull the progress log from the collection repo to ``report_path``.

    Lets ``--resume`` recover the checkpoint on a fresh (ephemeral) pod. Returns
    True if a file was fetched; never raises (a missing file just means fresh run).
    """
    from huggingface_hub import hf_hub_download

    try:
        cached = hf_hub_download(
            repo_id=cfg.repo_id,
            repo_type="dataset",
            filename=report_path.name,
            revision=cfg.branch,
        )
    except Exception as exc:  # noqa: BLE001 - no checkpoint on the Hub yet is normal
        logger.info("resume: no progress checkpoint on the Hub (%s); starting fresh", exc)
        return False
    report_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, report_path)
    logger.info("resume: fetched progress checkpoint from the Hub -> %s", report_path)
    return True


def _persist_progress_to_hub(cfg: CameraCurationConfig, report_path: Path, lock: "threading.Lock") -> None:
    """Best-effort: upload the progress log to the collection repo (under ``lock``).

    Serialized with the rename commits (same branch head) so uploads don't race
    concurrent renames. A failed checkpoint upload is logged, never fatal.
    """
    from huggingface_hub import HfApi

    try:
        with lock:
            HfApi().upload_file(
                path_or_fileobj=str(report_path),
                path_in_repo=report_path.name,
                repo_id=cfg.repo_id,
                repo_type="dataset",
                revision=cfg.branch,
                commit_message="curate: progress checkpoint (lerobot-curate-cameras)",
            )
    except Exception as exc:  # noqa: BLE001 - a lost checkpoint must not sink the sweep
        logger.warning("progress: failed to upload checkpoint to the Hub (%s)", exc)


def _summary_report(
    cfg: CameraCurationConfig, all_subpaths: list[str], agg: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the aggregated nested summary — counts plus lists of sub-datasets by
    outcome, no per-dataset detail (that lives in each dataset's own meta/info.json).

    The ``completed`` list is what ``--resume`` reads to skip finished sub-datasets.
    """
    return {
        "repo_id": cfg.repo_id,
        "mode": cfg.mode,
        "n_total": len(all_subpaths),
        "n_done": len(agg["completed"]),
        "n_renamed": len(agg["renamed"]),
        "n_failed": len(agg["failed"]),
        "n_rename_skipped": len(agg["rename_skipped"]),
        "n_conflicts": len(agg["conflicts"]),
        "n_with_unusable": len(agg["with_unusable"]),
        "n_with_name_collision": len(agg["with_name_collision"]),
        "renamed": sorted(agg["renamed"]),
        "with_unusable": sorted(agg["with_unusable"]),
        "with_name_collision": sorted(agg["with_name_collision"]),
        "conflicts": sorted(agg["conflicts"]),
        "rename_skipped": sorted(agg["rename_skipped"]),
        "failed": dict(sorted(agg["failed"].items())),
        "completed": sorted(agg["completed"]),
    }


def _run_nested(cfg: CameraCurationConfig, vlm: Any, subpaths: list[str]) -> None:
    """Curate each sub-dataset of a nested collection.

    Sub-datasets are processed with up to ``cfg.parallelism`` workers sharing the
    one VLM server; the work is download/latency-bound, so this overlaps I/O for a
    large speedup on a single GPU. Each worker uses its own temp dir (removed as
    soon as it is done), and Hub rename commits are serialized via ``commit_lock``
    so parallel renames don't race on the branch head.

    The aggregated report is rewritten after EACH sub-dataset completes (a progress
    log), and ``--resume`` skips sub-datasets already recorded there without an
    error — so an interrupted sweep can pick up where it left off (retrying only
    the failures) instead of redoing the whole collection.
    """
    if cfg.repo_id is None:
        raise ValueError("nested curation requires --repo_id (the collection repo on the Hub).")

    report_path = _collection_report_path(cfg)
    total = len(subpaths)
    commit_lock = threading.Lock()

    agg = _empty_aggregate()
    if cfg.resume:
        # On an ephemeral pod the local file is gone, so pull the checkpoint from
        # the Hub first when it is not on disk.
        if cfg.progress_to_hub and not report_path.exists():
            _fetch_progress_from_hub(cfg, report_path)
        _seed_aggregate_from_report(agg, report_path)
        if agg["completed"]:
            logger.info(
                "resume: %d/%d sub-dataset(s) already done in %s; processing the rest",
                len(agg["completed"]),
                total,
                report_path,
            )

    todo = [sp for sp in subpaths if sp not in agg["completed"]]

    def _record(sp: str, entry: dict[str, Any]) -> None:
        # Fold into the aggregate and persist after every completion so a
        # crashed/killed sweep can --resume. Only the summary is written — the
        # per-camera detail already lives in each sub-dataset's meta/info.json.
        _update_aggregate(agg, sp, entry)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_summary_report(cfg, subpaths, agg), indent=2), encoding="utf-8")
        if cfg.progress_to_hub:
            _persist_progress_to_hub(cfg, report_path, commit_lock)

    parallelism = max(1, min(cfg.parallelism, len(todo))) if todo else 1
    logger.info(
        "curate-cameras: %d sub-dataset(s) total, %d to process, %d already done (parallelism=%d)",
        total,
        len(todo),
        total - len(todo),
        parallelism,
    )
    if not todo:
        logger.info("curate-cameras: nothing to do — all sub-datasets already completed")
    elif parallelism == 1:
        for i, subpath in enumerate(todo, 1):
            sp, entry = _process_subdataset(cfg, vlm, subpath, i, len(todo), commit_lock)
            _record(sp, entry)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = [
                pool.submit(_process_subdataset, cfg, vlm, subpath, i, len(todo), commit_lock)
                for i, subpath in enumerate(todo, 1)
            ]
            for fut in as_completed(futures):
                sp, entry = fut.result()
                _record(sp, entry)

    # Top-level tallies, surfaced so they're easy to find/re-run:
    #  - failed: couldn't process the dataset at all (retried on the next --resume).
    #  - rename_skipped: whole-dataset rename skipped (on_collision="error").
    #  - conflicts: some cameras skipped for a label conflict but the rest renamed.
    report = _summary_report(cfg, subpaths, agg)
    logger.info(
        "curate-cameras: %d done, %d renamed, %d failed, %d whole-rename-skipped, "
        "%d with camera conflicts, %d with unusable view (of %d)",
        report["n_done"],
        report["n_renamed"],
        report["n_failed"],
        report["n_rename_skipped"],
        report["n_conflicts"],
        report["n_with_unusable"],
        total,
    )
    if report["failed"]:
        logger.warning("curate-cameras: failed sub-dataset(s): %s", sorted(report["failed"]))
    if report["rename_skipped"]:
        logger.warning(
            "curate-cameras: whole-rename-skipped sub-dataset(s): %s", sorted(report["rename_skipped"])
        )
    if report["conflicts"]:
        logger.warning("curate-cameras: sub-dataset(s) with skipped cameras: %s", sorted(report["conflicts"]))

    _write_and_echo_report(report, cfg, default_name=_COLLECTION_REPORT_NAME)


def _run_single(cfg: CameraCurationConfig, vlm: Any) -> None:
    """Curate one standalone dataset (root-level ``meta/``).

    Unless the user pointed at their own ``--root``, the first-episode download
    goes to a temp dir that is removed once curation (and any rename) is done, so
    no video files are left behind in the cache.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if cfg.root is None and cfg.repo_id is None:
        raise ValueError("Either --repo_id or --root must be provided.")

    user_root = cfg.root is not None
    root = Path(cfg.root) if user_root else Path(tempfile.mkdtemp(prefix="lerobot_curate_"))
    logger.info("curate-cameras: repo_id=%s root=%s mode=%s", cfg.repo_id, root, cfg.mode)

    try:
        # Only episode ``cfg.episode_index`` is fetched (a cheap partial download).
        dataset = LeRobotDataset(
            cfg.repo_id or "local",
            root=root,
            episodes=[cfg.episode_index],
            download_videos=True,
        )

        verdicts, mapping, rename_notes = _decide(
            dataset.root, dataset.meta, cfg, vlm, label=cfg.repo_id or str(root), dataset=dataset
        )
        # Stamp the verdict into info.json so a rename commit carries it.
        curator.stamp_verdicts_into_info(dataset.root, verdicts)
        report = curator.build_report(verdicts, mapping, cfg)
        report.update(rename_notes)  # "collisions" (partial skip) and/or "rename_error"
        _write_and_echo_report(report, cfg, default_name="camera_curation.json")

        if cfg.mode == "rename":
            if not mapping:
                logger.info("curate-cameras: nothing to rename (no confident new labels)")
            else:
                _apply_rename(dataset.root, dataset.meta, cfg, mapping, verdicts, dataset=dataset)
    finally:
        if not user_root:
            shutil.rmtree(root, ignore_errors=True)
            logger.info("curate-cameras: cleaned up temporary download at %s", root)


@parser.wrap()
def curate_cameras(cfg: CameraCurationConfig) -> None:
    """Run the camera-view curation pipeline over a dataset (or nested collection)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if cfg.mode not in ("report", "rename"):
        raise ValueError(f"--mode must be 'report' or 'rename', got {cfg.mode!r}")

    if cfg.job.is_remote:
        from lerobot.jobs.curate import submit_curate_to_hf

        return submit_curate_to_hf(cfg)

    require_package("datasets", "dataset")

    # Resolve which sub-datasets (if any) to process. Explicit --subpaths wins;
    # otherwise, on a Hub repo with no root meta/info.json, discover them all.
    subpaths: list[str] | None = list(cfg.subpaths) if cfg.subpaths else None
    if subpaths is None and cfg.repo_id is not None and cfg.root is None:
        discovered = _discover_subpaths(cfg.repo_id)
        if discovered is not None:
            logger.info("curate-cameras: nested collection — discovered %d sub-dataset(s)", len(discovered))
            subpaths = discovered

    if subpaths is not None and cfg.limit is not None:
        subpaths = subpaths[: cfg.limit]
        logger.info("curate-cameras: limited to first %d sub-dataset(s)", len(subpaths))

    vlm = make_vlm_client(cfg.vlm)
    if subpaths is not None:
        _run_nested(cfg, vlm, subpaths)
    else:
        _run_single(cfg, vlm)


def main() -> None:
    curate_cameras()


if __name__ == "__main__":
    main()
