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
and (2) label each view (``top``/``wrist``/``front``/…), then either records the
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


def _uniform_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    step = (n - 1) / (k - 1) if k > 1 else 0.0
    return sorted({round(i * step) for i in range(k)})


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
            for key in video_cameras:
                frames[key] = provider.video_for_episode(record, cfg.n_frames, camera_key=key)

    image_cameras = [k for k in cameras if k in image_keys]
    if image_cameras:
        if dataset is not None:
            n = len(dataset)
            for i in _uniform_indices(n, cfg.n_frames):
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
) -> None:
    """Apply the computed ``{old: new}`` camera-key mapping.

    Video datasets on the Hub → download-free server-side rename commit (scoped to
    ``path_prefix`` for a sub-dataset). Otherwise (image dataset, local-only, or a
    swap/cycle) → local ``rename_features`` over a full copy — single-dataset only.
    """
    video_keys = set(meta.video_keys)
    all_video = set(mapping) <= video_keys
    has_swap = bool(set(mapping.values()) & set(mapping))

    if cfg.repo_id is not None and all_video and not has_swap:
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
    from lerobot.datasets import LeRobotDataset as _LeRobotDataset
    from lerobot.datasets import remove_feature, rename_features

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
) -> tuple[list["curator.CameraVerdict"], dict[str, str], str | None]:
    """Sample frames, run the VLM, and compute the rename mapping for one dataset.

    Returns ``(verdicts, mapping, mapping_error)``. A label collision does NOT
    abort: the verdicts are kept, ``mapping`` is empty, and ``mapping_error``
    carries the message so the quality report survives and the rename is simply
    skipped for this dataset.
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
    try:
        mapping = curator.build_name_mapping(verdicts, meta.features, cfg)
        mapping_error = None
        logger.info("  [%s] proposed rename mapping: %s", label, mapping or "(none)")
    except ValueError as exc:
        mapping, mapping_error = {}, str(exc)
        logger.warning("  [%s] rename skipped, verdicts kept: %s", label, exc)
    return verdicts, mapping, mapping_error


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


def _run_nested(cfg: CameraCurationConfig, vlm: Any, subpaths: list[str]) -> None:
    """Curate each sub-dataset of a nested collection.

    Each sub-dataset is materialized into its own temp dir, and that dir is
    removed as soon as the sub-dataset is done — the disk footprint stays at one
    sub-dataset's first episode at a time, never the whole sweep.
    """
    if cfg.repo_id is None:
        raise ValueError("nested curation requires --repo_id (the collection repo on the Hub).")

    collection: dict[str, Any] = {}
    for i, subpath in enumerate(subpaths, 1):
        logger.info("===== sub-dataset %d/%d: %s =====", i, len(subpaths), subpath)
        sub_work = Path(tempfile.mkdtemp(prefix="lerobot_curate_sub_"))
        try:
            sub_root, meta = _materialize_subdataset(cfg.repo_id, subpath, sub_work, cfg.episode_index)
            verdicts, mapping, mapping_error = _decide(sub_root, meta, cfg, vlm, label=subpath)
            # Record verdicts first — a collision below must not lose the quality report.
            curator.stamp_verdicts_into_info(sub_root, verdicts)
            entry = curator.build_report(verdicts, mapping, cfg)
            if mapping_error:
                entry["rename_error"] = mapping_error
            if cfg.mode == "rename" and mapping:
                _apply_rename(sub_root, meta, cfg, mapping, verdicts, path_prefix=subpath)
            collection[subpath] = entry
        except Exception as exc:  # noqa: BLE001 - one bad sub-dataset shouldn't sink the sweep
            logger.error("sub-dataset %s failed: %s", subpath, exc)
            collection[subpath] = {"error": str(exc)}
        finally:
            # Drop this sub-dataset's downloaded files immediately (see rename cleanup).
            shutil.rmtree(sub_work, ignore_errors=True)
            logger.info("curate-cameras: cleaned up temporary download at %s", sub_work)

    # Hard failures (couldn't process at all) vs rename-skipped (verdicts kept,
    # rename skipped — e.g. a label collision). Both surfaced top-level so they
    # are easy to find and re-run rather than grepping every entry.
    failed = {sp: e["error"] for sp, e in collection.items() if isinstance(e, dict) and "error" in e}
    rename_skipped = {
        sp: e["rename_error"] for sp, e in collection.items() if isinstance(e, dict) and "rename_error" in e
    }
    logger.info(
        "curate-cameras: %d ok, %d failed, %d rename-skipped (of %d)",
        len(subpaths) - len(failed) - len(rename_skipped),
        len(failed),
        len(rename_skipped),
        len(subpaths),
    )
    if failed:
        logger.warning("curate-cameras: failed sub-dataset(s): %s", sorted(failed))
    if rename_skipped:
        logger.warning("curate-cameras: rename-skipped sub-dataset(s): %s", sorted(rename_skipped))

    report = {
        "repo_id": cfg.repo_id,
        "mode": cfg.mode,
        "n_total": len(subpaths),
        "n_failed": len(failed),
        "n_rename_skipped": len(rename_skipped),
        "failed": failed,
        "rename_skipped": rename_skipped,
        "subdatasets": collection,
    }
    _write_and_echo_report(report, cfg, default_name="camera_curation_collection.json")


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

        verdicts, mapping, mapping_error = _decide(
            dataset.root, dataset.meta, cfg, vlm, label=cfg.repo_id or str(root), dataset=dataset
        )
        # Stamp the verdict into info.json so a rename commit carries it.
        curator.stamp_verdicts_into_info(dataset.root, verdicts)
        report = curator.build_report(verdicts, mapping, cfg)
        if mapping_error:
            report["rename_error"] = mapping_error
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
