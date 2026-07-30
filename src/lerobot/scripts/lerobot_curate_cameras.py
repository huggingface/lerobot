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
"""

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
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import _datasets_available, require_package

if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def _resolve_root(cfg: CameraCurationConfig) -> Path:
    """Concrete, writable root for the dataset (never the symlinked snapshot cache)."""
    if cfg.root is not None:
        return Path(cfg.root)
    if cfg.repo_id is not None:
        return HF_LEROBOT_HOME / cfg.repo_id
    raise ValueError("Either --repo_id or --root must be provided.")


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


def _sample_frames(dataset: "LeRobotDataset", cfg: CameraCurationConfig) -> dict[str, list[Any]]:
    """Sample ``n_frames`` from the inspected episode for each (non-depth) camera.

    Video cameras go through the annotation frame provider (uint8 frames); image
    cameras are read straight from the dataset rows and scaled to uint8.
    """
    meta = dataset.meta
    depth_keys = set(meta.depth_keys)
    video_keys = set(meta.video_keys)
    image_keys = set(meta.image_keys)
    cameras = [k for k in meta.camera_keys if k not in depth_keys]

    frames: dict[str, list[Any]] = {k: [] for k in cameras}

    video_cameras = [k for k in cameras if k in video_keys]
    if video_cameras:
        provider = make_frame_provider(dataset.root, video_backend=cfg.video_backend)
        records = list(iter_episodes(dataset.root, only_episodes=(cfg.episode_index,)))
        record = records[0] if records else None
        if record is not None:
            for key in video_cameras:
                frames[key] = provider.video_for_episode(record, cfg.n_frames, camera_key=key)

    image_cameras = [k for k in cameras if k in image_keys]
    if image_cameras:
        n = len(dataset)
        for i in _uniform_indices(n, cfg.n_frames):
            item = dataset[i]
            for key in image_cameras:
                if key in item:
                    frames[key].append(_to_uint8_frame(item[key]))

    return frames


def _apply_rename(
    root: Path,
    dataset: "LeRobotDataset",
    cfg: CameraCurationConfig,
    mapping: dict[str, str],
    verdicts: list["curator.CameraVerdict"],
) -> None:
    """Apply the computed ``{old: new}`` camera-key mapping.

    Video datasets on the Hub → download-free server-side rename commit.
    Otherwise (image dataset, local-only, or a swap/cycle) → local
    ``rename_features`` over a full copy of the dataset.
    """
    video_keys = set(dataset.meta.video_keys)
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
                branch=cfg.branch,
                commit_message=cfg.push_commit_message,
            )
            oid = getattr(commit, "oid", None)
            ref = cfg.branch or "main"
            logger.info("Hub rename committed to %s@%s (%s)", cfg.repo_id, ref, oid)
        finally:
            shutil.rmtree(work, ignore_errors=True)
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


@parser.wrap()
def curate_cameras(cfg: CameraCurationConfig) -> None:
    """Run the camera-view curation pipeline over a dataset's first episode."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if cfg.mode not in ("report", "rename"):
        raise ValueError(f"--mode must be 'report' or 'rename', got {cfg.mode!r}")

    if cfg.job.is_remote:
        from lerobot.jobs.curate import submit_curate_to_hf

        return submit_curate_to_hf(cfg)

    require_package("datasets", "dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = _resolve_root(cfg)
    logger.info("curate-cameras: repo_id=%s root=%s mode=%s", cfg.repo_id, root, cfg.mode)

    # Only episode ``cfg.episode_index`` is fetched (a cheap partial download).
    dataset = LeRobotDataset(
        cfg.repo_id or "local",
        root=root,
        episodes=[cfg.episode_index],
        download_videos=True,
    )

    frames = _sample_frames(dataset, cfg)
    n_with_frames = sum(1 for v in frames.values() if v)
    logger.info("curate-cameras: %d camera(s), %d with sampled frames", len(frames), n_with_frames)

    vlm = make_vlm_client(cfg.vlm)
    verdicts = curator.curate_cameras(frames, cfg, vlm)
    for v in verdicts:
        logger.info(
            "  %s -> label=%s usable=%s%s",
            v.camera_key,
            v.view_label,
            v.usable,
            "" if v.usable else f" (blur_reason={v.blur_reason!r})",
        )

    mapping = curator.build_name_mapping(verdicts, dataset.meta.features, cfg)
    report_path = curator.write_report(dataset.root, verdicts, mapping, cfg)
    logger.info("curate-cameras: report written to %s", report_path)
    logger.info("curate-cameras: proposed rename mapping: %s", mapping or "(none)")

    if cfg.mode == "rename":
        if not mapping:
            logger.info("curate-cameras: nothing to rename (no confident labels differ from current keys)")
            return
        _apply_rename(dataset.root, dataset, cfg, mapping, verdicts)


def main() -> None:
    curate_cameras()


if __name__ == "__main__":
    main()
