#!/usr/bin/env python
"""Create a progress-conditioned copy of the blind SO101 space-bar dataset."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.dataset_tools import add_features, recompute_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from _spacebar_progress import (
    DERIVED_DATASET_NAME,
    EXPECTED_MOTOR_NAMES,
    PROGRESS_FEATURE_NAMES,
    SOURCE_REPO_ID,
    compute_progress_stats,
    episode_lengths_from_metadata,
    progress_vector,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-id", default=SOURCE_REPO_ID)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/datasets") / DERIVED_DATASET_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--recompute-stats",
        action="store_true",
        help="Recompute all numeric stats after the copy. By default only the new progress stats are patched.",
    )
    return parser.parse_args()


def infer_output_repo_id(output_repo_id: str | None) -> str:
    if output_repo_id:
        return output_repo_id
    try:
        user = HfApi().whoami().get("name")
    except Exception as exc:
        raise RuntimeError(
            "Could not infer your Hugging Face username. Pass "
            f"--output-repo-id=<hf_user>/{DERIVED_DATASET_NAME}."
        ) from exc
    if not user:
        raise RuntimeError(f"Could not infer HF username. Pass --output-repo-id=<hf_user>/{DERIVED_DATASET_NAME}.")
    return f"{user}/{DERIVED_DATASET_NAME}"


def feature_info() -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": (len(PROGRESS_FEATURE_NAMES),),
        "names": PROGRESS_FEATURE_NAMES,
    }


def audit_source_dataset(dataset: LeRobotDataset, episode_lengths: list[int]) -> None:
    logger.info("Source dataset: %s", dataset.repo_id)
    logger.info(
        "episodes=%d frames=%d fps=%s lengths=%s",
        dataset.meta.total_episodes,
        dataset.meta.total_frames,
        dataset.meta.fps,
        sorted(set(episode_lengths)),
    )
    for feature_key in (OBS_STATE, ACTION):
        feature = dataset.meta.features.get(feature_key)
        if feature is None:
            raise ValueError(f"Missing required feature: {feature_key}")
        logger.info("%s names: %s", feature_key, feature.get("names"))
        if feature.get("names") != EXPECTED_MOTOR_NAMES:
            logger.warning("Unexpected %s names. Expected %s", feature_key, EXPECTED_MOTOR_NAMES)
    if OBS_ENV_STATE in dataset.meta.features:
        raise ValueError(f"Source dataset already has {OBS_ENV_STATE}; refusing to modify in place.")


def patch_progress_stats(dataset: LeRobotDataset, episode_lengths: list[int]) -> None:
    template_keys = None
    if dataset.meta.stats:
        for key in (OBS_STATE, ACTION):
            if key in dataset.meta.stats:
                template_keys = dataset.meta.stats[key].keys()
                break
    stats = dict(dataset.meta.stats or {})
    stats[OBS_ENV_STATE] = compute_progress_stats(episode_lengths, template_keys)
    write_stats(stats, dataset.root)
    dataset.meta.stats = stats
    logger.info("Patched stats for %s: %s", OBS_ENV_STATE, stats[OBS_ENV_STATE])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    output_repo_id = infer_output_repo_id(args.output_repo_id)
    output_root = args.output_root.resolve()

    source = LeRobotDataset(args.source_repo_id, root=args.source_root)
    if source.root.resolve() == output_root:
        raise ValueError("Output root must differ from the source dataset root.")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    episode_lengths = episode_lengths_from_metadata(source.meta.episodes)
    audit_source_dataset(source, episode_lengths)

    length_by_episode = {episode_index: length for episode_index, length in enumerate(episode_lengths)}

    def make_progress(row: dict, episode_index: int, frame_index: int) -> list[float]:
        del row
        return progress_vector(int(frame_index), length_by_episode[int(episode_index)]).tolist()

    logger.info("Writing derived dataset to %s with repo_id=%s", output_root, output_repo_id)
    derived = add_features(
        source,
        features={OBS_ENV_STATE: (make_progress, feature_info())},
        output_dir=output_root,
        repo_id=output_repo_id,
    )

    if args.recompute_stats:
        derived = recompute_stats(derived, skip_image_video=True)
    else:
        patch_progress_stats(derived, episode_lengths)

    logger.info("Derived features: %s", sorted(derived.meta.features))
    logger.info("Derived dataset root: %s", derived.root)

    if args.push_to_hub:
        logger.info("Pushing %s to Hugging Face Hub", output_repo_id)
        derived.push_to_hub(private=args.private)


if __name__ == "__main__":
    main()
