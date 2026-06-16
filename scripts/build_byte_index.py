#!/usr/bin/env python
"""Build mmap-able byte-index sidecars for LeRobot streaming datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.datasets.byte_index_builder import (
    build_byte_index_tables,
    load_existing_file_ids,
    write_byte_index,
)
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LeRobot video byte-index sidecar.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data-root", required=True, help="fsspec root for videos/ + data/")
    parser.add_argument("--output", type=Path, required=True, help="Output meta/byte_index directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit episodes (debug/smoke)")
    parser.add_argument("--no-keyframes", action="store_true")
    args = parser.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)
    output = args.output
    existing = load_existing_file_ids(output)
    if existing:
        logger.info("resuming: %s files already indexed", len(existing))

    files_tbl, episodes_tbl, keyframes_tbl = build_byte_index_tables(
        meta,
        args.data_root,
        include_keyframes=not args.no_keyframes,
        workers=args.workers,
        existing_files=existing,
        max_episodes=args.max_episodes,
    )
    write_byte_index(output, files_tbl, episodes_tbl, keyframes_tbl, merge_existing=True)
    logger.info("wrote byte index to %s", output)


if __name__ == "__main__":
    main()
