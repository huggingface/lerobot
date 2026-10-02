#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Build MP4 streaming sidecars with optional, explicit publication."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec
from huggingface_hub import HfApi

from lerobot.streaming.location import LocationKind, StorageLocation
from lerobot.streaming.sidecar import SidecarSpec
from lerobot.utils.import_utils import _datasets_available, require_package

if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.streaming_sidecar import (
        SIDECAR_REPO_BRANCH,
        build_mp4_sidecar,
        make_sidecar_spec,
        published_sidecar_url,
    )


def parse_args() -> argparse.Namespace:
    """Parse the explicit sidecar build and optional publication arguments."""
    parser = argparse.ArgumentParser(description="Build a reusable MP4 byte-index sidecar for streaming.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=("dataset", "bucket"), default="dataset")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--range-backend", choices=("fsspec", "native-http"), default=None)
    parser.add_argument("--max-probe-mb", type=int, default=64)
    parser.add_argument(
        "--push",
        action="store_true",
        help="Publish to the repository's lerobot-sidecars branch, or directly to the bucket.",
    )
    return parser.parse_args()


def push_sidecar(local_path: str, spec: SidecarSpec) -> list[str]:
    """Explicitly publish an index without advancing the pinned dataset's source branch."""
    require_package("datasets", "dataset")
    location = StorageLocation.parse(spec.data_root)
    if not location.is_hf:
        raise ValueError("--push currently supports only hf:// data roots")

    fs = fsspec.filesystem("hf")
    remote = published_sidecar_url(spec)
    if location.kind is LocationKind.HF_DATASET and location.repo_id is not None:
        # Keep source commits immutable. Index publication advances only this separate branch.
        # published_sidecar_url above already required the root to be pinned to a commit.
        HfApi().create_branch(
            repo_id=location.repo_id,
            repo_type="dataset",
            branch=SIDECAR_REPO_BRANCH,
            revision=location.pinned_commit,
            exist_ok=True,
        )
    fs.put(str(Path(local_path)), remote)
    return [remote]


def main() -> None:
    """Build a local index and publish only a full-dataset sidecar when requested."""
    args = parse_args()
    require_package("datasets", "dataset")

    meta = LeRobotDatasetMetadata(args.repo_id, revision=args.revision, repo_type=args.repo_type)
    meta.ensure_readable()
    total = (
        int(meta.total_episodes) if args.episodes is None else min(args.episodes, int(meta.total_episodes))
    )
    spec = make_sidecar_spec(meta, args.data_root)
    if total != int(meta.total_episodes):
        selected_paths = {
            str(meta.get_video_file_path(ep_idx, key)) for ep_idx in range(total) for key in meta.video_keys
        }
        spec = SidecarSpec(
            repo_id=spec.repo_id,
            revision=spec.revision,
            data_root=spec.data_root,
            source_files=tuple(item for item in spec.source_files if item[0] in selected_paths),
            source_fingerprints=tuple(item for item in spec.source_fingerprints if item[0] in selected_paths),
        )

    start = time.perf_counter()
    build_mp4_sidecar(
        args.output,
        spec,
        range_backend=args.range_backend or StorageLocation.parse(args.data_root).range_backend,
        workers=args.workers,
        max_probe_bytes=args.max_probe_mb * 1024 * 1024,
    )
    elapsed = time.perf_counter() - start
    print(f"wrote {args.output}")
    print(f"episodes={total} files={len(spec.source_files)} elapsed_s={elapsed:.2f}")
    if args.push:
        if total != int(meta.total_episodes):
            raise ValueError("Only a complete dataset sidecar can be published")
        pushed = push_sidecar(args.output, spec)
        for remote in pushed:
            print(f"pushed {remote}")
    else:
        print("push_skipped: pass --push for explicit publication")


if __name__ == "__main__":
    main()
