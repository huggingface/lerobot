#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import time
from pathlib import Path

import fsspec

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.episode_video_streaming import EpisodeVideoManifest, assert_hf_hub_range_cache_branch

DEFAULT_REPO = "allenai/MolmoAct2-BimanualYAM-Dataset"
DEFAULT_REVISION = "e9f21ae15074330839f2ac25ed4b49d76dfa1f9c"
DEFAULT_DATA_ROOT = "hf://buckets/pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reusable MP4 byte-index sidecar for streaming.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--range-backend", choices=("fsspec", "native-http"), default="native-http")
    parser.add_argument("--max-probe-mb", type=int, default=64)
    parser.add_argument(
        "--no-push", action="store_true", help="Do not upload the sidecar to data_root/meta/mp4-sidecars."
    )
    parser.add_argument("--no-hub-branch-assert", action="store_true")
    return parser.parse_args()


def push_sidecar(local_path: str, data_root: str) -> list[str]:
    if not data_root.startswith("hf://"):
        return []

    local = Path(local_path)
    fs = fsspec.filesystem("hf")
    remote_dir = f"{data_root.rstrip('/')}/meta/mp4-sidecars"
    remote_paths = [f"{remote_dir}/{local.name}"]

    for remote in remote_paths:
        fs.put(str(local), remote)
    return remote_paths


def main() -> None:
    args = parse_args()
    if args.data_root.startswith("hf://") and not args.no_hub_branch_assert:
        assert_hf_hub_range_cache_branch()

    meta = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)
    meta.ensure_readable()
    total = (
        int(meta.total_episodes) if args.episodes is None else min(args.episodes, int(meta.total_episodes))
    )
    rel_paths = sorted(
        {str(meta.get_video_file_path(ep_idx, key)) for ep_idx in range(total) for key in meta.video_keys}
    )

    start = time.perf_counter()
    EpisodeVideoManifest.write_file_sidecar(
        args.output,
        rel_paths,
        args.data_root,
        range_backend=args.range_backend,
        workers=args.workers,
        max_probe_bytes=args.max_probe_mb * 1024 * 1024,
    )
    elapsed = time.perf_counter() - start
    print(f"wrote {args.output}")
    print(f"episodes={total} files={len(rel_paths)} elapsed_s={elapsed:.2f}")
    if args.no_push:
        print("push_skipped: --no-push")
    else:
        pushed = push_sidecar(args.output, args.data_root)
        for remote in pushed:
            print(f"pushed {remote}")


if __name__ == "__main__":
    main()
