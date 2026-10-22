#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Research-only raw human-POV MP4 to LeRobot hand/action annotations."""

import argparse
import json
import re
from pathlib import Path

import draccus

from lerobot.annotations.hands.config import HandPipelineConfig, estimate, public_catalog
from lerobot.annotations.hands.pipeline import run_hand_pipeline
from lerobot.annotations.hands.source import RawHandVideoSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="HandPipelineConfig JSON")
    parser.add_argument("--catalog", action="store_true", help="Print compute profiles and a setup template")
    parser.add_argument(
        "--estimate", action="store_true", help="Inspect inputs and estimate; no GPU or paid Jobs"
    )
    parser.add_argument("--hub-repo", help="Optional raw-MP4/calibration dataset to download first")
    parser.add_argument("--hub-revision", help="Immutable 40-character commit for --hub-repo")
    parser.add_argument("--upload-repo", help="Explicitly upload the resulting dataset as private")
    args = parser.parse_args()
    if args.catalog:
        print(json.dumps(public_catalog(), indent=2, allow_nan=False))
        return
    if not args.config:
        parser.error("Supply --config or --catalog")
    config = draccus.decode(HandPipelineConfig, json.loads(args.config.read_text()))
    if args.hub_repo:
        if not re.fullmatch(r"[0-9a-f]{40}", args.hub_revision or ""):
            parser.error("--hub-repo requires a pinned --hub-revision")
        if args.estimate:
            parser.error("Estimate local inputs first; this mode does not download raw media")
        if not config.backend.research_terms_accepted:
            parser.error("Accept the upstream research terms before downloading inputs for inference")
        from huggingface_hub import snapshot_download

        config.input_root = Path(
            snapshot_download(
                args.hub_repo,
                repo_type="dataset",
                revision=args.hub_revision,
                allow_patterns=["*.mp4", "*.intrinsics.json"],
            )
        )
    if args.estimate:
        config.validate()
        source = RawHandVideoSource(config.input_root, config.videos_per_group)
        result = estimate(config, source.groups)
    else:
        result = run_hand_pipeline(config)
        if args.upload_repo and result.get("episodes"):
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.upload_repo, repo_type="dataset", private=True, exist_ok=True)
            if not api.dataset_info(args.upload_repo).private:
                raise ValueError("Research outputs are uploaded only to private repositories by this command")
            api.upload_folder(repo_id=args.upload_repo, repo_type="dataset", folder_path=config.output_root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
