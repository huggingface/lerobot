#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""``lerobot-wandb``: move LeRobot datasets to/from W&B Artifacts. Never touches the Hub.

Deliberately ``argparse``-based rather than wired into the central Draccus config parser,
consistent with this repo's existing precedent for small standalone scripts. Transfer commands
always use an online W&B run: their success contract is a durable, cross-machine artifact, not a
local offline run that still needs a later ``wandb sync``.

Examples:

```shell
lerobot-wandb dataset upload --root ./my-dataset --entity my-team --project my-project \
    --name pick-cube --alias raw

lerobot-wandb dataset download --ref my-team/my-project/pick-cube:latest --root ./materialized

lerobot-wandb model upload --root ./outputs/train/pretrained_model --entity my-team \
    --project my-project --name pick-cube-policy --alias candidate \
    --registry-collection pick-cube-policy

lerobot-wandb model download --ref my-team/my-project/pick-cube-policy:latest --root ./policy
```
"""

import argparse
from pathlib import Path

import wandb

from lerobot.utils.utils import init_logging

from .inspect import (
    inspect_dataset_directory,
    inspect_model_directory,
    validate_dataset_directory,
    validate_model_directory,
)
from .refs import parse_artifact_ref
from .store import download_artifact, upload_directory

DATASET_ARTIFACT_TYPE = "dataset"
MODEL_ARTIFACT_TYPE = "model"


def cmd_dataset_upload(args: argparse.Namespace) -> None:
    # Validate — and pay any local, no-network cost of a bad directory — before a run ever starts.
    metadata = inspect_dataset_directory(args.root)
    aliases = args.aliases or ["latest"]

    run = wandb.init(entity=args.entity, project=args.project, job_type="dataset_upload", mode="online")
    try:
        result = upload_directory(
            run,
            args.root,
            name=args.name,
            artifact_type=DATASET_ARTIFACT_TYPE,
            aliases=aliases,
            metadata=metadata.to_wandb_metadata(),
        )
    finally:
        run.finish()

    print(f"Uploaded dataset artifact: {result.resolved_ref}")
    print(f"Aliases applied: {', '.join(aliases)}")


def cmd_dataset_download(args: argparse.Namespace) -> None:
    # Fail fast on a malformed ref before a run ever starts.
    parsed = parse_artifact_ref(args.ref)

    # The lineage run's own home defaults to the artifact's entity/project, but a caller with only
    # read access to the source project (e.g. a shared team dataset) needs to log it somewhere they
    # can actually create runs — `use_artifact` accepts a fully qualified ref regardless of which
    # project the run itself lives in, so overriding here never changes which artifact is fetched.
    run = wandb.init(
        entity=args.entity or parsed.entity,
        project=args.project or parsed.project,
        job_type="dataset_download",
        mode="online",
    )
    try:
        result = download_artifact(
            run,
            parsed,
            expected_type=DATASET_ARTIFACT_TYPE,
            download_root=args.root,
            validator=validate_dataset_directory,
        )
    finally:
        run.finish()

    print(f"Downloaded dataset artifact {result.resolved_ref} to: {result.local_path}")


def cmd_model_upload(args: argparse.Namespace) -> None:
    # Validate — and pay any local, no-network cost of a bad directory — before a run ever starts.
    metadata = inspect_model_directory(args.root)
    aliases = args.aliases or ["latest"]

    run = wandb.init(entity=args.entity, project=args.project, job_type="model_upload", mode="online")
    try:
        result = upload_directory(
            run,
            args.root,
            name=args.name,
            artifact_type=MODEL_ARTIFACT_TYPE,
            aliases=aliases,
            metadata=metadata.to_wandb_metadata(),
            registry_collection=args.registry_collection,
        )
    finally:
        run.finish()

    print(f"Uploaded model artifact: {result.resolved_ref}")
    print(f"Aliases applied: {', '.join(aliases)}")
    if result.registry_collection:
        print(f"Linked into registry collection: {result.registry_collection}")


def cmd_model_download(args: argparse.Namespace) -> None:
    # Fail fast on a malformed ref before a run ever starts.
    parsed = parse_artifact_ref(args.ref)

    # The lineage run's own home defaults to the artifact's entity/project, but a caller with only
    # read access to the source project (e.g. a shared team dataset) needs to log it somewhere they
    # can actually create runs — `use_artifact` accepts a fully qualified ref regardless of which
    # project the run itself lives in, so overriding here never changes which artifact is fetched.
    run = wandb.init(
        entity=args.entity or parsed.entity,
        project=args.project or parsed.project,
        job_type="model_download",
        mode="online",
    )
    try:
        result = download_artifact(
            run,
            parsed,
            expected_type=MODEL_ARTIFACT_TYPE,
            download_root=args.root,
        )
    finally:
        run.finish()

    validate_model_directory(result.local_path)
    print(
        f"Downloaded model artifact {result.resolved_ref} to: {result.local_path} "
        "(use directly as a rollout policy path)"
    )


def _add_upload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True, help="Local directory to upload.")
    parser.add_argument("--entity", default=None, help="W&B entity. Defaults to your W&B default entity.")
    parser.add_argument("--project", required=True, help="W&B project to upload into.")
    parser.add_argument("--name", required=True, help="Artifact collection name.")
    parser.add_argument(
        "--alias", dest="aliases", action="append", default=[], help="Repeatable. Defaults to ['latest']."
    )


def _add_download_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ref", required=True, help="Artifact reference: entity/project/name:version_or_alias"
    )
    parser.add_argument("--root", type=Path, required=True, help="Local directory to materialize into.")
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity to create the lineage run in. Defaults to the artifact's own entity (--ref). "
        "Override if you can read the artifact but can't create runs in its project.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project to create the lineage run in. Defaults to the artifact's own project (--ref).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-wandb", description="Move LeRobot datasets and models to/from W&B Artifacts."
    )
    resource_subparsers = parser.add_subparsers(dest="resource", required=True)

    dataset_parser = resource_subparsers.add_parser("dataset", help="Upload/download a dataset artifact.")
    dataset_action_subparsers = dataset_parser.add_subparsers(dest="action", required=True)

    dataset_upload_parser = dataset_action_subparsers.add_parser(
        "upload", help="Validate and upload a local dataset directory as a versioned W&B Artifact."
    )
    _add_upload_args(dataset_upload_parser)
    dataset_upload_parser.set_defaults(func=cmd_dataset_upload)

    dataset_download_parser = dataset_action_subparsers.add_parser(
        "download", help="Download a dataset Artifact into a local, LeRobotDataset-ready directory."
    )
    _add_download_args(dataset_download_parser)
    dataset_download_parser.set_defaults(func=cmd_dataset_download)

    model_parser = resource_subparsers.add_parser("model", help="Upload/download a model artifact.")
    model_action_subparsers = model_parser.add_subparsers(dest="action", required=True)

    model_upload_parser = model_action_subparsers.add_parser(
        "upload", help="Validate and upload a local model directory as a versioned W&B Artifact."
    )
    _add_upload_args(model_upload_parser)
    model_upload_parser.add_argument(
        "--registry-collection",
        default=None,
        help="If set, link the uploaded version into this unified-Registry collection "
        "(wandb-registry-model/<name>).",
    )
    model_upload_parser.set_defaults(func=cmd_model_upload)

    model_download_parser = model_action_subparsers.add_parser(
        "download", help="Download a model Artifact into a local, rollout-ready policy directory."
    )
    _add_download_args(model_download_parser)
    model_download_parser.set_defaults(func=cmd_model_download)

    return parser


def main(argv: list[str] | None = None) -> None:
    init_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
