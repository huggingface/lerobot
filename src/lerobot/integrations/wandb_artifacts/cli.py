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

lerobot-wandb rollout upload --root ./rollout_pick-cube --entity my-team --project my-project \
    --name pick-cube-rollout --model-ref my-team/my-project/pick-cube-policy:v3 \
    --episodes-succeeded 7
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
from .rollout import (
    ROLLOUT_ARTIFACT_TYPE,
    RolloutSummary,
    select_representative_video,
    validate_success_count,
)
from .store import declare_input, download_artifact, upload_directory

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
            validator=validate_model_directory,
        )
    finally:
        run.finish()

    print(
        f"Downloaded model artifact {result.resolved_ref} to: {result.local_path} "
        "(use directly as a rollout policy path)"
    )


def cmd_rollout_upload(args: argparse.Namespace) -> None:
    # Everything local and fallible happens before `wandb.init` creates a run: a bad directory, an
    # impossible success count or a malformed model ref must not leave an empty run behind.
    metadata = inspect_dataset_directory(args.root)
    validate_success_count(args.episodes_succeeded, metadata.total_episodes)
    parsed_model_ref = parse_artifact_ref(args.model_ref)
    video = select_representative_video(args.root)
    aliases = args.aliases or ["latest"]

    run = wandb.init(entity=args.entity, project=args.project, job_type="rollout_upload", mode="online")
    try:
        # Lineage only: the model that produced this rollout is referenced, never downloaded.
        model = declare_input(run, parsed_model_ref, expected_type=MODEL_ARTIFACT_TYPE)
        summary = RolloutSummary.build(
            metadata,
            successes=args.episodes_succeeded,
            model_requested_ref=model.requested_ref,
            model_resolved_ref=model.resolved_ref,
            video=video,
        )
        result = upload_directory(
            run,
            args.root,
            name=args.name,
            artifact_type=ROLLOUT_ARTIFACT_TYPE,
            aliases=aliases,
            metadata={**metadata.to_wandb_metadata(), **summary.to_wandb_metadata()},
        )
        run.summary.update(summary.to_wandb_metadata())
        if video is not None:
            # Exactly one video reaches the run UI. The rest stay in the Artifact: a rollout can be
            # hundreds of episodes, and logging each as run media would duplicate the whole dataset
            # into W&B's media store for no added information.
            run.log(
                {"rollout_video": wandb.Video(str(args.root / video.path), fps=metadata.fps, format="mp4")}
            )
    finally:
        run.finish()

    print(f"Uploaded rollout artifact: {result.resolved_ref}")
    print(f"Aliases applied: {', '.join(aliases)}")
    print(f"Model input (lineage): {model.resolved_ref}")
    print(
        f"Episodes: {summary.episodes} | successes: {summary.successes} "
        f"| success rate: {summary.success_rate:.1%} | duration: {summary.duration_s:.1f}s"
    )
    if video is None:
        print("No video in this rollout dataset: nothing logged as run media.")
    else:
        print(
            f"Representative video: {video.path} ({video.video_key}, "
            f"episode(s) {', '.join(str(index) for index in video.episodes)})"
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

    rollout_parser = resource_subparsers.add_parser("rollout", help="Upload a rollout result.")
    rollout_action_subparsers = rollout_parser.add_subparsers(dest="action", required=True)

    rollout_upload_parser = rollout_action_subparsers.add_parser(
        "upload",
        help="Validate and upload a local rollout dataset as a versioned W&B Artifact, with the "
        "model that produced it declared as a run input.",
    )
    _add_upload_args(rollout_upload_parser)
    rollout_upload_parser.add_argument(
        "--model-ref",
        required=True,
        help="Model artifact that produced this rollout: entity/project/name:version_or_alias. "
        "Referenced for lineage only — never downloaded.",
    )
    rollout_upload_parser.add_argument(
        "--episodes-succeeded",
        type=int,
        required=True,
        help="How many episodes the operator judged successful. Not auto-detected.",
    )
    rollout_upload_parser.set_defaults(func=cmd_rollout_upload)

    return parser


def main(argv: list[str] | None = None) -> None:
    init_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
