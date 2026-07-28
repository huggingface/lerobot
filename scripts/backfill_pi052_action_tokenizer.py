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

"""Audit or backfill checkpoint-local FAST artifacts for PI052 model repositories."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

DEFAULT_REPOSITORIES = (
    "pepijn223/pi052_atomic4_01_baseline",
    "pepijn223/pi052_atomic4_02_lr_1e5",
    "pepijn223/pi052_atomic4_03_recipe_50_50",
    "pepijn223/pi052_atomic4_04_flow_weight_10",
    "pepijn223/pi052_atomic4_05_flow_repeat_1",
    "pepijn223/pi052_atomic4_06_ki_off",
)
CHECKPOINT_DIRECTORIES = (
    "",
    "checkpoints/003000/pretrained_model",
    "checkpoints/006000/pretrained_model",
    "checkpoints/009000/pretrained_model",
    "checkpoints/012000/pretrained_model",
)
TOKENIZER_DIRECTORY = "action_tokenizer"


def artifact_fingerprint(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative_path, content in sorted(files):
        encoded_path = relative_path.encode()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def tokenizer_files(tokenizer_path: Path) -> list[tuple[str, Path]]:
    return [
        (path.relative_to(tokenizer_path).as_posix(), path)
        for path in sorted(tokenizer_path.rglob("*"))
        if path.is_file()
    ]


def _repo_path(directory: str, filename: str) -> str:
    return (PurePosixPath(directory) / filename).as_posix() if directory else filename


def _download_json(repo_id: str, path_in_repo: str, revision: str | None = None) -> dict[str, Any]:
    path = hf_hub_download(repo_id, path_in_repo, repo_type="model", revision=revision)
    return json.loads(Path(path).read_text())


def make_portable_preprocessor(config: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    action_steps = [
        step for step in config["steps"] if step.get("registry_name") == "action_tokenizer_processor"
    ]
    if len(action_steps) != 1:
        raise ValueError(f"Expected one action tokenizer step, found {len(action_steps)}")
    action_step = action_steps[0]
    action_step["config"]["action_tokenizer_name"] = TOKENIZER_DIRECTORY
    action_step["artifacts"] = {"action_tokenizer_name": TOKENIZER_DIRECTORY}

    recipe_steps = [
        step for step in config["steps"] if step.get("registry_name") == "render_messages_processor"
    ]
    if len(recipe_steps) != 1 or not recipe_steps[0].get("config", {}).get("recipe"):
        raise ValueError("PI052 preprocessor does not contain an embedded training recipe")
    return config


def _json_operation(path_in_repo: str, content: dict[str, Any]) -> CommitOperationAdd:
    serialized = (json.dumps(content, indent=2) + "\n").encode()
    return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=io.BytesIO(serialized))


def prepare_operations(
    repo_id: str,
    tokenizer_path: Path,
    revision: str | None = None,
) -> list[CommitOperationAdd]:
    operations: list[CommitOperationAdd] = []
    files = tokenizer_files(tokenizer_path)
    for directory in CHECKPOINT_DIRECTORIES:
        preprocessor_path = _repo_path(directory, "policy_preprocessor.json")
        operations.append(
            _json_operation(
                preprocessor_path,
                make_portable_preprocessor(_download_json(repo_id, preprocessor_path, revision)),
            )
        )
        for relative_path, local_path in files:
            operations.append(
                CommitOperationAdd(
                    path_in_repo=_repo_path(
                        directory,
                        f"{TOKENIZER_DIRECTORY}/{relative_path}",
                    ),
                    path_or_fileobj=str(local_path),
                )
            )
    return operations


def audit_repository(
    api: HfApi,
    repo_id: str,
    expected_tokenizer_fingerprint: str,
    revision: str | None = None,
) -> None:
    info = api.model_info(repo_id, revision=revision)
    repository_files = {sibling.rfilename for sibling in info.siblings or []}

    for directory in CHECKPOINT_DIRECTORIES:
        preprocessor_path = _repo_path(directory, "policy_preprocessor.json")
        policy_config_path = _repo_path(directory, "config.json")
        postprocessor_path = _repo_path(directory, "policy_postprocessor.json")
        for required_path in (preprocessor_path, policy_config_path, postprocessor_path):
            if required_path not in repository_files:
                raise FileNotFoundError(f"{repo_id}@{revision or 'main'} is missing {required_path}")

        preprocessor = _download_json(repo_id, preprocessor_path, revision)
        portable_preprocessor = make_portable_preprocessor(preprocessor)
        if preprocessor != portable_preprocessor:
            raise ValueError(f"{repo_id}:{preprocessor_path} is not portable")

        normalizer_steps = [
            step for step in preprocessor["steps"] if step.get("registry_name") == "normalizer_processor"
        ]
        if len(normalizer_steps) != 1 or "state_file" not in normalizer_steps[0]:
            raise ValueError(f"{repo_id}:{preprocessor_path} is missing normalizer state metadata")
        normalizer_path = _repo_path(directory, normalizer_steps[0]["state_file"])
        if normalizer_path not in repository_files:
            raise FileNotFoundError(f"{repo_id} is missing {normalizer_path}")

        remote_tokenizer_files: list[tuple[str, bytes]] = []
        for relative_path in _tokenizer_relative_paths(repository_files, directory):
            path_in_repo = _repo_path(directory, f"{TOKENIZER_DIRECTORY}/{relative_path}")
            downloaded = hf_hub_download(repo_id, path_in_repo, repo_type="model", revision=revision)
            remote_tokenizer_files.append((relative_path, Path(downloaded).read_bytes()))
        fingerprint = artifact_fingerprint(remote_tokenizer_files)
        if fingerprint != expected_tokenizer_fingerprint:
            raise ValueError(
                f"{repo_id}:{_repo_path(directory, TOKENIZER_DIRECTORY)} fingerprint "
                f"{fingerprint} != {expected_tokenizer_fingerprint}"
            )


def _tokenizer_relative_paths(repository_files: set[str], directory: str) -> list[str]:
    prefix = _repo_path(directory, TOKENIZER_DIRECTORY).rstrip("/") + "/"
    paths = sorted(path.removeprefix(prefix) for path in repository_files if path.startswith(prefix))
    if not paths:
        raise FileNotFoundError(f"Missing tokenizer artifact directory {prefix.rstrip('/')}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--repo-id", action="append", dest="repo_ids")
    parser.add_argument("--revision")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = args.tokenizer_path.resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_path}")

    files = tokenizer_files(tokenizer_path)
    fingerprint = artifact_fingerprint([(relative_path, path.read_bytes()) for relative_path, path in files])
    api = HfApi()
    repositories = tuple(args.repo_ids or DEFAULT_REPOSITORIES)
    print(f"Tokenizer fingerprint: {fingerprint}")

    for repo_id in repositories:
        if args.audit_only:
            audit_repository(api, repo_id, fingerprint, args.revision)
            print(f"AUDIT OK {repo_id}@{args.revision or 'main'}")
            continue

        operations = prepare_operations(repo_id, tokenizer_path, args.revision)
        if args.dry_run:
            print(f"DRY RUN {repo_id}: {len(operations)} files")
            for operation in operations:
                print(f"  {operation.path_in_repo}")
            continue

        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message="Embed fitted FAST tokenizer for portable PI052 checkpoints",
            revision=args.revision,
        )
        audit_repository(api, repo_id, fingerprint, commit.oid)
        print(f"BACKFILLED {repo_id}@{commit.oid}")


if __name__ == "__main__":
    main()
