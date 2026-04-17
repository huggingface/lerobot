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

"""Publish benchmark rows and lightweight artifacts to a Hub dataset."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lerobot.utils.history_repo import UploadTarget, make_hub_file_url, upload_targets, utc_timestamp_slug


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def find_latest_train_config_path(run_root: Path) -> Path | None:
    checkpoints_dir = run_root / "train" / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    candidates = sorted(
        checkpoints_dir.glob("*/pretrained_model/train_config.json"),
        key=lambda path: path.parts[-3],
    )
    return candidates[-1] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--results_repo", required=True)
    parser.add_argument("--git_commit", required=True)
    parser.add_argument("--num_gpus", required=True, type=int)
    parser.add_argument("--microbatch_per_gpu", required=True, type=int)
    parser.add_argument("--gradient_accumulation_steps", required=True, type=int)
    parser.add_argument("--effective_batch_size", required=True, type=int)
    parser.add_argument("--train_wall_time_s", required=True, type=float)
    parser.add_argument("--eval_wall_time_s", required=True, type=float)
    parser.add_argument("--slurm_job_id", default="")
    parser.add_argument("--docker_image", required=True)
    return parser.parse_args()


def build_row(args: argparse.Namespace) -> tuple[dict[str, Any], list[UploadTarget]]:
    now = datetime.now(UTC)
    created_at = now.isoformat()
    timestamp = utc_timestamp_slug(now)
    run_id = f"{timestamp}__{args.benchmark}__{args.policy}__{args.slurm_job_id or 'manual'}"
    eval_info = load_json_if_exists(args.run_root / "eval" / "eval_info.json") or {}
    train_config_path = find_latest_train_config_path(args.run_root)
    train_config = load_json_if_exists(train_config_path) or {}

    artifact_prefix = f"artifacts/{args.benchmark}/{args.policy}/{run_id}"
    row_path_in_repo = f"rows/{args.benchmark}/{args.policy}/{run_id}.json"

    row = {
        "schema_version": 1,
        "created_at": created_at,
        "run_id": run_id,
        "benchmark": args.benchmark,
        "policy": args.policy,
        "git_commit": args.git_commit,
        "slurm_job_id": args.slurm_job_id or None,
        "docker_image": args.docker_image,
        "resources": {
            "num_gpus": args.num_gpus,
            "microbatch_per_gpu": args.microbatch_per_gpu,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.effective_batch_size,
        },
        "timings": {
            "train_wall_time_s": args.train_wall_time_s,
            "eval_wall_time_s": args.eval_wall_time_s,
            "total_wall_time_s": args.train_wall_time_s + args.eval_wall_time_s,
        },
        "eval": {
            "overall": eval_info.get("overall", {}),
            "per_group": eval_info.get("per_group", {}),
            "per_task_count": len(eval_info.get("per_task", [])),
        },
        "paths": {
            "run_root": str(args.run_root),
            "train_dir": str(args.run_root / "train"),
            "eval_dir": str(args.run_root / "eval"),
        },
        "train_config": train_config,
        "artifact_urls": {
            "row": make_hub_file_url(args.results_repo, row_path_in_repo),
        },
    }

    row_path = args.run_root / "benchmark_row.json"
    row_path.parent.mkdir(parents=True, exist_ok=True)
    upload_list = [UploadTarget(local_path=row_path, path_in_repo=row_path_in_repo)]

    eval_info_path = args.run_root / "eval" / "eval_info.json"
    if eval_info_path.exists():
        row["artifact_urls"]["eval_info"] = make_hub_file_url(
            args.results_repo, f"{artifact_prefix}/eval_info.json"
        )
        upload_list.append(
            UploadTarget(local_path=eval_info_path, path_in_repo=f"{artifact_prefix}/eval_info.json")
        )

    if train_config_path is not None and train_config_path.exists():
        row["artifact_urls"]["train_config"] = make_hub_file_url(
            args.results_repo, f"{artifact_prefix}/train_config.json"
        )
        upload_list.append(
            UploadTarget(local_path=train_config_path, path_in_repo=f"{artifact_prefix}/train_config.json")
        )

    row_path.write_text(json.dumps(row, indent=2, sort_keys=True))
    return row, upload_list


def main() -> int:
    args = parse_args()
    row, upload_list = build_row(args)
    uploaded = upload_targets(
        repo_id=args.results_repo,
        targets=upload_list,
        repo_type="dataset",
        private=False,
        commit_message=f"Add benchmark row {row['run_id']}",
    )
    row["uploaded_paths"] = uploaded
    row_path = args.run_root / "benchmark_row.json"
    row_path.write_text(json.dumps(row, indent=2, sort_keys=True))
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
