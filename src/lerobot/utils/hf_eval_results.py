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

"""Helpers for building and uploading HF-native `.eval_results` YAML rows.

The `.eval_results` format is consumed by the HuggingFace leaderboard surface.
See https://huggingface.co/docs/evaluate/en/evaluation-on-hub for the spec.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import yaml


def default_eval_date() -> str:
    """Return today's UTC date as an ISO-8601 string (YYYY-MM-DD)."""
    return date.today().isoformat()


def build_eval_results_rows(
    *,
    info: dict,
    env_type: str,
    env_task: str | None,
    benchmark_dataset_id: str,
    eval_date: str,
    source_url: str | None = None,
    notes: str | None = None,
) -> list[dict[str, Any]]:
    """Build a list of `.eval_results` rows from an eval info dict.

    Each row represents one (task_group, metric) combination.  When no
    per-group breakdown is available a single overall row is emitted.

    Args:
        info: The dict returned by ``_aggregate_eval_from_per_task`` / ``eval_policy_all``.
            Expected keys: ``overall``, optionally ``per_group``.
        env_type: Environment type string (e.g. ``"libero_plus"``).
        env_task: The env task string from eval config (may be ``None``).
        benchmark_dataset_id: HF dataset repo-id of the consolidated benchmark dataset.
        eval_date: ISO-8601 date string (use ``default_eval_date()``).
        source_url: Optional URL to the evaluation run / report.
        notes: Optional free-text notes about the evaluation setup.

    Returns:
        A list of row dicts ready for serialisation with ``upload_eval_results_yaml``.
    """
    rows: list[dict[str, Any]] = []
    task_name = env_task or env_type

    def _safe(v: float) -> float:
        return 0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

    def _make_row(config_name: str, pc_success: float, n_episodes: int) -> dict[str, Any]:
        row: dict[str, Any] = {
            "task": {
                "type": "robotics",
                "name": task_name,
            },
            "dataset": {
                "name": benchmark_dataset_id,
                "type": benchmark_dataset_id,
                "config": config_name,
                "split": "test",
            },
            "metrics": [
                {
                    "type": "success_rate",
                    "value": _safe(pc_success),
                    "name": "Success Rate (%)",
                },
                {
                    "type": "n_episodes",
                    "value": n_episodes,
                    "name": "Number of Episodes",
                },
            ],
            "evaluated_at": eval_date,
        }
        if source_url:
            row["source_url"] = source_url
        if notes:
            row["notes"] = notes
        return row

    per_group: dict = info.get("per_group", {})
    if per_group:
        for group_name, group_metrics in per_group.items():
            rows.append(
                _make_row(
                    config_name=group_name,
                    pc_success=group_metrics.get("pc_success", float("nan")),
                    n_episodes=group_metrics.get("n_episodes", 0),
                )
            )
    else:
        overall = info.get("overall", {})
        rows.append(
            _make_row(
                config_name=env_type,
                pc_success=overall.get("pc_success", float("nan")),
                n_episodes=overall.get("n_episodes", 0),
            )
        )

    return rows


def upload_eval_results_yaml(
    *,
    api: Any,
    repo_id: str,
    rows: list[dict[str, Any]],
    env_type: str,
    env_task: str | None,
    output_dir: Path,
) -> str:
    """Serialise ``rows`` to YAML and upload to the Hub model repo.

    The file is written locally to ``output_dir/eval_results.yaml`` and
    then uploaded to ``eval/{env_type}/eval_results.yaml`` in ``repo_id``.

    Args:
        api: An instantiated ``huggingface_hub.HfApi`` object.
        repo_id: HF model repo (e.g. ``"user/my_policy"``).
        rows: Rows produced by ``build_eval_results_rows``.
        env_type: Environment type string (used for the Hub path prefix).
        env_task: The env task string (unused, kept for API symmetry).
        output_dir: Local directory to write the YAML before uploading.

    Returns:
        URL of the Hub commit containing the uploaded file.
    """
    yaml_path = Path(output_dir) / "eval_results.yaml"
    yaml_path.write_text(yaml.dump({"eval_results": rows}, sort_keys=False, allow_unicode=True))

    commit_url = api.upload_file(
        path_or_fileobj=str(yaml_path),
        path_in_repo=f"eval/{env_type}/eval_results.yaml",
        repo_id=repo_id,
        commit_message=f"Upload eval results YAML for {env_type}",
    )
    return str(commit_url)
