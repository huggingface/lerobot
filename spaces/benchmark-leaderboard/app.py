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

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px
from huggingface_hub import HfApi, hf_hub_download

RESULTS_REPO = os.environ.get("BENCHMARK_RESULTS_REPO", "lerobot/benchmark-history")
CACHE_DIR = Path("/tmp/benchmark-leaderboard-cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_S = 300

_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}


def _row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    overall = row.get("eval", {}).get("overall", {})
    resources = row.get("resources", {})
    timings = row.get("timings", {})
    artifact_urls = row.get("artifact_urls", {})
    return {
        "created_at": row.get("created_at"),
        "benchmark": row.get("benchmark"),
        "policy": row.get("policy"),
        "success_rate": overall.get("pc_success"),
        "n_episodes": overall.get("n_episodes"),
        "avg_sum_reward": overall.get("avg_sum_reward"),
        "train_wall_time_s": timings.get("train_wall_time_s"),
        "eval_wall_time_s": timings.get("eval_wall_time_s"),
        "total_wall_time_s": timings.get("total_wall_time_s"),
        "num_gpus": resources.get("num_gpus"),
        "microbatch_per_gpu": resources.get("microbatch_per_gpu"),
        "gradient_accumulation_steps": resources.get("gradient_accumulation_steps"),
        "effective_batch_size": resources.get("effective_batch_size"),
        "git_commit": row.get("git_commit"),
        "row_url": artifact_urls.get("row"),
        "eval_info_url": artifact_urls.get("eval_info"),
        "train_config_url": artifact_urls.get("train_config"),
    }


def load_rows(repo_id: str = RESULTS_REPO) -> pd.DataFrame:
    cache_key = f"rows::{repo_id}"
    cached = _CACHE.get(cache_key)
    if cached is not None and (time.monotonic() - cached[0]) < CACHE_TTL_S:
        return cached[1]

    api = HfApi()
    files = [path for path in api.list_repo_files(repo_id=repo_id, repo_type="dataset") if path.startswith("rows/")]
    records: list[dict[str, Any]] = []
    for path_in_repo in sorted(files, reverse=True):
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=path_in_repo, cache_dir=CACHE_DIR)
        with open(local_path) as f:
            row = json.load(f)
        records.append(_row_to_record(row))

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
    _CACHE[cache_key] = (time.monotonic(), df)
    return df


def make_latest_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    latest = (
        df.sort_values("created_at", ascending=False)
        .groupby(["benchmark", "policy"], as_index=False)
        .first()
        .sort_values(["benchmark", "success_rate"], ascending=[True, False], na_position="last")
    )
    return latest[
        [
            "benchmark",
            "policy",
            "success_rate",
            "n_episodes",
            "train_wall_time_s",
            "eval_wall_time_s",
            "num_gpus",
            "effective_batch_size",
            "git_commit",
            "row_url",
            "eval_info_url",
            "train_config_url",
        ]
    ]


def make_history_figure(df: pd.DataFrame, benchmark: str, policy: str | None) -> Any:
    filtered = df[df["benchmark"] == benchmark]
    if policy and policy != "All":
        filtered = filtered[filtered["policy"] == policy]
    if filtered.empty:
        return px.line(title="No benchmark rows found")
    fig = px.line(
        filtered.sort_values("created_at"),
        x="created_at",
        y="success_rate",
        color="policy",
        markers=True,
        hover_data=["git_commit", "num_gpus", "train_wall_time_s", "eval_wall_time_s"],
        title=f"{benchmark} success rate history",
    )
    fig.update_layout(yaxis_title="Success rate (%)", xaxis_title="Run time")
    return fig


def make_run_markdown(df: pd.DataFrame, benchmark: str, policy: str | None) -> str:
    filtered = df[df["benchmark"] == benchmark]
    if policy and policy != "All":
        filtered = filtered[filtered["policy"] == policy]
    if filtered.empty:
        return "No matching runs yet."
    latest = filtered.sort_values("created_at", ascending=False).iloc[0]
    row_link = latest["row_url"] if pd.notna(latest["row_url"]) else None
    eval_link = latest["eval_info_url"] if pd.notna(latest["eval_info_url"]) else None
    train_link = latest["train_config_url"] if pd.notna(latest["train_config_url"]) else None
    lines = [
        f"Latest run: `{latest['policy']}` on `{latest['benchmark']}`",
        f"Success rate: `{latest['success_rate']}`",
        f"GPUs: `{latest['num_gpus']}`",
        f"Effective batch size: `{latest['effective_batch_size']}`",
        f"Commit: `{latest['git_commit']}`",
    ]
    if row_link:
        lines.append(f"Row JSON: [open]({row_link})")
    if eval_link:
        lines.append(f"Eval Info: [open]({eval_link})")
    if train_link:
        lines.append(f"Train Config: [open]({train_link})")
    return "\n\n".join(lines)


def refresh_view(benchmark: str, policy: str) -> tuple[pd.DataFrame, dict[str, Any], Any, str]:
    df = load_rows()
    latest_table = make_latest_table(df)
    benchmark_names = sorted(df["benchmark"].dropna().unique().tolist()) if not df.empty else []
    if benchmark not in benchmark_names and benchmark_names:
        benchmark = benchmark_names[0]
    policy_choices = ["All"]
    if benchmark and not df.empty:
        policy_choices.extend(sorted(df[df["benchmark"] == benchmark]["policy"].dropna().unique().tolist()))
    if policy not in policy_choices:
        policy = "All"
    history = make_history_figure(df, benchmark, policy)
    summary = make_run_markdown(df, benchmark, policy)
    return latest_table, gr.update(choices=policy_choices, value=policy), history, summary


with gr.Blocks(title="LeRobot Benchmark Leaderboard") as demo:
    gr.Markdown(
        f"""
# LeRobot Benchmark Leaderboard

Results dataset: [`{RESULTS_REPO}`](https://huggingface.co/datasets/{RESULTS_REPO})
"""
    )

    with gr.Row():
        benchmark_dropdown = gr.Dropdown(label="Benchmark", choices=[])
        policy_dropdown = gr.Dropdown(label="Policy", choices=["All"], value="All")
        refresh_button = gr.Button("Refresh")

    latest_table = gr.Dataframe(label="Latest Results", interactive=False)
    history_plot = gr.Plot(label="History")
    latest_summary = gr.Markdown()

    def _initial_state():
        df = load_rows()
        benchmarks = sorted(df["benchmark"].dropna().unique().tolist()) if not df.empty else []
        benchmark = benchmarks[0] if benchmarks else ""
        latest, policy_choices, history, summary = refresh_view(benchmark, "All")
        return (
            gr.update(choices=benchmarks, value=benchmark),
            policy_choices,
            latest,
            history,
            summary,
        )

    demo.load(
        _initial_state,
        outputs=[benchmark_dropdown, policy_dropdown, latest_table, history_plot, latest_summary],
    )
    refresh_button.click(
        refresh_view,
        inputs=[benchmark_dropdown, policy_dropdown],
        outputs=[latest_table, policy_dropdown, history_plot, latest_summary],
    )
    benchmark_dropdown.change(
        refresh_view,
        inputs=[benchmark_dropdown, policy_dropdown],
        outputs=[latest_table, policy_dropdown, history_plot, latest_summary],
    )
    policy_dropdown.change(
        refresh_view,
        inputs=[benchmark_dropdown, policy_dropdown],
        outputs=[latest_table, policy_dropdown, history_plot, latest_summary],
    )


if __name__ == "__main__":
    demo.launch()
