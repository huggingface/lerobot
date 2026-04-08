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

"""LeRobot CI Health Dashboard.

Pulls live data from the GitHub Actions API — no separate data store needed.
Benchmark smoke-test results (success rate, duration) come from a small
metrics.json artifact that each benchmark CI job uploads.

Required Space secret: GITHUB_RO_TOKEN
  Fine-grained token for huggingface/lerobot with Actions=read, Metadata=read.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr
import plotly.graph_objects as go
import requests  # type: ignore[import-untyped]

# ── Config ────────────────────────────────────────────────────────────────────

REPO = "huggingface/lerobot"
GH_TOKEN = os.environ.get("GITHUB_RO_TOKEN", "")

CACHE_DIR = Path("/tmp/dashboard-cache")  # nosec B108 — only writable path in HF Spaces
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_CACHE_TTL = 300  # 5 min — avoids hammering GitHub on every page load

# Maps CI job name fragment → display info.
# "artifact" is the actions/upload-artifact name for the rollout video.
# "metrics_artifact" is the artifact name for metrics.json.
BENCHMARKS: dict[str, dict[str, str]] = {
    "libero-integration-test": {
        "label": "LIBERO",
        "video_artifact": "libero-rollout-video",
        "metrics_artifact": "libero-metrics",
    },
    "metaworld-integration-test": {
        "label": "MetaWorld",
        "video_artifact": "metaworld-rollout-video",
        "metrics_artifact": "metaworld-metrics",
    },
}

WORKFLOW_LABELS: dict[str, str] = {
    "Benchmark Integration Tests": "Benchmarks",
    "Fast Tests": "Fast Tests",
    "Full Tests": "Full Tests",
    "Quality": "Quality",
    "Security": "Security",
}

# ── GitHub API helpers ────────────────────────────────────────────────────────

_api_cache: dict[str, tuple[Any, float]] = {}
_api_lock = threading.Lock()


def _gh_get(path: str, **kwargs: Any) -> Any:
    """Authenticated GitHub API GET with in-memory TTL cache."""
    key = path + str(kwargs)
    with _api_lock:
        if key in _api_cache:
            val, ts = _api_cache[key]
            if time.monotonic() - ts < API_CACHE_TTL:
                return val

    headers: dict[str, str] = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GH_TOKEN:
        headers["Authorization"] = f"Bearer {GH_TOKEN}"

    url = f"https://api.github.com{path}"
    resp = requests.get(url, headers=headers, timeout=20, **kwargs)
    resp.raise_for_status()
    data = resp.json()

    with _api_lock:
        _api_cache[key] = (data, time.monotonic())
    return data


def _gh_download(url: str) -> bytes:
    """Download a URL with auth (follows redirects, e.g. artifact zip → S3)."""
    headers: dict[str, str] = {}
    if GH_TOKEN:
        headers["Authorization"] = f"Bearer {GH_TOKEN}"
    resp = requests.get(url, headers=headers, allow_redirects=True, timeout=120)
    resp.raise_for_status()
    return resp.content


# ── Data fetchers ─────────────────────────────────────────────────────────────


def fetch_recent_runs(branch: str, n: int = 40) -> list[dict]:
    data = _gh_get(f"/repos/{REPO}/actions/runs", params={"branch": branch, "per_page": n})
    return data.get("workflow_runs", [])


def fetch_jobs(run_id: int) -> list[dict]:
    # Jobs are immutable once a run completes — cache forever (use long TTL via completed_at check).
    data = _gh_get(f"/repos/{REPO}/actions/runs/{run_id}/jobs", params={"per_page": 100})
    return data.get("jobs", [])


def fetch_artifacts(run_id: int) -> list[dict]:
    data = _gh_get(f"/repos/{REPO}/actions/runs/{run_id}/artifacts", params={"per_page": 100})
    return data.get("artifacts", [])


def download_metrics_json(artifact_id: int) -> dict | None:
    """Download and parse metrics.json from a zip artifact. Caches to disk."""
    cache_path = CACHE_DIR / f"metrics_{artifact_id}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache_path.unlink(missing_ok=True)

    try:
        raw = _gh_download(f"https://api.github.com/repos/{REPO}/actions/artifacts/{artifact_id}/zip")
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            if "metrics.json" in zf.namelist():
                data = json.loads(zf.read("metrics.json"))
                cache_path.write_text(json.dumps(data))
                return data
    except Exception as exc:
        print(f"[dashboard] Could not fetch metrics artifact {artifact_id}: {exc}")
    return None


def download_video(artifact_id: int, label: str) -> Path | None:
    """Download the first .mp4 from a zip artifact. Caches to disk."""
    cache_path = CACHE_DIR / f"video_{artifact_id}.mp4"
    if cache_path.exists():
        return cache_path

    try:
        raw = _gh_download(f"https://api.github.com/repos/{REPO}/actions/artifacts/{artifact_id}/zip")
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            mp4s = [n for n in zf.namelist() if n.endswith(".mp4")]
            if mp4s:
                cache_path.write_bytes(zf.read(mp4s[0]))
                return cache_path
    except Exception as exc:
        print(f"[dashboard] Could not fetch video artifact {artifact_id} ({label}): {exc}")
    return None


# ── Data aggregation ──────────────────────────────────────────────────────────


def _job_duration_minutes(job: dict) -> float | None:
    started = job.get("started_at")
    completed = job.get("completed_at")
    if not started or not completed:
        return None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        delta = datetime.strptime(completed, fmt) - datetime.strptime(started, fmt)
        return delta.total_seconds() / 60
    except ValueError:
        return None


def aggregate(branch: str) -> dict:
    """Pull GitHub data and reshape into what the UI needs."""
    runs = fetch_recent_runs(branch, n=40)

    # Per-benchmark history (ordered newest-first from the API)
    bench_history: dict[str, list[dict]] = {k: [] for k in BENCHMARKS}

    # Per-workflow latest status + last few runs for the summary table
    workflow_latest: dict[str, dict] = {}

    for run in runs:
        wf_name = run["name"]
        conclusion = run["conclusion"]  # "success" | "failure" | "cancelled" | None
        created_at = run["created_at"]
        run_id = run["id"]
        run_url = run["html_url"]

        # Track latest status per workflow
        if wf_name not in workflow_latest:
            workflow_latest[wf_name] = {
                "conclusion": conclusion,
                "created_at": created_at,
                "run_url": run_url,
            }

        if wf_name != "Benchmark Integration Tests":
            continue

        # Drill into jobs for this benchmark run
        jobs = fetch_jobs(run_id)
        artifacts = fetch_artifacts(run_id)
        art_by_name = {a["name"]: a for a in artifacts if not a.get("expired")}

        for job in jobs:
            job_name = job["name"]  # e.g. "Libero — build image + 1-episode eval"
            matched_key = next(
                (k for k in BENCHMARKS if k in job_name.lower().replace(" ", "-")),
                None,
            )
            if matched_key is None:
                continue

            info = BENCHMARKS[matched_key]
            metrics: dict | None = None
            if info["metrics_artifact"] in art_by_name:
                metrics = download_metrics_json(art_by_name[info["metrics_artifact"]]["id"])

            bench_history[matched_key].append(
                {
                    "run_id": run_id,
                    "run_url": run_url,
                    "created_at": created_at,
                    "conclusion": job["conclusion"],
                    "duration_min": _job_duration_minutes(job),
                    "pc_success": metrics.get("pc_success") if metrics else None,
                    "n_episodes": metrics.get("n_episodes") if metrics else None,
                    "video_artifact_id": art_by_name.get(info["video_artifact"], {}).get("id"),
                }
            )

    return {
        "bench_history": bench_history,
        "workflow_latest": workflow_latest,
        "fetched_at": datetime.now(UTC).isoformat(),
    }


# ── UI helpers ────────────────────────────────────────────────────────────────

_STATUS_STYLE = {
    "success": ("✓ passing", "#16a34a"),
    "failure": ("✗ failing", "#dc2626"),
    "cancelled": ("⚠ cancelled", "#d97706"),
    None: ("◌ pending", "#6b7280"),
}


def _badge(conclusion: str | None) -> str:
    label, color = _STATUS_STYLE.get(conclusion, ("? unknown", "#6b7280"))
    return (
        f'<span style="background:{color};color:#fff;padding:1px 9px;border-radius:12px;'
        f'font-size:12px;font-weight:600;font-family:monospace">{label}</span>'
    )


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    return iso[:10]


def render_status_table(data: dict) -> str:
    bench_history = data["bench_history"]
    workflow_latest = data["workflow_latest"]

    rows = []

    # ── Benchmark rows ──────────────────────────────────────────────
    for key, info in BENCHMARKS.items():
        history = bench_history.get(key, [])
        if history:
            latest = history[0]
            badge = _badge(latest["conclusion"])
            date = _fmt_date(latest["created_at"])
            pc = latest.get("pc_success")
            sr_str = f"{pc:.1f}%" if pc is not None else "—"
            n_ep = latest.get("n_episodes") or "—"
            link = f'<a href="{latest["run_url"]}" target="_blank">#{latest["run_id"]}</a>'
        else:
            badge = _badge(None)
            date = sr_str = n_ep = link = "—"

        rows.append(
            f"<tr>"
            f"<td><b>{info['label']}</b></td>"
            f"<td>{badge}</td>"
            f"<td>{date}</td>"
            f"<td>{sr_str}</td>"
            f"<td>{n_ep}</td>"
            f"<td>{link}</td>"
            f"</tr>"
        )

    # ── Other workflow rows ─────────────────────────────────────────
    for wf_name, label in WORKFLOW_LABELS.items():
        if wf_name == "Benchmark Integration Tests":
            continue  # already shown above
        latest_run = workflow_latest.get(wf_name)
        if latest_run:
            badge = _badge(latest_run["conclusion"])
            date = _fmt_date(latest_run["created_at"])
            link = f'<a href="{latest_run["run_url"]}" target="_blank">run</a>'
        else:
            badge = _badge(None)
            date = link = "—"

        rows.append(
            f"<tr>"
            f"<td><b>{label}</b></td>"
            f"<td>{badge}</td>"
            f"<td>{date}</td>"
            f"<td>—</td><td>—</td>"
            f"<td>{link}</td>"
            f"</tr>"
        )

    header = (
        "<tr style='border-bottom:1px solid #e5e7eb'>"
        "<th align='left' style='padding:6px 12px'>Job</th>"
        "<th align='left' style='padding:6px 12px'>Status</th>"
        "<th align='left' style='padding:6px 12px'>Last run</th>"
        "<th align='left' style='padding:6px 12px'>Success rate</th>"
        "<th align='left' style='padding:6px 12px'>Episodes</th>"
        "<th align='left' style='padding:6px 12px'>Link</th>"
        "</tr>"
    )
    table_rows = "\n".join(rows)
    return (
        "<table style='width:100%;border-collapse:collapse;font-family:sans-serif;font-size:14px'>"
        f"{header}{table_rows}"
        "</table>"
    )


def render_success_rate_chart(data: dict) -> go.Figure:
    fig = go.Figure()
    for key, info in BENCHMARKS.items():
        history = [e for e in data["bench_history"].get(key, []) if e.get("pc_success") is not None]
        if history:
            fig.add_trace(
                go.Scatter(
                    x=[e["created_at"][:10] for e in history],
                    y=[e["pc_success"] for e in history],
                    mode="lines+markers",
                    name=info["label"],
                    line={"width": 2},
                    marker={"size": 6},
                )
            )
    fig.update_layout(
        title="Benchmark Success Rate (%) over time",
        yaxis={"title": "Success rate (%)", "range": [0, 105]},
        xaxis={"title": ""},
        height=320,
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def render_duration_chart(data: dict) -> go.Figure:
    fig = go.Figure()
    for key, info in BENCHMARKS.items():
        history = [e for e in data["bench_history"].get(key, []) if e.get("duration_min") is not None]
        if history:
            fig.add_trace(
                go.Bar(
                    x=[e["created_at"][:10] for e in history],
                    y=[round(e["duration_min"], 1) for e in history],
                    name=info["label"],
                    opacity=0.85,
                )
            )
    fig.update_layout(
        title="Benchmark CI Duration (minutes)",
        yaxis={"title": "Duration (min)"},
        xaxis={"title": ""},
        barmode="group",
        height=320,
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def fetch_latest_videos(data: dict) -> dict[str, str | None]:
    """Return {bench_key: local_mp4_path_or_None} for the latest successful run of each benchmark."""
    results: dict[str, str | None] = {}
    for key, info in BENCHMARKS.items():
        history = data["bench_history"].get(key, [])
        path = None
        for entry in history:
            art_id = entry.get("video_artifact_id")
            if art_id:
                downloaded = download_video(art_id, info["label"])
                if downloaded:
                    path = str(downloaded)
                    break
        results[key] = path
    return results


# ── Gradio app ────────────────────────────────────────────────────────────────


def refresh(branch: str) -> tuple:
    if not GH_TOKEN:
        err = "<p style='color:red'><b>GITHUB_RO_TOKEN secret not set.</b> Add it in Space settings.</p>"
        return err, go.Figure(), go.Figure(), None, None, "Error: no token"

    try:
        data = aggregate(branch)
    except requests.HTTPError as exc:
        err = f"<p style='color:red'>GitHub API error: {exc}</p>"
        return err, go.Figure(), go.Figure(), None, None, str(exc)

    status_html = render_status_table(data)
    sr_chart = render_success_rate_chart(data)
    dur_chart = render_duration_chart(data)
    videos = fetch_latest_videos(data)

    updated = datetime.now(UTC).strftime("Last updated: %Y-%m-%d %H:%M UTC")

    bench_keys = list(BENCHMARKS.keys())
    video_0 = videos.get(bench_keys[0]) if len(bench_keys) > 0 else None
    video_1 = videos.get(bench_keys[1]) if len(bench_keys) > 1 else None

    return status_html, sr_chart, dur_chart, video_0, video_1, updated


with gr.Blocks(title="LeRobot Health Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🤖 LeRobot — CI Health Dashboard\n"
        "Live view of benchmark smoke tests, CI job health, and latest rollout videos. "
        "Data pulled from the GitHub Actions API."
    )

    with gr.Row():
        branch_dd = gr.Dropdown(
            choices=["main", "feat/benchmark-ci"],
            value="main",
            label="Branch",
            scale=1,
        )
        refresh_btn = gr.Button("Refresh", variant="primary", scale=0)
        updated_md = gr.Markdown("Click Refresh or wait for auto-load.", scale=3)

    gr.Markdown("## Status")
    status_html = gr.HTML()

    with gr.Row():
        sr_plot = gr.Plot(label="Success Rate Trend")
        dur_plot = gr.Plot(label="Duration Trend")

    gr.Markdown("## Latest Rollout Videos")
    bench_labels = [v["label"] for v in BENCHMARKS.values()]
    with gr.Row():
        video_0 = gr.Video(
            label=bench_labels[0] if len(bench_labels) > 0 else "Benchmark 0", interactive=False
        )
        video_1 = gr.Video(
            label=bench_labels[1] if len(bench_labels) > 1 else "Benchmark 1", interactive=False
        )

    outputs = [status_html, sr_plot, dur_plot, video_0, video_1, updated_md]

    refresh_btn.click(fn=refresh, inputs=[branch_dd], outputs=outputs)
    demo.load(fn=refresh, inputs=[branch_dd], outputs=outputs)

if __name__ == "__main__":
    demo.launch()
