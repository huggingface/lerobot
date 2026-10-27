#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Dependency-free policy discovery, benchmark launching, and result summaries."""

import argparse
import ast
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
SPACE_URL = "https://huggingface.co/spaces/lerobot/health-dashboard"
HISTORY_URL = "https://lerobot-health-dashboard.hf.space/api/history"
LATEST_URL = "https://lerobot-health-dashboard.hf.space/api/runs/latest"
SETUP_FIELDS = (
    "device_name",
    "device_total_memory_bytes",
    "batch_size",
    "parameter_dtype",
    "mixed_precision",
    "warmup",
    "iterations",
    "dataset",
    "dataset_revision",
    "dataset_episodes",
    "checkpoint",
    "checkpoint_revision",
    "config_fingerprint",
    "gradient_checkpointing",
)


def policy_matrix(root: Path = ROOT) -> list[dict[str, str]]:
    # Read the registrations without importing optional policy dependencies.
    names = set()
    for path in (root / "src/lerobot/policies").glob("*/configuration_*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id == "PreTrainedConfig"
                    and decorator.func.attr == "register_subclass"
                ):
                    names.add(ast.literal_eval(decorator.args[0]))
    aliases = {"pi0": "pi", "pi0_fast": "pi", "pi05": "pi", "wall_x": "wallx"}
    base_policies = {"act", "tdmpc", "vqbet", "gaussian_actor"}
    return [
        {"policy": name, "extra": "" if name in base_policies else aliases.get(name, name)}
        for name in sorted(names)
    ]


def policy_args(policy: str, gradient_checkpointing: bool = False) -> list[str]:
    dataset = (
        "lerobot/pusht"
        if policy in {"act", "diffusion", "tdmpc", "vqbet", "gaussian_actor", "flux3"}
        else "lerobot/libero_plus"
    )
    args = [
        f"--policy.type={policy}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={dataset}",
        "--dataset.episodes=[0]",
        "--batch_size=1",
        "--num_workers=0",
        "--wandb.enable=false",
    ]
    if policy == "multi_task_dit":
        args += ["--policy.horizon=32", "--policy.n_action_steps=30"]
    if policy in {"dm05", "pi05", "molmoact2", "fineart_vla"}:
        args += ["--compute_missing_quantiles=true"]
    if policy == "tdmpc":
        args += ["--policy.horizon=3", "--policy.n_action_steps=1"]
    if policy == "pi0_fast":
        # Autoregressive action decoding needs learned action tokens, not a random language model.
        args += [
            "--policy.pretrained_path=lerobot/pi0fast-libero",
            "--policy.pretrained_revision=840f4b503f4c09110421c33c810a85b6684fd658",
            "--policy.action_tokenizer_name=jadechoghari/tokenizer-lib-mean",
            "--policy.chunk_size=10",
            "--policy.n_action_steps=10",
            "--policy.empty_cameras=1",
        ]
    if policy == "flux3":
        args += ["--policy.camera_layout=single"]
    if policy == "lawam":
        args += [
            '--policy.primary_image_features=["observation.images.front"]',
            '--policy.wrist_image_features=["observation.images.wrist"]',
        ]
    if policy == "lingbot_va":
        # Block-causal flow-matching training requires the policy's FlexAttention backend.
        args += [
            '--policy.obs_cam_keys=["observation.images.front","observation.images.wrist"]',
            "--policy.attn_mode=flex",
        ]
    if policy == "vqbet":
        # Finish the codebook stage during the default warmup, then time BeT updates.
        args += ["--policy.n_vqvae_training_steps=1"]
    if gradient_checkpointing:
        if policy in {"eo1", "pi0", "pi05", "pi0_fast", "fineart_vla", "molmoact2"}:
            args += ["--policy.gradient_checkpointing=true"]
        elif policy == "fastwam":
            args += ["--policy.use_gradient_checkpointing=true"]
        elif policy == "dm05":
            args += ["--policy.vlm_gradient_checkpointing=true", "--policy.ae_gradient_checkpointing=true"]
    return args


def run_policy(
    policy: str, output_dir: Path, overrides: list[str], gradient_checkpointing: bool = False
) -> int:
    if policy not in {entry["policy"] for entry in policy_matrix()}:
        raise ValueError(f"Unknown policy: {policy}")
    run_dir = output_dir / policy
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "result.json"
    output.write_text(json.dumps({"policy": policy, "status": "failed", "error": "Benchmark did not finish"}))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/profile_policies.py"),
            *policy_args(policy, gradient_checkpointing),
            *overrides,
            f"--output_dir={run_dir / 'run'}",
        ],
        check=False,
    )
    if (run_dir / "run/result.json").exists():
        shutil.copyfile(run_dir / "run/result.json", output)
    else:
        output.write_text(
            json.dumps(
                {
                    "policy": policy,
                    "status": "failed",
                    "error": f"Setup failed (exit {result.returncode}); see job logs",
                }
            )
        )
    return result.returncode


def mark_timeout(policy: str, output_dir: Path, mode: str) -> None:
    """Keep finished measurements and mark unfinished operations after the policy budget."""
    run_dir = output_dir / policy
    output = run_dir / "result.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    partial = run_dir / "run/result.json"
    row = json.loads(partial.read_text()) if partial.exists() else {"policy": policy}
    row.setdefault("profile_mode", mode)
    error = "Time limit: profiling stopped to keep the job under 10 minutes"
    row.update(status="failed", error=error)
    for operation in ("inference", "training"):
        if mode in ("both", operation) and operation not in row:
            row[operation] = {"status": "failed", "error": error}
    output.write_text(json.dumps(row, indent=2) + "\n")


def measured(row: dict, mode: str) -> float | None:
    result = row.get(mode, {})
    value = result.get("median_ms")
    if (
        result.get("status") == "success"
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    ):
        return value
    return None


def comparison(row: dict, mode: str, points: list[dict], started_at: datetime) -> dict:
    """Nearest matching measurement to four weeks ago, within three days either side."""
    value = measured(row, mode)
    result: dict[str, Any] = {"median_ms": value, "change_pct": None, "baseline": None}
    if value is None or not row.get("config_fingerprint") or any(key not in row for key in SETUP_FIELDS):
        return result
    target = started_at - timedelta(weeks=4)
    candidates = []
    for point in points:
        run, old = point["run"], point["result"]
        if run.get("status") != "completed" or run.get("conclusion") == "cancelled":
            continue
        if old.get("policy") != row["policy"] or measured(old, mode) is None:
            continue
        if mode == "inference" and old.get("inference_context") != row.get("inference_context"):
            continue
        if old.get("cpu_threads") != row.get("cpu_threads"):
            continue
        if any(key not in old or old[key] != row[key] for key in SETUP_FIELDS):
            continue
        date = run.get("run_started_at") or run.get("created_at")
        if not date:
            continue
        timestamp = datetime.fromisoformat(date.replace("Z", "+00:00"))
        distance = abs(timestamp - target)
        if distance <= timedelta(days=3):
            candidates.append((distance, -timestamp.timestamp(), -run.get("run_attempt", 1), point))
    if candidates:
        point = min(candidates, key=lambda candidate: candidate[:3])[3]
        run, old = point["run"], point["result"]
        previous = measured(old, mode)
        assert previous is not None  # Candidates above contain successful, finite measurements.
        result.update(
            change_pct=(value / previous - 1) * 100,
            baseline={
                "run": run.get("key", str(run["id"])),
                "started_at": run.get("run_started_at") or run["created_at"],
                "median_ms": previous,
                "url": run["html_url"],
            },
        )
    return result


def metric_label(row: dict, mode: str, metric: dict) -> str:
    if row.get("profile_mode") not in (None, "both", mode):
        return "—"
    if metric["median_ms"] is not None:
        return f"{metric['median_ms']:.2f}"
    error = (row.get(mode, {}).get("error") or row.get("error", "")).lower()
    if "time limit" in error or "timed out" in error:
        return "Time limit"
    if "out of memory" in error or "outofmemory" in error:
        return "GPU limit"
    if "gated" in error or "403" in error or "401" in error:
        return "Access"
    return "Failed" if error or row.get(mode, {}).get("status") == "failed" else "—"


def delta_label(metric: dict) -> str:
    return f"{metric['change_pct']:+.1f}%" if metric["change_pct"] is not None else "—"


def summary(
    output_dir: Path,
    policies: list[str] | None = None,
    *,
    points: list[dict] | None = None,
    started_at: datetime | None = None,
    history_note: str = "",
    latest: bool = False,
) -> str:
    rows: list[dict[str, Any]] = []
    for policy in policies or [entry["policy"] for entry in policy_matrix()]:
        path = output_dir / policy / "result.json"
        rows.append(
            json.loads(path.read_text())
            if path.exists()
            else {
                "policy": policy,
                "status": "failed",
                "error": "No result artifact; see job logs",
            }
        )
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    started_at = started_at or datetime.now(UTC)
    comparisons = [
        {
            "policy": row["policy"],
            **{
                mode: comparison(
                    row,
                    mode,
                    points or [],
                    datetime.fromisoformat(row["source_run"]["run_started_at"].replace("Z", "+00:00"))
                    if row.get("source_run", {}).get("run_started_at")
                    else started_at,
                )
                for mode in ("inference", "training")
            },
        }
        for row in rows
    ]
    note = (
        "Median latency in ms; Δ4w: − faster, + slower. — means no matching baseline. "
        "Baseline: nearest run to 28 days before each measurement (±3 days), with matching GPU and configuration. "
        "* after a policy means gradient checkpointing is on."
    )
    if any(row.get("inference_context") == "cached_episode_prompt" for row in rows):
        note += " LingBot inference reuses its episode prompt."
    source_note = (
        "Latest archived measurements from multiple source runs; setups and source dates are retained in results.json."
        if latest
        else ""
    )
    lines = [
        "# LeRobot model profiling",
        "",
        f"[Results and history in the public health dashboard]({SPACE_URL})",
        "",
        "Fixed batch; warmup excluded; latency measured without the Torch profiler.",
        "",
        note,
        source_note,
        history_note,
        "",
        "| Policy | Status | Inference ms | Δ4w | Training ms | Δ4w | Error |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    table = [f"{'Policy':<15} {'Infer ms':>10} {'Δ4w':>8} {'Train ms':>10} {'Δ4w':>8}"]
    for row, compared in zip(rows, comparisons, strict=True):
        values = []
        for mode in ("inference", "training"):
            values.extend([metric_label(row, mode, compared[mode]), delta_label(compared[mode])])
        label = row["policy"] + ("*" if row.get("gradient_checkpointing") else "")
        table.append(f"{label:<15} {values[0]:>10} {values[1]:>8} {values[2]:>10} {values[3]:>8}")
        errors = [row.get("error", "")] + [row.get(m, {}).get("error", "") for m in ("inference", "training")]
        error = "; ".join(filter(None, errors)).replace("|", "/").replace("\n", " ")[:250]
        lines.append(f"| {label} | {row['status']} | {' | '.join(values)} | {error} |")
    report = "\n".join(lines) + "\n"
    (output_dir / "README.md").write_text(report)
    (output_dir / "comparison.json").write_text(
        json.dumps(
            {
                "started_at": started_at.isoformat(),
                "note": note,
                "history_note": history_note,
                "rows": comparisons,
            },
            indent=2,
        )
        + "\n"
    )
    counts = ", ".join(
        f"{sum(measured(row, mode) is not None for row in rows)}/{len(rows)} {mode}"
        for mode in ("inference", "training")
    )
    run_url = os.environ.get("PROFILE_RUN_URL", "")
    payload: dict[str, Any] = {
        "text": f"LeRobot profiling: {counts}. Results and history: {SPACE_URL}",
        "unfurl_links": False,
        "unfurl_media": False,
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": "LeRobot model profiling"}},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{started_at:%Y-%m-%d}* · {counts}\nFresh-action inference / training update · median ms",
                },
            },
            {"type": "section", "text": {"type": "mrkdwn", "text": "```\n" + "\n".join(table) + "\n```"}},
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": " ".join(filter(None, [note, source_note, history_note]))}
                ],
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Check Action results"},
                        "url": SPACE_URL,
                    }
                ],
            },
        ],
    }
    if run_url:
        payload["blocks"].append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"<{run_url}|GitHub run and raw artifacts>"}],
            }
        )
    (output_dir / "slack.json").write_text(json.dumps(payload, indent=2) + "\n")
    status = {
        "context": "Profiling report",
        "state": "success",  # Report availability; failed GPU jobs retain their own conclusions.
        "target_url": SPACE_URL,
        "description": f"{counts} measurements",
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return report


def current_run() -> dict:
    run_id = os.environ["GITHUB_RUN_ID"]
    request = Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}",
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(request, timeout=30) as response:  # nosec B310: fixed GitHub HTTPS host above
        return json.load(response)


def archive(output_dir: Path) -> Path:
    """Prepare a permanent, public report without including caches or credentials."""
    run_id = os.environ["GITHUB_RUN_ID"]
    run = current_run()
    rows = json.loads((output_dir / "results.json").read_text())
    metadata = {
        key: run[key] for key in ("id", "head_sha", "head_branch", "created_at", "run_started_at", "html_url")
    }
    metadata.update(
        key=f"{run_id}-{os.environ['GITHUB_RUN_ATTEMPT']}",
        run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
        status="completed",
        conclusion="success" if all(row["status"] == "success" for row in rows) else "failure",
        profiling_variant=(
            "checkpointing_trial" if os.environ.get("GRADIENT_CHECKPOINTING") == "true" else "baseline"
        ),
    )
    folder = output_dir / "dashboard"
    folder.mkdir(exist_ok=True)
    shutil.copyfile(output_dir / "results.json", folder / "results.json")
    (folder / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with zipfile.ZipFile(folder / "report.zip", "w", zipfile.ZIP_DEFLATED) as report:
        for name in ("results.json", "README.md", "comparison.json", "slack.json"):
            if (output_dir / name).exists():
                report.write(output_dir / name, name)
    return folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["matrix", "init", "run", "summary", "archive", "timeout"])
    parser.add_argument("--policy")
    parser.add_argument("--output-dir", type=Path, default=Path("profiling-results"))
    parser.add_argument("--mode", choices=["inference", "training", "both"], default="both")
    parser.add_argument("--gradient-checkpointing", choices=["true", "false"], default="false")
    parser.add_argument("--compare-history", action="store_true")
    parser.add_argument("--publish-latest", choices=["true", "false"], default="false")
    args, overrides = parser.parse_known_args()
    if args.command in ("matrix", "summary"):
        selected = set(filter(None, (v.strip() for v in os.environ.get("POLICIES", "").split(","))))
        matrix = policy_matrix()
        unknown = selected - {row["policy"] for row in matrix}
        if unknown:
            parser.error(f"Unknown policies: {sorted(unknown)}")
        matrix = [row for row in matrix if not selected or row["policy"] in selected]
        if args.command == "matrix":
            print(json.dumps({"include": matrix}))
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if args.publish_latest == "true":
                with urlopen(LATEST_URL, timeout=30) as response:  # nosec B310: fixed public HTTPS URL
                    rows = json.load(response)["rows"]
                for row in rows:
                    if row["policy"] in {entry["policy"] for entry in matrix}:
                        path = args.output_dir / row["policy"] / "result.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(row))
            points, started_at, history_note = [], None, ""
            if args.compare_history:
                run = current_run()
                started_at = datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00"))
                try:
                    with urlopen(HISTORY_URL, timeout=30) as response:  # nosec B310: fixed public HTTPS URL
                        points = json.load(response)["points"]
                except (OSError, ValueError, KeyError):
                    history_note = "History unavailable; four-week comparisons omitted."
            print(
                summary(
                    args.output_dir,
                    [row["policy"] for row in matrix],
                    points=points,
                    started_at=started_at,
                    history_note=history_note,
                    latest=args.publish_latest == "true",
                )
            )
    elif args.command == "timeout":
        mark_timeout(args.policy, args.output_dir, args.mode)
    elif args.command == "run":
        return run_policy(
            args.policy, args.output_dir.resolve(), overrides, args.gradient_checkpointing == "true"
        )
    elif args.command == "archive":
        print(archive(args.output_dir))
    else:
        path = args.output_dir / args.policy / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "policy": args.policy,
                    "status": "failed",
                    "error": "Dependency installation or container failed; see job logs",
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
