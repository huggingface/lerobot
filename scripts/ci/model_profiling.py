#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Dependency-free policy discovery, benchmark launching, and result summaries."""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


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


def policy_args(policy: str) -> list[str]:
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
    if policy == "fineart_vla":
        # The default FAST tokenizer fit needs a representative action distribution.
        args += ["--dataset.episodes=[0,1,2,3,4,5,6,7,8,9]"]
    if policy in {"dm05", "pi05", "molmoact2"}:
        args += ["--compute_missing_quantiles=true"]
    if policy == "tdmpc":
        args += ["--policy.horizon=3", "--policy.n_action_steps=1"]
    if policy == "flux3":
        args += ["--policy.camera_layout=single"]
    if policy == "lawam":
        args += [
            '--policy.primary_image_features=["observation.images.front"]',
            '--policy.wrist_image_features=["observation.images.wrist"]',
        ]
    if policy == "vqbet":
        # Finish the codebook stage during the default warmup, then time BeT updates.
        args += ["--policy.n_vqvae_training_steps=1"]
    return args


def run_policy(policy: str, output_dir: Path, overrides: list[str]) -> int:
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
            *policy_args(policy),
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


def summary(output_dir: Path, policies: list[str] | None = None) -> str:
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
    lines = [
        "# LeRobot model profiling",
        "",
        "Fixed batch; warmup excluded; latency measured without the Torch profiler.",
        "",
        "| Policy | Status | Inference median ms | Training median ms | Error |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        values = []
        for mode in ("inference", "training"):
            value = row.get(mode, {}).get("median_ms")
            values.append(f"{value:.2f}" if value is not None else "—")
        errors = [row.get("error", "")] + [row.get(m, {}).get("error", "") for m in ("inference", "training")]
        error = "; ".join(filter(None, errors)).replace("|", "/").replace("\n", " ")[:250]
        lines.append(f"| {row['policy']} | {row['status']} | {' | '.join(values)} | {error} |")
    report = "\n".join(lines) + "\n"
    (output_dir / "README.md").write_text(report)
    return report


def archive(output_dir: Path) -> Path:
    """Prepare a permanent, public report without including caches or credentials."""
    run_id = os.environ["GITHUB_RUN_ID"]
    request = Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}",
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(request, timeout=30) as response:  # nosec B310: fixed GitHub HTTPS host above
        run = json.load(response)
    rows = json.loads((output_dir / "results.json").read_text())
    metadata = {key: run[key] for key in ("id", "head_sha", "head_branch", "created_at", "html_url")}
    metadata.update(
        key=f"{run_id}-{os.environ['GITHUB_RUN_ATTEMPT']}",
        run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
        status="completed",
        conclusion="success" if all(row["status"] == "success" for row in rows) else "failure",
    )
    folder = output_dir / "dashboard"
    folder.mkdir(exist_ok=True)
    shutil.copyfile(output_dir / "results.json", folder / "results.json")
    (folder / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with zipfile.ZipFile(folder / "report.zip", "w", zipfile.ZIP_DEFLATED) as report:
        for name in ("results.json", "README.md"):
            report.write(output_dir / name, name)
    return folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["matrix", "init", "run", "summary", "archive"])
    parser.add_argument("--policy")
    parser.add_argument("--output-dir", type=Path, default=Path("profiling-results"))
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
            print(summary(args.output_dir, [row["policy"] for row in matrix]))
    elif args.command == "run":
        return run_policy(args.policy, args.output_dir.resolve(), overrides)
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
