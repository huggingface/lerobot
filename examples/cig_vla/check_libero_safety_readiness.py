#!/usr/bin/env python
"""Strict LIBERO-Safety integration readiness gate for CIG-VLA."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from lerobot.datasets.adapters.libero_safety import load_libero_safety_contract
from lerobot.datasets.adapters.libero_safety_v21 import (
    LiberoSafetyV21Dataset,
    build_episode_safe_action_chunk,
)
from lerobot.policies.cig_vla.trajectory_geometry import TrajectoryGeometryTargetBuilder

CHECK_NAMES = (
    "Metadata contract",
    "Parquet loading",
    "Front RGB decode",
    "Wrist RGB decode",
    "Instruction mapping",
    "Action contract",
    "Episode-safe chunking",
    "Normalization stats",
    "Physical denormalization",
    "Trajectory targets",
    "Target leakage check",
    "CIG-VLA one-batch",
    "Actual Qwen loading",
    "Actual Qwen backward",
    "10-step smoke",
)


def run_checks(diagnostics_path):
    checks = dict.fromkeys(CHECK_NAMES, False)
    details = {}
    try:
        contract = load_libero_safety_contract()
        checks["Metadata contract"] = contract.codebase_version == "v2.1" and (
            contract.total_episodes,
            contract.total_frames,
            contract.total_tasks,
            contract.fps,
        ) == (19664, 3443735, 15, 20)
        dataset = LiberoSafetyV21Dataset(episodes=[0], chunk_size=4)
        sample = dataset[0]
        checks["Parquet loading"] = sample["observation.state"].shape == (8,)
        checks["Front RGB decode"] = sample["observation.image"].shape == (3, 256, 256)
        checks["Wrist RGB decode"] = sample["observation.wrist_image"].shape == (3, 256, 256)
        checks["Instruction mapping"] = sample["task"] == contract.tasks[int(sample["task_index"])]
        checks["Action contract"] = sample["action"].shape == (4, 7)

        rows = [{"actions": [1.0] * 7}, {"actions": [9.0] * 7}]
        chunk, padding = build_episode_safe_action_chunk(rows, 1, 4)
        checks["Episode-safe chunking"] = (
            padding.tolist() == [False, True, True, True] and not chunk[1:].any()
        )

        stats = dataset.meta.stats["action"]
        checks["Normalization stats"] = (
            stats["mean"].shape == (7,)
            and stats["std"].shape == (7,)
            and torch.isfinite(stats["mean"]).all()
            and torch.isfinite(stats["std"]).all()
            and (stats["std"] > 0).all()
        )
        normalized = (sample["action"] - stats["mean"]) / (stats["std"] + 1e-8)
        restored = normalized * stats["std"] + stats["mean"]
        checks["Physical denormalization"] = torch.allclose(restored, sample["action"], atol=1e-7, rtol=1e-5)
        target = TrajectoryGeometryTargetBuilder().build(
            normalized[None],
            sample["observation.state"][None],
            dataset.meta.stats,
            sample["action_is_pad"][None],
        )
        checks["Trajectory targets"] = (
            target.valid_mask.item()
            and torch.isfinite(target.translation_goal).all()
            and target.rotation_goal is None
        )
    except Exception as error:
        details["dataset_error"] = repr(error)

    unit = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/policies/cig_vla/test_full_policy_forward.py::test_future_action_changes_target_but_not_stage_a_prediction",
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    checks["Target leakage check"] = unit.returncode == 0
    if unit.returncode:
        details["leakage_test"] = unit.stdout + unit.stderr

    mock_path = Path("/tmp/cig_libero_dataset_mock_smoke/smoke_diagnostics.json")
    if mock_path.exists():
        mock = json.loads(mock_path.read_text())
        checks["CIG-VLA one-batch"] = (
            not mock["actual_qwen"]
            and mock["parameter_update"]
            and mock["reload_finite"]
            and all(torch.isfinite(torch.tensor(item["loss"])) for item in mock["losses"])
        )
    else:
        details["mock_smoke"] = f"missing {mock_path}"

    if diagnostics_path.exists():
        actual = json.loads(diagnostics_path.read_text())
        checks["Actual Qwen loading"] = (
            actual.get("actual_qwen") is True and actual.get("total_parameters", 0) > 2_000_000_000
        )
        checks["Actual Qwen backward"] = (
            actual.get("gradient_norm", 0) > 0
            and actual.get("parameter_update") is True
            and len(actual.get("losses", [])) >= 1
        )
        checks["10-step smoke"] = (
            actual.get("steps") == 10
            and len(actual.get("losses", [])) == 10
            and actual.get("reload_finite") is True
            and actual.get("fixed_inference_shape_match") is True
            and actual.get("fixed_inference_reload_match") is True
        )
    else:
        details["actual_qwen"] = f"missing {diagnostics_path}"
    return checks, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path("outputs/cig_vla/libero_safety_qwen_smoke/smoke_diagnostics.json"),
    )
    parser.add_argument("--run-actual-qwen", action="store_true")
    args = parser.parse_args()
    if args.run_actual_qwen:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "examples/cig_vla/libero_safety_smoke.py",
                "--actual-qwen",
                "--local-response",
                "--steps",
                "10",
                "--output-dir",
                str(args.diagnostics.parent),
                "--model-path",
                "/home/user_lerobot/.cache/modelscope/models/Qwen--Qwen3-VL-2B-Instruct/snapshots/master",
            ],
            check=True,
        )
    checks, details = run_checks(args.diagnostics)
    print("LIBERO-Safety integration readiness")
    print("------------------------------------")
    for name in CHECK_NAMES:
        status = "PASS" if checks[name] else "FAIL"
        print(f"{name}: {status}")
    ready = all(checks.values())
    print("\nVerdict:")
    print("READY FOR A0/A1/A3 PILOT" if ready else "NOT READY")
    if details:
        print("\nDetails:")
        for name, detail in details.items():
            print(f"- {name}: {detail}")
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
