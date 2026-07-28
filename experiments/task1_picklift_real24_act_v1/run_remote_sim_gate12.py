"""Run the authorized hardware-free Task1 ACT Real-to-Sim gate12.

This runner is intentionally gate12-only. It imports the frozen ACT inference
interface and the deployed Remote Nexus adapter, but starts no service, accesses
no camera/serial/robot hardware, and writes no LeRobot Dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


EXPERIMENT_DIR = Path(
    "/home/ubuntu24/Teleop/lerobot/experiments/"
    "task1_picklift_real24_act_v1"
)
REMOTE_ROOT = Path("/home/ubuntu24/SO101QuestRemote")
REMOTE_ADAPTER_DIR = REMOTE_ROOT / "robot-host"
REMOTE_DEPLOYED_COMMIT_FILE = REMOTE_ROOT / ".deployed-commit"
EXPECTED_REMOTE_DEPLOYED_COMMIT = (
    "a5914b14261488193b007e004c81ceefb1eed254"
)
EXPECTED_PLAN_SHA256 = (
    "c26bf3caa788708cf8ccb332e2f84771d271b05795634a9b1022ed97d06b8c86"
)
EXPECTED_MODEL_SHA256 = (
    "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
)
EXPECTED_GATE_EPISODES = 12
LEGAL_TERMINATION_REASONS = {
    "environment_terminated",
    "environment_truncated",
    "max_steps_reached",
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(handle: Any, payload: Any) -> None:
    handle.write(canonical_json(payload) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_contract() -> tuple[Any, dict[str, Any], tuple[Any, ...]]:
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))
    if str(REMOTE_ADAPTER_DIR) not in sys.path:
        sys.path.insert(0, str(REMOTE_ADAPTER_DIR))

    from nexus_picklift_policy_adapter import (
        GATE_PHASE_ID,
        build_real24_act100k_plan,
        experiment_contract,
    )

    if GATE_PHASE_ID != "gate12":
        raise RuntimeError("Remote gate phase id drifted.")
    contract = experiment_contract()
    plan_sha = contract["plan"]["plan_sha256"]
    if plan_sha != EXPECTED_PLAN_SHA256:
        raise RuntimeError(
            f"Remote plan SHA mismatch: {plan_sha} != {EXPECTED_PLAN_SHA256}"
        )
    trials = build_real24_act100k_plan()[GATE_PHASE_ID]
    if len(trials) != EXPECTED_GATE_EPISODES:
        raise RuntimeError("Remote gate plan must contain exactly 12 episodes.")
    if [trial.phase_episode_index for trial in trials] != list(
        range(EXPECTED_GATE_EPISODES)
    ):
        raise RuntimeError("Remote gate trial ordering drifted.")
    if [trial.cell_name for trial in trials] != [
        f"r{row}c{column}"
        for row in range(1, 4)
        for column in range(1, 5)
    ]:
        raise RuntimeError("Remote gate cells are not frozen row-major 3x4.")
    return GATE_PHASE_ID, contract, trials


def validate_deployment() -> dict[str, Any]:
    deployed_commit = REMOTE_DEPLOYED_COMMIT_FILE.read_text(
        encoding="utf-8"
    ).strip()
    if deployed_commit != EXPECTED_REMOTE_DEPLOYED_COMMIT:
        raise RuntimeError(
            "Remote deployed commit mismatch; refusing to execute gate12."
        )
    gate_phase_id, contract, trials = load_contract()
    if contract["upstream_act_binding"]["model_sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Remote contract checkpoint binding drifted.")
    if contract["plan"]["evaluation_phase_id"] != "frozen120":
        raise RuntimeError("Remote evaluation phase contract drifted.")
    if contract["clock"]["maximum_policy_ticks"] != 600:
        raise RuntimeError("Remote 30-second policy-tick limit drifted.")
    if contract["clock"]["maximum_env_steps"] != 1500:
        raise RuntimeError("Remote 30-second environment-step limit drifted.")
    return {
        "gate_phase_id": gate_phase_id,
        "gate_episode_count": len(trials),
        "plan_sha256": contract["plan"]["plan_sha256"],
        "remote_deployed_commit": deployed_commit,
        "model_sha256": contract["upstream_act_binding"]["model_sha256"],
        "clock": contract["clock"],
        "status": "pass",
    }


def validate_policy_inputs(inputs: dict[str, Any]) -> None:
    if set(inputs) != {
        "observation.state",
        "observation.images.front",
    }:
        raise RuntimeError("Remote observation keys drifted.")
    state = inputs["observation.state"]
    image = inputs["observation.images.front"]
    if (
        not isinstance(state, np.ndarray)
        or state.shape != (6,)
        or state.dtype != np.float32
        or not bool(np.isfinite(state).all())
    ):
        raise RuntimeError("Remote state is not finite float32[6].")
    if (
        not isinstance(image, np.ndarray)
        or image.shape != (480, 640, 3)
        or image.dtype != np.uint8
    ):
        raise RuntimeError("Remote front image is not uint8[480,640,3] RGB.")


def episode_record(
    *,
    runtime_summary: dict[str, Any],
    raw_action_count: int,
    calibration_clipped_action_count: int,
    relative_clipped_action_count: int,
    calibration_clipped_joint_value_count: int,
    relative_clipped_joint_value_count: int,
    sent_action_count: int,
    valid_observation_count: int,
    expected_policy_ticks: int,
    interface_error: str | None,
) -> dict[str, Any]:
    record = dict(runtime_summary)
    record.update(
        {
            "raw_action_count": raw_action_count,
            "calibration_clipped_action_count": (
                calibration_clipped_action_count
            ),
            "relative_clipped_action_count": relative_clipped_action_count,
            "calibration_clipped_joint_value_count": (
                calibration_clipped_joint_value_count
            ),
            "relative_clipped_joint_value_count": (
                relative_clipped_joint_value_count
            ),
            "sent_action_count": sent_action_count,
            "valid_observation_count": valid_observation_count,
            "interface_error": interface_error,
        }
    )
    legal_end = (
        record["termination_reason"] in LEGAL_TERMINATION_REASONS
        and (
            bool(record["terminated"])
            or bool(record["truncated"])
            or bool(record["timeout"])
        )
    )
    record["interface_valid"] = bool(
        interface_error is None
        and valid_observation_count == raw_action_count
        and raw_action_count == sent_action_count
        and 0 < raw_action_count <= expected_policy_ticks
        and (
            not bool(record["timeout"])
            or raw_action_count == expected_policy_ticks
        )
        and int(record["env_steps"]) > 0
        and legal_end
    )
    if interface_error is not None:
        record["failure_type"] = "interface_error"
    return record


def summarize_gate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_cell[str(episode["cell"])].append(episode)

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(bool(item["success"]) for item in group)
        return {
            "episodes": len(group),
            "interface_valid_episodes": sum(
                bool(item["interface_valid"]) for item in group
            ),
            "successes": successes,
            "success_rate": successes / len(group),
            "env_steps": sum(int(item["env_steps"]) for item in group),
            "raw_action_count": sum(
                int(item["raw_action_count"]) for item in group
            ),
            "calibration_clipped_action_count": sum(
                int(item["calibration_clipped_action_count"])
                for item in group
            ),
            "relative_clipped_action_count": sum(
                int(item["relative_clipped_action_count"]) for item in group
            ),
            "calibration_clipped_joint_value_count": sum(
                int(item["calibration_clipped_joint_value_count"])
                for item in group
            ),
            "relative_clipped_joint_value_count": sum(
                int(item["relative_clipped_joint_value_count"])
                for item in group
            ),
            "sent_action_count": sum(
                int(item["sent_action_count"]) for item in group
            ),
            "failure_types": dict(
                Counter(
                    str(item["failure_type"])
                    for item in group
                    if not item["success"]
                )
            ),
            "termination_reasons": dict(
                Counter(str(item["termination_reason"]) for item in group)
            ),
        }

    overall = summarize_group(episodes)
    expected_cells = {
        f"r{row}c{column}"
        for row in range(1, 4)
        for column in range(1, 5)
    }
    gate_pass = bool(
        len(episodes) == EXPECTED_GATE_EPISODES
        and set(by_cell) == expected_cells
        and all(len(group) == 1 for group in by_cell.values())
        and overall["interface_valid_episodes"] == EXPECTED_GATE_EPISODES
    )
    return {
        "schema_version": 1,
        "status": "diagnostic_only_not_a_real_robot_or_paper_result",
        "phase_id": "gate12",
        "gate_definition": (
            "all 12 trials reset, produced exact state/RGB, completed finite "
            "action stepping, and produced legal termination/timeout evidence; "
            "task success is not required"
        ),
        "interface_gate_pass": gate_pass,
        "overall": overall,
        "by_cell": {
            cell: summarize_group(by_cell[cell]) for cell in sorted(by_cell)
        },
    }


def write_hashes(output_dir: Path) -> dict[str, str]:
    names = (
        "episodes.jsonl",
        "ticks.jsonl",
        "run_manifest.json",
        "summary.json",
    )
    hashes = {name: sha256_file(output_dir / name) for name in names}
    (output_dir / "hashes.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
        encoding="utf-8",
    )
    return hashes


def run_gate(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(
            f"Evidence directory already exists; refusing overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    validation = validate_deployment()
    _, contract, trials = load_contract()

    from nexus_picklift_policy_adapter import (
        create_nexus_picklift_policy_adapter,
    )
    from sim_policy_inference import Task1ActSimInference

    source_commit = git_head(EXPERIMENT_DIR.parents[1])
    run_manifest = {
        "schema_version": 1,
        "experiment_run_id": output_dir.name,
        "phase_id": "gate12",
        "status": "running",
        "research_status": (
            "diagnostic_only_not_real_robot_evaluation_not_a_paper_result"
        ),
        "started_at_utc": started_at.isoformat(),
        "source_commit": source_commit,
        "remote_deployed_commit": validation["remote_deployed_commit"],
        "remote_adapter_path": str(
            REMOTE_ADAPTER_DIR / "nexus_picklift_policy_adapter.py"
        ),
        "remote_adapter_sha256": sha256_file(
            REMOTE_ADAPTER_DIR / "nexus_picklift_policy_adapter.py"
        ),
        "plan_sha256": validation["plan_sha256"],
        "checkpoint_path": contract["upstream_act_binding"]["checkpoint_path"],
        "model_sha256": validation["model_sha256"],
        "python_executable": sys.executable,
        "python_version": sys.version,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "hardware_accessed": False,
        "gateway_or_quest_started": False,
        "lerobot_dataset_written": False,
        "phase_scope": "gate12_only_frozen120_not_authorized",
        "contract": contract,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    policy = Task1ActSimInference()
    if policy.model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Loaded policy model SHA mismatch.")
    adapter = create_nexus_picklift_policy_adapter()
    episodes: list[dict[str, Any]] = []
    try:
        with (
            (output_dir / "ticks.jsonl").open(
                "x", encoding="utf-8", buffering=1
            ) as ticks_handle,
            (output_dir / "episodes.jsonl").open(
                "x", encoding="utf-8", buffering=1
            ) as episodes_handle,
        ):
            for trial in trials:
                policy.reset_episode()
                raw_action_count = 0
                calibration_clipped_action_count = 0
                relative_clipped_action_count = 0
                calibration_clipped_joint_value_count = 0
                relative_clipped_joint_value_count = 0
                sent_action_count = 0
                valid_observation_count = 0
                interface_error: str | None = None
                try:
                    observation = adapter.reset(trial)
                    while adapter.phase == "active":
                        inputs = observation.policy_inputs()
                        validate_policy_inputs(inputs)
                        valid_observation_count += 1
                        policy_step = policy.infer(
                            inputs["observation.state"],
                            inputs["observation.images.front"],
                        )
                        raw_action_count += 1
                        calibration_clipped_action_count += int(
                            bool(policy_step.calibration_clip_mask.any())
                        )
                        relative_clipped_action_count += int(
                            bool(policy_step.relative_clip_mask.any())
                        )
                        calibration_clipped_joint_value_count += int(
                            policy_step.calibration_clip_mask.sum()
                        )
                        relative_clipped_joint_value_count += int(
                            policy_step.relative_clip_mask.sum()
                        )
                        tick_result = adapter.apply_action(
                            policy_step.sent_action
                        )
                        sent_action_count += 1
                        append_jsonl(
                            ticks_handle,
                            {
                                "phase_id": trial.phase_id,
                                "cell": trial.cell_name,
                                "seed": trial.seed,
                                "observation": observation.evidence(),
                                "policy": policy_step.to_jsonable(),
                                "remote": tick_result.evidence(),
                            },
                        )
                        if adapter.phase == "active":
                            observation = adapter.observe()
                except Exception as exc:
                    interface_error = f"{type(exc).__name__}: {exc}"

                manifest = adapter.episode_manifest()
                runtime_summary = manifest["runtime_summary"]
                record = episode_record(
                    runtime_summary=runtime_summary,
                    raw_action_count=raw_action_count,
                    calibration_clipped_action_count=(
                        calibration_clipped_action_count
                    ),
                    relative_clipped_action_count=(
                        relative_clipped_action_count
                    ),
                    calibration_clipped_joint_value_count=(
                        calibration_clipped_joint_value_count
                    ),
                    relative_clipped_joint_value_count=(
                        relative_clipped_joint_value_count
                    ),
                    sent_action_count=sent_action_count,
                    valid_observation_count=valid_observation_count,
                    expected_policy_ticks=contract["clock"][
                        "maximum_policy_ticks"
                    ],
                    interface_error=interface_error,
                )
                record.update(
                    {
                        "phase_episode_index": trial.phase_episode_index,
                        "repeat_index": trial.repeat_index,
                        "trial_manifest": trial.manifest(),
                        "experiment_run_id": output_dir.name,
                        "source_commit": source_commit,
                        "remote_deployed_commit": validation[
                            "remote_deployed_commit"
                        ],
                        "model_sha256": policy.model_hash,
                    }
                )
                append_jsonl(episodes_handle, record)
                episodes.append(record)
                print(
                    canonical_json(
                        {
                            "cell": record["cell"],
                            "episode": len(episodes),
                            "interface_valid": record["interface_valid"],
                            "success": record["success"],
                            "env_steps": record["env_steps"],
                            "termination_reason": record[
                                "termination_reason"
                            ],
                        }
                    ),
                    flush=True,
                )
    finally:
        adapter.close()

    summary = summarize_gate(episodes)
    summary["experiment_run_id"] = output_dir.name
    summary["plan_sha256"] = validation["plan_sha256"]
    summary["model_sha256"] = policy.model_hash
    summary["runtime_seconds"] = time.monotonic() - started_monotonic
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "summary.json", summary)

    run_manifest.update(
        {
            "status": (
                "gate_pass"
                if summary["interface_gate_pass"]
                else "gate_fail"
            ),
            "completed_at_utc": summary["completed_at_utc"],
            "runtime_seconds": summary["runtime_seconds"],
            "episode_count": len(episodes),
        }
    )
    write_json(output_dir / "run_manifest.json", run_manifest)
    hashes = write_hashes(output_dir)
    print(
        canonical_json(
            {
                "interface_gate_pass": summary["interface_gate_pass"],
                "output_dir": str(output_dir),
                "hashes": hashes,
            }
        ),
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hardware-free Task1 ACT Remote gate12 only."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-contract-only",
        action="store_true",
        help="Validate frozen identities without creating a Nexus environment.",
    )
    args = parser.parse_args()
    if args.validate_contract_only == (args.output_dir is not None):
        parser.error(
            "choose exactly one of --validate-contract-only or --output-dir"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.validate_contract_only:
        print(json.dumps(validate_deployment(), indent=2, sort_keys=True))
        return
    summary = run_gate(args.output_dir.resolve())
    if not summary["interface_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
