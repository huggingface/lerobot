from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_remote_sim import (
    COMPARISON_BOUNDARY,
    EXPECTED_MODEL_SHA256,
    EXPECTED_OWNER_PLAN_SHA256,
    EXPECTED_REMOTE_DEPLOYED_COMMIT,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
    return rows


def verify_core_hashes(output_dir: Path) -> dict[str, str]:
    rows = (
        output_dir / "hashes.sha256"
    ).read_text(encoding="utf-8").splitlines()
    expected = {}
    for row in rows:
        digest, name = row.split("  ", 1)
        expected[name] = digest
    required = {
        "episodes.jsonl",
        "ticks.jsonl",
        "run_manifest.json",
        "summary.json",
    }
    if set(expected) != required:
        raise RuntimeError("Core hash manifest file set drifted.")
    for name, digest in expected.items():
        if sha256_file(output_dir / name) != digest:
            raise RuntimeError(f"Evidence hash mismatch: {name}")
    return expected


def verify(output_dir: Path, phase: str) -> dict[str, Any]:
    expected_episodes = 12 if phase == "gate12" else 120
    expected_ticks = expected_episodes * 600
    core_hashes = verify_core_hashes(output_dir)
    episodes = read_jsonl(output_dir / "episodes.jsonl")
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    if len(episodes) != expected_episodes:
        raise RuntimeError("Episode count differs from the frozen phase.")
    if manifest["phase_id"] != phase or summary["phase_id"] != phase:
        raise RuntimeError("Phase identity mismatch.")
    if manifest["model_sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Manifest model SHA mismatch.")
    if summary["model_sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Summary model SHA mismatch.")
    if manifest["owner_plan_sha256"] != EXPECTED_OWNER_PLAN_SHA256:
        raise RuntimeError("Owner plan SHA mismatch.")
    if manifest["remote_deployed_commit"] != EXPECTED_REMOTE_DEPLOYED_COMMIT:
        raise RuntimeError("Remote deployed commit mismatch.")
    if manifest["hardware_accessed"] is not False:
        raise RuntimeError("Evidence unexpectedly reports hardware access.")
    if manifest["gateway_or_quest_started"] is not False:
        raise RuntimeError("Evidence unexpectedly reports gateway/Quest.")
    if manifest["lerobot_dataset_written"] is not False:
        raise RuntimeError("Evidence unexpectedly reports Dataset writes.")
    if manifest["comparison_boundary"] != COMPARISON_BOUNDARY:
        raise RuntimeError("Comparison boundary drifted.")

    tick_counts: Counter[int] = Counter()
    environment_clip_ticks: Counter[int] = Counter()
    environment_clip_values: Counter[int] = Counter()
    total_ticks = 0
    with (output_dir / "ticks.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid tick JSON at line {line_number}"
                ) from exc
            episode_index = int(row["phase_episode_index"])
            tick_counts[episode_index] += 1
            total_ticks += 1
            policy = row["policy"]
            remote = row["remote"]
            raw = np.asarray(policy["raw_action"], dtype=np.float32)
            requested = np.asarray(
                policy["requested_action"],
                dtype=np.float32,
            )
            adapter_received = np.asarray(
                remote["sent_action"],
                dtype=np.float32,
            )
            environment_action = np.asarray(
                remote["environment_action"],
                dtype=np.float32,
            )
            mask = np.asarray(remote["environment_clipped_mask"], dtype=bool)
            if any(vector.shape != (6,) for vector in (
                raw,
                requested,
                adapter_received,
                environment_action,
                mask,
            )):
                raise RuntimeError("Tick action evidence must have shape [6].")
            if not all(
                bool(np.isfinite(vector).all())
                for vector in (raw, requested, adapter_received, environment_action)
            ):
                raise RuntimeError("Tick action evidence contains non-finite values.")
            if not np.array_equal(raw, requested):
                raise RuntimeError("Runner modified ACT raw action.")
            if not np.array_equal(requested, adapter_received):
                raise RuntimeError("Adapter did not receive requested action exactly.")
            expected_mask = ~np.isclose(
                requested,
                environment_action,
                rtol=0.0,
                atol=1.0e-6,
            )
            if not np.array_equal(mask, expected_mask):
                raise RuntimeError("Nexus environment clip mask is inconsistent.")
            if bool(mask.any()):
                environment_clip_ticks[episode_index] += 1
            environment_clip_values[episode_index] += int(mask.sum())
            if policy["runner_state_projection"] != "none":
                raise RuntimeError("Unexpected runner state projection.")
            if policy["runner_absolute_calibration_clamp"] != "none":
                raise RuntimeError("Unexpected runner absolute clamp.")
            if policy["runner_relative_clamp"] != "none":
                raise RuntimeError("Unexpected runner relative clamp.")
    if total_ticks != expected_ticks:
        raise RuntimeError("Total tick count differs from the frozen phase.")
    if set(tick_counts) != set(range(expected_episodes)):
        raise RuntimeError("Tick episode indices are incomplete.")
    if any(tick_counts[index] != 600 for index in range(expected_episodes)):
        raise RuntimeError("Every episode must contain exactly 600 policy ticks.")

    for index, episode in enumerate(episodes):
        if episode["phase_episode_index"] != index:
            raise RuntimeError("Episode ordering drifted.")
        if episode["interface_valid"] is not True:
            raise RuntimeError(f"Episode {index} is not interface-valid.")
        if episode["ready_pose_tick0_valid"] is not True:
            raise RuntimeError(f"Episode {index} ready pose is invalid.")
        if episode["object_spawn_plan_valid"] is not True:
            raise RuntimeError(f"Episode {index} object spawn is invalid.")
        if episode["raw_action_count"] != 600:
            raise RuntimeError(f"Episode {index} raw action count drifted.")
        if episode["requested_action_count"] != 600:
            raise RuntimeError(f"Episode {index} requested action count drifted.")
        if episode["env_steps"] != 1500:
            raise RuntimeError(f"Episode {index} env step count drifted.")
        if episode["termination_reason"] != "max_steps_reached":
            raise RuntimeError(f"Episode {index} termination drifted.")
        if episode["environment_clipped_action_count"] != (
            environment_clip_ticks[index]
        ):
            raise RuntimeError(f"Episode {index} clip tick count mismatch.")
        if episode["environment_clipped_joint_value_count"] != (
            environment_clip_values[index]
        ):
            raise RuntimeError(f"Episode {index} clip value count mismatch.")

    successes = sum(bool(row["success"]) for row in episodes)
    clip_ticks = sum(environment_clip_ticks.values())
    clip_values = sum(environment_clip_values.values())
    if summary["interface_pass"] is not True:
        raise RuntimeError("Summary interface gate did not pass.")
    if summary["overall"]["successes"] != successes:
        raise RuntimeError("Summary success count mismatch.")
    if summary["overall"]["environment_clipped_action_count"] != clip_ticks:
        raise RuntimeError("Summary environment clip tick count mismatch.")
    if (
        summary["overall"]["environment_clipped_joint_value_count"]
        != clip_values
    ):
        raise RuntimeError("Summary environment clip value count mismatch.")
    for cell, group in summary["by_cell"].items():
        source = [row for row in episodes if row["cell"] == cell]
        if group["successes"] != sum(bool(row["success"]) for row in source):
            raise RuntimeError(f"Cell {cell} success count mismatch.")
    return {
        "schema_version": 1,
        "status": "independent_verification_passed",
        "phase_id": phase,
        "output_dir": str(output_dir),
        "episode_count": len(episodes),
        "tick_count": total_ticks,
        "interface_valid_episodes": expected_episodes,
        "successes": successes,
        "success_rate": successes / expected_episodes,
        "environment_clipped_action_count": clip_ticks,
        "environment_clipped_joint_value_count": clip_values,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "owner_plan_sha256": EXPECTED_OWNER_PLAN_SHA256,
        "remote_deployed_commit": EXPECTED_REMOTE_DEPLOYED_COMMIT,
        "core_hashes": core_hashes,
        "comparison_boundary": COMPARISON_BOUNDARY,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("gate12", "frozen120"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    verification_path = output_dir / "verification.json"
    verification_hash_path = output_dir / "verification.sha256"
    if verification_path.exists() or verification_hash_path.exists():
        raise RuntimeError("Refusing to overwrite independent verification.")
    result = verify(output_dir, args.phase)
    verification_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verification_hash_path.write_text(
        (
            f"{sha256_file(verification_path)}  verification.json\n"
            f"{sha256_file(output_dir / 'hashes.sha256')}  hashes.sha256\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
