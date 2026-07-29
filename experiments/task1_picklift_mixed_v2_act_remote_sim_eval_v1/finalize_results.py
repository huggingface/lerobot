from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

from verify_evidence import verify

EXPERIMENT_DIR = Path(__file__).resolve().parent
PLAN_PATH = EXPERIMENT_DIR / "evaluation_plan.json"
EXPECTED_PLAN_SHA256 = (
    "f06a201d7261f75b8dbb24a8510a9c4ab52f48c475cb2744ee2547b6f51a4aa3"
)
EXPECTED_MODEL_SHA256 = (
    "b7faae880393bdbf5e44ebeaab1f399f732d6ee325be698f999c90eb865cee68"
)
MIXED_V1_RESULT_INDEX = Path(
    "/home/ubuntu24/Teleop/lerobot/experiments/"
    "task1_picklift_mixed_act_remote_sim_eval_v1/result_index.json"
)
MIXED_V1_RESULT_INDEX_SHA256 = (
    "711e59df28ab2982652587eb56cba2df49c285e1a492f4077b33991f018e8221"
)
MIXED_V1_FROZEN_DIR = Path(
    "/home/ubuntu24/Teleop/artifacts/evaluation/"
    "task1_picklift_mixed_act_remote_sim_eval_v1/"
    "frozen120_mixed_act100k_remote_1dfac5_20260729_v1"
)
MIXED_V1_SUMMARY_SHA256 = (
    "de94cd50a02dd768b6636b521fc68b9437fdf62ea37569af86b66479f9853ee9"
)
COMPARISON_BOUNDARY = (
    "Mixed v2 versus Mixed v1 is a same-Remote-contract, single-seed, "
    "small-sample descriptive diagnostic. It does not isolate the bundled "
    "engineering changes, establish a causal effect, or constitute a "
    "real-robot or paper performance conclusion."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_index(output_dir: Path, phase: str) -> dict[str, Any]:
    recomputed = verify(output_dir, phase)
    frozen_verification = load_json(output_dir / "verification.json")
    if recomputed != frozen_verification:
        raise RuntimeError(f"{phase} verification differs from recomputation.")
    summary = load_json(output_dir / "summary.json")
    overall = summary["overall"]
    return {
        "evidence_dir": str(output_dir),
        "episodes": overall["episodes"],
        "interface_valid": overall["interface_valid_episodes"],
        "policy_ticks": overall["raw_action_count"],
        "environment_steps": overall["env_steps"],
        "official_successes": overall["successes"],
        "official_success_rate": overall["success_rate"],
        "episodes_with_any_official_success_step": len(
            overall["first_success_steps"]
        ),
        "median_first_official_success_step": (
            median(overall["first_success_steps"])
            if overall["first_success_steps"]
            else None
        ),
        "median_confirmed_success_step": (
            median(overall["confirmed_success_steps"])
            if overall["confirmed_success_steps"]
            else None
        ),
        "maximum_lift_m": overall["maximum_lift_m"],
        "final_is_grasped_episodes": overall["final_is_grasped_episodes"],
        "environment_clipped_action_count": overall[
            "environment_clipped_action_count"
        ],
        "environment_clipped_action_rate": (
            overall["environment_clipped_action_count"]
            / overall["raw_action_count"]
        ),
        "environment_clipped_joint_value_count": overall[
            "environment_clipped_joint_value_count"
        ],
        "environment_clipped_joint_value_rate": (
            overall["environment_clipped_joint_value_count"]
            / (overall["raw_action_count"] * 6)
        ),
        "failure_types": overall["failure_types"],
        "termination_reasons": overall["termination_reasons"],
        "runtime_seconds": summary["runtime_seconds"],
        "by_cell": summary["by_cell"],
        "core_hashes": recomputed["core_hashes"],
        "hashes_sha256": sha256_file(output_dir / "hashes.sha256"),
        "verification_sha256": sha256_file(output_dir / "verification.json"),
        "verification_index_sha256": sha256_file(
            output_dir / "verification.sha256"
        ),
    }


def build_result_index() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Mixed v2 plan hash mismatch.")
    plan = load_json(PLAN_PATH)
    if sha256_file(MIXED_V1_RESULT_INDEX) != MIXED_V1_RESULT_INDEX_SHA256:
        raise RuntimeError("Mixed v1 result index hash mismatch.")
    if (
        sha256_file(MIXED_V1_FROZEN_DIR / "summary.json")
        != MIXED_V1_SUMMARY_SHA256
    ):
        raise RuntimeError("Mixed v1 frozen summary hash mismatch.")
    mixed_v1_index = load_json(MIXED_V1_RESULT_INDEX)
    mixed_v1 = mixed_v1_index["frozen120"]
    if mixed_v1["official_successes"] != 52 or mixed_v1["episodes"] != 120:
        raise RuntimeError("Mixed v1 frozen result identity drifted.")

    gate = phase_index(Path(plan["phases"]["gate12"]["output_dir"]), "gate12")
    frozen = phase_index(
        Path(plan["phases"]["frozen120"]["output_dir"]),
        "frozen120",
    )
    if gate["interface_valid"] != 12 or frozen["interface_valid"] != 120:
        raise RuntimeError("Mixed v2 interface validity is incomplete.")

    mixed_v2_successes = frozen["official_successes"]
    mixed_v2_rate = frozen["official_success_rate"]
    per_cell = {}
    for cell in sorted(frozen["by_cell"]):
        v1_cell = mixed_v1["by_cell"][cell]
        v2_cell = frozen["by_cell"][cell]
        per_cell[cell] = {
            "mixed_v1_successes": v1_cell["successes"],
            "mixed_v2_successes": v2_cell["successes"],
            "success_difference_v2_minus_v1": (
                v2_cell["successes"] - v1_cell["successes"]
            ),
            "mixed_v1_success_rate": v1_cell["success_rate"],
            "mixed_v2_success_rate": v2_cell["success_rate"],
            "success_rate_difference_v2_minus_v1": (
                v2_cell["success_rate"] - v1_cell["success_rate"]
            ),
        }
    comparison = {
        "schema_version": 1,
        "status": "same_remote_contract_descriptive_diagnostic_only",
        "comparison_boundary": COMPARISON_BOUNDARY,
        "contract_relationship": plan["comparison_reference"]["relationship"],
        "mixed_v1": {
            "model_sha256": mixed_v1_index["model"]["model_sha256"],
            "official_successes": mixed_v1["official_successes"],
            "episodes": mixed_v1["episodes"],
            "official_success_rate": mixed_v1["official_success_rate"],
            "failure_types": mixed_v1["failure_types"],
            "environment_clipped_action_count": mixed_v1[
                "environment_clipped_action_count"
            ],
            "result_index_path": str(MIXED_V1_RESULT_INDEX),
            "result_index_sha256": MIXED_V1_RESULT_INDEX_SHA256,
            "frozen_summary_sha256": MIXED_V1_SUMMARY_SHA256,
        },
        "mixed_v2": {
            "model_sha256": EXPECTED_MODEL_SHA256,
            "official_successes": mixed_v2_successes,
            "episodes": frozen["episodes"],
            "official_success_rate": mixed_v2_rate,
            "failure_types": frozen["failure_types"],
            "environment_clipped_action_count": frozen[
                "environment_clipped_action_count"
            ],
        },
        "overall_difference_v2_minus_v1": {
            "successes": mixed_v2_successes - mixed_v1["official_successes"],
            "success_rate": mixed_v2_rate - mixed_v1["official_success_rate"],
            "percentage_points": 100
            * (mixed_v2_rate - mixed_v1["official_success_rate"]),
        },
        "by_cell": per_cell,
    }
    manifest = load_json(
        Path(plan["phases"]["frozen120"]["output_dir"]) / "run_manifest.json"
    )
    result_index = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "status": "complete_independently_verified_diagnostic_only",
        "source_commit": manifest["source_commit"],
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "model": {
            "checkpoint": plan["model"]["checkpoint"],
            "model_sha256": EXPECTED_MODEL_SHA256,
            "checkpoint_owned_processor": manifest[
                "checkpoint_owned_processor"
            ],
        },
        "remote": {
            "deployed_commit": plan["remote"]["deployed_commit"],
            "adapter_sha256": plan["remote"]["adapter_sha256"],
            "plan_sha256": plan["remote"]["remote_plan_sha256"],
            "reset_profile": plan["remote"]["reset_profile_id"],
            "ready_pose_profile": plan["remote"]["ready_pose_profile_id"],
            "ready_pose_state_sha256": plan["remote"][
                "ready_pose_state_sha256"
            ],
            "camera_spawn": "formal v5",
            "success": (
                "official MuJoCoPickLift-v1 info.success for 25 consecutive "
                "environment steps"
            ),
        },
        "action_contract": {
            "raw_equals_requested": True,
            "follower_calibration_state_gate": False,
            "sim_state_projection": False,
            "max_relative_target": None,
            "runner_absolute_calibration_clamp": False,
            "runner_relative_clamp": False,
            "nexus_environment_clip_preserved": True,
        },
        "gate12": gate,
        "frozen120": frozen,
        "comparison": comparison,
        "hardware_access": {
            "serial": False,
            "camera": False,
            "robot": False,
            "torque": False,
            "follower_12v": False,
            "gateway": False,
            "quest": False,
            "real": False,
        },
    }
    return result_index, comparison


def main() -> None:
    result_path = EXPERIMENT_DIR / "result_index.json"
    comparison_path = EXPERIMENT_DIR / "comparison_result.json"
    if result_path.exists() or comparison_path.exists():
        raise RuntimeError("Refusing to overwrite frozen Mixed v2 result files.")
    result, comparison = build_result_index()
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_index": str(result_path),
                "result_index_sha256": sha256_file(result_path),
                "comparison_result": str(comparison_path),
                "comparison_result_sha256": sha256_file(comparison_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
