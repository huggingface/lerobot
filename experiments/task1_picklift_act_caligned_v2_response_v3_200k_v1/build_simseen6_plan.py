from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path("/home/ubuntu24/Teleop/lerobot")
HERE = REPO / "experiments/task1_picklift_act_caligned_v2_response_v3_200k_v1"
SOURCE_EXP = REPO / "experiments/task1_picklift_act_additive_three_model_200k_v1"
SOURCE_PLAN = SOURCE_EXP / "evaluation_plan.json"
SOURCE_RESULT = SOURCE_EXP / "training_result_v1.json"
OUTPUT = HERE / "bound_simseen6_evaluation_plan.json"
EVALUATION_ID = "task1_picklift_csource_vs_response_v3_simseen6_paired_eval_v1"
EXPECTED_RESEARCH_SHA = "f488689e72e2f51e580f8c26a2cecffd539fcd98dca7bc7e3f4d930eac2aeaad"
EXPECTED_SOURCE_PLAN_SHA = "ad711d39df8f2e5334665add1c2fdfee00e4a57e4550fa9960cffaf32728c8c3"
EXPECTED_SOURCE_MODEL_SHA = "dd5a7002d850da8ea45dc8097a14de89e51e98432fab05dc898b35e2cc34811f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-result-sha256", required=True)
    parser.add_argument("--research-eval-contract", type=Path, required=True)
    parser.add_argument("--research-eval-contract-sha256", default=EXPECTED_RESEARCH_SHA)
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256(SOURCE_PLAN) != EXPECTED_SOURCE_PLAN_SHA:
        raise RuntimeError("frozen additive Eval24 plan drift")
    if sha256(args.research_eval_contract) != args.research_eval_contract_sha256:
        raise RuntimeError("Sim-seen6 research contract hash mismatch")
    if args.research_eval_contract_sha256 != EXPECTED_RESEARCH_SHA:
        raise RuntimeError("unexpected Sim-seen6 research identity")
    if sha256(args.training_result) != args.training_result_sha256:
        raise RuntimeError("C-aligned-v2 training result hash mismatch")
    aligned_result = json.loads(args.training_result.read_text())
    if aligned_result.get("status") != "offline_training_complete_ready_for_simseen6_software_gate":
        raise RuntimeError("C-aligned-v2 training is incomplete")
    if aligned_result.get("selected_step") != 200000:
        raise RuntimeError("C-aligned-v2 checkpoint is not fixed step200000")
    research = json.loads(args.research_eval_contract.read_text())
    if research.get("evaluation_id") != EVALUATION_ID:
        raise RuntimeError("Sim-seen6 evaluation identity mismatch")
    if research["balance_invariants"] != {
        "poses": 6,
        "unique_cells": 6,
        "yaw_counts": {"0": 3, "45": 3},
        "position_kind_counts": {"center": 3, "offset": 3},
        "models": 2,
        "valid_trials": 12,
        "first_model_counts": {"C_source": 3, "C_aligned_v2_response_v3": 3},
    }:
        raise RuntimeError("Sim-seen6 balance contract drift")
    source_result = json.loads(SOURCE_RESULT.read_text())
    source_model = source_result["models"]["C"]
    if source_model["model_sha256"] != EXPECTED_SOURCE_MODEL_SHA:
        raise RuntimeError("C-source model drift")
    source_plan = json.loads(SOURCE_PLAN.read_text())
    pose_by_id = {row["eval_pose_id"]: row for row in source_plan["trials"][::3]}
    if len(source_plan["trials"]) != 72 or len(pose_by_id) != 24:
        raise RuntimeError("frozen additive Eval24 is not 24 contiguous three-model pose groups")
    for offset in range(0, 72, 3):
        group = source_plan["trials"][offset : offset + 3]
        if len({row["eval_pose_id"] for row in group}) != 1:
            raise RuntimeError("frozen additive Eval24 pose grouping drift")
    ordered_poses = research["ordered_poses"]
    if len(ordered_poses) != 6 or len({row["eval_pose_id"] for row in ordered_poses}) != 6:
        raise RuntimeError("Sim-seen6 pose identity is not six unique poses")
    models = {
        "S": {**source_model, "role": "C_source"},
        "A": {**aligned_result["model"], "role": "C_aligned_v2_response_v3"},
    }
    trial_contract = research["ordered_trials"]
    if len(trial_contract) != 12:
        raise RuntimeError("Sim-seen6 trial contract is not paired12")
    trials = []
    for frozen_trial in trial_contract:
        pose = ordered_poses[frozen_trial["pose_order"] - 1]
        source_pose = pose_by_id.get(pose["eval_pose_id"])
        if source_pose is None:
            raise RuntimeError(f"Sim-seen6 pose missing from frozen Eval24: {pose['eval_pose_id']}")
        exact = (
            source_pose["cell"],
            source_pose["nominal_x_forward_m"],
            source_pose["nominal_y_lateral_m"],
            source_pose["nominal_yaw_degrees_modulo_90"],
        )
        expected = (pose["cell"], pose["x_m"], pose["y_m"], pose["yaw_deg"])
        if exact != expected:
            raise RuntimeError(f"Sim-seen6 pose value drift: {exact} != {expected}")
        model_key = "S" if frozen_trial["model"] == "C_source" else "A"
        row = {
            key: value
            for key, value in source_pose.items()
            if key
            not in (
                "order",
                "trial_id",
                "artifact_stem",
                "spawn_region",
                "within_pose_order",
                "model_key",
                "model_id",
            )
        }
        row.update(
            {
                "order": len(trials) + 1,
                "trial_id": frozen_trial["trial_id"],
                "artifact_stem": frozen_trial["trial_id"],
                "spawn_region": frozen_trial["trial_id"],
                "pose_order": pose["pose_order"],
                "within_pose_order": frozen_trial["pair_position"],
                "model_key": model_key,
                "model_id": models[model_key]["model_id"],
                "localsim_plan_item_id": pose["localsim_plan_item_id"],
                "operator_placement_prompt_zh": source_pose["operator_placement_prompt_zh"],
                "restore_nominal_cube_pose_before_this_trial": True,
            }
        )
        trials.append(row)
    expected_order = [("S", "A"), ("A", "S"), ("S", "A"), ("A", "S"), ("S", "A"), ("A", "S")]
    if [tuple(row["model_key"] for row in trials[index : index + 2]) for index in range(0, 12, 2)] != expected_order:
        raise RuntimeError("Sim-seen6 SA/AS frozen order mismatch")
    plan = {
        "schema_version": 1,
        "evaluation_id": EVALUATION_ID,
        "status": "software_gate_frozen_hardware_not_authorized",
        "comparison_role": "paired C-source versus response-v3 C-aligned-v2 Sim-seen6 development gate",
        "source_identities": {
            "research_eval_contract_path": str(args.research_eval_contract),
            "research_eval_contract_sha256": args.research_eval_contract_sha256,
            "source_eval24_plan_sha256": sha256(SOURCE_PLAN),
            "source_training_result_sha256": sha256(SOURCE_RESULT),
            "c_aligned_training_result_path": str(args.training_result),
            "c_aligned_training_result_sha256": args.training_result_sha256,
        },
        "models": models,
        "setup": source_plan["setup"],
        "execution_engine": source_plan["execution_engine"],
        "evaluation_profile": source_plan["evaluation_profile"],
        "success_early_stop": {
            **source_plan["success_early_stop"],
            "marker_root": f"/home/ubuntu24/Teleop/artifacts/evaluation/{EVALUATION_ID}/success_markers",
        },
        "success_contract": source_plan["success_contract"],
        "replacement_contract": source_plan["replacement_contract"],
        "pose_bank": {
            "selection": "exact frozen Eval24 intersection with response-v3 LocalSim-gap24 x/y/yaw membership",
            "poses": 6,
            "result_used_for_selection": False,
            "automatic_full_eval24_fallback": False,
        },
        "model_order": {
            "same_pose_contiguous": True,
            "restore_nominal_cube_pose_before_every_trial": True,
            "source_first": 3,
            "aligned_first": 3,
            "pattern": "SA/AS alternating",
        },
        "planned_valid_rollouts": 12,
        "trials": trials,
        "evidence_root": f"/home/ubuntu24/Teleop/artifacts/evaluation/{EVALUATION_ID}",
        "canonical_video_blind_review_required": True,
        "hardware_authorized": False,
    }
    OUTPUT.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(OUTPUT), "sha256": sha256(OUTPUT), "trials": 12, "hardware_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
