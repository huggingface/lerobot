from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULT = HERE / "training_result_v1.json"
HALF = HERE / "research_control/task1-picklift-eval48-half24-poses-v1.json"
SOURCE_PLAN = REPO / "experiments/task1_picklift_real48_vs_real96_eval48_v1/evaluation_plan.json"
PROFILE = REPO / "experiments/task1_picklift_real48_vs_real96_eval48_v1/real_evaluation_profile.json"
EARLY = REPO / "experiments/task1_picklift_real24_act_v1/real_evaluation_success_early_stop_profile_v1.json"
OUTPUT = HERE / "evaluation_plan.json"
EVALUATION_ID = "task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1"
PERMS = ("ABC", "BCA", "CAB", "ACB", "CBA", "BAC")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prompt(pose: dict, keep: bool) -> str:
    prefix = "保持方块不动" if keep else "摆放方块"
    yaw = "边与网格线平行（0°）" if pose["nominal_yaw_degrees_modulo_90"] == 0 else "绕中心转45°"
    return (f"{prefix}：{pose['cell']}（第{pose['row']}行第{pose['column']}列），红块中心放在"
            f"X={pose['nominal_x_forward_m']*100:g} cm、Y={pose['nominal_y_lateral_m']*100:+g} cm；{yaw}。")


def main() -> None:
    if sha(HALF) != "5a61888b54ef99b5daf1e7875319b41ce6fa410aa3550f38eef14dbd3f60e7d7":
        raise RuntimeError("Frozen half-bank hash mismatch")
    if sha(SOURCE_PLAN) != "7de77eed859898a6397265244ab2f4c189d91dbb565202b99a0a5bdd208214f1":
        raise RuntimeError("Source Eval48 plan mismatch")
    result = json.loads(RESULT.read_text())
    if result["status"] != "three_models_offline_training_complete":
        raise RuntimeError("Training result is not complete")
    half = json.loads(HALF.read_text())
    source = json.loads(SOURCE_PLAN.read_text())
    source_by_order = {int(t["pose_order"]): t for t in source["trials"][::2]}
    models = result["models"]
    if tuple(models) != ("A", "B", "C"):
        raise RuntimeError("Expected fixed A/B/C model result")
    trials = []
    for pose_index, selected in enumerate(half["ordered_selected_poses"], start=1):
        source_pose = source_by_order[int(selected["source_order"])]
        permutation = PERMS[(pose_index - 1) % len(PERMS)]
        for within, key in enumerate(permutation, start=1):
            order = len(trials) + 1
            trial_id = f"t{order:03d}_h{pose_index:02d}_{key.lower()}"
            row, column = int(selected["cell"][1]), int(selected["cell"][3])
            trial = {
                "order": order, "trial_id": trial_id, "artifact_stem": trial_id, "spawn_region": trial_id,
                "pose_order": pose_index, "source_eval48_pose_order": selected["source_order"],
                "within_pose_order": within, "model_key": key, "model_id": models[key]["model_id"],
                "eval_pose_id": selected["eval_pose_id"], "cell": selected["cell"], "row": row, "column": column,
                "coverage_tier": selected["coverage_tier"], "quadrant": source_pose.get("quadrant"),
                "nominal_x_forward_m": source_pose["nominal_x_forward_m"],
                "nominal_y_lateral_m": source_pose["nominal_y_lateral_m"],
                "nominal_yaw_degrees_modulo_90": selected["yaw"],
                "operator_placement_prompt_zh": "", "manual_pose_is_measurement_truth": False,
                "policy_failure_retry_allowed": False,
            }
            trial["operator_placement_prompt_zh"] = prompt(trial, keep=within > 1)
            trials.append(trial)
    base_setup = source["setup"]
    plan = {
        "schema_version": 1, "evaluation_id": EVALUATION_ID,
        "status": "software_gate_frozen_hardware_not_authorized",
        "comparison_role": "matched additive data ACT200k minimal Eval24 engineering comparison",
        "research_contract": {
            "research_repo_commit": "cc252373b71ad38032e2a5e418fbe57f1efa541d",
            "half_bank_sha256": sha(HALF), "source_eval48_plan_sha256": sha(SOURCE_PLAN),
            "training_result_sha256": sha(RESULT), "result_data_used_for_pose_selection": False,
        },
        "execution_engine": {"path": "experiments/task1_picklift_real24_act_v1/evaluate_real.py",
                             "source_sha256": sha(REPO / "experiments/task1_picklift_real24_act_v1/evaluate_real.py")},
        "evaluation_profile": {"path": str(PROFILE.relative_to(REPO)),
            "profile_id": "task1_real48_vs_real96_eval48_official_send_interpolated3s_tolerance3_v1",
            "sha256": sha(PROFILE)},
        "evidence_root": f"/home/ubuntu24/Teleop/artifacts/evaluation/{EVALUATION_ID}",
        "models": models, "setup": base_setup,
        "success_early_stop": {"enabled": True, "explicit_opt_in": True,
            "profile_path": str(EARLY.relative_to(REPO)), "profile_sha256": sha(EARLY),
            "marker_root": f"/home/ubuntu24/Teleop/artifacts/evaluation/{EVALUATION_ID}/success_markers"},
        "success_contract": {"within_scored_window": True, "bilateral_finger_grasp": True,
            "unsupported_lift_strictly_greater_than_m": 0.05, "continuous_hold_seconds_minimum": 0.5,
            "must_remain_held_until_timeout": False, "operator_label_required": True,
            "canonical_video_review_required": True, "success_early_stop_enabled": True,
            "changes_policy_action_window": True},
        "replacement_contract": {"model_or_task_failure_retry_allowed": False,
            "maximum_linked_replacements_per_original": 1,
            "allowed_only_for": ["policy_window_never_started", "confirmed_operator_placement_error", "infrastructure_error"],
            "original_evidence_preserved": True},
        "balance": half["balance_invariants"], "model_order": half["model_order_contract"],
        "planned_valid_rollouts": 72, "trials": trials,
        "hardware_authorized": False,
    }
    OUTPUT.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"plan": str(OUTPUT), "sha256": sha(OUTPUT), "trials": len(trials)}, indent=2))


if __name__ == "__main__": main()
