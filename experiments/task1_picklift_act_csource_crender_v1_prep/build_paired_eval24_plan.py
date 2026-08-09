from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path("/home/ubuntu24/Teleop/lerobot")
HERE = REPO / "experiments/task1_picklift_act_csource_crender_v1_prep"
SOURCE_EXP = REPO / "experiments/task1_picklift_act_additive_three_model_200k_v1"
SOURCE_PLAN = SOURCE_EXP / "evaluation_plan.json"
SOURCE_RESULT = SOURCE_EXP / "training_result_v1.json"
OUTPUT = HERE / "bound_evaluation_plan.json"
EVALUATION_ID = "task1_picklift_act_csource_vs_crender_eval24_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-render-result", type=Path, required=True)
    parser.add_argument("--c-render-result-sha256", required=True)
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not args.c_render_result.is_file() or sha256(args.c_render_result) != args.c_render_result_sha256:
        raise RuntimeError("C-render result identity mismatch")
    render_result = json.loads(args.c_render_result.read_text())
    if render_result.get("status") != "offline_training_complete_ready_for_separately_authorized_real_eval":
        raise RuntimeError("C-render training is not complete")
    if render_result.get("selected_step") != 200000:
        raise RuntimeError("C-render selected checkpoint is not fixed step200000")
    source_result = json.loads(SOURCE_RESULT.read_text())
    source_model = source_result["models"]["C"]
    if source_model["model_sha256"] != "dd5a7002d850da8ea45dc8097a14de89e51e98432fab05dc898b35e2cc34811f":
        raise RuntimeError("C-source model drift")
    render_model = render_result["model"]
    source_plan = json.loads(SOURCE_PLAN.read_text())
    pose_rows = [source_plan["trials"][index] for index in range(0, 72, 3)]
    if len(pose_rows) != 24:
        raise RuntimeError("source Eval24 pose count mismatch")
    models = {
        "S": {**source_model, "role": "C-source"},
        "R": {**render_model, "role": "C-render"},
    }
    trials = []
    for pose_index, pose in enumerate(pose_rows, 1):
        order = ("S", "R") if pose_index % 2 else ("R", "S")
        for within, key in enumerate(order, 1):
            trial_number = len(trials) + 1
            trial_id = f"t{trial_number:03d}_h{pose_index:02d}_{key.lower()}"
            row = {key_: value for key_, value in pose.items() if key_ not in ("order", "trial_id", "artifact_stem", "spawn_region", "within_pose_order", "model_key", "model_id", "operator_placement_prompt_zh")}
            row.update({
                "order": trial_number, "trial_id": trial_id, "artifact_stem": trial_id, "spawn_region": trial_id,
                "pose_order": pose_index, "within_pose_order": within, "model_key": key,
                "model_id": models[key]["model_id"],
                "operator_placement_prompt_zh": pose["operator_placement_prompt_zh"] if within == 1 else pose["operator_placement_prompt_zh"].replace("摆放方块", "保持方块不动"),
            })
            trials.append(row)
    plan = {
        "schema_version": 1,
        "evaluation_id": EVALUATION_ID,
        "status": "software_gate_frozen_hardware_not_authorized",
        "comparison_role": "paired C-source versus RGB-only C-render engineering comparison",
        "source_identities": {
            "source_eval24_plan_sha256": sha256(SOURCE_PLAN),
            "source_training_result_sha256": sha256(SOURCE_RESULT),
            "c_render_training_result_path": str(args.c_render_result),
            "c_render_training_result_sha256": args.c_render_result_sha256,
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
        "pose_bank": {"source_plan_sha256": sha256(SOURCE_PLAN), "poses": 24, "result_used_for_selection": False},
        "model_order": {"same_pose_contiguous": True, "source_first": 12, "render_first": 12, "pattern": "SR/RS alternating"},
        "planned_valid_rollouts": 48,
        "trials": trials,
        "evidence_root": f"/home/ubuntu24/Teleop/artifacts/evaluation/{EVALUATION_ID}",
        "canonical_video_blind_review_required": True,
        "hardware_authorized": False,
    }
    OUTPUT.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(OUTPUT), "sha256": sha256(OUTPUT), "trials": len(trials), "hardware_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
