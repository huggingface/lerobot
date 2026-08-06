from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PLAN = HERE / "evaluation_plan.json"
BASE_PATH = REPO / "experiments/task1_picklift_real24_localsim24gap_vs_localsim48full_eval48_v1/paired_evaluator.py"
PROFILE_SHA = "6b031bb4c980467addb3e69d68a16032ceae7e45fb3f8e2288d8a4989ff3cbf3"
CALIBRATION_SHA = "c78e4f7e1383571c6aa496f62996f518b3e4122f78244d2bbc094658bc0cb8a0"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_base():
    spec = importlib.util.spec_from_file_location("task1_eval24_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan["status"] != "software_gate_frozen_hardware_not_authorized" or plan["hardware_authorized"] is not False:
        raise RuntimeError("Plan is not at the software-only gate")
    if len(plan["trials"]) != 72 or [t["order"] for t in plan["trials"]] != list(range(1, 73)):
        raise RuntimeError("Expected 72 contiguous trials")
    pose_groups = [plan["trials"][i:i+3] for i in range(0, 72, 3)]
    if any(len({t["eval_pose_id"] for t in group}) != 1 for group in pose_groups):
        raise RuntimeError("Each pose must run three models consecutively")
    expected = ["ABC", "BCA", "CAB", "ACB", "CBA", "BAC"] * 4
    if ["".join(t["model_key"] for t in group) for group in pose_groups] != expected:
        raise RuntimeError("Frozen six-permutation model order changed")
    positions = {key: [0, 0, 0] for key in "ABC"}
    for group in pose_groups:
        for pos, trial in enumerate(group): positions[trial["model_key"]][pos] += 1
    if any(value != [8, 8, 8] for value in positions.values()):
        raise RuntimeError("Model positions are not balanced 8/8/8")
    return plan


def verify_static(plan: dict) -> dict:
    if sha(REPO / plan["execution_engine"]["path"]) != plan["execution_engine"]["source_sha256"]:
        raise RuntimeError("Official-send engine hash mismatch")
    profile = REPO / plan["evaluation_profile"]["path"]
    if sha(profile) != PROFILE_SHA or plan["evaluation_profile"]["sha256"] != PROFILE_SHA:
        raise RuntimeError("Evaluation profile mismatch")
    calibration = Path(plan["setup"]["follower_calibration_path"])
    if sha(calibration) != CALIBRATION_SHA:
        raise RuntimeError("Follower calibration drift")
    if plan["setup"]["max_relative_target"] is not None or plan["setup"]["custom_absolute_action_clamp"] is not False:
        raise RuntimeError("Official-send action path changed")
    if plan["setup"]["control_fps"] != 20 or plan["setup"]["maximum_trial_seconds"] != 30:
        raise RuntimeError("Timing changed")
    models = {}
    for key, model in plan["models"].items():
        ckpt = Path(model["checkpoint"])
        paths = {
            "model_sha256": ckpt / "model.safetensors", "config_sha256": ckpt / "config.json",
            "train_config_sha256": ckpt / "train_config.json",
            "policy_preprocessor_sha256": ckpt / "policy_preprocessor.json",
            "processor_stats_sha256": ckpt / "policy_preprocessor_step_3_normalizer_processor.safetensors",
        }
        for field, path in paths.items():
            if sha(path) != model[field]: raise RuntimeError(f"{key} {field} mismatch")
        cfg = json.loads((ckpt / "config.json").read_text())
        if cfg["type"] != "act" or cfg["chunk_size"] != 67 or cfg["n_action_steps"] != 67:
            raise RuntimeError(f"{key} ACT configuration mismatch")
        stats = load_file(paths["processor_stats_sha256"])
        if not np.allclose(stats["observation.images.front.mean"].numpy().reshape(-1), [.485,.456,.406], atol=1e-7):
            raise RuntimeError(f"{key} visual normalization mismatch")
        models[key] = {field: sha(path) for field, path in paths.items()}
    return {"status": "pass", "models": models, "hardware_accessed": False}


def configure_base(base, plan: dict, plan_path: Path) -> None:
    base.MODEL_IDS = tuple(plan["models"])
    base.EXPECTED_MODEL_SHA256 = {k: v["model_sha256"] for k, v in plan["models"].items()}
    base.EXPECTED_PLAN_SHA256 = sha(plan_path)
    base.EXPECTED_ENGINE_SHA256 = plan["execution_engine"]["source_sha256"]
    base.CURRENT_COMPATIBLE_ENGINE_SHA256 = plan["execution_engine"]["source_sha256"]
    base.EXPECTED_PROFILE_SHA256 = PROFILE_SHA
    base.EXPECTED_EVALUATION_ID = plan["evaluation_id"]
    base.verify_static_files = verify_static
    def write_sidecar(plan_: dict, trial: dict, artifact_stem: str, replacement_for: str | None) -> None:
        root = Path(plan_["evidence_root"]) / "trials"
        evidence_path = root / f"{artifact_stem}.json"
        if not evidence_path.exists(): return
        sidecar_path = root / f"{artifact_stem}.eval24.json"
        if sidecar_path.exists(): return
        evidence = json.loads(evidence_path.read_text())
        pre_action = base.build_pre_action_evidence(evidence, trial, root, artifact_stem)
        sidecar = {"schema":"task1_additive_three_model_eval24_trial_sidecar_v1",
            "evaluation_id":plan_["evaluation_id"],"trial":trial,"artifact_stem":artifact_stem,
            "replacement_for":replacement_for,"engine_evidence":{"path":str(evidence_path),"sha256":sha(evidence_path)},
            "actual_policy_ticks":evidence["steps_jsonl"]["lines"],
            "wall_duration_seconds":(datetime.fromisoformat(evidence["ended_at_utc"])-datetime.fromisoformat(evidence["started_at_utc"])).total_seconds(),
            "pre_action_frame":pre_action,"operator_label":{"status":"pending"},
            "canonical_video_review_label":{"status":"pending"},"success_contract":plan_["success_contract"],
            "return":evidence["automatic_return"],"torque_disable_verified":evidence["torque_disable_verified"]}
        sidecar_path.write_text(json.dumps(sidecar,indent=2,sort_keys=True)+"\n")
    base.write_eval48_sidecar = write_sidecar


def dry_run(base, plan: dict) -> dict:
    static = verify_static(plan)
    fake = base.run_fake_protocol(plan)
    ready = base.run_fake_interpolated_ready_probe(plan)
    if fake["trials_exercised"] != 72 or fake["policy_reset_calls"] != {k: 24 for k in "ABC"}:
        raise RuntimeError("72-trial fake protocol mismatch")
    if not all((fake["all_ready_before_policy"], fake["all_ready_after_trial"],
                fake["all_canonical_rgb_640x480"], fake["all_official_sent_equals_requested"],
                fake["all_torque_disabled"], ready["commands_sent"] == 60)):
        raise RuntimeError("Fake protocol invariant failed")
    return {"status": "software_dry_run_pass_hardware_not_accessed", "static": static,
            "trials": 72, "model_reset_calls": fake["policy_reset_calls"],
            "ready_return": ready, "hardware_access": {"serial": False, "camera": False,
            "robot": False, "torque": False, "rollout": False}}


def main() -> None:
    ap = argparse.ArgumentParser(); mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--software-dry-run", action="store_true"); mode.add_argument("--execute-hardware", action="store_true")
    ap.add_argument("--plan", type=Path, default=PLAN); ap.add_argument("--trial-id")
    ap.add_argument("--replacement", action="store_true"); ap.add_argument("--operator-confirmed-ready", action="store_true")
    ap.add_argument("--follower-port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110904-if00")
    ap.add_argument("--camera-device", default="/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0")
    ap.add_argument("--calibration", type=Path, default=Path("/home/ubuntu24/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_main.json"))
    args = ap.parse_args(); plan = load_plan(args.plan); base = load_base(); configure_base(base, plan, args.plan)
    if args.software_dry_run:
        result = dry_run(base, plan)
        out = Path(plan["evidence_root"]) / "software_preparation_v1/dry_run.json"
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
        print(json.dumps(result, indent=2, sort_keys=True)); return
    base.execute_hardware(args, plan)


if __name__ == "__main__": main()
