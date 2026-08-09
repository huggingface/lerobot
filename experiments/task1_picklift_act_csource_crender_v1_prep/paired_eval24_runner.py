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
PLAN = HERE / "bound_evaluation_plan.json"
BASE_PATH = REPO / "experiments/task1_picklift_real24_localsim24gap_vs_localsim48full_eval48_v1/paired_evaluator.py"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base():
    spec = importlib.util.spec_from_file_location("csource_crender_eval_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan["status"] != "software_gate_frozen_hardware_not_authorized" or plan["hardware_authorized"] is not False:
        raise RuntimeError("plan is not at the software-only gate")
    trials = plan["trials"]
    if len(trials) != 48 or [row["order"] for row in trials] != list(range(1, 49)):
        raise RuntimeError("expected 48 contiguous trials")
    pairs = [trials[index:index + 2] for index in range(0, 48, 2)]
    if any(len({row["eval_pose_id"] for row in pair}) != 1 for pair in pairs):
        raise RuntimeError("each pose must run as one contiguous pair")
    expected = [("S", "R") if pose % 2 else ("R", "S") for pose in range(1, 25)]
    if [tuple(row["model_key"] for row in pair) for pair in pairs] != expected:
        raise RuntimeError("SR/RS frozen order mismatch")
    if {key: sum(row["model_key"] == key for row in trials) for key in ("S", "R")} != {"S": 24, "R": 24}:
        raise RuntimeError("model trial counts mismatch")
    return plan


def verify_static(plan: dict) -> dict:
    engine = REPO / plan["execution_engine"]["path"]
    if sha(engine) != plan["execution_engine"]["source_sha256"]:
        raise RuntimeError("official-send engine hash mismatch")
    profile = REPO / plan["evaluation_profile"]["path"]
    if sha(profile) != plan["evaluation_profile"]["sha256"]:
        raise RuntimeError("evaluation profile hash mismatch")
    early = REPO / plan["success_early_stop"]["profile_path"]
    if sha(early) != plan["success_early_stop"]["profile_sha256"]:
        raise RuntimeError("success early-stop profile mismatch")
    setup = plan["setup"]
    if setup["max_relative_target"] is not None or setup["custom_absolute_action_clamp"] is not False:
        raise RuntimeError("official-send action path changed")
    if setup["custom_relative_step_limit_degrees"] is not None or setup["control_fps"] != 20:
        raise RuntimeError("timing/action contract changed")
    models = {}
    for key, model in plan["models"].items():
        checkpoint = Path(model["checkpoint"])
        paths = {
            "model_sha256": checkpoint / "model.safetensors",
            "config_sha256": checkpoint / "config.json",
            "train_config_sha256": checkpoint / "train_config.json",
            "policy_preprocessor_sha256": checkpoint / "policy_preprocessor.json",
            "processor_stats_sha256": checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors",
        }
        for field, file in paths.items():
            if sha(file) != model[field]:
                raise RuntimeError(f"{key} {field} mismatch")
        config = json.loads((checkpoint / "config.json").read_text())
        if config["type"] != "act" or config["chunk_size"] != 67 or config["n_action_steps"] != 67:
            raise RuntimeError(f"{key} ACT contract mismatch")
        tensors = load_file(paths["processor_stats_sha256"])
        if not np.allclose(tensors["observation.images.front.mean"].numpy().reshape(-1), [.485, .456, .406], atol=1e-7):
            raise RuntimeError(f"{key} ImageNet stats mismatch")
        models[key] = {field: sha(file) for field, file in paths.items()}
    return {"status": "pass", "models": models, "hardware_accessed": False}


def configure_base(base, plan: dict, plan_path: Path) -> None:
    base.MODEL_IDS = tuple(plan["models"])
    base.EXPECTED_MODEL_SHA256 = {key: value["model_sha256"] for key, value in plan["models"].items()}
    base.EXPECTED_PLAN_SHA256 = sha(plan_path)
    base.EXPECTED_ENGINE_SHA256 = plan["execution_engine"]["source_sha256"]
    base.CURRENT_COMPATIBLE_ENGINE_SHA256 = plan["execution_engine"]["source_sha256"]
    base.EXPECTED_PROFILE_SHA256 = plan["evaluation_profile"]["sha256"]
    base.EXPECTED_EVALUATION_ID = plan["evaluation_id"]
    base.verify_static_files = verify_static


def dry_run(base, plan: dict) -> dict:
    static = verify_static(plan)
    fake = base.run_fake_protocol(plan)
    ready = base.run_fake_interpolated_ready_probe(plan)
    if fake["trials_exercised"] != 48 or fake["policy_reset_calls"] != {"S": 24, "R": 24}:
        raise RuntimeError("48-trial fake protocol mismatch")
    if not all((fake["all_ready_before_policy"], fake["all_ready_after_trial"],
                fake["all_canonical_rgb_640x480"], fake["all_official_sent_equals_requested"],
                fake["all_torque_disabled"], ready["commands_sent"] == 60)):
        raise RuntimeError("fake protocol invariant failed")
    return {
        "status": "software_dry_run_pass_hardware_not_accessed",
        "static": static,
        "trials": 48,
        "model_reset_calls": fake["policy_reset_calls"],
        "ready_return": ready,
        "success_early_stop_symmetric": plan["success_early_stop"]["enabled"] is True,
        "hardware_access": {"serial": False, "camera": False, "robot": False, "torque": False, "rollout": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--software-dry-run", action="store_true")
    mode.add_argument("--execute-hardware", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--trial-id")
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--operator-confirmed-ready", action="store_true")
    parser.add_argument("--follower-port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110904-if00")
    parser.add_argument("--camera-device", default="/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0")
    parser.add_argument("--calibration", type=Path, default=Path("/home/ubuntu24/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_main.json"))
    args = parser.parse_args()
    plan = load_plan(args.plan)
    base = load_base()
    configure_base(base, plan, args.plan)
    if args.software_dry_run:
        result = dry_run(base, plan)
        output = Path(plan["evidence_root"]) / "software_preparation_v1/dry_run.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    base.execute_hardware(args, plan)


if __name__ == "__main__":
    main()
