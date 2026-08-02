from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


EXPERIMENT_ID = "task1_picklift_real48_vs_real96_act_v1"
ARTIFACT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/training") / EXPERIMENT_ID
EVIDENCE_ROOT = Path("/home/ubuntu24/Teleop/artifacts/evidence") / EXPERIMENT_ID / "data_and_contract_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparable_saved_config(value: dict) -> dict:
    result = deepcopy(value)
    for key in ("repo_id", "root", "episodes"):
        result["dataset"].pop(key)
    result.pop("output_dir")
    result.pop("job_name")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify matched Real48/Real96 ACT results")
    parser.add_argument("--summary", type=Path, default=ARTIFACT_ROOT / "run_summary.json")
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT / "result_independent_verification.json")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if summary["status"] != "offline_training_and_validation_complete_no_rollout_started":
        raise RuntimeError("Run summary status is incomplete")
    dataset_audit = json.loads((EVIDENCE_ROOT / "dataset_audit.json").read_text())
    if dataset_audit["datasets"]["real48"]["tree"]["tree_sha256"] != (
        "c4534befc536c10217638da91f5cbbaff59b0795ec91f0633e53e8a6d99507b9"
    ):
        raise RuntimeError("Real48 dataset identity mismatch")
    if dataset_audit["datasets"]["real96"]["tree"]["tree_sha256"] != (
        "58a5f8fa907c6b4433750c816f0eb80743ee861b06a1dd1356811fbc6800b1a1"
    ):
        raise RuntimeError("Real96 dataset identity mismatch")
    if dataset_audit["discard_boundary"]["frames"] != {
        "raw_attempts": 17681,
        "accepted_real96": 17439,
        "discard_excluded": 242,
    }:
        raise RuntimeError("Discard-frame boundary mismatch")

    recomputed: dict[str, dict[str, str]] = {}
    saved_configs = {}
    for condition in ("real48", "real96"):
        condition_result = summary["conditions"][condition]
        if condition_result["offline_validation"]["status"] != "pass":
            raise RuntimeError(f"{condition} validation not passing")
        if condition_result["offline_validation"]["output_shape"] != [1, 6] or not condition_result["offline_validation"]["output_finite"]:
            raise RuntimeError(f"{condition} inference shape/finite mismatch")
        if condition_result["full_training"]["initialized_from_checkpoint"] is not None:
            raise RuntimeError(f"{condition} was not trained from scratch")
        recomputed[condition] = {}
        for step, identity in condition_result["full_training"]["checkpoints"].items():
            model = Path(identity["checkpoint"]) / "model.safetensors"
            observed = sha256_file(model)
            if observed != identity["model_sha256"]:
                raise RuntimeError(f"{condition} checkpoint {step} model hash mismatch")
            recomputed[condition][step] = observed
        if condition_result["full_training"]["selected_checkpoint_step"] != 100000:
            raise RuntimeError(f"{condition} selected checkpoint is not fixed step100000")
        selected = Path(condition_result["full_training"]["selected_checkpoint"])
        saved = json.loads((selected / "train_config.json").read_text())
        if saved["steps"] != 100000 or saved["seed"] != 1000 or saved["checkpoint_path"] is not None:
            raise RuntimeError(f"{condition} saved run identity mismatch")
        if saved["optimizer"]["grad_clip_norm"] != 10.0 or saved["policy"]["use_amp"]:
            raise RuntimeError(f"{condition} saved optimizer/runtime mismatch")
        saved_configs[condition] = saved
        log = Path(condition_result["full_training"]["log_path"])
        if sha256_file(log) != condition_result["full_training"]["log_sha256"] or "End of training" not in log.read_text(errors="replace"):
            raise RuntimeError(f"{condition} log identity/completion mismatch")
    if comparable_saved_config(saved_configs["real48"]) != comparable_saved_config(saved_configs["real96"]):
        raise RuntimeError("Saved full-run configurations differ beyond dataset/output identity")

    result = {
        "schema": "task1_picklift_real48_vs_real96_act_result_independent_verification_v1",
        "status": "pass",
        "dataset_hashes_and_counts": "pass",
        "discard_242_frames_excluded": True,
        "all_ten_checkpoint_model_hashes_recomputed": recomputed,
        "saved_full_configs_match_after_allowed_identity_fields": True,
        "both_selected_steps": 100000,
        "both_trained_from_scratch": True,
        "both_cuda_reload_inference_finite_shape_1x6": True,
        "hardware_accessed": False,
        "rollout_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
