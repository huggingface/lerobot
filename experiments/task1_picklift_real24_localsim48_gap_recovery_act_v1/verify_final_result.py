from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPERIMENT_ID = "task1_picklift_real24_localsim48_gap_recovery_act_v1"
REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / EXPERIMENT_ID
RESULT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/evidence") / EXPERIMENT_ID / "training_result_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash_entry(entry: dict) -> None:
    path = Path(entry["path"])
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise RuntimeError(f"Frozen evidence mismatch: {path}")


def verify_condition(condition_id: str, condition: dict) -> int:
    checked = 0
    if condition["status"] != "offline_training_and_validation_complete_no_rollout_started":
        raise RuntimeError(f"Condition {condition_id} is incomplete")
    for entry in condition["configs"].values():
        verify_hash_entry(entry)
        checked += 1
    for phase in ("smoke", "full_training"):
        section = condition[phase]
        for key in ("log", "domain_counts_file", "offline_validation"):
            verify_hash_entry(section[key])
            checked += 1
        stored_counts = json.loads(Path(section["domain_counts_file"]["path"]).read_text(encoding="utf-8"))
        if stored_counts != section["domain_counts"]:
            raise RuntimeError(f"Condition {condition_id} {phase} count payload mismatch")
        validation = json.loads(Path(section["offline_validation"]["path"]).read_text(encoding="utf-8"))
        if validation["status"] != "pass" or not validation["all_outputs_shape_1x6_and_finite"]:
            raise RuntimeError(f"Condition {condition_id} {phase} validation is not pass")
        if [sample["domain"] for sample in validation["samples"]] != [
            "real",
            "simulation",
        ] or not all(sample["output_finite"] for sample in validation["samples"]):
            raise RuntimeError(f"Condition {condition_id} {phase} domains are invalid")
    expected_counts = {
        "smoke": {"real": 2000, "simulation": 2000},
        "full_training": {"real": 400000, "simulation": 400000},
    }
    for phase, expected in expected_counts.items():
        if condition[phase]["domain_counts"]["actual_samples_seen_by_main_process"] != expected:
            raise RuntimeError(f"Condition {condition_id} {phase} sampling count mismatch")
    full = condition["full_training"]
    verify_hash_entry(full["saved_train_config"])
    checked += 1
    checkpoint = Path(full["selected_checkpoint"])
    saved_config = json.loads((checkpoint / "train_config.json").read_text(encoding="utf-8"))
    expected_config = json.loads(
        (EXPERIMENT_ROOT / "configs" / f"{condition_id}_full.json").read_text(encoding="utf-8")
    )
    contract_pairs = (
        (saved_config["dataset"]["repo_id"], expected_config["dataset"]["repo_id"]),
        (saved_config["dataset"]["root"], expected_config["dataset"]["root"]),
        (
            saved_config["dataset"]["domain_balanced_episode_groups"],
            expected_config["dataset"]["domain_balanced_episode_groups"],
        ),
        (saved_config["dataset"]["use_imagenet_stats"], True),
        (saved_config["seed"], 1000),
        (saved_config["steps"], 100000),
        (saved_config["batch_size"], 8),
        (saved_config["resume"], False),
        (saved_config["output_dir"], expected_config["output_dir"]),
        (saved_config["policy"]["type"], "act"),
        (saved_config["policy"]["chunk_size"], 67),
        (saved_config["policy"]["n_action_steps"], 67),
        (saved_config["policy"]["pretrained_path"], None),
    )
    if any(actual != expected for actual, expected in contract_pairs):
        raise RuntimeError(f"Condition {condition_id} saved training contract mismatch")
    for step, expected_sha in full["checkpoints"].items():
        model = checkpoint.parents[1] / step / "pretrained_model" / "model.safetensors"
        if sha256_file(model) != expected_sha:
            raise RuntimeError(f"Condition {condition_id} checkpoint {step} mismatch")
        checked += 1
    selected_model = checkpoint / "model.safetensors"
    if sha256_file(selected_model) != full["selected_model_sha256"]:
        raise RuntimeError(f"Condition {condition_id} selected model SHA mismatch")
    processor = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    if sha256_file(processor) != full["processor_sha256"]:
        raise RuntimeError(f"Condition {condition_id} processor SHA mismatch")
    full_validation = json.loads(Path(full["offline_validation"]["path"]).read_text(encoding="utf-8"))
    if (
        full_validation["model_sha256"] != full["selected_model_sha256"]
        or full_validation["processor_sha256"] != full["processor_sha256"]
    ):
        raise RuntimeError(f"Condition {condition_id} validation/checkpoint mismatch")
    return checked


def main() -> None:
    hashes_path = RESULT_ROOT / "hashes.json"
    manifest_path = RESULT_ROOT / "manifest.json"
    summary_path = RESULT_ROOT / "run_summary.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    checked = 0
    for entry in hashes["entries"].values():
        verify_hash_entry(entry)
        checked += 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest["hashes_sha256"] != sha256_file(hashes_path) or manifest["run_summary_sha256"] != sha256_file(
        summary_path
    ):
        raise RuntimeError("Manifest does not bind hashes/run summary")
    result_index = json.loads((EXPERIMENT_ROOT / "result_index.json").read_text(encoding="utf-8"))
    if result_index["manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("Result index does not bind manifest")
    if manifest["status"] != "frozen_both_models_offline_complete":
        raise RuntimeError("Pair result manifest is not complete")
    if summary["status"] != "both_models_offline_training_and_validation_complete":
        raise RuntimeError("Pair run summary is not complete")
    for condition_id in ("C", "D"):
        condition = summary["conditions"][condition_id]
        condition_path = RESULT_ROOT / f"condition_{condition_id}.json"
        if json.loads(condition_path.read_text(encoding="utf-8")) != condition:
            raise RuntimeError(f"Condition {condition_id} summary/file mismatch")
        checked += verify_condition(condition_id, condition)
        if (
            manifest[f"condition_{condition_id}_model_sha256"]
            != condition["full_training"]["selected_model_sha256"]
        ):
            raise RuntimeError(f"Manifest model binding mismatch for condition {condition_id}")
    result = {
        "status": "pass",
        "checked_hash_entries": checked,
        "conditions": ["C", "D"],
        "both_selected_checkpoints_match": True,
        "both_domain_sampling_counts_match": True,
        "hardware_accessed": False,
        "rollout_started": False,
    }
    output = RESULT_ROOT / "independent_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
