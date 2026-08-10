from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE / "bound_simseen6_evaluation_plan.json"
TRAINING_RESULT = HERE / "training_result_v1.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    dry_path = Path(plan["evidence_root"]) / "software_preparation_v1/dry_run.json"
    dry = json.loads(dry_path.read_text())
    result = json.loads(TRAINING_RESULT.read_text())
    if len(plan["trials"]) != 12 or dry["status"] != "software_dry_run_pass_hardware_not_accessed":
        raise RuntimeError("Sim-seen6 software gate evidence incomplete")
    if result["status"] != "offline_training_complete_ready_for_simseen6_software_gate":
        raise RuntimeError("C-aligned-v2 training result is not eligible")
    if plan["pose_bank"]["automatic_full_eval24_fallback"] is not False:
        raise RuntimeError("forbidden full Eval24 fallback")
    output = dry_path.parent
    if (output / "manifest.json").exists():
        raise FileExistsError(output / "manifest.json")
    manifest = {
        "schema": "task1_act_csource_caligned_response_v3_simseen6_software_gate_v1",
        "status": "pass_hardware_not_authorized",
        "evaluation_id": plan["evaluation_id"],
        "plan": {"path": str(PLAN), "sha256": sha(PLAN)},
        "training_result": {"path": str(TRAINING_RESULT), "sha256": sha(TRAINING_RESULT)},
        "dry_run": {"path": str(dry_path), "sha256": sha(dry_path)},
        "trials": 12,
        "poses": 6,
        "models": {key: value["model_sha256"] for key, value in plan["models"].items()},
        "model_order": plan["model_order"],
        "success_early_stop": plan["success_early_stop"],
        "first_trial": plan["trials"][0],
        "automatic_full_eval24_fallback": False,
        "hardware_access": {"serial": False, "camera": False, "robot": False, "torque": False, "rollout": False},
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    files = [PLAN, TRAINING_RESULT, dry_path, manifest_path]
    (output / "hashes.sha256").write_text("".join(f"{sha(path)}  {path}\n" for path in files))
    (HERE / "software_gate_result_index.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
