from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE / "bound_evaluation_plan.json"
TRAINING_RESULT = HERE / "training_result_v1.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    dry_path = Path(plan["evidence_root"]) / "software_preparation_v1/dry_run.json"
    dry = json.loads(dry_path.read_text())
    result = json.loads(TRAINING_RESULT.read_text())
    if len(plan["trials"]) != 48 or dry["status"] != "software_dry_run_pass_hardware_not_accessed":
        raise RuntimeError("software gate evidence incomplete")
    if result["status"] != "offline_training_complete_ready_for_separately_authorized_real_eval":
        raise RuntimeError("training result not eligible")
    first = plan["trials"][0]
    output = dry_path.parent
    manifest = {
        "schema": "task1_act_csource_crender_paired_eval24_software_gate_v1",
        "status": "pass_hardware_not_authorized",
        "evaluation_id": plan["evaluation_id"],
        "plan": {"path": str(PLAN), "sha256": sha(PLAN)},
        "training_result": {"path": str(TRAINING_RESULT), "sha256": sha(TRAINING_RESULT)},
        "dry_run": {"path": str(dry_path), "sha256": sha(dry_path)},
        "trials": 48,
        "poses": 24,
        "models": {key: value["model_sha256"] for key, value in plan["models"].items()},
        "model_order": plan["model_order"],
        "success_early_stop": plan["success_early_stop"],
        "first_trial": first,
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
