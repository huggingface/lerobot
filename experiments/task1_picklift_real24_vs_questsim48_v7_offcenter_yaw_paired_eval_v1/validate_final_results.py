from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/evaluation/"
    "task1_picklift_real24_vs_questsim48_v7_offcenter_yaw_paired_eval24_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash_list(path: Path) -> int:
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        digest, raw_path = raw.split("  ", 1)
        artifact = Path(raw_path)
        if not artifact.exists() or sha256_file(artifact) != digest:
            raise RuntimeError(f"Hash verification failed: {artifact}")
        count += 1
    return count


def main() -> None:
    operator_manifest = json.loads((ROOT / "operator_manifest_v1.json").read_text())
    operator_summary = json.loads((ROOT / "operator_summary_v1.json").read_text())
    review_root = ROOT / "canonical_video_review_v1"
    review_manifest = json.loads((review_root / "manifest.json").read_text())
    review_summary = json.loads((review_root / "summary.json").read_text())
    review_rows = [
        json.loads(raw)
        for raw in (review_root / "trials.jsonl").read_text().splitlines()
    ]
    if len(operator_manifest["scored_trials"]) != 24 or len(review_rows) != 24:
        raise RuntimeError("Final result does not contain exactly 24 scored rows.")
    expected = {
        "real24_only": (0, 12),
        "questsim48_v7": (3, 12),
    }
    for model_id, (successes, trials) in expected.items():
        operator = operator_summary["by_model"][model_id]
        review = review_summary["by_model"][model_id]
        if (operator["operator_successes"], operator["trials"]) != (successes, trials):
            raise RuntimeError(f"Operator summary mismatch for {model_id}.")
        if (review["reviewed_successes"], review["trials"]) != (successes, trials):
            raise RuntimeError(f"Review summary mismatch for {model_id}.")
    if review_summary["operator_review_disagreements"]:
        raise RuntimeError("Unexpected operator/review disagreement.")
    if review_summary["video_contract"] != {
        "videos": 24,
        "width": 640,
        "height": 480,
        "encoded_fps": 20,
        "all_frame_counts_match_steps": True,
        "all_policy_windows_reach_at_least_29p9_seconds": True,
    }:
        raise RuntimeError("Canonical video contract mismatch.")
    if review_manifest["status"] != "canonical_video_review_frozen":
        raise RuntimeError("Canonical-video review is not frozen.")
    result = {
        "status": "independent_final_validation_passed",
        "evaluation_id": operator_manifest["evaluation_id"],
        "operator_hash_entries": verify_hash_list(
            ROOT / "operator_evidence_hashes_v1.sha256"
        ),
        "review_hash_entries": verify_hash_list(review_root / "hashes.sha256"),
        "scored_trials": 24,
        "infrastructure_invalid_originals": len(
            operator_manifest["infrastructure_invalid_originals"]
        ),
        "reviewed_successes": {"real24_only": 0, "questsim48_v7": 3},
        "operator_review_disagreements": 0,
        "operator_manifest_sha256": sha256_file(ROOT / "operator_manifest_v1.json"),
        "operator_summary_sha256": sha256_file(ROOT / "operator_summary_v1.json"),
        "operator_hashes_sha256": sha256_file(ROOT / "operator_evidence_hashes_v1.sha256"),
        "review_manifest_sha256": sha256_file(review_root / "manifest.json"),
        "review_summary_sha256": sha256_file(review_root / "summary.json"),
        "review_trials_sha256": sha256_file(review_root / "trials.jsonl"),
        "review_hashes_sha256": sha256_file(review_root / "hashes.sha256"),
    }
    output = ROOT / "final_validation_v1.json"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite final validation: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "output_sha256": sha256_file(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
