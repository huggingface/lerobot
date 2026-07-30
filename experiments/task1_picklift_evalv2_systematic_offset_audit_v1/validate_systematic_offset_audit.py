from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


AUDIT_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/analysis/"
    "task1_picklift_evalv2_systematic_offset_audit_v1"
)
EXPECTED_AUDIT_ID = "task1_picklift_evalv2_systematic_offset_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    output_path = AUDIT_ROOT / "independent_validation.json"
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite validation: {output_path}")
    manifest = read_json(AUDIT_ROOT / "audit_manifest.json")
    summary = read_json(AUDIT_ROOT / "summary.json")
    trials = read_jsonl(AUDIT_ROOT / "trials.jsonl")
    training = read_jsonl(AUDIT_ROOT / "real24_training_coverage.jsonl")
    if manifest["audit_id"] != EXPECTED_AUDIT_ID or summary["audit_id"] != EXPECTED_AUDIT_ID:
        raise RuntimeError("Audit identity mismatch.")

    hash_rows = []
    for line in (AUDIT_ROOT / "hashes.sha256").read_text(encoding="utf-8").splitlines():
        digest, path = line.split("  ", 1)
        target = Path(path)
        if sha256_file(target) != digest:
            raise RuntimeError(f"Frozen output hash mismatch: {target}")
        hash_rows.append((digest, target))
    if len(trials) != 12 or [row["order"] for row in trials] != list(range(1, 13)):
        raise RuntimeError("Trial order/count mismatch.")
    if len(training) != 24 or [row["episode_index"] for row in training] != list(range(24)):
        raise RuntimeError("Training coverage count mismatch.")
    if sum(row["review_success"] for row in trials) != 1:
        raise RuntimeError("Reviewed success count mismatch.")
    if "__replacement1" not in trials[7]["artifact_stem"]:
        raise RuntimeError("Scored t08 is not the linked replacement.")
    if any(row["upstream_action_modified_events"] != 0 for row in trials):
        raise RuntimeError("Unexpected action modification evidence.")

    placement_px = np.asarray(
        [row["expected_visible_centroid_residual_norm_px"] for row in trials]
    )
    placement_m = np.asarray(
        [row["visible_nominal_task_residual_norm_m"] for row in trials]
    )
    nearest = np.asarray(
        [
            row["training_coverage"]["nearest_real24_early_frame_distance_px"]
            for row in trials
        ]
    )
    policy = np.asarray(
        [row["demo_calibrated_policy_residual_xy_m"] for row in trials]
    )
    checks = {
        "audit_id": True,
        "frozen_output_hashes": len(hash_rows),
        "trial_count_and_order": 12,
        "training_episode_count_and_order": 24,
        "reviewed_result_1_of_12": True,
        "t08_replacement_only": True,
        "action_modification_events_zero": True,
        "placement_median_px": close(
            float(np.median(placement_px)),
            summary["placement_and_grid_mapping"]["median_expected_visible_residual_px"],
        ),
        "placement_maximum_px": close(
            float(np.max(placement_px)),
            summary["placement_and_grid_mapping"]["maximum_expected_visible_residual_px"],
        ),
        "placement_median_m": close(
            float(np.median(placement_m)),
            summary["placement_and_grid_mapping"]["median_approx_task_residual_m"],
        ),
        "training_nearest_median_px": close(
            float(np.median(nearest)),
            summary["real24_image_position_coverage"]["eval_nearest_early_distance_px"][
                "median"
            ],
        ),
        "training_nearest_maximum_px": close(
            float(np.max(nearest)),
            summary["real24_image_position_coverage"]["eval_nearest_early_distance_px"][
                "maximum"
            ],
        ),
        "training_outside_support_count": (
            sum(
                row["training_coverage"]["outside_sampled_early_support_radius"]
                for row in trials
            )
            == summary["real24_image_position_coverage"][
                "outside_sampled_early_support_trials"
            ]
        ),
        "policy_residual_median_norm": close(
            float(np.median(np.linalg.norm(policy, axis=1))),
            summary["policy_offset"]["median_demo_calibrated_residual_norm_m"],
        ),
        "policy_residual_maximum_norm": close(
            float(np.max(np.linalg.norm(policy, axis=1))),
            summary["policy_offset"]["maximum_demo_calibrated_residual_norm_m"],
        ),
        "all_approach_frames_present": all(
            Path(row["approach_frame"]["path"]).exists() for row in trials
        ),
        "all_three_overlays_present": all(
            (AUDIT_ROOT / name).exists()
            for name in (
                "placement_expected_visible_overlay.png",
                "approach_policy_offset_overlay.png",
                "real24_eval_image_position_coverage.png",
            )
        ),
    }
    if not all(value is True or isinstance(value, int) and value > 0 for value in checks.values()):
        failed = [name for name, value in checks.items() if value is not True and not (isinstance(value, int) and value > 0)]
        raise RuntimeError(f"Independent validation failed: {failed}")
    result = {
        "schema_version": 1,
        "audit_id": EXPECTED_AUDIT_ID,
        "status": "pass",
        "checks": checks,
        "recomputed": {
            "placement_residual_px": {
                "median": float(np.median(placement_px)),
                "maximum": float(np.max(placement_px)),
            },
            "placement_residual_m_approx": {
                "median": float(np.median(placement_m)),
                "maximum": float(np.max(placement_m)),
            },
            "nearest_real24_early_frame_distance_px": {
                "median": float(np.median(nearest)),
                "maximum": float(np.max(nearest)),
            },
            "policy_residual_norm_m": {
                "median": float(np.median(np.linalg.norm(policy, axis=1))),
                "maximum": float(np.max(np.linalg.norm(policy, axis=1))),
            },
        },
        "verified_artifact_hashes": len(hash_rows),
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
