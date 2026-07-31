from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
PLAN_PATH = EXPERIMENT_DIR / "evaluation_plan.json"
EXPECTED_PLAN_SHA256 = "f3f6e797ce001752b46f5f278dcb94df712a29d3e1e29708228810a82074406e"
EXPECTED_PROFILE_SHA256 = "243ef491eda81c38ceba85f7bc01c63c7dc7c14e39e5b4261a109f8a00736525"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_new_json(path: Path, payload: dict) -> None:
    require(not path.exists(), f"Refusing to overwrite frozen evidence: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    require(sha256_file(PLAN_PATH) == EXPECTED_PLAN_SHA256, "Plan hash mismatch.")
    plan = read_json(PLAN_PATH)
    root = Path(plan["evidence_root"])
    trials_root = root / "trials"
    labels_root = root / "labels"
    manifest_path = root / "operator_manifest_v1.json"
    summary_path = root / "operator_summary_v1.json"
    hashes_path = root / "operator_evidence_hashes_v1.sha256"
    for path in (manifest_path, summary_path, hashes_path):
        require(not path.exists(), f"Output already exists: {path}")

    scored = []
    invalid = []
    files: set[Path] = {PLAN_PATH}
    for trial in plan["trials"]:
        original_stem = trial["spawn_region"]
        original_path = trials_root / f"{original_stem}.json"
        require(original_path.exists(), f"Missing original evidence: {original_path}")
        marker_path = trials_root / f"{original_stem}.infrastructure_invalid.json"
        artifact_stem = original_stem
        if marker_path.exists():
            marker = read_json(marker_path)
            require(marker["status"] == "infrastructure_invalid", "Invalid marker status.")
            require(marker["scored_trial"] is False, "Invalid original cannot be scored.")
            require(marker["replacement_allowed"] is True, "Replacement was not allowed.")
            require(
                marker["original_evidence"]["sha256"] == sha256_file(original_path),
                "Invalid marker does not bind original evidence.",
            )
            artifact_stem = f"{original_stem}__replacement1"
            invalid.append(
                {
                    "trial_id": trial["trial_id"],
                    "artifact_stem": original_stem,
                    "reason": marker["reason"],
                    "evidence_path": str(original_path),
                    "evidence_sha256": sha256_file(original_path),
                    "marker_path": str(marker_path),
                    "marker_sha256": sha256_file(marker_path),
                    "replacement_artifact_stem": artifact_stem,
                }
            )
            files.update({original_path, marker_path})

        evidence_path = trials_root / f"{artifact_stem}.json"
        sidecar_path = trials_root / f"{artifact_stem}.paired_evalv2.json"
        label_path = labels_root / f"{artifact_stem}.operator.json"
        for path in (evidence_path, sidecar_path, label_path):
            require(path.exists(), f"Missing scored artifact: {path}")
        evidence = read_json(evidence_path)
        sidecar = read_json(sidecar_path)
        label = read_json(label_path)
        require(evidence["status"] == "completed_pending_operator_annotation", "Trial incomplete.")
        require(evidence["termination"] == "maximum_duration", "Unexpected termination.")
        require(evidence["evaluation_plan_sha256"] == EXPECTED_PLAN_SHA256, "Plan SHA mismatch.")
        require(
            evidence["evaluation_profile_sha256"] == EXPECTED_PROFILE_SHA256,
            "Profile SHA mismatch.",
        )
        require(
            evidence["model_sha256"] == plan["models"][trial["model_id"]]["model_sha256"],
            "Model SHA mismatch.",
        )
        require(evidence["paired_evalv2_trial"] == trial, "Trial metadata mismatch.")
        require(
            evidence["replacement_for"]
            == (original_stem if artifact_stem != original_stem else None),
            "Replacement link mismatch.",
        )
        require(
            evidence["ready_pose_alignment"]["result"]["status"] == "ready_pose_observed",
            "Ready pose not observed.",
        )
        require(
            evidence["automatic_return"]["result"]["status"] == "ready_pose_observed",
            "Return pose not observed.",
        )
        require(evidence["torque_disable_verified"] is True, "Torque disable not verified.")
        require(evidence["video"]["exists"] is True, "Canonical video missing.")
        require(
            evidence["video"]["frames"] == evidence["steps_jsonl"]["lines"],
            "Video/tick count mismatch.",
        )
        require(evidence["upstream_action_modified_events"] == 0, "Action was modified upstream.")
        require(sidecar["trial"] == trial, "Sidecar trial mismatch.")
        require(label["trial_id"] == trial["trial_id"], "Operator label trial mismatch.")
        require(label["artifact_stem"] == artifact_stem, "Operator label artifact mismatch.")
        require(isinstance(label["success"], bool), "Operator label incomplete.")

        related = [
            evidence_path,
            sidecar_path,
            label_path,
            Path(evidence["steps_jsonl"]["path"]),
            Path(evidence["video"]["path"]),
            Path(evidence["ready_pose_alignment"]["trajectory"]["path"]),
            Path(evidence["automatic_return"]["trajectory"]["path"]),
        ]
        for path in related:
            require(path.exists(), f"Missing evidence artifact: {path}")
            files.add(path)
        scored.append(
            {
                "order": trial["order"],
                "trial_id": trial["trial_id"],
                "source_pose_id": trial["source_pose_id"],
                "cell_id": trial["cell_id"],
                "model_id": trial["model_id"],
                "model_sha256": evidence["model_sha256"],
                "artifact_stem": artifact_stem,
                "replacement_for": evidence["replacement_for"],
                "operator_success": label["success"],
                "failure_category": label.get("failure_category"),
                "operator_notes": label.get("notes"),
                "policy_ticks": evidence["steps_jsonl"]["lines"],
                "video_path": evidence["video"]["path"],
                "video_sha256": evidence["video"]["sha256"],
                "steps_path": evidence["steps_jsonl"]["path"],
                "steps_sha256": evidence["steps_jsonl"]["sha256"],
                "evidence_path": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
                "ready_maximum_absolute_error_degrees": evidence["ready_pose_alignment"]["result"][
                    "maximum_absolute_error"
                ],
                "return_maximum_absolute_error_degrees": evidence["automatic_return"]["result"][
                    "maximum_absolute_error"
                ],
                "canonical_video_review_status": "pending",
            }
        )

    require(len(scored) == 24, "Expected exactly 24 scored trials.")
    by_model = {}
    for model_id in plan["models"]:
        rows = [row for row in scored if row["model_id"] == model_id]
        successes = sum(row["operator_success"] for row in rows)
        by_model[model_id] = {
            "trials": len(rows),
            "operator_successes": successes,
            "operator_failures": len(rows) - successes,
            "operator_success_rate": successes / len(rows),
            "failure_categories": dict(
                Counter(row["failure_category"] for row in rows if not row["operator_success"])
            ),
        }

    paired = []
    paired_counts = Counter()
    for pose_order in range(1, 13):
        rows = [row for row in scored if plan["trials"][row["order"] - 1]["source_pose_order"] == pose_order]
        require(len(rows) == 2, "Each source pose must have two scored model rows.")
        cell = {row["model_id"]: row for row in rows}
        real = cell["real24_only"]["operator_success"]
        sim = cell["questsim48_v7"]["operator_success"]
        outcome = (
            "both_success"
            if real and sim
            else "both_failure"
            if not real and not sim
            else "real24_only_only"
            if real
            else "questsim48_v7_only"
        )
        paired_counts[outcome] += 1
        paired.append(
            {
                "source_pose_order": pose_order,
                "source_pose_id": rows[0]["source_pose_id"],
                "cell_id": rows[0]["cell_id"],
                "real24_only_success": real,
                "questsim48_v7_success": sim,
                "paired_outcome": outcome,
            }
        )

    manifest = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "result_scope": "operator labels only; canonical-video review pending",
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "scored_trials": scored,
        "infrastructure_invalid_originals": invalid,
    }
    summary = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "result_scope": "operator labels only; canonical-video review pending",
        "by_model": by_model,
        "paired_pose_results": paired,
        "paired_outcomes": dict(paired_counts),
        "infrastructure_invalid_original_count": len(invalid),
        "canonical_video_review_status": "pending",
    }
    write_new_json(manifest_path, manifest)
    write_new_json(summary_path, summary)
    files.update({manifest_path, summary_path})
    hashes_path.write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in sorted(files)),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "hashes": str(hashes_path),
                "hashes_sha256": sha256_file(hashes_path),
                "by_model": by_model,
                "paired_outcomes": dict(paired_counts),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
