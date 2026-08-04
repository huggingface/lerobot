from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PLAN_PATH = EXPERIMENT_DIR / "evaluation_plan_ready_tolerance3_v2.json"
EXPECTED_PLAN_SHA256 = (
    "1eb82cc573a661f85d7c21ced5651a9cc64c8809873a3299d13e86d59fd19ada"
)
EXPECTED_PROFILE_SHA256 = (
    "60025f6478a63bcf9b301a75cd27124b19f1cf4f6b142f8576e6b10abf4f95a5"
)
OUTPUT_NAMES = (
    "operator_manifest_v1.json",
    "operator_summary_v1.json",
    "operator_evidence_hashes_v1.sha256",
)


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
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite frozen evidence: {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    require(
        sha256_file(PLAN_PATH) == EXPECTED_PLAN_SHA256,
        "Paired evaluation plan hash mismatch.",
    )
    plan = read_json(PLAN_PATH)
    evidence_root = Path(plan["evidence_root"])
    trials_root = evidence_root / "trials"
    labels_root = evidence_root / "labels"
    for name in OUTPUT_NAMES:
        require(not (evidence_root / name).exists(), f"Output already exists: {name}")

    selected: list[dict] = []
    files_to_hash: set[Path] = {PLAN_PATH}
    invalid_originals: list[dict] = []
    for trial in plan["trials"]:
        original_stem = trial["spawn_region"]
        original_path = trials_root / f"{original_stem}.json"
        require(original_path.exists(), f"Missing original evidence: {original_path}")
        original = read_json(original_path)
        artifact_stem = original_stem
        evidence = original
        if original.get("status") == "aborted_with_error":
            replacement_stem = f"{original_stem}__replacement1"
            replacement_path = trials_root / f"{replacement_stem}.json"
            require(
                replacement_path.exists(),
                f"Missing linked infrastructure replacement: {replacement_path}",
            )
            artifact_stem = replacement_stem
            evidence = read_json(replacement_path)
            invalid_originals.append(
                {
                    "trial_id": trial["trial_id"],
                    "artifact_stem": original_stem,
                    "status": original["status"],
                    "run_error": original["run_error"],
                    "evidence_path": str(original_path),
                    "evidence_sha256": sha256_file(original_path),
                    "replacement_artifact_stem": replacement_stem,
                }
            )
            files_to_hash.add(original_path)
            for suffix in (".ready.jsonl", ".paired.json"):
                path = trials_root / f"{original_stem}{suffix}"
                if path.exists():
                    files_to_hash.add(path)

        evidence_path = trials_root / f"{artifact_stem}.json"
        paired_path = trials_root / f"{artifact_stem}.paired.json"
        label_path = labels_root / f"{artifact_stem}.operator.json"
        require(paired_path.exists(), f"Missing paired sidecar: {paired_path}")
        require(label_path.exists(), f"Missing operator label: {label_path}")
        paired = read_json(paired_path)
        label = read_json(label_path)

        require(
            evidence["status"] == "completed_pending_operator_annotation",
            f"{trial['trial_id']} did not complete its scored policy window.",
        )
        require(
            evidence["termination"] == "maximum_duration",
            f"{trial['trial_id']} has unexpected termination.",
        )
        require(
            evidence["evaluation_plan_sha256"] == EXPECTED_PLAN_SHA256,
            f"{trial['trial_id']} plan SHA mismatch.",
        )
        require(
            evidence["evaluation_profile_sha256"] == EXPECTED_PROFILE_SHA256,
            f"{trial['trial_id']} profile SHA mismatch.",
        )
        require(
            evidence["model_sha256"]
            == plan["models"][trial["model_id"]]["model_sha256"],
            f"{trial['trial_id']} model SHA mismatch.",
        )
        require(
            evidence["paired_trial"] == trial,
            f"{trial['trial_id']} trial metadata mismatch.",
        )
        require(
            evidence["replacement_for"]
            == (original_stem if artifact_stem != original_stem else None),
            f"{trial['trial_id']} replacement link mismatch.",
        )
        require(
            evidence["ready_pose_alignment"]["result"]["status"]
            == "ready_pose_observed",
            f"{trial['trial_id']} ready pose was not observed.",
        )
        require(
            evidence["ready_pose_alignment"]["result"]["maximum_absolute_error"]
            <= 3.0,
            f"{trial['trial_id']} exceeded ready tolerance.",
        )
        require(
            evidence["automatic_return"]["result"]["status"]
            == "ready_pose_observed",
            f"{trial['trial_id']} did not return to ready.",
        )
        require(
            evidence["automatic_return"]["result"]["maximum_absolute_error"]
            <= 3.0,
            f"{trial['trial_id']} exceeded return tolerance.",
        )
        require(
            evidence["torque_disable_verified"] is True,
            f"{trial['trial_id']} torque disable was not verified.",
        )
        require(
            evidence["video"]["exists"] is True,
            f"{trial['trial_id']} canonical video is missing.",
        )
        require(
            evidence["video"]["frames"] == evidence["steps_jsonl"]["lines"],
            f"{trial['trial_id']} video/tick count mismatch.",
        )
        require(
            evidence["steps_jsonl"]["lines"] > 0,
            f"{trial['trial_id']} has no policy ticks.",
        )
        require(
            evidence["upstream_action_modified_events"] == 0,
            f"{trial['trial_id']} had an upstream action modification.",
        )
        require(
            label["trial_id"] == trial["trial_id"],
            f"{trial['trial_id']} operator label mismatch.",
        )
        require(
            isinstance(label["success"], bool),
            f"{trial['trial_id']} operator label is incomplete.",
        )

        related_paths = [
            evidence_path,
            paired_path,
            label_path,
            Path(evidence["steps_jsonl"]["path"]),
            Path(evidence["video"]["path"]),
            Path(evidence["ready_pose_alignment"]["trajectory"]["path"]),
            Path(evidence["automatic_return"]["trajectory"]["path"]),
        ]
        for path in related_paths:
            require(path.exists(), f"Missing evidence artifact: {path}")
            files_to_hash.add(path)
        selected.append(
            {
                "trial_id": trial["trial_id"],
                "order": trial["order"],
                "cell_id": trial["cell_id"],
                "model_id": trial["model_id"],
                "model_sha256": evidence["model_sha256"],
                "artifact_stem": artifact_stem,
                "replacement_for": evidence["replacement_for"],
                "operator_success": label["success"],
                "failure_category": label.get("failure_category"),
                "operator_notes": label.get("notes"),
                "policy_ticks": evidence["steps_jsonl"]["lines"],
                "ready_maximum_absolute_error_degrees": evidence[
                    "ready_pose_alignment"
                ]["result"]["maximum_absolute_error"],
                "return_maximum_absolute_error_degrees": evidence[
                    "automatic_return"
                ]["result"]["maximum_absolute_error"],
                "upstream_action_modified_events": evidence[
                    "upstream_action_modified_events"
                ],
                "evidence_path": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
                "operator_label_path": str(label_path),
                "operator_label_sha256": sha256_file(label_path),
                "video_path": evidence["video"]["path"],
                "video_sha256": evidence["video"]["sha256"],
                "steps_path": evidence["steps_jsonl"]["path"],
                "steps_sha256": evidence["steps_jsonl"]["sha256"],
                "canonical_video_review_status": "pending",
            }
        )

    require(len(selected) == 24, "Expected exactly 24 scored trials.")
    by_model: dict[str, dict] = {}
    for model_id in plan["models"]:
        rows = [row for row in selected if row["model_id"] == model_id]
        successes = sum(row["operator_success"] for row in rows)
        by_model[model_id] = {
            "trials": len(rows),
            "operator_successes": successes,
            "operator_failures": len(rows) - successes,
            "operator_success_rate": successes / len(rows),
            "failure_categories": dict(
                Counter(
                    row["failure_category"]
                    for row in rows
                    if not row["operator_success"]
                )
            ),
        }

    paired_rows: list[dict] = []
    discordance = Counter()
    for cell_id in sorted({row["cell_id"] for row in selected}):
        cell = {row["model_id"]: row for row in selected if row["cell_id"] == cell_id}
        real_success = cell["real24_only"]["operator_success"]
        mixed_success = cell["real24_questsim24"]["operator_success"]
        if real_success and not mixed_success:
            outcome = "real24_only_only"
        elif mixed_success and not real_success:
            outcome = "mixed_only"
        elif real_success:
            outcome = "both_success"
        else:
            outcome = "both_failure"
        discordance[outcome] += 1
        paired_rows.append(
            {
                "cell_id": cell_id,
                "real24_only_success": real_success,
                "mixed_success": mixed_success,
                "paired_outcome": outcome,
            }
        )

    manifest = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "result_scope": "operator labels only; canonical-video review pending",
        "plan_path": str(PLAN_PATH),
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "ready_pose_arrival_tolerance_degrees": 3.0,
        "scored_trials": selected,
        "preserved_infrastructure_invalid_originals": invalid_originals,
    }
    summary = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "status": "operator_labels_complete_canonical_video_review_pending",
        "paper_conclusion": False,
        "scored_trials": len(selected),
        "total_policy_ticks": sum(row["policy_ticks"] for row in selected),
        "all_trials_have_video_and_tick_evidence": True,
        "all_trials_returned_to_same_ready_pose": True,
        "all_trials_torque_disable_verified": True,
        "total_upstream_action_modified_events": sum(
            row["upstream_action_modified_events"] for row in selected
        ),
        "by_model": by_model,
        "paired_outcomes": dict(discordance),
        "by_cell": paired_rows,
        "infrastructure_invalid_originals_preserved": len(invalid_originals),
        "canonical_video_review": {
            "status": "pending",
            "operator_labels_are_not_silently_promoted_to_review_labels": True,
        },
        "interpretation_boundary": (
            "Descriptive paired operator result only. Canonical-video review is "
            "required before the reviewed result is used for a paper claim."
        ),
    }
    manifest_path = evidence_root / OUTPUT_NAMES[0]
    summary_path = evidence_root / OUTPUT_NAMES[1]
    write_new_json(manifest_path, manifest)
    write_new_json(summary_path, summary)
    files_to_hash.update({manifest_path, summary_path})
    hashes_path = evidence_root / OUTPUT_NAMES[2]
    hash_rows = []
    for path in sorted(files_to_hash, key=lambda value: str(value)):
        hash_rows.append(f"{sha256_file(path)}  {path}\n")
    hashes_path.write_text("".join(hash_rows), encoding="utf-8")
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
                "paired_outcomes": dict(discordance),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
