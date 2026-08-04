from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "evaluation_plan.json"
EXPECTED_PLAN_SHA256 = "31c2aeea619729018876aa2886713a40b4a43be650de234adea29fc0d6a4eed6"
EXPECTED_MODEL_SHA256 = "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
REVIEW_ID = "task1_picklift_real24_offcenter_yaw_eval_v2_pilot_canonical_review_v1"

ARTIFACT_STEMS = {
    "evalv2_pilot_v2_r3c3": "t01_evalv2_pilot_v2_r3c3_real24_only",
    "evalv2_pilot_v2_r3c4": "t02_evalv2_pilot_v2_r3c4_real24_only",
    "evalv2_pilot_v2_r2c4": "t03_evalv2_pilot_v2_r2c4_real24_only",
    "evalv2_pilot_v2_r2c1": "t04_evalv2_pilot_v2_r2c1_real24_only",
    "evalv2_pilot_v2_r2c2": "t05_evalv2_pilot_v2_r2c2_real24_only",
    "evalv2_pilot_v2_r2c3": "t06_evalv2_pilot_v2_r2c3_real24_only",
    "evalv2_pilot_v2_r1c2": "t07_evalv2_pilot_v2_r1c2_real24_only",
    "evalv2_pilot_v2_r1c4": "t08_evalv2_pilot_v2_r1c4_real24_only__replacement1",
    "evalv2_pilot_v2_r1c1": "t09_evalv2_pilot_v2_r1c1_real24_only",
    "evalv2_pilot_v2_r1c3": "t10_evalv2_pilot_v2_r1c3_real24_only",
    "evalv2_pilot_v2_r3c2": "t11_evalv2_pilot_v2_r3c2_real24_only",
    "evalv2_pilot_v2_r3c1": "t12_evalv2_pilot_v2_r3c1_real24_only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def probe_video(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream["r_frame_rate"],
        "avg_frame_rate": stream["avg_frame_rate"],
        "nb_frames": int(stream["nb_frames"]),
        "nb_read_frames": int(stream["nb_read_frames"]),
    }


def rate(successes: int, total: int) -> dict:
    return {
        "successes": successes,
        "trials": total,
        "success_rate": successes / total if total else None,
    }


def main() -> None:
    if sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Frozen evaluation plan hash mismatch.")
    plan = read_json(PLAN_PATH)
    if len(plan["trials"]) != 12:
        raise RuntimeError("Expected exactly 12 frozen planned trials.")
    if plan["models"]["real24_only"]["model_sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Frozen model identity mismatch.")

    evidence_root = Path(plan["evidence_root"])
    trials_root = evidence_root / "trials"
    labels_root = evidence_root / "labels"
    output_root = evidence_root / "canonical_video_review_v1"
    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite immutable review: {output_root}")

    rows = []
    total_ticks = 0
    failure_types: Counter[str] = Counter()
    operator_review_disagreements = []
    grouped: dict[str, list[bool]] = defaultdict(list)

    for trial in sorted(plan["trials"], key=lambda item: item["order"]):
        stem = ARTIFACT_STEMS[trial["trial_id"]]
        evidence_path = trials_root / f"{stem}.json"
        operator_path = labels_root / f"{stem}.operator.json"
        review_path = labels_root / f"{stem}.review.json"
        evidence = read_json(evidence_path)
        operator = read_json(operator_path)
        review = read_json(review_path)

        if evidence["model_sha256"] != EXPECTED_MODEL_SHA256:
            raise RuntimeError(f"{stem}: model hash mismatch.")
        if evidence["run_error"] is not None or not evidence["torque_disable_verified"]:
            raise RuntimeError(f"{stem}: invalid execution or missing torque-disable evidence.")
        if operator["source"] != "operator" or review["source"] != "review":
            raise RuntimeError(f"{stem}: label source mismatch.")
        if operator["trial_id"] != trial["trial_id"] or review["trial_id"] != trial["trial_id"]:
            raise RuntimeError(f"{stem}: label trial identity mismatch.")

        steps_path = Path(evidence["steps_jsonl"]["path"])
        video_path = Path(evidence["video"]["path"])
        if sha256_file(steps_path) != evidence["steps_jsonl"]["sha256"]:
            raise RuntimeError(f"{stem}: steps hash mismatch.")
        if sha256_file(video_path) != evidence["video"]["sha256"]:
            raise RuntimeError(f"{stem}: video hash mismatch.")

        ticks = line_count(steps_path)
        probe = probe_video(video_path)
        if ticks != evidence["steps_jsonl"]["lines"]:
            raise RuntimeError(f"{stem}: steps line count mismatch.")
        if probe["width"] != 640 or probe["height"] != 480:
            raise RuntimeError(f"{stem}: canonical video dimensions mismatch.")
        if probe["r_frame_rate"] != "20/1" or probe["avg_frame_rate"] != "20/1":
            raise RuntimeError(f"{stem}: canonical video frame rate mismatch.")
        if probe["nb_frames"] != ticks or probe["nb_read_frames"] != ticks:
            raise RuntimeError(f"{stem}: canonical video frame/tick mismatch.")
        if evidence["video"]["frames"] != ticks or evidence["video"]["encoded_fps"] != 20:
            raise RuntimeError(f"{stem}: trial video manifest mismatch.")
        if evidence["termination"] != "maximum_duration":
            raise RuntimeError(f"{stem}: unexpected termination.")

        agrees = operator["success"] == review["success"]
        if not agrees:
            operator_review_disagreements.append(trial["trial_id"])
        if not review["success"]:
            failure_types[review["failure_category"]] += 1
        total_ticks += ticks
        grouped[f"yaw_{trial['nominal_yaw_degrees_modulo_90']}"].append(review["success"])
        grouped[f"quadrant_{trial['quadrant']}"].append(review["success"])

        rows.append(
            {
                "order": trial["order"],
                "trial_id": trial["trial_id"],
                "artifact_stem": stem,
                "cell_id": trial["cell_id"],
                "quadrant": trial["quadrant"],
                "nominal_x_forward_m": trial["nominal_x_forward_m"],
                "nominal_y_lateral_m": trial["nominal_y_lateral_m"],
                "nominal_yaw_degrees_modulo_90": trial["nominal_yaw_degrees_modulo_90"],
                "operator_success": operator["success"],
                "operator_failure_category": operator["failure_category"],
                "operator_notes": operator["notes"],
                "review_success": review["success"],
                "review_failure_category": review["failure_category"],
                "review_notes": review["notes"],
                "operator_review_agree": agrees,
                "termination": evidence["termination"],
                "run_error": evidence["run_error"],
                "torque_disable_verified": evidence["torque_disable_verified"],
                "policy_ticks": ticks,
                "upstream_action_modified_events": evidence["upstream_action_modified_events"],
                "video": {
                    "path": str(video_path),
                    "sha256": evidence["video"]["sha256"],
                    **probe,
                },
                "evidence": {
                    "path": str(evidence_path),
                    "sha256": sha256_file(evidence_path),
                },
                "operator_label": {
                    "path": str(operator_path),
                    "sha256": sha256_file(operator_path),
                },
                "review_label": {
                    "path": str(review_path),
                    "sha256": sha256_file(review_path),
                },
            }
        )

    successes = sum(row["review_success"] for row in rows)
    operator_successes = sum(row["operator_success"] for row in rows)
    grouped_rates = {
        name: rate(sum(values), len(values)) for name, values in sorted(grouped.items())
    }
    invalid_marker = (
        trials_root / "t08_evalv2_pilot_v2_r1c4_real24_only.infrastructure_invalid.json"
    )
    if not invalid_marker.exists():
        raise RuntimeError("Expected preserved placement-invalid t08 evidence marker.")

    output_root.mkdir(parents=True)
    trials_path = output_root / "trials.jsonl"
    trials_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "review_id": REVIEW_ID,
        "reviewed_result": rate(successes, len(rows)),
        "operator_result": rate(operator_successes, len(rows)),
        "operator_review_agreement": {
            "agreements": len(rows) - len(operator_review_disagreements),
            "trials": len(rows),
            "disagreement_trial_ids": operator_review_disagreements,
        },
        "failure_types_reviewed": dict(sorted(failure_types.items())),
        "breakdowns": grouped_rates,
        "total_policy_ticks_and_video_frames": total_ticks,
        "interface_valid_trials": len(rows),
        "expected_trials": 12,
        "excluded_evidence": {
            "reason": "operator_placement_invalid",
            "original_trial_id": "evalv2_pilot_v2_r1c4",
            "original_artifact_stem": "t08_evalv2_pilot_v2_r1c4_real24_only",
            "linked_scored_replacement": ARTIFACT_STEMS["evalv2_pilot_v2_r1c4"],
            "marker_path": str(invalid_marker),
            "marker_sha256": sha256_file(invalid_marker),
        },
        "protocol_revision_caveat": (
            "Orders 1-4 used the original direct ready/return command; orders 5-12 used the "
            "versioned three-second interpolated ready/return trajectory. Ready/return occurred "
            "outside the scored policy window; policy, model, placements, camera, action path, "
            "and 30-second success contract were unchanged."
        ),
        "interpretation_boundary": (
            "This is a single-model, single-session difficulty pilot. It shows that the tested "
            "off-center/yaw pose set is not near ceiling, but it is not a paper effect estimate "
            "and does not by itself identify a causal failure mechanism."
        ),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "review_id": REVIEW_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_id": plan["evaluation_id"],
        "evaluation_plan": {
            "path": str(PLAN_PATH),
            "sha256": EXPECTED_PLAN_SHA256,
        },
        "model_sha256": EXPECTED_MODEL_SHA256,
        "evidence_root": str(evidence_root),
        "canonical_review_root": str(output_root),
        "success_contract": plan["success_contract"],
        "video_contract": {
            "source": "canonical_rgb_act_input",
            "width": 640,
            "height": 480,
            "fps": 20,
            "frame_count_must_equal_policy_tick_count": True,
        },
        "scored_trial_count": 12,
        "review_method": (
            "Independent visual inspection of the complete canonical policy-input videos. "
            "Operator labels were preserved as separate immutable files."
        ),
        "files": {
            "trials_jsonl": {
                "path": str(trials_path),
                "sha256": sha256_file(trials_path),
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
        },
    }
    manifest_path = output_root / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashed_files = [
        trials_path,
        summary_path,
        manifest_path,
        *sorted(labels_root.glob("*.operator.json")),
        *sorted(labels_root.glob("*.review.json")),
        *[Path(row["video"]["path"]) for row in rows],
        *[Path(row["evidence"]["path"]) for row in rows],
    ]
    hashes_path = output_root / "hashes.sha256"
    hashes_path.write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in hashed_files),
        encoding="utf-8",
    )

    verification = {
        "schema_version": 1,
        "review_id": REVIEW_ID,
        "status": "pass",
        "checks": {
            "evaluation_plan_hash": True,
            "model_hash_identity": True,
            "scored_trials": 12,
            "operator_labels": 12,
            "review_labels": 12,
            "canonical_video_640x480_20fps": 12,
            "video_frames_equal_policy_ticks": 12,
            "run_error_null": 12,
            "torque_disable_verified": 12,
            "preserved_invalid_t08_marker": True,
        },
        "artifact_hashes": {
            "trials_jsonl": sha256_file(trials_path),
            "summary": sha256_file(summary_path),
            "review_manifest": sha256_file(manifest_path),
            "hashes_sha256": sha256_file(hashes_path),
        },
    }
    verification_path = output_root / "independent_verification.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
