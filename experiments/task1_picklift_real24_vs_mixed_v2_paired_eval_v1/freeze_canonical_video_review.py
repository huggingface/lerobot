from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
PLAN_PATH = EXPERIMENT_DIR / "evaluation_plan.json"
DECISIONS_PATH = EXPERIMENT_DIR / "canonical_video_review_v1.json"
EVIDENCE_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_vs_mixed_v2_paired_real24_eval24_v1"
)
REVIEW_ROOT = EVIDENCE_ROOT / "canonical_video_review_v1"
OPERATOR_MANIFEST_PATH = EVIDENCE_ROOT / "operator_manifest_v1.json"
OPERATOR_SUMMARY_PATH = EVIDENCE_ROOT / "operator_summary_v1.json"
OPERATOR_HASHES_PATH = EVIDENCE_ROOT / "operator_evidence_hashes_v1.sha256"
OPERATOR_INDEX_PATH = EXPERIMENT_DIR / "operator_result_index_v1.json"

EXPECTED_PLAN_SHA256 = "d59d07a5abcc1644ba89c8d8234e00b113a676b226c10503054a341cae2d9dc5"
EXPECTED_PROFILE_SHA256 = "60025f6478a63bcf9b301a75cd27124b19f1cf4f6b142f8576e6b10abf4f95a5"
EXPECTED_OPERATOR_MANIFEST_SHA256 = "9ffffd0f376e2f33dadece6fa9738309611654a0008c30081939d013f3a4edfe"
EXPECTED_OPERATOR_SUMMARY_SHA256 = "74392a3ebb007405fd6697580c210a7aa84b606b2c9a7abd2199b021ff226b68"
EXPECTED_OPERATOR_HASHES_SHA256 = "68ca70102ccb1e33c98a038a8bc2244ac16aabd66a7a94dc66d2ab4cc2f19de6"
EXPECTED_OPERATOR_RESULT_SOURCE_COMMIT = "f0832638f6087f7fdc16b8184793754eb3070c97"
EXPECTED_REVIEW_ID = "task1_picklift_real24_vs_mixed_v2_paired_real24_eval24_canonical_video_review_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            ("stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration"),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    require(len(streams) == 1, f"Expected one video stream: {path}")
    stream = streams[0]
    return {
        "codec_name": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream["r_frame_rate"],
        "avg_frame_rate": stream["avg_frame_rate"],
        "frames": int(stream["nb_frames"]),
        "encoded_duration_seconds": float(stream["duration"]),
    }


def read_steps_contract(path: Path) -> dict:
    first = None
    last = None
    lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if first is None:
                first = row
            last = row
            lines += 1
    require(first is not None and last is not None, f"Empty steps JSONL: {path}")
    return {
        "lines": lines,
        "first_step": first["step"],
        "last_step": last["step"],
        "first_tick_elapsed_seconds": first["tick_started_elapsed_seconds"],
        "last_tick_elapsed_seconds": last["tick_started_elapsed_seconds"],
    }


def parse_operator_hash_list() -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for line in OPERATOR_HASHES_PATH.read_text(encoding="utf-8").splitlines():
        digest, raw_path = line.split("  ", maxsplit=1)
        path = Path(raw_path)
        require(path not in expected, f"Duplicate operator hash path: {path}")
        require(path.exists(), f"Missing operator evidence artifact: {path}")
        require(
            sha256_file(path) == digest,
            f"Operator evidence hash mismatch: {path}",
        )
        expected[path] = digest
    return expected


def summarize_by_model(rows: list[dict], model_ids: list[str]) -> dict:
    result = {}
    for model_id in model_ids:
        selected = [row for row in rows if row["model_id"] == model_id]
        successes = sum(row["review_success"] for row in selected)
        result[model_id] = {
            "trials": len(selected),
            "reviewed_successes": successes,
            "reviewed_failures": len(selected) - successes,
            "reviewed_success_rate": successes / len(selected),
            "failure_types": dict(
                Counter(row["review_failure_type"] for row in selected if not row["review_success"])
            ),
        }
    return result


def summarize_pairs(rows: list[dict]) -> tuple[list[dict], dict]:
    by_cell = []
    paired_outcomes = Counter()
    for cell_id in sorted({row["cell_id"] for row in rows}):
        cell_rows = {row["model_id"]: row for row in rows if row["cell_id"] == cell_id}
        real_success = cell_rows["real24_only"]["review_success"]
        mixed_success = cell_rows["mixed_v2"]["review_success"]
        if real_success and mixed_success:
            outcome = "both_success"
        elif not real_success and not mixed_success:
            outcome = "both_failure"
        elif real_success:
            outcome = "real24_only_only"
        else:
            outcome = "mixed_only"
        paired_outcomes[outcome] += 1
        by_cell.append(
            {
                "cell_id": cell_id,
                "real24_only_review_success": real_success,
                "mixed_v2_review_success": mixed_success,
                "paired_review_outcome": outcome,
            }
        )
    return by_cell, dict(paired_outcomes)


def main() -> None:
    require(not REVIEW_ROOT.exists(), f"Refusing to overwrite: {REVIEW_ROOT}")
    require(
        sha256_file(PLAN_PATH) == EXPECTED_PLAN_SHA256,
        "Evaluation plan SHA mismatch.",
    )
    require(
        sha256_file(OPERATOR_MANIFEST_PATH) == EXPECTED_OPERATOR_MANIFEST_SHA256,
        "Operator manifest SHA mismatch.",
    )
    require(
        sha256_file(OPERATOR_SUMMARY_PATH) == EXPECTED_OPERATOR_SUMMARY_SHA256,
        "Operator summary SHA mismatch.",
    )
    require(
        sha256_file(OPERATOR_HASHES_PATH) == EXPECTED_OPERATOR_HASHES_SHA256,
        "Operator evidence hashes SHA mismatch.",
    )

    plan = read_json(PLAN_PATH)
    operator_manifest = read_json(OPERATOR_MANIFEST_PATH)
    read_json(OPERATOR_SUMMARY_PATH)
    decisions_source = read_json(DECISIONS_PATH)
    require(
        decisions_source["review_id"] == EXPECTED_REVIEW_ID,
        "Review identity mismatch.",
    )
    require(
        decisions_source["operator_result_source_commit"] == EXPECTED_OPERATOR_RESULT_SOURCE_COMMIT,
        "Operator result source commit mismatch.",
    )
    require(
        decisions_source["classification_independent_of_operator_label"] is True,
        "Review source is not marked independent.",
    )

    operator_hashes = parse_operator_hash_list()
    require(len(operator_hashes) == 171, "Expected 171 frozen operator artifacts.")
    trials = operator_manifest["scored_trials"]
    decisions = decisions_source["decisions"]
    require(len(trials) == 24, "Expected 24 scored operator trials.")
    require(len(decisions) == 24, "Expected 24 review decisions.")
    decisions_by_id = {row["trial_id"]: row for row in decisions}
    require(len(decisions_by_id) == 24, "Review trial IDs are not unique.")
    require(
        [row["trial_id"] for row in decisions] == [row["trial_id"] for row in trials],
        "Review decisions are not in frozen trial order.",
    )

    decision_source_sha256 = sha256_file(DECISIONS_PATH)
    rows: list[dict] = []
    label_payloads: list[tuple[str, dict]] = []
    video_validations = []
    for operator_row in trials:
        decision = decisions_by_id[operator_row["trial_id"]]
        require(
            decision["artifact_stem"] == operator_row["artifact_stem"],
            f"{operator_row['trial_id']} artifact stem mismatch.",
        )
        require(
            decision["video_sha256"] == operator_row["video_sha256"],
            f"{operator_row['trial_id']} review video identity mismatch.",
        )
        video_path = Path(operator_row["video_path"])
        steps_path = Path(operator_row["steps_path"])
        label_path = Path(operator_row["operator_label_path"])
        evidence_path = Path(operator_row["evidence_path"])
        require(
            sha256_file(video_path) == operator_row["video_sha256"],
            f"{operator_row['trial_id']} video SHA mismatch.",
        )
        require(
            sha256_file(steps_path) == operator_row["steps_sha256"],
            f"{operator_row['trial_id']} steps SHA mismatch.",
        )
        require(
            sha256_file(label_path) == operator_row["operator_label_sha256"],
            f"{operator_row['trial_id']} operator label SHA mismatch.",
        )
        require(
            sha256_file(evidence_path) == operator_row["evidence_sha256"],
            f"{operator_row['trial_id']} evidence SHA mismatch.",
        )

        probe = probe_video(video_path)
        steps = read_steps_contract(steps_path)
        require(
            probe["width"] == 640 and probe["height"] == 480,
            f"{operator_row['trial_id']} is not 640x480.",
        )
        require(
            Fraction(probe["avg_frame_rate"]) == Fraction(20, 1),
            f"{operator_row['trial_id']} is not 20 FPS.",
        )
        require(
            probe["frames"] == steps["lines"] == operator_row["policy_ticks"],
            f"{operator_row['trial_id']} video/tick/frame mismatch.",
        )
        require(
            steps["first_step"] == 0 and steps["last_step"] == steps["lines"] - 1,
            f"{operator_row['trial_id']} has non-contiguous tick numbering.",
        )
        require(
            29.9 <= steps["last_tick_elapsed_seconds"] <= 30.1,
            f"{operator_row['trial_id']} does not cover the 30-second window.",
        )
        require(
            isinstance(decision["review_success"], bool),
            f"{operator_row['trial_id']} review label is incomplete.",
        )
        if decision["review_success"]:
            require(
                decision["failure_type"] is None,
                f"{operator_row['trial_id']} success has a failure type.",
            )
            require(
                decision["confirmation_time_seconds"] - decision["first_visible_qualifying_time_seconds"]
                >= 0.5,
                f"{operator_row['trial_id']} lacks >=0.5 s visual confirmation.",
            )
        else:
            require(
                isinstance(decision["failure_type"], str),
                f"{operator_row['trial_id']} failure lacks a failure type.",
            )
            require(
                decision["first_visible_qualifying_time_seconds"] is None
                and decision["confirmation_time_seconds"] is None,
                f"{operator_row['trial_id']} failure has success timestamps.",
            )

        operator_label = read_json(label_path)
        agreement = operator_label["success"] == decision["review_success"]
        review_label_payload = {
            "schema_version": 1,
            "review_id": EXPECTED_REVIEW_ID,
            "evaluation_id": plan["evaluation_id"],
            "trial_id": operator_row["trial_id"],
            "order": operator_row["order"],
            "cell_id": operator_row["cell_id"],
            "model_id": operator_row["model_id"],
            "artifact_stem": operator_row["artifact_stem"],
            "review_success": decision["review_success"],
            "failure_type": decision["failure_type"],
            "first_visible_qualifying_time_seconds": decision["first_visible_qualifying_time_seconds"],
            "confirmation_time_seconds": decision["confirmation_time_seconds"],
            "visible_evidence": decision["visible_evidence"],
            "success_definition": decisions_source["success_definition"],
            "review_method": decisions_source["review_method"],
            "video_path": str(video_path),
            "video_sha256": operator_row["video_sha256"],
            "review_decisions_source_path": str(DECISIONS_PATH),
            "review_decisions_source_sha256": decision_source_sha256,
            "operator_label_comparison": {
                "operator_label_path": str(label_path),
                "operator_label_sha256": operator_row["operator_label_sha256"],
                "operator_success": operator_label["success"],
                "operator_failure_category": operator_label.get("failure_category"),
                "operator_notes": operator_label.get("notes"),
                "agreement": agreement,
            },
            "adjudication_status": ("not_required" if agreement else "append_only_adjudication_required"),
        }
        label_name = f"{operator_row['artifact_stem']}.review.json"
        label_payloads.append((label_name, review_label_payload))
        row = {
            "trial_id": operator_row["trial_id"],
            "order": operator_row["order"],
            "cell_id": operator_row["cell_id"],
            "model_id": operator_row["model_id"],
            "model_sha256": operator_row["model_sha256"],
            "artifact_stem": operator_row["artifact_stem"],
            "replacement_for": operator_row["replacement_for"],
            "operator_success": operator_label["success"],
            "operator_failure_category": operator_label.get("failure_category"),
            "operator_notes": operator_label.get("notes"),
            "review_success": decision["review_success"],
            "review_failure_type": decision["failure_type"],
            "review_visible_evidence": decision["visible_evidence"],
            "first_visible_qualifying_time_seconds": decision["first_visible_qualifying_time_seconds"],
            "confirmation_time_seconds": decision["confirmation_time_seconds"],
            "operator_review_agreement": agreement,
            "adjudication_status": review_label_payload["adjudication_status"],
            "policy_ticks": steps["lines"],
            "last_tick_elapsed_seconds": steps["last_tick_elapsed_seconds"],
            "video_width": probe["width"],
            "video_height": probe["height"],
            "video_fps": float(Fraction(probe["avg_frame_rate"])),
            "video_frames": probe["frames"],
            "video_encoded_duration_seconds": probe["encoded_duration_seconds"],
            "video_path": str(video_path),
            "video_sha256": operator_row["video_sha256"],
            "steps_path": str(steps_path),
            "steps_sha256": operator_row["steps_sha256"],
            "operator_label_path": str(label_path),
            "operator_label_sha256": operator_row["operator_label_sha256"],
            "review_label_path": str(REVIEW_ROOT / "labels" / label_name),
        }
        rows.append(row)
        video_validations.append(
            {
                "trial_id": operator_row["trial_id"],
                "artifact_stem": operator_row["artifact_stem"],
                "video_path": str(video_path),
                "video_sha256": operator_row["video_sha256"],
                "ffprobe": probe,
                "steps_contract": steps,
                "frame_tick_match": True,
                "full_policy_window_covered": True,
            }
        )

    disagreements = [row for row in rows if not row["operator_review_agreement"]]
    by_model = summarize_by_model(rows, list(plan["models"]))
    by_cell, paired_outcomes = summarize_pairs(rows)
    total_policy_ticks = sum(row["policy_ticks"] for row in rows)
    source_index_discrepancies = []
    if OPERATOR_INDEX_PATH.exists():
        operator_index = read_json(OPERATOR_INDEX_PATH)
        index_total = operator_index["evidence"]["total_policy_ticks"]
        if index_total != total_policy_ticks:
            source_index_discrepancies.append(
                {
                    "field": "evidence.total_policy_ticks",
                    "operator_result_index_v1_value": index_total,
                    "primary_operator_manifest_sum": total_policy_ticks,
                    "resolution": (
                        "The reviewed result uses the immutable operator manifest, "
                        "operator summary, per-trial steps files, and video frame "
                        "counts, which all resolve to 14234."
                    ),
                }
            )

    REVIEW_ROOT.mkdir(parents=False)
    labels_root = REVIEW_ROOT / "labels"
    labels_root.mkdir()
    for label_name, payload in label_payloads:
        write_json(labels_root / label_name, payload)
    for row in rows:
        row["review_label_sha256"] = sha256_file(Path(row["review_label_path"]))

    trials_path = REVIEW_ROOT / "review_trials_v1.jsonl"
    summary_path = REVIEW_ROOT / "reviewed_summary_v1.json"
    manifest_path = REVIEW_ROOT / "review_manifest_v1.json"
    validation_path = REVIEW_ROOT / "validation_v1.json"
    hashes_path = REVIEW_ROOT / "review_hashes_v1.sha256"
    write_jsonl(trials_path, rows)
    summary = {
        "schema_version": 1,
        "review_id": EXPECTED_REVIEW_ID,
        "evaluation_id": plan["evaluation_id"],
        "status": "canonical_video_review_complete",
        "paper_conclusion": False,
        "success_definition": decisions_source["success_definition"],
        "reviewed_trials": len(rows),
        "operator_review_agreements": len(rows) - len(disagreements),
        "operator_review_disagreements": len(disagreements),
        "disagreements": [
            {
                "trial_id": row["trial_id"],
                "operator_success": row["operator_success"],
                "review_success": row["review_success"],
                "adjudication_status": row["adjudication_status"],
            }
            for row in disagreements
        ],
        "by_model": by_model,
        "paired_outcomes": paired_outcomes,
        "by_cell": by_cell,
        "review_failure_types": dict(
            Counter(row["review_failure_type"] for row in rows if not row["review_success"])
        ),
        "video_contract": {
            "videos_verified": len(rows),
            "width": 640,
            "height": 480,
            "fps": 20,
            "total_video_frames": total_policy_ticks,
            "total_policy_ticks": total_policy_ticks,
            "all_frame_counts_equal_policy_tick_counts": True,
            "all_last_policy_ticks_cover_29.9_to_30.1_seconds": True,
            "note": (
                "The MP4 contains one canonical RGB frame per actual policy tick. "
                "The no-catch-up loop produced 593 or 594 ticks over each 30-second "
                "wall-clock policy window, so encoded playback at fixed 20 FPS is "
                "29.65 or 29.70 seconds while the final tick timestamps are "
                "29.95-30.00 seconds."
            ),
        },
        "source_index_discrepancies": source_index_discrepancies,
        "interpretation_boundary": (
            "Descriptive canonical-video review of the frozen paired real "
            "evaluation only; no causal model-performance or paper conclusion."
        ),
    }
    write_json(summary_path, summary)
    validation = {
        "schema_version": 1,
        "review_id": EXPECTED_REVIEW_ID,
        "operator_input_hash_list_entries_verified": len(operator_hashes),
        "operator_input_hashes_verified": True,
        "review_decisions_count": len(decisions),
        "review_labels_count": len(label_payloads),
        "videos": video_validations,
        "all_videos_640x480_at_20_fps": True,
        "all_video_frames_equal_policy_ticks": True,
        "all_trials_cover_full_30_second_wall_clock_window": True,
        "all_review_labels_bound_to_exact_video_sha256": True,
        "operator_labels_preserved_and_compared_separately": True,
        "adjudication_files_required": len(disagreements),
    }
    write_json(validation_path, validation)

    future_output_paths = [labels_root / name for name, _ in label_payloads] + [
        trials_path,
        summary_path,
        validation_path,
        manifest_path,
    ]
    files_to_hash = set(operator_hashes)
    files_to_hash.update(
        {
            PLAN_PATH,
            DECISIONS_PATH,
            OPERATOR_MANIFEST_PATH,
            OPERATOR_SUMMARY_PATH,
            OPERATOR_HASHES_PATH,
            OPERATOR_INDEX_PATH,
        }
    )
    files_to_hash.update(future_output_paths)
    manifest = {
        "schema_version": 1,
        "review_id": EXPECTED_REVIEW_ID,
        "evaluation_id": plan["evaluation_id"],
        "result_scope": "independent canonical-video review complete",
        "operator_result_source_commit": EXPECTED_OPERATOR_RESULT_SOURCE_COMMIT,
        "plan_path": str(PLAN_PATH),
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "operator_inputs": {
            "manifest_path": str(OPERATOR_MANIFEST_PATH),
            "manifest_sha256": EXPECTED_OPERATOR_MANIFEST_SHA256,
            "summary_path": str(OPERATOR_SUMMARY_PATH),
            "summary_sha256": EXPECTED_OPERATOR_SUMMARY_SHA256,
            "hashes_path": str(OPERATOR_HASHES_PATH),
            "hashes_sha256": EXPECTED_OPERATOR_HASHES_SHA256,
            "hashed_artifacts_verified": len(operator_hashes),
        },
        "review_decisions_source_path": str(DECISIONS_PATH),
        "review_decisions_source_sha256": decision_source_sha256,
        "review_method": decisions_source["review_method"],
        "success_definition": decisions_source["success_definition"],
        "trial_table_path": str(trials_path),
        "reviewed_summary_path": str(summary_path),
        "validation_path": str(validation_path),
        "review_labels": [
            {
                "trial_id": row["trial_id"],
                "review_label_path": row["review_label_path"],
                "review_label_sha256": row["review_label_sha256"],
            }
            for row in rows
        ],
        "operator_review_disagreements": len(disagreements),
        "adjudication_policy": (
            "Operator labels are immutable. Any disagreement requires a new "
            "append-only adjudication artifact; no original label may be edited."
        ),
        "adjudication_artifacts": [],
        "source_index_discrepancies": source_index_discrepancies,
        "hash_manifest_path": str(hashes_path),
        "hash_manifest_excludes_itself": True,
        "hashed_artifact_count": len(files_to_hash),
    }
    write_json(manifest_path, manifest)
    require(
        len(files_to_hash) == len(set(files_to_hash)),
        "Hash path set is not unique.",
    )
    with hashes_path.open("w", encoding="utf-8") as handle:
        for path in sorted(files_to_hash, key=lambda value: str(value)):
            require(path.exists(), f"Missing hash target: {path}")
            handle.write(f"{sha256_file(path)}  {path}\n")

    print(
        json.dumps(
            {
                "review_root": str(REVIEW_ROOT),
                "review_manifest_sha256": sha256_file(manifest_path),
                "review_trials_sha256": sha256_file(trials_path),
                "reviewed_summary_sha256": sha256_file(summary_path),
                "validation_sha256": sha256_file(validation_path),
                "review_hashes_sha256": sha256_file(hashes_path),
                "hashed_artifact_count": len(files_to_hash),
                "operator_review_disagreements": len(disagreements),
                "by_model": by_model,
                "paired_outcomes": paired_outcomes,
                "source_index_discrepancies": source_index_discrepancies,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
