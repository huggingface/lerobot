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
    "/home/ubuntu24/Teleop/artifacts/evaluation/"
    "task1_picklift_real24_vs_questsim48_v7_offcenter_yaw_paired_eval24_v1"
)
OPERATOR_MANIFEST_PATH = EVIDENCE_ROOT / "operator_manifest_v1.json"
OPERATOR_SUMMARY_PATH = EVIDENCE_ROOT / "operator_summary_v1.json"
OPERATOR_HASHES_PATH = EVIDENCE_ROOT / "operator_evidence_hashes_v1.sha256"
REVIEW_ROOT = EVIDENCE_ROOT / "canonical_video_review_v1"

EXPECTED_PLAN_SHA256 = "f3f6e797ce001752b46f5f278dcb94df712a29d3e1e29708228810a82074406e"
EXPECTED_DECISIONS_SHA256 = "41d0fe567e22863c9ccdf55639b9f12887fe34d86f2624fe9818089e6253ed6e"
EXPECTED_OPERATOR_MANIFEST_SHA256 = "e11f387fd54b31c12d17897a36fb988ecc19cb42dc82ae5f6fe161ca043024f4"
EXPECTED_OPERATOR_SUMMARY_SHA256 = "46f59adf40d42cba5fef0894f5ebdad20618c83b3fe464e5e78d57e114f50650"
EXPECTED_OPERATOR_HASHES_SHA256 = "5bb74a79facf5e99a66b95538cc9f13b5cc0752e880e2e3704ddeda88c3e33f4"


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


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
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
        "duration_seconds": float(stream["duration"]),
    }


def inspect_steps(path: Path) -> dict:
    first = None
    last = None
    lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            first = row if first is None else first
            last = row
            lines += 1
    require(first is not None and last is not None, f"Empty steps file: {path}")
    return {
        "lines": lines,
        "first_step": first["step"],
        "last_step": last["step"],
        "first_tick_elapsed_seconds": first["tick_started_elapsed_seconds"],
        "last_tick_elapsed_seconds": last["tick_started_elapsed_seconds"],
    }


def main() -> None:
    require(not REVIEW_ROOT.exists(), f"Refusing to overwrite review root: {REVIEW_ROOT}")
    expected_inputs = {
        PLAN_PATH: EXPECTED_PLAN_SHA256,
        DECISIONS_PATH: EXPECTED_DECISIONS_SHA256,
        OPERATOR_MANIFEST_PATH: EXPECTED_OPERATOR_MANIFEST_SHA256,
        OPERATOR_SUMMARY_PATH: EXPECTED_OPERATOR_SUMMARY_SHA256,
        OPERATOR_HASHES_PATH: EXPECTED_OPERATOR_HASHES_SHA256,
    }
    for path, digest in expected_inputs.items():
        require(path.exists(), f"Missing review input: {path}")
        require(sha256_file(path) == digest, f"Review input hash mismatch: {path}")
    for raw in OPERATOR_HASHES_PATH.read_text(encoding="utf-8").splitlines():
        digest, raw_path = raw.split("  ", 1)
        path = Path(raw_path)
        require(path.exists(), f"Missing operator-frozen artifact: {path}")
        require(sha256_file(path) == digest, f"Operator artifact hash mismatch: {path}")

    plan = read_json(PLAN_PATH)
    operator_manifest = read_json(OPERATOR_MANIFEST_PATH)
    decisions_source = read_json(DECISIONS_PATH)
    require(
        decisions_source["review_labels_are_separate_from_operator_labels"] is True,
        "Review labels are not declared separately.",
    )
    trials = operator_manifest["scored_trials"]
    decisions = decisions_source["decisions"]
    require(len(trials) == 24 and len(decisions) == 24, "Expected 24 scored review rows.")
    require(
        [row["trial_id"] for row in trials] == [row["trial_id"] for row in decisions],
        "Review decision order differs from the frozen trial order.",
    )
    REVIEW_ROOT.mkdir(parents=True)
    rows = []
    paths_to_hash: set[Path] = set(expected_inputs)
    for trial, decision in zip(trials, decisions, strict=True):
        video_path = Path(trial["video_path"])
        steps_path = Path(trial["steps_path"])
        require(sha256_file(video_path) == trial["video_sha256"], "Video SHA mismatch.")
        require(sha256_file(steps_path) == trial["steps_sha256"], "Steps SHA mismatch.")
        video = probe_video(video_path)
        steps = inspect_steps(steps_path)
        require((video["width"], video["height"]) == (640, 480), "Video is not 640x480.")
        require(Fraction(video["avg_frame_rate"]) == 20, "Video is not encoded at 20 FPS.")
        require(video["frames"] == trial["policy_ticks"], "Video frame count mismatch.")
        require(video["frames"] == steps["lines"], "Video/steps line count mismatch.")
        require(steps["first_step"] == 0, "Steps do not start at zero.")
        require(steps["last_step"] == steps["lines"] - 1, "Steps are not contiguous.")
        require(
            steps["last_tick_elapsed_seconds"] >= 29.9,
            "Policy evidence does not reach the full 30-second wall window.",
        )
        require(
            decision["review_failure_type"] is None
            if decision["review_success"]
            else decision["review_failure_type"] == "missed_grasp",
            "Review success/failure type is inconsistent.",
        )
        rows.append(
            {
                **trial,
                "review_success": decision["review_success"],
                "review_failure_type": decision["review_failure_type"],
                "review_interval_seconds": decision["review_interval_seconds"],
                "review_visible_evidence": decision["visible_evidence"],
                "operator_review_agree": (
                    trial["operator_success"] == decision["review_success"]
                ),
                "video_probe": video,
                "steps_probe": steps,
            }
        )
        paths_to_hash.update({video_path, steps_path})

    by_model = {}
    for model_id in plan["models"]:
        selected = [row for row in rows if row["model_id"] == model_id]
        successes = sum(row["review_success"] for row in selected)
        by_model[model_id] = {
            "trials": len(selected),
            "reviewed_successes": successes,
            "reviewed_failures": len(selected) - successes,
            "reviewed_success_rate": successes / len(selected),
            "failure_types": dict(
                Counter(row["review_failure_type"] for row in selected if not row["review_success"])
            ),
        }
    paired = []
    paired_counts = Counter()
    for source_pose_order in range(1, 13):
        selected = [
            row
            for row in rows
            if plan["trials"][row["order"] - 1]["source_pose_order"] == source_pose_order
        ]
        require(len(selected) == 2, "Paired source pose is incomplete.")
        by_id = {row["model_id"]: row for row in selected}
        real = by_id["real24_only"]["review_success"]
        sim = by_id["questsim48_v7"]["review_success"]
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
                "source_pose_order": source_pose_order,
                "source_pose_id": selected[0]["source_pose_id"],
                "cell_id": selected[0]["cell_id"],
                "real24_only_review_success": real,
                "questsim48_v7_review_success": sim,
                "paired_review_outcome": outcome,
            }
        )
    disagreements = [
        {
            "trial_id": row["trial_id"],
            "operator_success": row["operator_success"],
            "review_success": row["review_success"],
        }
        for row in rows
        if not row["operator_review_agree"]
    ]
    rows_path = REVIEW_ROOT / "trials.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "review_id": decisions_source["review_id"],
        "evaluation_id": plan["evaluation_id"],
        "by_model": by_model,
        "paired_pose_results": paired,
        "paired_outcomes": dict(paired_counts),
        "operator_review_disagreements": disagreements,
        "operator_review_agreement_count": 24 - len(disagreements),
        "infrastructure_invalid_original_count": len(
            operator_manifest["infrastructure_invalid_originals"]
        ),
        "video_contract": {
            "videos": 24,
            "width": 640,
            "height": 480,
            "encoded_fps": 20,
            "all_frame_counts_match_steps": True,
            "all_policy_windows_reach_at_least_29p9_seconds": True,
        },
        "interpretation_boundary": (
            "Fresh paired engineering evaluation; do not generalize beyond this "
            "single 12-pose set or treat it as a paper-scale effect estimate."
        ),
    }
    summary_path = REVIEW_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "review_id": decisions_source["review_id"],
        "evaluation_id": plan["evaluation_id"],
        "status": "canonical_video_review_frozen",
        "inputs": {str(path): digest for path, digest in expected_inputs.items()},
        "trials_jsonl": {"path": str(rows_path), "sha256": sha256_file(rows_path)},
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "review_method": decisions_source["review_method"],
    }
    manifest_path = REVIEW_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths_to_hash.update({rows_path, summary_path, manifest_path})
    hashes_path = REVIEW_ROOT / "hashes.sha256"
    hashes_path.write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in sorted(paths_to_hash)),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "review_root": str(REVIEW_ROOT),
                "manifest_sha256": sha256_file(manifest_path),
                "summary_sha256": sha256_file(summary_path),
                "trials_sha256": sha256_file(rows_path),
                "hashes_sha256": sha256_file(hashes_path),
                "by_model": by_model,
                "paired_outcomes": dict(paired_counts),
                "operator_review_disagreements": disagreements,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
