from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from paired_evaluator import DEFAULT_PLAN, load_frozen_plan


MODEL_KEYS = ("real48", "real96")
FAILURE_CATEGORIES = ("missed_grasp", "spatial_offset", "post_grasp_drop", "other")


def resolved_label(sidecar: dict) -> dict:
    operator = sidecar["operator_label"]
    review = sidecar["canonical_video_review_label"]
    if operator.get("status") != "labeled" or review.get("status") != "reviewed":
        raise RuntimeError("Operator and canonical-video review labels must both be frozen.")
    if operator.get("success") == review.get("success"):
        return {
            "success": bool(review["success"]),
            "failure_category": None if review["success"] else review.get("failure_category"),
            "source": "operator_review_agreement",
        }
    adjudication = sidecar.get("adjudication", {})
    if adjudication.get("status") != "complete":
        raise RuntimeError("Operator/review disagreement requires immutable adjudication.")
    return {
        "success": bool(adjudication["success"]),
        "failure_category": None if adjudication["success"] else adjudication.get("failure_category"),
        "source": "adjudication",
    }


def rate_record(successes: int, total: int) -> dict:
    return {"successes": successes, "trials": total, "success_rate": successes / total}


def summarize_rows(rows: list[dict]) -> dict:
    if len(rows) != 96:
        raise RuntimeError("Reviewed Eval48 summary requires exactly 96 scored trials.")
    if {row["model_key"] for row in rows} != set(MODEL_KEYS):
        raise RuntimeError("Reviewed rows do not contain both frozen model keys.")
    overall = {}
    by_tier: dict[str, dict] = defaultdict(dict)
    by_cell: dict[str, dict] = defaultdict(dict)
    by_yaw: dict[str, dict] = defaultdict(dict)
    failures = {}
    for model_key in MODEL_KEYS:
        selected = [row for row in rows if row["model_key"] == model_key]
        overall[model_key] = rate_record(sum(row["success"] for row in selected), len(selected))
        failures[model_key] = dict(
            Counter(row["failure_category"] for row in selected if not row["success"])
        )
        for tier in ("seen_by_real48", "added_by_real96", "unseen_by_both"):
            group = [row for row in selected if row["coverage_tier"] == tier]
            by_tier[tier][model_key] = rate_record(sum(row["success"] for row in group), len(group))
        for cell in sorted({row["cell"] for row in selected}):
            group = [row for row in selected if row["cell"] == cell]
            by_cell[cell][model_key] = rate_record(sum(row["success"] for row in group), len(group))
        for yaw in (0, 45):
            group = [row for row in selected if row["yaw"] == yaw]
            by_yaw[str(yaw)][model_key] = rate_record(sum(row["success"] for row in group), len(group))
    pair_deltas = []
    for pose_order in range(1, 49):
        pair = {row["model_key"]: row for row in rows if row["pose_order"] == pose_order}
        if set(pair) != set(MODEL_KEYS):
            raise RuntimeError(f"Pose {pose_order} is not a complete frozen pair.")
        pair_deltas.append(
            {
                "pose_order": pose_order,
                "eval_pose_id": pair["real48"]["eval_pose_id"],
                "real48_success": pair["real48"]["success"],
                "real96_success": pair["real96"]["success"],
                "real96_minus_real48": int(pair["real96"]["success"]) - int(pair["real48"]["success"]),
            }
        )
    return {
        "overall": overall,
        "by_coverage_tier": dict(by_tier),
        "by_cell": dict(by_cell),
        "by_yaw": dict(by_yaw),
        "failure_categories": failures,
        "paired_real96_minus_real48": {
            "success_count_difference": overall["real96"]["successes"] - overall["real48"]["successes"],
            "success_rate_difference": overall["real96"]["success_rate"] - overall["real48"]["success_rate"],
            "per_pose": pair_deltas,
        },
    }


def load_reviewed_rows(plan: dict, trials_root: Path) -> list[dict]:
    rows = []
    allowed_reasons = set(plan["replacement_contract"]["allowed_only_for"])
    for trial in plan["trials"]:
        stem = trial["artifact_stem"]
        original = trials_root / f"{stem}.paired_eval48.json"
        replacement = trials_root / f"{stem}__replacement1.paired_eval48.json"
        selected = original
        if replacement.exists():
            marker = trials_root / f"{stem}.infrastructure_invalid.json"
            if not original.exists() or not marker.exists():
                raise RuntimeError(f"Replacement for {stem} is not linked to preserved original evidence.")
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            if marker_payload.get("reason") not in allowed_reasons:
                raise RuntimeError(f"Replacement for {stem} has a prohibited reason.")
            selected = replacement
        if not selected.exists():
            raise RuntimeError(f"Missing scored reviewed sidecar for {stem}.")
        sidecar = json.loads(selected.read_text(encoding="utf-8"))
        label = resolved_label(sidecar)
        if not label["success"] and label["failure_category"] not in FAILURE_CATEGORIES:
            raise RuntimeError(f"Unfrozen failure category for {stem}.")
        rows.append(
            {
                "trial_id": trial["trial_id"],
                "scored_sidecar": str(selected),
                "pose_order": trial["pose_order"],
                "eval_pose_id": trial["eval_pose_id"],
                "model_key": trial["model_key"],
                "coverage_tier": trial["coverage_tier"],
                "cell": trial["cell"],
                "yaw": trial["nominal_yaw_degrees_modulo_90"],
                "success": label["success"],
                "failure_category": label["failure_category"],
                "resolved_label_source": label["source"],
                "operator_label": sidecar["operator_label"],
                "canonical_video_review_label": sidecar["canonical_video_review_label"],
                "adjudication": sidecar.get("adjudication"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze reviewed Task1 Real48 versus Real96 Eval48 summary")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite reviewed summary: {args.output}")
    plan = load_frozen_plan(args.plan)
    rows = load_reviewed_rows(plan, args.trials_root)
    payload = {
        "schema": "task1_picklift_real48_vs_real96_eval48_reviewed_summary_v1",
        "evaluation_id": plan["evaluation_id"],
        "status": "review_complete",
        "rows": rows,
        "summary": summarize_rows(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
