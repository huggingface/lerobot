#!/usr/bin/env python
"""Aggregates `final_eval_paper_multivariant.py`'s `episode_results.jsonl` into `summary.json`:
native VLABench success (per task, overall) and mechanism-diagnostic means, per model variant.
Pure aggregation -- no rollout, no model, no training.

Usage:
    uv run python examples/safediff_vla/build_final_eval_summary.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from final_eval_paper_multivariant import MACHINE, MODEL_VARIANTS, OUT_DIR, SEEDS, TASKS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)


def main() -> None:
    jsonl_path = OUT_DIR / "episode_results.jsonl"
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]

    per_model: dict[str, dict] = {}
    for model in MODEL_VARIANTS:
        model_records = [r for r in records if r["model_variant"] == model]
        valid = [r for r in model_records if r["status"] == "OK"]
        invalid = [r for r in model_records if r["status"] == "INVALID_ENVIRONMENT"]

        per_task = {}
        for task in TASKS:
            task_valid = [r for r in valid if r["task"] == task]
            n_success = sum(1 for r in task_valid if r["native_success"])
            per_task[task] = {
                "native_success": n_success,
                "n_episodes": len(task_valid),
                "native_sr_str": f"{n_success}/{len(task_valid)}",
            }

        overall_success = sum(1 for r in valid if r["native_success"])
        overall_n = len(valid)

        def mean_of(key: str, vals=valid) -> float | None:
            xs = [r[key] for r in vals if r.get(key) is not None]
            return sum(xs) / len(xs) if xs else None

        per_model[model] = {
            "per_task_native_sr": per_task,
            "overall_native_sr": f"{overall_success}/{overall_n}",
            "overall_native_success_rate": overall_success / overall_n if overall_n else None,
            "n_invalid_environment": len(invalid),
            "contact_based_grasp_diagnostic_rate": mean_of("contact_based_grasp_diagnostic"),
            "insufficient_approach_rate": mean_of("insufficient_approach"),
            "mean_terminal_ee_to_target_dist_m": mean_of("terminal_ee_to_target_dist_m"),
            "mean_target_localization_error_m": mean_of("target_localization_error_m"),
            "entered_local_grasp_rate": mean_of("entered_local_grasp"),
            "mean_episode_len": mean_of("episode_len"),
        }
        logger.info("[%s] overall native SR = %d/%d", model, overall_success, overall_n)

    summary = {
        "machine": MACHINE,
        "seeds": SEEDS,
        "tasks": TASKS,
        "n_total_episodes": len(records),
        "per_model": per_model,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
