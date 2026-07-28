from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "phase_id",
    "cell",
    "seed",
    "initial_pose",
    "success",
    "env_steps",
    "first_success_step",
    "confirmed_success_step",
    "max_lift_m",
    "is_grasped",
    "terminated",
    "truncated",
    "timeout",
    "termination_reason",
    "raw_action_count",
    "calibration_clipped_action_count",
    "relative_clipped_action_count",
    "sent_action_count",
    "failure_type",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    episodes = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            raise RuntimeError(f"{path}:{line_number} missing fields: {missing}")
        if not isinstance(record["success"], bool):
            raise RuntimeError(f"{path}:{line_number} success must be boolean.")
        if int(record["env_steps"]) <= 0:
            raise RuntimeError(f"{path}:{line_number} env_steps must be positive.")
        if int(record["raw_action_count"]) != int(record["sent_action_count"]):
            raise RuntimeError(f"{path}:{line_number} raw/sent action counts differ.")
        episodes.append(record)
    if not episodes:
        raise RuntimeError("No episode evidence records found.")
    return episodes


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_cell[str(episode["cell"])].append(episode)

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(bool(item["success"]) for item in group)
        return {
            "episodes": len(group),
            "successes": successes,
            "success_rate": successes / len(group),
            "mean_env_steps": sum(int(item["env_steps"]) for item in group) / len(group),
            "calibration_clipped_action_count": sum(
                int(item["calibration_clipped_action_count"]) for item in group
            ),
            "relative_clipped_action_count": sum(
                int(item["relative_clipped_action_count"]) for item in group
            ),
            "failure_types": dict(Counter(str(item["failure_type"]) for item in group if not item["success"])),
        }

    return {
        "status": "diagnostic_only_not_a_real_robot_or_paper_result",
        "overall": group_summary(episodes),
        "by_cell": {cell: group_summary(by_cell[cell]) for cell in sorted(by_cell)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Task1 Real-to-Sim diagnostic evidence.")
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(load_jsonl(args.episodes_jsonl))
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
