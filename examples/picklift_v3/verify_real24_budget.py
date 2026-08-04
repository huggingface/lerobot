from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.picklift_v3.finalize_real96 import tree_inventory, write_json
from examples.picklift_v3.materialize_real24_budget import (
    PLAN_ITEM_IDS,
    SOURCE_TREE_SHA256,
    TARGET_REPO_ID,
    load_selection,
    sha256_bytes,
    verify_dataset,
)
from lerobot.datasets import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.evidence_root / "materialization_summary.json").read_text())
    _, source_sha, _ = tree_inventory(args.source_root)
    if source_sha != SOURCE_TREE_SHA256:
        raise RuntimeError("source Real48 tree identity failed")
    inventory, tree_sha, total_bytes = tree_inventory(args.dataset_root)
    if tree_sha != summary["tree_sha256"]:
        raise RuntimeError("Real24 derived tree identity failed")
    mapping_path = args.dataset_root / "provenance/source_episode_map.jsonl"
    mapping = [json.loads(line) for line in mapping_path.read_text().splitlines()]
    if [row["plan_item_id"] for row in mapping] != list(PLAN_ITEM_IDS):
        raise RuntimeError("frozen plan-item sequence failed")
    if [row["derived_episode_index"] for row in mapping] != list(range(24)):
        raise RuntimeError("derived episode mapping failed")
    if len({row["source_real48_episode_index"] for row in mapping}) != 24:
        raise RuntimeError("source episode uniqueness failed")
    source_selection = load_selection(args.source_root)
    if [row["plan_item_id"] for row in source_selection] != list(PLAN_ITEM_IDS):
        raise RuntimeError("strict Real48 subset proof failed")
    action_rows_checked = 0
    for row in mapping:
        attempt_dir = Path(row["raw_root"]) / "provenance/attempts"
        matches = [
            json.loads(path.read_text())
            for path in attempt_dir.glob("*.json")
            if json.loads(path.read_text())["attempt_id"] == row["attempt_id"]
        ]
        if len(matches) != 1:
            raise RuntimeError("raw attempt linkage failed")
        attempt = matches[0]
        if (
            attempt["result"] != "success"
            or attempt["action_mapping"] != "official_so101_direct_absolute"
            or attempt["action_transform"] != "none"
            or attempt["state_action_order"] != "pre_action_follower_state_then_actual_sent_target"
            or attempt["field_applicability"]["actual_applied_action"] != "required_action_field"
        ):
            raise RuntimeError("actual-sent action provenance failed")
        action_rows_checked += 1
    dataset = LeRobotDataset(TARGET_REPO_ID, root=args.dataset_root)
    loader_qa = verify_dataset(dataset, summary["frames"])
    manifest_path = args.evidence_root / "real24_subset_manifest.jsonl"
    if sha256_bytes(manifest_path.read_bytes()) != summary["manifest_sha256"]:
        raise RuntimeError("subset manifest identity failed")
    verification = {
        "verification_id": "task1_picklift_real24_budget_independent_verification_v1",
        "source_real48_tree_sha256": source_sha,
        "dataset_tree_sha256": tree_sha,
        "dataset_file_count": len(inventory),
        "dataset_total_bytes": total_bytes,
        "manifest_sha256": summary["manifest_sha256"],
        "plan_item_sequence": "passed",
        "real24_strict_real48_subset": "passed",
        "source_and_derived_episode_indices_unique": "passed",
        "actual_sent_action_provenance_rows_checked": action_rows_checked,
        "loader_qa": loader_qa,
        "hardware_accessed": False,
        "training_started": False,
        "result": "passed",
    }
    write_json(args.evidence_root / "independent_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
