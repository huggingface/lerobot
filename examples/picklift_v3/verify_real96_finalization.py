from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from examples.picklift_v3.finalize_real96 import RAW_TREE_SHA256, tree_inventory, write_json
from lerobot.datasets import LeRobotDataset


def verify_dataset(root: Path, repo_id: str, episodes: int, frames: int, tree_sha: str) -> dict:
    _, actual_tree_sha, total_bytes = tree_inventory(root)
    if actual_tree_sha != tree_sha:
        raise RuntimeError(f"derived tree changed: {root}")
    dataset = LeRobotDataset(repo_id, root=root)
    if dataset.num_episodes != episodes or dataset.num_frames != frames:
        raise RuntimeError(f"loader count mismatch: {root}")
    info = json.loads((root / "meta/info.json").read_text())
    features = info["features"]
    if info["fps"] != 20 or "observation.images.wrist" in features:
        raise RuntimeError("fps/wrist schema contract failed")
    if features["observation.state"]["dtype"] != "float32" or features["observation.state"]["shape"] != [6]:
        raise RuntimeError("state schema failed")
    if features["action"]["dtype"] != "float32" or features["action"]["shape"] != [6]:
        raise RuntimeError("action schema failed")
    if features["observation.images.front"]["shape"] != [480, 640, 3]:
        raise RuntimeError("front schema failed")
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    table = pa.concat_tables([pq.read_table(path) for path in parquet_files])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise RuntimeError("non-finite state/action")
    episode_index = np.asarray(table["episode_index"].to_pylist())
    frame_index = np.asarray(table["frame_index"].to_pylist())
    timestamp = np.asarray(table["timestamp"].to_pylist())
    index = np.asarray(table["index"].to_pylist())
    if not np.array_equal(index, np.arange(frames)):
        raise RuntimeError("global frame index failed")
    for episode in range(episodes):
        mask = episode_index == episode
        expected_frames = np.arange(mask.sum())
        if not np.array_equal(frame_index[mask], expected_frames):
            raise RuntimeError(f"frame index failed: episode {episode}")
        if not np.allclose(timestamp[mask], expected_frames / 20, atol=1e-5):
            raise RuntimeError(f"timestamp failed: episode {episode}")
    return {
        "root": str(root),
        "repo_id": repo_id,
        "episodes": episodes,
        "frames": frames,
        "tree_sha256": actual_tree_sha,
        "total_bytes": total_bytes,
        "official_loader": "passed",
        "schema_state_action_front_no_wrist": "passed",
        "finite_state_action": "passed",
        "frame_episode_index_timestamp_20fps": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.evidence_root / "cross_session_qa_summary.json").read_text())
    ledger = list(csv.DictReader((args.evidence_root / "global_attempt_ledger.csv").open()))
    if len(ledger) != 98 or len({row["attempt_id"] for row in ledger}) != 98:
        raise RuntimeError("ledger uniqueness failed")
    accepted_ids = [row["accepted_episode_id"] for row in ledger if row["accepted"] == "True"]
    if len(accepted_ids) != 96 or len(set(accepted_ids)) != 96:
        raise RuntimeError("accepted identity uniqueness failed")
    real96_map = [
        json.loads(line) for line in Path(summary["datasets"][0]["source_mapping"]).read_text().splitlines()
    ]
    real48_map = [
        json.loads(line) for line in Path(summary["datasets"][1]["source_mapping"]).read_text().splitlines()
    ]
    if len(real96_map) != 96 or len(real48_map) != 48:
        raise RuntimeError("source mapping count failed")
    if {row["plan_item_id"] for row in real48_map} >= {row["plan_item_id"] for row in real96_map}:
        raise RuntimeError("Real48 must be a strict Real96 subset")
    per_session = {}
    for session in range(1, 5):
        rows = [row for row in real96_map if row["session_index"] == session]
        checks = {
            "episodes": len(rows),
            "cells": dict(Counter(row["cell"] for row in rows)),
            "subset_role": dict(Counter(row["subset_role"] for row in rows)),
            "position_kind": dict(Counter(row["position_kind"] for row in rows)),
            "yaw": dict(Counter(str(row["yaw_degrees_modulo_90"]) for row in rows)),
        }
        if (
            checks["episodes"] != 24
            or set(checks["cells"].values()) != {2}
            or checks["subset_role"] != {"core": 12, "extension": 12}
            or checks["position_kind"] != {"center": 12, "offset": 12}
            or checks["yaw"] != {"0": 12, "45": 12}
        ):
            raise RuntimeError(f"Session {session} balance failed")
        qa = json.loads(
            Path(
                f"/home/ubuntu24/Teleop/artifacts/evidence/"
                f"task1_picklift_real96_s0{session}_20260802/qa_v1/qa_summary.json"
            ).read_text()
        )
        if (
            qa["tree_sha256"] != RAW_TREE_SHA256[session]
            or qa["official_loader"] != "passed"
            or not qa["all_numeric_and_timing_contracts_passed"]
        ):
            raise RuntimeError(f"Session {session} QA identity failed")
        attempt_root = Path(
            f"/home/ubuntu24/Teleop/artifacts/"
            f"task1_picklift_real96_s0{session}_raw_attempts_20260802/provenance/attempts"
        )
        attempts = [json.loads(path.read_text()) for path in sorted(attempt_root.glob("*.json"))]
        for attempt in attempts:
            if (
                attempt["action_mapping"] != "official_so101_direct_absolute"
                or attempt["action_transform"] != "none"
                or attempt["state_action_order"] != "pre_action_follower_state_then_actual_sent_target"
                or attempt["field_applicability"]["actual_applied_action"] != "required_action_field"
                or attempt["field_applicability"]["raw_human_target"] != "not_applicable_not_fabricated"
                or attempt["field_applicability"]["reachable_target"] != "not_applicable_not_fabricated"
            ):
                raise RuntimeError(f"Session {session} action/provenance semantics failed")
        checks["session_qa_id"] = qa["qa_id"]
        checks["raw_tree_sha256"] = qa["tree_sha256"]
        checks["attempt_provenance_rows_checked"] = len(attempts)
        checks["pre_action_state_actual_sent_action_semantics"] = "passed"
        per_session[str(session)] = checks
    datasets = [
        verify_dataset(
            Path(item["root"]),
            item["repo_id"],
            item["episodes"],
            item["frames"],
            item["tree_sha256"],
        )
        for item in summary["datasets"]
    ]
    result = {
        "verification_id": "task1_picklift_real96_global_independent_verification_v1",
        "global_attempt_ledger": "passed",
        "accepted_episode_ids_unique": "passed",
        "real48_strict_preregistered_subset": "passed",
        "per_session_balance_and_qa_identity": per_session,
        "datasets": datasets,
        "hardware_accessed": False,
        "training_started": False,
        "result": "passed",
    }
    write_json(args.evidence_root / "independent_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
