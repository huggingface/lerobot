from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from examples.picklift_v3.finalize_real96 import tree_inventory, write_csv, write_json, write_jsonl
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import split_dataset

MATERIALIZATION_ID = "task1_picklift_real24_budget_dataset_materialization_v1"
TARGET_REPO_ID = "local/task1_picklift_real24_budget_extension_v1_accepted"
SOURCE_TREE_SHA256 = "c4534befc536c10217638da91f5cbbaff59b0795ec91f0633e53e8a6d99507b9"
RESEARCH_MANIFEST_SHA256 = "3c2162aa8a9b559921d69bef0c2d2c12e9327c4963a92979458ccea305a54ab3"
ACT_PLAN_SHA256 = "29349a9631814219386ecd84406f1f1fbe35abba62a060be9d6069d8db483b3f"
SELECTION_VECTOR = "011100100101"
SELECTION_SEQUENCE_SHA256 = "054055ec11e3430ce81ae67c3b1b009c3580f1db31182d15b90c403132086ff4"
PLAN_ITEM_IDS = (
    "real96_s01_r1c1_core_center_rep01_yaw00",
    "real96_s01_r2c2_core_center_rep01_yaw00",
    "real96_s01_r2c4_core_center_rep01_yaw00",
    "real96_s01_r1c2_core_center_rep01_yaw00",
    "real96_s01_r2c3_core_center_rep01_yaw00",
    "real96_s01_r1c4_core_center_rep01_yaw00",
    "real96_s02_r3c4_core_q1_rep01_yaw45",
    "real96_s02_r3c3_core_q0_rep01_yaw45",
    "real96_s02_r2c1_core_q2_rep01_yaw45",
    "real96_s02_r3c1_core_q2_rep01_yaw45",
    "real96_s02_r1c3_core_q0_rep01_yaw45",
    "real96_s02_r3c2_core_q3_rep01_yaw45",
    "real96_s03_r2c1_core_center_rep01_yaw45",
    "real96_s03_r3c1_core_center_rep01_yaw45",
    "real96_s03_r3c3_core_center_rep01_yaw45",
    "real96_s03_r1c3_core_center_rep01_yaw45",
    "real96_s03_r3c4_core_center_rep01_yaw45",
    "real96_s03_r3c2_core_center_rep01_yaw45",
    "real96_s04_r1c1_core_q0_rep01_yaw00",
    "real96_s04_r1c4_core_q3_rep01_yaw00",
    "real96_s04_r2c2_core_q1_rep01_yaw00",
    "real96_s04_r1c2_core_q1_rep01_yaw00",
    "real96_s04_r2c4_core_q3_rep01_yaw00",
    "real96_s04_r2c3_core_q2_rep01_yaw00",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_selection(source_root: Path) -> list[dict]:
    map_path = source_root / "provenance/source_episode_map.jsonl"
    rows = [json.loads(line) for line in map_path.read_text().splitlines()]
    if len(rows) != 48:
        raise RuntimeError("source Real48 mapping count failed")
    by_id = {}
    for source_episode_index, row in enumerate(rows):
        by_id[row["plan_item_id"]] = {
            **row,
            "source_real48_episode_index": source_episode_index,
            "source_real96_episode_index": row["derived_episode_index"],
        }
    if set(PLAN_ITEM_IDS) - set(by_id):
        raise RuntimeError("frozen Real24 member missing from Real48")
    selection = [by_id[item_id] for item_id in PLAN_ITEM_IDS]
    if [row["global_order"] for row in selection] != sorted(row["global_order"] for row in selection):
        raise RuntimeError("selection is not in frozen global order")
    return selection


def validate_balance(selection: list[dict]) -> dict:
    checks = {
        "cells": dict(Counter(row["cell"] for row in selection)),
        "sessions": dict(Counter(str(row["session_index"]) for row in selection)),
        "position_kind": dict(Counter(row["position_kind"] for row in selection)),
        "yaw": dict(Counter(str(row["yaw_degrees_modulo_90"]) for row in selection)),
        "quadrants": dict(Counter(row["quadrant"] for row in selection if row["quadrant"])),
    }
    if (
        len(selection) != 24
        or set(checks["cells"].values()) != {2}
        or set(checks["sessions"].values()) != {6}
        or checks["position_kind"] != {"center": 12, "offset": 12}
        or checks["yaw"] != {"0": 12, "45": 12}
        or set(checks["quadrants"].values()) != {3}
    ):
        raise RuntimeError(f"Real24 balance failed: {checks}")
    return checks


def parquet_table(root: Path) -> pa.Table:
    return pa.concat_tables([pq.read_table(path) for path in sorted((root / "data").rglob("*.parquet"))])


def verify_dataset(dataset: LeRobotDataset, expected_frames: int) -> dict:
    if dataset.num_episodes != 24 or dataset.num_frames != expected_frames:
        raise RuntimeError("official loader count failed")
    info = json.loads((dataset.root / "meta/info.json").read_text())
    features = info["features"]
    if info["fps"] != 20 or "observation.images.wrist" in features:
        raise RuntimeError("fps/wrist contract failed")
    if features["observation.state"]["dtype"] != "float32" or features["observation.state"]["shape"] != [6]:
        raise RuntimeError("state schema failed")
    if features["action"]["dtype"] != "float32" or features["action"]["shape"] != [6]:
        raise RuntimeError("action schema failed")
    if features["observation.images.front"]["shape"] != [480, 640, 3]:
        raise RuntimeError("front schema failed")
    table = parquet_table(dataset.root)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise RuntimeError("non-finite state/action")
    episode = np.asarray(table["episode_index"].to_pylist())
    frame = np.asarray(table["frame_index"].to_pylist())
    timestamp = np.asarray(table["timestamp"].to_pylist())
    index = np.asarray(table["index"].to_pylist())
    if not np.array_equal(index, np.arange(expected_frames)):
        raise RuntimeError("global index failed")
    for episode_index in range(24):
        mask = episode == episode_index
        expected = np.arange(mask.sum())
        if not np.array_equal(frame[mask], expected):
            raise RuntimeError(f"frame index failed at episode {episode_index}")
        if not np.allclose(timestamp[mask], expected / 20, atol=1e-5):
            raise RuntimeError(f"timestamp failed at episode {episode_index}")
    for index_value in (0, expected_frames // 2, expected_frames - 1):
        if tuple(dataset[index_value]["observation.images.front"].shape) != (3, 480, 640):
            raise RuntimeError("RGB decode failed")
    return {
        "official_lerobot_0_6_1_loader": "passed",
        "episodes": dataset.num_episodes,
        "frames": dataset.num_frames,
        "front_640x480_rgb_20fps": "passed",
        "state_action_float32_6_finite": "passed",
        "no_wrist": "passed",
        "frame_episode_index_timestamp": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-parent", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    target_root = args.target_parent / "accepted"
    if args.target_parent.exists() or args.evidence_root.exists():
        raise FileExistsError("target/evidence already exists")
    source_inventory, source_sha, source_bytes = tree_inventory(args.source_root)
    if source_sha != SOURCE_TREE_SHA256:
        raise RuntimeError(f"source Real48 tree changed: {source_sha}")
    selection = load_selection(args.source_root)
    balances = validate_balance(selection)
    source_dataset = LeRobotDataset("local/task1_picklift_real96_accepted_v1_accepted", root=args.source_root)
    source_indices = [row["source_real48_episode_index"] for row in selection]
    expected_frames = sum(row["recorded_frames"] for row in selection)
    split = split_dataset(source_dataset, {"accepted": source_indices}, output_dir=args.target_parent)[
        "accepted"
    ]
    if split.root != target_root:
        raise RuntimeError("unexpected target root")
    materialized = LeRobotDataset(TARGET_REPO_ID, root=target_root)
    provenance = target_root / "provenance"
    provenance.mkdir()
    mapping = [
        {
            **row,
            "derived_episode_index": index,
            "selection_source": "frozen_plan_item_id_membership",
        }
        for index, row in enumerate(selection)
    ]
    write_jsonl(provenance / "source_episode_map.jsonl", mapping)
    write_json(
        provenance / "materialization_identity.json",
        {
            "materialization_id": MATERIALIZATION_ID,
            "research_contract_commit": "428bdcd",
            "research_manifest_sha256": RESEARCH_MANIFEST_SHA256,
            "act_plan_sha256": ACT_PLAN_SHA256,
            "selection_vector": SELECTION_VECTOR,
            "selection_sequence_sha256": SELECTION_SEQUENCE_SHA256,
            "source_root": str(args.source_root),
            "source_tree_sha256": source_sha,
            "source_real48_mapping_note": (
                "source_episode_map line order is the actual Real48 episode order; its inherited "
                "derived_episode_index is the upstream Real96 episode index"
            ),
            "payload_handling": (
                "Official LeRobot split_dataset copied/reindexed state/action/data without "
                "resampling and re-encoded AV1 video because selected episodes are non-contiguous."
            ),
            "hardware_accessed": False,
            "training_started": False,
        },
    )
    loader_qa = verify_dataset(materialized, expected_frames)
    target_inventory, target_sha, target_bytes = tree_inventory(target_root)
    args.evidence_root.mkdir(parents=True)
    write_csv(args.evidence_root / "tree_inventory.csv", target_inventory)
    write_jsonl(args.evidence_root / "real24_subset_manifest.jsonl", mapping)
    source_values = parquet_table(args.source_root)
    target_values = parquet_table(target_root)
    selected_source_frames = []
    source_episode = np.asarray(source_values["episode_index"].to_pylist())
    for source_index in source_indices:
        selected_source_frames.extend(np.flatnonzero(source_episode == source_index).tolist())
    for key in ("observation.state", "action"):
        source_array = np.asarray(source_values[key].to_pylist())[selected_source_frames]
        target_array = np.asarray(target_values[key].to_pylist())
        if not np.array_equal(source_array, target_array):
            raise RuntimeError(f"{key} changed during materialization")
    summary = {
        "materialization_id": MATERIALIZATION_ID,
        "dataset_root": str(target_root),
        "repo_id": materialized.repo_id,
        "source_tree_sha256": source_sha,
        "source_file_count": len(source_inventory),
        "source_total_bytes": source_bytes,
        "episodes": materialized.num_episodes,
        "frames": materialized.num_frames,
        "file_count": len(target_inventory),
        "total_bytes": target_bytes,
        "tree_sha256": target_sha,
        "manifest_sha256": sha256_bytes((args.evidence_root / "real24_subset_manifest.jsonl").read_bytes()),
        "balances": balances,
        "numeric_payload_identity": "state/action arrays exactly equal selected source frames",
        "video_payload": "official AV1 re-encode; no episode/frame resampling",
        "loader_qa": loader_qa,
        "real24_is_strict_frozen_real48_subset": True,
        "hardware_accessed": False,
        "training_started": False,
        "result": "passed",
    }
    write_json(args.evidence_root / "materialization_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
