from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from examples.picklift_v3.real96_plan import real96_items
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets, split_dataset

FINALIZATION_ID = "task1_picklift_real96_global_finalization_v1"
RAW_TREE_SHA256 = {
    1: "f628ab551aadea2c40402a48b7bda9625342d2b7921e960035e9620f2970fd2a",
    2: "ed02dda9b55400b3a4be6a837c98fff3db83ff4c7ffd5de2466c64eb39773f4b",
    3: "4cc2705007a6be46c1493321d75ab137dacbeb687d809699a42aef9706638c42",
    4: "ee6f822a7717280b0ad861bed63677a5dc1a11dc3d8a3cb3534bee51ea96a1b7",
}
EXPECTED_FRAMES = {1: 5451, 2: 4485, 3: 3993, 4: 3752}
EXPECTED_RAW_FRAMES = 17681
EXPECTED_ACCEPTED_FRAMES = 17439


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_inventory(root: Path) -> tuple[list[dict], str, int]:
    rows = []
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_digest = sha256_file(path)
        size = path.stat().st_size
        rows.append({"path": relative, "bytes": size, "sha256": file_digest})
        digest.update(f"{file_digest}  {relative}\n".encode())
        total_bytes += size
    return rows, digest.hexdigest(), total_bytes


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(raw_roots: dict[int, Path], evidence_root: Path) -> tuple[list[dict], list[dict]]:
    planned = {item["plan_item_id"]: item for item in real96_items()}
    ledger = []
    accepted = []
    identity_reference = None
    identity_fields = (
        "task_contract_id",
        "task_version",
        "task",
        "camera_profile_id",
        "ready_pose_profile",
        "ready_pose_state_sha256",
        "record_fps",
        "control_hz",
        "pose_manifest_sha256",
        "collection_plan_sha256",
        "subset_manifest_sha256",
    )
    for session_index, root in raw_roots.items():
        inventory, tree_sha, total_bytes = tree_inventory(root)
        if tree_sha != RAW_TREE_SHA256[session_index]:
            raise RuntimeError(f"Session {session_index} raw tree changed: {tree_sha}")
        write_json(
            evidence_root / f"s{session_index:02d}_raw_rehash.json",
            {
                "root": str(root),
                "tree_sha256": tree_sha,
                "file_count": len(inventory),
                "total_bytes": total_bytes,
            },
        )
        session = json.loads((root / "provenance/session.json").read_text())
        current_identity = {key: session[key] for key in identity_fields}
        if identity_reference is None:
            identity_reference = current_identity
        elif current_identity != identity_reference:
            raise RuntimeError(f"Session {session_index} frozen identity drift")
        attempts = [
            json.loads(path.read_text()) for path in sorted((root / "provenance/attempts").glob("*.json"))
        ]
        if session["accepted_dataset_episode_indices"] != [
            item["episode_index"] for item in attempts if item["success"]
        ]:
            raise RuntimeError(f"Session {session_index} accepted index mismatch")
        table = pq.read_table(root / "data/chunk-000/file-000.parquet")
        if len(table) != EXPECTED_FRAMES[session_index]:
            raise RuntimeError(f"Session {session_index} frame count mismatch")
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if state.shape != (len(table), 6) or action.shape != (len(table), 6):
            raise RuntimeError("state/action shape mismatch")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise RuntimeError("state/action non-finite")
        if "observation.images.wrist" in table.column_names:
            raise RuntimeError("unexpected wrist field")
        dataset = LeRobotDataset(session["repo_id"], root=root)
        if dataset.num_episodes != len(attempts) or dataset.num_frames != len(table):
            raise RuntimeError(f"Session {session_index} official loader mismatch")
        for attempt in attempts:
            item = planned[attempt["plan_item_id"]]
            accepted_episode_id = (
                f"{attempt['session_id']}:episode_{attempt['episode_index']:06d}"
                if attempt["success"]
                else None
            )
            row = {
                "session_id": attempt["session_id"],
                "session_index": attempt["session_index"],
                "attempt_id": attempt["attempt_id"],
                "source_episode_index": attempt["episode_index"],
                "plan_item_id": attempt["plan_item_id"],
                "global_order": attempt["global_order"],
                "result": attempt["result"],
                "accepted": bool(attempt["success"]),
                "accepted_episode_id": accepted_episode_id,
                "recorded_frames": attempt["recorded_frames"],
                "real48_member": item["real48_member"],
                "raw_root": str(root),
                "freeze_id": f"task1_picklift_real96_s{session_index:02d}_raw_attempts_freeze_v1",
                "raw_tree_sha256": tree_sha,
            }
            ledger.append(row)
            if row["accepted"]:
                accepted.append({**row, **item})
    return ledger, accepted


def validate_global(ledger: list[dict], accepted: list[dict]) -> dict:
    if len(ledger) != 98 or Counter(row["result"] for row in ledger) != {
        "success": 96,
        "discard": 2,
    }:
        raise RuntimeError("global attempt ledger cardinality failed")
    if len({row["attempt_id"] for row in ledger}) != 98:
        raise RuntimeError("attempt_id uniqueness failed")
    if len({row["accepted_episode_id"] for row in accepted}) != 96:
        raise RuntimeError("accepted episode identity uniqueness failed")
    if Counter(row["plan_item_id"] for row in accepted) != Counter(
        item["plan_item_id"] for item in real96_items()
    ):
        raise RuntimeError("accepted plan membership failed")
    if sorted(row["global_order"] for row in accepted) != list(range(1, 97)):
        raise RuntimeError("global order coverage failed")
    real48 = [row for row in accepted if row["real48_member"]]
    overall = {
        "cells": Counter(row["cell"] for row in accepted),
        "subset_role": Counter(row["subset_role"] for row in accepted),
        "position_kind": Counter(row["position_kind"] for row in accepted),
        "yaw": Counter(row["yaw_degrees_modulo_90"] for row in accepted),
        "quadrants": Counter(row["quadrant"] for row in accepted if row["quadrant"]),
        "sessions": Counter(row["session_index"] for row in accepted),
    }
    subset = {
        "cells": Counter(row["cell"] for row in real48),
        "position_kind": Counter(row["position_kind"] for row in real48),
        "yaw": Counter(row["yaw_degrees_modulo_90"] for row in real48),
        "quadrants": Counter(row["quadrant"] for row in real48 if row["quadrant"]),
    }
    if set(overall["cells"].values()) != {8} or overall["subset_role"] != {
        "core": 48,
        "extension": 48,
    }:
        raise RuntimeError("Real96 balance failed")
    if overall["position_kind"] != {"center": 48, "offset": 48}:
        raise RuntimeError("Real96 position balance failed")
    if overall["yaw"] != {0: 48, 45: 48} or set(overall["quadrants"].values()) != {12}:
        raise RuntimeError("Real96 yaw/quadrant balance failed")
    if overall["sessions"] != {1: 24, 2: 24, 3: 24, 4: 24}:
        raise RuntimeError("session balance failed")
    if len(real48) != 48 or set(subset["cells"].values()) != {4}:
        raise RuntimeError("Real48 membership balance failed")
    if subset["position_kind"] != {"center": 24, "offset": 24}:
        raise RuntimeError("Real48 position balance failed")
    if subset["yaw"] != {0: 24, 45: 24} or set(subset["quadrants"].values()) != {6}:
        raise RuntimeError("Real48 yaw/quadrant balance failed")
    return {"real96": overall, "real48": subset}


def derive_datasets(
    raw_roots: dict[int, Path], accepted: list[dict], output_root: Path, evidence_root: Path
) -> list[dict]:
    staging = output_root / "staging_accepted_sessions"
    sources = []
    for session_index, root in raw_roots.items():
        dataset = LeRobotDataset(f"local/task1_real96_s{session_index:02d}_raw_attempts", root=root)
        keep = sorted(
            row["source_episode_index"] for row in accepted if row["session_index"] == session_index
        )
        split = split_dataset(dataset, {"accepted": keep}, output_dir=staging / f"s{session_index:02d}")
        sources.append(split["accepted"])
    real96_root = output_root / "task1_picklift_real96_accepted_v1"
    real96 = merge_datasets(
        sources,
        output_repo_id="local/task1_picklift_real96_accepted_v1",
        output_dir=real96_root,
    )
    real96_map = []
    for derived_index, row in enumerate(sorted(accepted, key=lambda item: item["global_order"])):
        real96_map.append({"derived_episode_index": derived_index, **row})
    provenance = real96_root / "provenance"
    provenance.mkdir()
    write_jsonl(provenance / "source_episode_map.jsonl", real96_map)
    real48_indices = [row["derived_episode_index"] for row in real96_map if row["real48_member"]]
    real48_container = output_root / "task1_picklift_real48_accepted_v1"
    real48 = split_dataset(real96, {"accepted": real48_indices}, output_dir=real48_container)["accepted"]
    real48_root = real48.root
    real48_source = [row for row in real96_map if row["real48_member"]]
    provenance = real48_root / "provenance"
    provenance.mkdir()
    write_jsonl(
        provenance / "source_episode_map.jsonl",
        [{"derived_episode_index": index, **row} for index, row in enumerate(real48_source)],
    )
    outputs = []
    for subset_id, dataset, root, expected_episodes in (
        ("Real96", real96, real96_root, 96),
        ("Real48", real48, real48_root, 48),
    ):
        loaded = LeRobotDataset(dataset.repo_id, root=root)
        if loaded.num_episodes != expected_episodes:
            raise RuntimeError(f"{subset_id} loader episode count failed")
        expected_frames = sum(
            row["recorded_frames"] for row in accepted if subset_id == "Real96" or row["real48_member"]
        )
        if loaded.num_frames != expected_frames:
            raise RuntimeError(f"{subset_id} loader frame count failed")
        for index in (0, loaded.num_frames // 2, loaded.num_frames - 1):
            if tuple(loaded[index]["observation.images.front"].shape) != (3, 480, 640):
                raise RuntimeError(f"{subset_id} RGB decode failed")
        inventory, tree_sha, total_bytes = tree_inventory(root)
        inventory_path = evidence_root / f"{subset_id.lower()}_tree_inventory.csv"
        write_csv(inventory_path, inventory)
        freeze = {
            "dataset_id": f"task1_picklift_{subset_id.lower()}_accepted_v1",
            "root": str(root),
            "repo_id": loaded.repo_id,
            "lerobot_version": "0.6.1",
            "episodes": loaded.num_episodes,
            "frames": loaded.num_frames,
            "file_count": len(inventory),
            "total_bytes": total_bytes,
            "tree_sha256": tree_sha,
            "source_mapping": str(root / "provenance/source_episode_map.jsonl"),
        }
        write_json(evidence_root / f"{subset_id.lower()}_dataset_freeze.json", freeze)
        outputs.append(freeze)
    if outputs[0]["frames"] != EXPECTED_ACCEPTED_FRAMES:
        raise RuntimeError("Real96 accepted frame total failed")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    if args.evidence_root.exists():
        raise FileExistsError(args.evidence_root)
    output_root = args.artifacts_root / "derived"
    if output_root.exists():
        raise FileExistsError(output_root)
    args.evidence_root.mkdir(parents=True)
    output_root.mkdir(parents=True)
    raw_roots = {
        index: args.artifacts_root / f"task1_picklift_real96_s0{index}_raw_attempts_20260802"
        for index in range(1, 5)
    }
    ledger, accepted = load_inputs(raw_roots, args.evidence_root)
    balances = validate_global(ledger, accepted)
    real96_manifest = sorted(accepted, key=lambda row: row["global_order"])
    real48_manifest = [row for row in real96_manifest if row["real48_member"]]
    write_csv(args.evidence_root / "global_attempt_ledger.csv", ledger)
    write_jsonl(args.evidence_root / "real96_accepted_manifest.jsonl", real96_manifest)
    write_jsonl(args.evidence_root / "real48_accepted_manifest.jsonl", real48_manifest)
    outputs = derive_datasets(raw_roots, accepted, output_root, args.evidence_root)
    summary = {
        "finalization_id": FINALIZATION_ID,
        "research_contract_commit": "73908355df1add52cd04753216c13f8b1c0b400a",
        "attempts": len(ledger),
        "outcomes": dict(Counter(row["result"] for row in ledger)),
        "real96_accepted": len(real96_manifest),
        "real48_accepted": len(real48_manifest),
        "real48_is_strict_preregistered_subset": True,
        "raw_attempt_frames": sum(row["recorded_frames"] for row in ledger),
        "accepted_real96_frames": sum(row["recorded_frames"] for row in real96_manifest),
        "discard_frames_excluded": sum(row["recorded_frames"] for row in ledger if not row["accepted"]),
        "frame_count_reconciliation": (
            "17681 is the 98-attempt raw total; accepted Real96 is 17439 after excluding "
            "the two retained DISCARD attempts (79 + 163 frames)."
        ),
        "balances": balances,
        "datasets": outputs,
        "raw_unchanged": True,
        "hardware_accessed": False,
        "training_started": False,
        "qa": "passed",
    }
    write_json(args.evidence_root / "cross_session_qa_summary.json", summary)
    if summary["raw_attempt_frames"] != EXPECTED_RAW_FRAMES:
        raise RuntimeError("raw attempt frame total failed")
    print(json.dumps(summary, indent=2, default=dict))


if __name__ == "__main__":
    main()
