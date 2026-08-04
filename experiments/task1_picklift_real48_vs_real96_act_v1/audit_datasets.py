from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset


EVIDENCE_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/"
    "task1_picklift_real48_vs_real96_act_v1/data_and_contract_v1"
)
GLOBAL_EVIDENCE = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_real96_global_finalization_v1"
)
DATASETS = {
    "real48": {
        "root": Path(
            "/home/ubuntu24/Teleop/artifacts/derived/"
            "task1_picklift_real48_accepted_v1/accepted"
        ),
        "repo_id": "local/task1_picklift_real96_accepted_v1_accepted",
        "episodes": 48,
        "frames": 8955,
        "files": 10,
        "bytes": 291460330,
        "tree_sha256": "c4534befc536c10217638da91f5cbbaff59b0795ec91f0633e53e8a6d99507b9",
        "accepted_manifest": GLOBAL_EVIDENCE / "real48_accepted_manifest.jsonl",
        "accepted_manifest_sha256": (
            "c5e40de9adf7f96f8b17cefbc9f5b152d2aa0bf82dfd8c3ad36bf7f01576edb1"
        ),
    },
    "real96": {
        "root": Path(
            "/home/ubuntu24/Teleop/artifacts/derived/task1_picklift_real96_accepted_v1"
        ),
        "repo_id": "local/task1_picklift_real96_accepted_v1",
        "episodes": 96,
        "frames": 17439,
        "files": 10,
        "bytes": 574870325,
        "tree_sha256": "58a5f8fa907c6b4433750c816f0eb80743ee861b06a1dd1356811fbc6800b1a1",
        "accepted_manifest": GLOBAL_EVIDENCE / "real96_accepted_manifest.jsonl",
        "accepted_manifest_sha256": (
            "20cd26f57afdf79ab81b2ce98a9591656910389db0a56673c62bf38c783da668"
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_inventory(root: Path) -> tuple[list[dict], str, int]:
    rows: list[dict] = []
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def audit_dataset(condition: str, spec: dict) -> dict:
    root = spec["root"]
    inventory, tree_sha, total_bytes = tree_inventory(root)
    observed = {
        "files": len(inventory),
        "bytes": total_bytes,
        "tree_sha256": tree_sha,
    }
    expected = {key: spec[key] for key in ("files", "bytes", "tree_sha256")}
    if observed != expected:
        raise RuntimeError(f"{condition} immutable tree mismatch: {observed} != {expected}")
    accepted_manifest_sha = sha256_file(spec["accepted_manifest"])
    if accepted_manifest_sha != spec["accepted_manifest_sha256"]:
        raise RuntimeError(f"{condition} accepted manifest hash mismatch")

    info_path = root / "meta/info.json"
    stats_path = root / "meta/stats.json"
    info = json.loads(info_path.read_text())
    expected_features = {
        "observation.state": ("float32", [6]),
        "action": ("float32", [6]),
        "observation.images.front": ("video", [480, 640, 3]),
    }
    for key, (dtype, shape) in expected_features.items():
        feature = info["features"].get(key)
        if feature is None or feature["dtype"] != dtype or feature["shape"] != shape:
            raise RuntimeError(f"{condition} feature contract mismatch for {key}: {feature}")
    if any(key.startswith("observation.images.") and key != "observation.images.front" for key in info["features"]):
        raise RuntimeError(f"{condition} contains a forbidden extra camera")
    if info["fps"] != 20 or info["total_episodes"] != spec["episodes"] or info["total_frames"] != spec["frames"]:
        raise RuntimeError(f"{condition} metadata cardinality/fps mismatch")
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    expected_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    if state_names != expected_names or action_names != expected_names:
        raise RuntimeError(f"{condition} joint order mismatch")

    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    if len(table) != spec["frames"]:
        raise RuntimeError(f"{condition} parquet frame count mismatch")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.shape != (spec["frames"], 6) or action.shape != (spec["frames"], 6):
        raise RuntimeError(f"{condition} state/action shape mismatch")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise RuntimeError(f"{condition} contains non-finite state/action")

    dataset = LeRobotDataset(spec["repo_id"], root=root, video_backend="pyav")
    if dataset.num_episodes != spec["episodes"] or dataset.num_frames != spec["frames"]:
        raise RuntimeError(f"{condition} official loader cardinality mismatch")
    sample_shapes = []
    for index in (0, spec["frames"] - 1):
        sample = dataset[index]
        shapes = {
            "sample_index": index,
            "observation.state": list(sample["observation.state"].shape),
            "observation.images.front": list(sample["observation.images.front"].shape),
            "action": list(sample["action"].shape),
        }
        if shapes["observation.state"] != [6] or shapes["observation.images.front"] != [3, 480, 640] or shapes["action"] != [6]:
            raise RuntimeError(f"{condition} official loader sample shape mismatch: {shapes}")
        sample_shapes.append(shapes)

    mapping_rows = [
        json.loads(line)
        for line in (root / "provenance/source_episode_map.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(mapping_rows) != spec["episodes"]:
        raise RuntimeError(f"{condition} provenance mapping count mismatch")

    stats = json.loads(stats_path.read_text())
    return {
        "status": "pass",
        "repo_id": spec["repo_id"],
        "root": str(root),
        "episodes": spec["episodes"],
        "frames": spec["frames"],
        "fps": 20,
        "tree": observed,
        "accepted_manifest": {
            "path": str(spec["accepted_manifest"]),
            "sha256": accepted_manifest_sha,
        },
        "info_sha256": sha256_file(info_path),
        "stats_sha256": sha256_file(stats_path),
        "state_stats": stats["observation.state"],
        "action_stats": stats["action"],
        "joint_order": expected_names,
        "feature_contract": expected_features,
        "official_loader": "pass",
        "sample_shapes": sample_shapes,
        "finite_state_action": True,
        "provenance_rows": len(mapping_rows),
        "inventory": inventory,
    }


def audit_discard_boundary() -> dict:
    ledger_path = GLOBAL_EVIDENCE / "global_attempt_ledger.csv"
    rows = list(csv.DictReader(ledger_path.open(newline="")))
    counts = {"success": 0, "discard": 0}
    frames = {"success": 0, "discard": 0}
    for row in rows:
        result = row["result"]
        counts[result] += 1
        frames[result] += int(row["recorded_frames"])
    if len(rows) != 98 or counts != {"success": 96, "discard": 2}:
        raise RuntimeError("Global attempt ledger outcome mismatch")
    if frames != {"success": 17439, "discard": 242}:
        raise RuntimeError(f"Global attempt frame reconciliation mismatch: {frames}")
    if sum(frames.values()) != 17681:
        raise RuntimeError("Raw attempt frame total mismatch")
    expected_hashes = {
        "global_attempt_ledger.csv": "7cbc3776b457c2304c644c5dda01185a30339148af904b3104bb1174aee0c5b7",
        "cross_session_qa_summary.json": "77843c913a120176f0758e3f440e9c9a432412ed7899610a30bb5d0ba350346b",
        "independent_verification.json": "6209b2b4072b56d6cc89dc8053b14b548553c9e6ec8fffcd678d8edb2a5e271f",
    }
    observed_hashes = {
        name: sha256_file(GLOBAL_EVIDENCE / name) for name in expected_hashes
    }
    if observed_hashes != expected_hashes:
        raise RuntimeError("Global finalization evidence hash mismatch")
    return {
        "status": "pass",
        "attempts": len(rows),
        "outcomes": counts,
        "frames": {
            "raw_attempts": 17681,
            "accepted_real96": frames["success"],
            "discard_excluded": frames["discard"],
        },
        "evidence_hashes": observed_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Task1 Real48/Real96 ACT inputs")
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT / "dataset_audit.json")
    args = parser.parse_args()
    result = {
        "schema": "task1_picklift_real48_real96_dataset_audit_v1",
        "status": "pass",
        "hardware_accessed": False,
        "datasets": {name: audit_dataset(name, spec) for name, spec in DATASETS.items()},
        "discard_boundary": audit_discard_boundary(),
        "compatibility": {
            "fps": 20,
            "front_only_640x480": True,
            "state_action_float32_6": True,
            "joint_order_and_units_matched": True,
            "standard_pure_real_frame_sampling": True,
        },
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
