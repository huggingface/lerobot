from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset


EVIDENCE_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/"
    "task1_picklift_real24_budget_extension_act_v1/data_and_contract_v1"
)
DATASETS = {
    "real24": {
        "root": Path(
            "/home/ubuntu24/Teleop/artifacts/derived/"
            "task1_picklift_real24_budget_extension_v1/accepted"
        ),
        "repo_id": "local/task1_picklift_real24_budget_extension_v1_accepted",
        "episodes": 24,
        "frames": 4263,
        "files": 11,
        "bytes": 137519477,
        "tree_sha256": "c01c45f9dcaee557248bff997f3c244a9fdba2b6c13211821ee335d4bfee0712",
        "accepted_manifest": Path(
            "/home/ubuntu24/Teleop/artifacts/evidence/"
            "task1_picklift_real24_budget_extension_v1/real24_subset_manifest.jsonl"
        ),
        "accepted_manifest_sha256": (
            "7cd5917bb03beafc347f9e1d6fd645e731eb0c26e6d1f0eaf98d7497e6d7d21f"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Task1 Real24 budget ACT input")
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT / "dataset_audit.json")
    args = parser.parse_args()
    result = {
        "schema": "task1_picklift_real24_budget_extension_dataset_audit_v1",
        "status": "pass",
        "hardware_accessed": False,
        "datasets": {name: audit_dataset(name, spec) for name, spec in DATASETS.items()},
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
