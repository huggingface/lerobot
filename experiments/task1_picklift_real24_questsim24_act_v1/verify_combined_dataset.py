from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_combined_dataset import (
    DEFAULT_EVIDENCE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DERIVED_REPO_ID,
    derived_tree_identity,
    loader_audit,
    source_audit,
    write_json,
)


def numeric_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(numeric_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(numeric_tree(item) for item in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen Combined48 Dataset v3")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    args = parser.parse_args()

    source, _, _ = source_audit()
    freeze = json.loads((args.evidence_root / "freeze_manifest.json").read_text())
    tree_sha, file_count, total_bytes, inventory = derived_tree_identity(
        args.dataset_root
    )
    if tree_sha != freeze["tree_sha256"]:
        raise RuntimeError(f"Derived tree SHA mismatch: {tree_sha}")
    if inventory != (args.evidence_root / "tree_inventory.sha256").read_bytes():
        raise RuntimeError("Derived inventory bytes do not match freeze evidence")

    loader_dataset, loader = loader_audit(DERIVED_REPO_ID, args.dataset_root)
    if loader_dataset.meta.total_episodes != 48 or len(loader_dataset) != 7966:
        raise RuntimeError("Official loader count mismatch")

    frame_counts = Counter()
    global_indices = []
    for path in sorted((args.dataset_root / "data").glob("*/*.parquet")):
        frame = pd.read_parquet(path, columns=["episode_index", "index"])
        frame_counts.update(int(item) for item in frame["episode_index"])
        global_indices.extend(int(item) for item in frame["index"])
    if sorted(frame_counts) != list(range(48)):
        raise RuntimeError("Derived episode indices are not exactly 0..47")
    if sorted(global_indices) != list(range(7966)):
        raise RuntimeError("Derived global indices are not exactly 0..7965")

    rows = [
        json.loads(line)
        for line in (
            args.dataset_root / "provenance/source_episodes.jsonl"
        ).read_text().splitlines()
        if line
    ]
    if len(rows) != 48:
        raise RuntimeError("Provenance map does not contain exactly 48 rows")
    if [row["derived_episode_index"] for row in rows] != list(range(48)):
        raise RuntimeError("Provenance derived episode indices are not exactly 0..47")
    domains = Counter(row["source_domain"] for row in rows)
    if domains != {"real": 24, "remote_mujoco_human_quest": 24}:
        raise RuntimeError(f"Unexpected domain episode counts: {domains}")
    for row in rows:
        episode = row["derived_episode_index"]
        if row["frame_count"] != frame_counts[episode]:
            raise RuntimeError(f"Frame count mismatch for episode {episode}")

    stats = json.loads((args.dataset_root / "meta/stats.json").read_text())
    policy_stats = {
        key: stats[key]
        for key in ("observation.state", "observation.images.front", "action")
    }
    if not numeric_tree(policy_stats):
        raise RuntimeError("Policy normalization stats contain a non-numeric value")
    for key in ("observation.state", "action"):
        if stats[key]["count"] != [7966]:
            raise RuntimeError(f"Unexpected {key} stats count")
    image_count = 7966 * 480 * 640
    if stats["observation.images.front"]["count"] != [image_count]:
        raise RuntimeError("Unexpected image stats pixel count")
    for key in policy_stats:
        for stat in ("mean", "std"):
            if not np.isfinite(np.asarray(policy_stats[key][stat])).all():
                raise RuntimeError(f"Non-finite {key}.{stat}")

    result = {
        "schema": "task1_picklift_combined48_verification_v1",
        "status": "pass",
        "source_identity_reverified": source["source_identity"],
        "derived": {
            "root": str(args.dataset_root),
            "repo_id": DERIVED_REPO_ID,
            "tree_sha256": tree_sha,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "episodes": 48,
            "frames": 7966,
            "episode_indices": "0..47 contiguous",
            "global_indices": "0..7965 contiguous",
            "domain_episode_counts": dict(sorted(domains.items())),
            "domain_frame_counts": {
                domain: sum(
                    row["frame_count"]
                    for row in rows
                    if row["source_domain"] == domain
                )
                for domain in sorted(domains)
            },
            "high_resolution_sidecar_paths": 0,
        },
        "official_loader": loader,
        "policy_normalization_stats": {
            "all_numeric_and_finite": True,
            "state_action_frame_count": 7966,
            "image_decoded_frame_count": 7966,
            "image_pixel_count_per_channel": image_count,
        },
        "note": (
            "LeRobot 0.6.1 aggregate metadata retains source-relative summary "
            "stats for bookkeeping keys such as episode_index. The underlying "
            "parquet indices were independently checked above; policy "
            "normalization uses only state, front RGB, and action stats."
        ),
    }
    write_json(args.evidence_root / "verification.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
