from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

from lerobot.datasets.dataset_tools import merge_datasets, recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

EXPERIMENT_ID = "task1_picklift_real24_localsim48_gap_recovery_act_v1"
REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / EXPERIMENT_ID
BINDINGS_PATH = EXPERIMENT_ROOT / "source_bindings.json"
CANONICAL_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CANONICAL_TASK = "Pick up the red cube."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_inventory_tree(root: Path) -> tuple[str, int, int, bytes]:
    """Legacy Real24 tree: SHA-256 of '<file sha>  <relative path>\n'."""
    rows: list[str] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"Symlink is forbidden in source Dataset: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        total_bytes += path.stat().st_size
        rows.append(f"{sha256_file(path)}  {relative}\n")
    encoded = "".join(rows).encode()
    return hashlib.sha256(encoded).hexdigest(), len(rows), total_bytes, encoded


def length_prefixed_tree(root: Path) -> tuple[str, int, int, bytes]:
    """Final LocalSim tree identity used by postcollection finalization."""
    digest = hashlib.sha256()
    rows: list[dict] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"Symlink is forbidden in source Dataset: {path}")
        if not path.is_file():
            continue
        relative_text = path.relative_to(root).as_posix()
        relative = relative_text.encode()
        size = path.stat().st_size
        file_sha = sha256_file(path)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_sha))
        rows.append({"relative_path": relative_text, "sha256": file_sha, "size_bytes": size})
        total_bytes += size
    inventory = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    return digest.hexdigest(), len(rows), total_bytes, inventory


def derived_tree(root: Path) -> tuple[str, int, int, bytes]:
    rows: list[str] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        rows.append(f"{sha256_file(path)}  {size}  {relative}\n")
        total_bytes += size
    inventory = "".join(rows).encode()
    return hashlib.sha256(inventory).hexdigest(), len(rows), total_bytes, inventory


def load_bindings() -> dict:
    bindings = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))
    finalization_path = EXPERIMENT_ROOT / bindings["postcollection_finalization"]["result_manifest_file"]
    expected_sha = bindings["postcollection_finalization"]["result_manifest_sha256"]
    if sha256_file(finalization_path) != expected_sha:
        raise RuntimeError("LocalSim postcollection result manifest SHA mismatch")
    result = json.loads(finalization_path.read_text(encoding="utf-8"))
    expected_status = bindings["postcollection_finalization"]["status"]
    if result["status"] != expected_status:
        raise RuntimeError(f"LocalSim result status is not training-ready: {result['status']}")
    for key, result_key in (("gap24", "sim24_gap"), ("overlap24", "sim24_overlap"), ("full48", "sim48_full")):
        expected = bindings["simulation"][key]
        actual = result["datasets"][result_key]
        checks = {
            "tree": (actual["dataset_tree_sha256"], expected["tree_sha256"]),
            "episodes": (actual["episode_count"], expected["episodes"]),
            "frames": (actual["frame_count"], expected["frames"]),
            "membership": (actual["membership_sha256"], expected["membership_sha256"]),
            "mapping": (
                actual["source_to_derived_mapping_sha256"],
                expected["mapping_sha256"],
            ),
        }
        mismatches = {name: pair for name, pair in checks.items() if pair[0] != pair[1]}
        if mismatches:
            raise RuntimeError(f"Finalization result disagrees with {key} binding: {mismatches}")
    return bindings


def episode_lengths(root: Path) -> dict[int, int]:
    parts = [
        pd.read_parquet(path, columns=["episode_index", "length"])
        for path in sorted((root / "meta/episodes").glob("*/*.parquet"))
    ]
    combined = pd.concat(parts, ignore_index=True)
    return {int(row.episode_index): int(row.length) for row in combined.itertuples(index=False)}


def source_loader(repo_id: str, root: Path, episodes: int, frames: int) -> LeRobotDataset:
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    if dataset.meta.total_episodes != episodes or len(dataset) != frames or dataset.meta.fps != 20:
        raise RuntimeError(
            f"Source count mismatch for {root}: {dataset.meta.total_episodes}/{len(dataset)}/{dataset.meta.fps}"
        )
    required = {"observation.state", "observation.images.front", "action"}
    if not required.issubset(dataset.meta.features):
        raise RuntimeError(f"Missing required features in {root}")
    for index in (0, frames - 1):
        sample = dataset[index]
        if sample["observation.state"].shape != (6,) or sample["action"].shape != (6,):
            raise RuntimeError(f"Bad state/action shape in {root} at {index}")
        if sample["observation.images.front"].shape != (3, 480, 640):
            raise RuntimeError(f"Bad image shape in {root} at {index}")
        if not all(
            bool(torch.isfinite(sample[key]).all())
            for key in ("observation.state", "action", "observation.images.front")
        ):
            raise RuntimeError(f"Non-finite source sample in {root} at {index}")
    return dataset


def verify_source(bindings: dict, sim_key: str) -> tuple[dict, LeRobotDataset, LeRobotDataset]:
    real = bindings["real"]
    real_root = Path(real["root"])
    real_sha, real_files, real_bytes, real_inventory = file_inventory_tree(real_root)
    if real_sha != real["tree_sha256"]:
        raise RuntimeError(f"Real24 tree mismatch: {real_sha}")

    sim = bindings["simulation"][sim_key]
    sim_base = Path(bindings["simulation"]["base_root"])
    sim_root = sim_base / sim["root_relative"]
    sim_sha, sim_files, sim_bytes, sim_inventory = length_prefixed_tree(sim_root)
    if sim_sha != sim["tree_sha256"]:
        raise RuntimeError(f"LocalSim {sim_key} tree mismatch: {sim_sha}")
    mapping = sim_base / sim["mapping_relative"]
    if sha256_file(mapping) != sim["mapping_sha256"]:
        raise RuntimeError(f"LocalSim {sim_key} source mapping mismatch")

    real_dataset = source_loader(real["repo_id"], real_root, real["episodes"], real["frames"])
    sim_dataset = source_loader(sim["repo_id"], sim_root, sim["episodes"], sim["frames"])
    audit = {
        "status": "pass",
        "real": {
            **real,
            "file_count": real_files,
            "total_bytes": real_bytes,
            "inventory_sha256": hashlib.sha256(real_inventory).hexdigest(),
        },
        "simulation": {
            **sim,
            "root": str(sim_root),
            "file_count": sim_files,
            "total_bytes": sim_bytes,
            "inventory_sha256": hashlib.sha256(sim_inventory).hexdigest(),
        },
        "postcollection_result_sha256": bindings["postcollection_finalization"]["result_manifest_sha256"],
        "raw_sources_modified": False,
    }
    return audit, real_dataset, sim_dataset


def normalize_staging_metadata(root: Path) -> None:
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["robot_type"] = "so101"
    for key in ("observation.state", "action"):
        info["features"][key]["names"] = CANONICAL_JOINT_NAMES
    video_info = info["features"]["observation.images.front"].get("info", {})
    video_info["video.codec"] = None
    video_info["video.pix_fmt"] = None
    write_json(info_path, info)

    tasks_path = root / "meta/tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    if tasks.index.name == "task":
        tasks.index = pd.Index([CANONICAL_TASK] * len(tasks), name="task")
        tasks.to_parquet(tasks_path)
    else:
        tasks["task"] = CANONICAL_TASK
        tasks.to_parquet(tasks_path, index=False)
    for path in sorted((root / "meta/episodes").glob("*/*.parquet")):
        episodes = pd.read_parquet(path)
        episodes["tasks"] = [[CANONICAL_TASK] for _ in range(len(episodes))]
        episodes.to_parquet(path, index=False)


def exact_video_stats(root: Path, expected_frames: int) -> dict:
    histogram = np.zeros((3, 256), dtype=np.uint64)
    decoded_frames = 0
    for video_path in sorted((root / "videos/observation.images.front").glob("*/*.mp4")):
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                image = frame.to_ndarray(format="rgb24")
                if image.shape != (480, 640, 3):
                    raise RuntimeError(f"Unexpected video shape {image.shape} in {video_path}")
                for channel in range(3):
                    histogram[channel] += np.bincount(image[:, :, channel].reshape(-1), minlength=256).astype(
                        np.uint64
                    )
                decoded_frames += 1
    if decoded_frames != expected_frames:
        raise RuntimeError(f"Decoded {decoded_frames} frames, expected {expected_frames}")
    values = np.arange(256, dtype=np.float64)
    count = int(histogram[0].sum())
    mean_255 = (histogram * values).sum(axis=1) / count
    second_255 = (histogram * values**2).sum(axis=1) / count
    std_255 = np.sqrt(np.maximum(second_255 - mean_255**2, 0.0))

    def shaped(values_: np.ndarray) -> list:
        return [[[float(value)]] for value in values_]

    def quantile(probability: float) -> np.ndarray:
        target = int(np.floor(probability * (count - 1)))
        cumulative = np.cumsum(histogram, axis=1)
        return np.asarray(
            [np.searchsorted(cumulative[channel], target + 1) for channel in range(3)],
            dtype=np.float64,
        )

    minimum = np.asarray([np.flatnonzero(histogram[c])[0] for c in range(3)], dtype=np.float64)
    maximum = np.asarray([np.flatnonzero(histogram[c])[-1] for c in range(3)], dtype=np.float64)
    return {
        "min": shaped(minimum / 255.0),
        "max": shaped(maximum / 255.0),
        "mean": shaped(mean_255 / 255.0),
        "std": shaped(std_255 / 255.0),
        "count": [count],
        "q01": shaped(quantile(0.01) / 255.0),
        "q10": shaped(quantile(0.10) / 255.0),
        "q50": shaped(quantile(0.50) / 255.0),
        "q90": shaped(quantile(0.90) / 255.0),
        "q99": shaped(quantile(0.99) / 255.0),
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def materialize(condition_id: str, bindings: dict) -> dict:
    condition = bindings["conditions"][condition_id]
    sim_key = condition["simulation_key"]
    audit, real_dataset, sim_dataset = verify_source(bindings, sim_key)
    output_root = Path(condition["dataset_root"])
    evidence_root = Path(condition["evidence_root"])
    if output_root.exists() or evidence_root.exists():
        raise FileExistsError(f"Refusing existing output/evidence for {condition_id}")

    real = bindings["real"]
    sim = bindings["simulation"][sim_key]
    sim_base = Path(bindings["simulation"]["base_root"])
    sim_root = sim_base / sim["root_relative"]
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{condition['dataset_id']}_", dir=output_root.parent))
    try:
        real_stage = staging / "real"
        sim_stage = staging / "simulation"
        shutil.copytree(Path(real["root"]), real_stage)
        shutil.copytree(sim_root, sim_stage)
        normalize_staging_metadata(real_stage)
        normalize_staging_metadata(sim_stage)
        staged_real = LeRobotDataset(real["repo_id"], root=real_stage, video_backend="pyav")
        staged_sim = LeRobotDataset(sim["repo_id"], root=sim_stage, video_backend="pyav")
        merged = merge_datasets(
            [staged_real, staged_sim],
            output_repo_id=condition["repo_id"],
            output_dir=output_root,
            concatenate_videos=False,
            concatenate_data=False,
        )
        recompute_stats(merged, skip_image_video=True)
        total_frames = real["frames"] + sim["frames"]
        stats_path = output_root / "meta/stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["observation.images.front"] = exact_video_stats(output_root, total_frames)
        write_json(stats_path, stats)

        real_lengths = episode_lengths(Path(real["root"]))
        sim_lengths = episode_lengths(sim_root)
        real_map = {
            int(row["derived_episode_index"]): row
            for row in read_jsonl(Path(real["root"]) / "provenance/source_episode_map.jsonl")
        }
        sim_map = {
            int(row["dataset_episode_index"]): row for row in read_jsonl(sim_base / sim["mapping_relative"])
        }
        rows: list[dict] = []
        for episode in sorted(real_lengths):
            rows.append(
                {
                    "derived_episode_index": episode,
                    "frame_count": real_lengths[episode],
                    "source_domain": "real",
                    "source_episode_index": episode,
                    "source": real_map[episode],
                }
            )
        for episode in sorted(sim_lengths):
            rows.append(
                {
                    "derived_episode_index": real["episodes"] + episode,
                    "frame_count": sim_lengths[episode],
                    "source_domain": "simulation",
                    "source_episode_index": episode,
                    "source": sim_map[episode],
                }
            )
        provenance_root = output_root / "provenance"
        provenance_root.mkdir(parents=True)
        with (provenance_root / "source_episodes.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        write_json(
            provenance_root / "manifest.json",
            {
                "schema": "task1_picklift_real24_localsim_combined_provenance_v1",
                "condition": condition_id,
                "dataset_id": condition["dataset_id"],
                "repo_id": condition["repo_id"],
                "domain_episode_counts": {"real": real["episodes"], "simulation": sim["episodes"]},
                "domain_frame_counts": {"real": real["frames"], "simulation": sim["frames"]},
                "construction": "every bound source episode exactly once; no frame resampling or duplication",
                "training_sampling": "deterministic 4 Real + 4 Sim per optimizer batch",
                "metadata_only_normalization": {
                    "robot_type": "so101",
                    "joint_names": CANONICAL_JOINT_NAMES,
                    "task": CANONICAL_TASK,
                },
                "state_action_values_modified": False,
                "source_bindings_sha256": sha256_file(BINDINGS_PATH),
                "postcollection_result_sha256": bindings["postcollection_finalization"][
                    "result_manifest_sha256"
                ],
            },
        )

        derived = LeRobotDataset(condition["repo_id"], root=output_root, video_backend="pyav")
        expected_episodes = real["episodes"] + sim["episodes"]
        if derived.meta.total_episodes != expected_episodes or len(derived) != total_frames:
            raise RuntimeError("Combined Dataset episode/frame count mismatch")
        derived_state = np.stack(derived.hf_dataset["observation.state"])
        derived_action = np.stack(derived.hf_dataset["action"])
        real_state = np.stack(real_dataset.hf_dataset["observation.state"])
        real_action = np.stack(real_dataset.hf_dataset["action"])
        sim_state = np.stack(sim_dataset.hf_dataset["observation.state"])
        sim_action = np.stack(sim_dataset.hf_dataset["action"])
        split = real["frames"]
        if not (
            np.array_equal(derived_state[:split], real_state)
            and np.array_equal(derived_action[:split], real_action)
            and np.array_equal(derived_state[split:], sim_state)
            and np.array_equal(derived_action[split:], sim_action)
        ):
            raise RuntimeError("Combined state/action values do not exactly match sources")
        for index in (0, split - 1, split, total_frames - 1):
            sample = derived[index]
            if sample["observation.images.front"].shape != (3, 480, 640):
                raise RuntimeError(f"Combined loader image shape mismatch at {index}")
            if not bool(torch.isfinite(sample["observation.images.front"]).all()):
                raise RuntimeError(f"Combined loader non-finite image at {index}")

        evidence_root.mkdir(parents=True)
        write_json(evidence_root / "source_binding_audit.json", audit)
        tree_sha, file_count, total_bytes, inventory = derived_tree(output_root)
        (evidence_root / "tree_inventory.sha256").write_bytes(inventory)
        freeze = {
            "schema": "task1_picklift_real24_localsim_combined_dataset_freeze_v1",
            "status": "frozen_official_loader_pass",
            "condition": condition_id,
            "dataset_id": condition["dataset_id"],
            "repo_id": condition["repo_id"],
            "root": str(output_root),
            "tree_sha256": tree_sha,
            "tree_algorithm": "sha256 of sorted '<file_sha>  <bytes>  <relative_path>\\n' inventory",
            "file_count": file_count,
            "total_bytes": total_bytes,
            "episodes": expected_episodes,
            "frames": total_frames,
            "domain_episode_counts": {"real": real["episodes"], "simulation": sim["episodes"]},
            "domain_frame_counts": {"real": real["frames"], "simulation": sim["frames"]},
            "state_action_identity_against_sources": True,
            "image_frames_fully_decoded_for_statistics": total_frames,
            "hardware_accessed": False,
            "rollout_started": False,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(evidence_root / "freeze_manifest.json", freeze)
        write_json(
            evidence_root / "verification.json",
            {
                "status": "pass",
                "condition": condition_id,
                "source_binding": audit,
                "derived": freeze,
                "source_episode_rows": len(rows),
                "source_episodes_exactly_once": len(rows) == expected_episodes,
                "state_action_identity_against_sources": True,
                "loader_boundary_samples": [0, split - 1, split, total_frames - 1],
            },
        )
        return freeze
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("C", "D"), required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    bindings = load_bindings()
    condition = bindings["conditions"][args.condition]
    if args.audit_only:
        audit, _, _ = verify_source(bindings, condition["simulation_key"])
        print(json.dumps(audit, indent=2, sort_keys=True))
        return
    result = materialize(args.condition, bindings)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
