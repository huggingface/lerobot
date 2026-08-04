from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

from lerobot.datasets.dataset_tools import merge_datasets, recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset


PLAN_ID = "task1-picklift-real24-questsim24-act-v1"
DERIVED_DATASET_ID = "task1_picklift_real24_questsim24_combined48_v2"
DERIVED_REPO_ID = f"local/{DERIVED_DATASET_ID}"
DERIVED_FREEZE_ID = f"{DERIVED_DATASET_ID}_freeze_v1"

REAL_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/task1_picklift_formal24_s03_20260728"
)
REAL_REPO_ID = "local/task1_picklift_formal24_s03_20260728"
REAL_FREEZE = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/"
    "task1_picklift_formal24_s03_20260728/qa_v1/freeze_manifest.json"
)
REAL_SUBSET = REAL_FREEZE.with_name("accepted_subset_manifest.json")
REAL_TREE_SHA256 = "251cbdc079b304425ccdfbd7a08f15d34858ea0dd8c19345544b8da9f3adb9f2"
REAL_SUBSET_SHA256 = "a0449033d3f2b447cf6b0774ac5d363c8073ab10667799bf3920b61805373e1d"

SIM_SESSION_ROOT = Path(
    "/home/ubuntu24/SO101QuestRemote-data/formal/"
    "task1_picklift_quest_remote_mujoco_formal24_20260729_r1"
)
SIM_ROOT = SIM_SESSION_ROOT / "mujoco_picklift_nexus_v1"
SIM_REPO_ID = "physicalai/so101_quest_remote_mujoco_picklift_formal24_20260729_r1"
SIM_FREEZE_ROOT = Path(
    "/home/ubuntu24/SO101QuestRemote-data/evidence/"
    "task1_picklift_quest_remote_mujoco_formal24_20260729_r1/freeze_v1"
)
SIM_FREEZE = SIM_FREEZE_ROOT / "freeze_manifest.json"
SIM_SUBSET = SIM_FREEZE_ROOT / "accepted_subset_manifest.json"
SIM_TREE_SHA256 = "1e78c636c8c6d6a508bafda557177bd096a75469680b563544b798077f2761fd"
SIM_SUBSET_SHA256 = "9f19cf5c3839a73b6d6727fd7480e36696669cefb9a952bac10ddd1cf60847a9"

DEFAULT_OUTPUT_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/datasets/"
    "task1_picklift_real24_questsim24_act_v1/combined48_v2"
)
DEFAULT_EVIDENCE_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/"
    "task1_picklift_real24_questsim24_act_v1/combined48_freeze_v2"
)

CANONICAL_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CANONICAL_TASK = (
    "Task 1 PickLift v1: grasp the 2 cm red cube and lift it >=5 cm "
    "with bilateral finger hold"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def real_tree_identity(root: Path) -> tuple[str, int, int, bytes]:
    rows = []
    total_bytes = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        rows.append(f"{sha256_file(path)}  {relative}\n")
    inventory = "".join(rows).encode()
    return hashlib.sha256(inventory).hexdigest(), len(rows), total_bytes, inventory


def sim_tree_identity(root: Path) -> tuple[str, int, int, bytes]:
    rows = []
    total_bytes = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == ".mujoco_picklift_nexus_v1.recording.lock":
            continue
        size = path.stat().st_size
        total_bytes += size
        rows.append(f"{sha256_file(path)}  {size}  {relative}\n")
    inventory = "".join(rows).encode()
    return hashlib.sha256(inventory).hexdigest(), len(rows), total_bytes, inventory


def derived_tree_identity(root: Path) -> tuple[str, int, int, bytes]:
    rows = []
    total_bytes = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        rows.append(f"{sha256_file(path)}  {size}  {relative}\n")
    inventory = "".join(rows).encode()
    return hashlib.sha256(inventory).hexdigest(), len(rows), total_bytes, inventory


def load_tasks(root: Path) -> list[str]:
    frame = pd.read_parquet(root / "meta/tasks.parquet")
    if frame.index.name == "task":
        return [str(value) for value in frame.index.tolist()]
    return [str(value) for value in frame["task"].tolist()]


def episode_lengths(root: Path) -> dict[int, int]:
    frames = [
        pd.read_parquet(path, columns=["episode_index", "length"])
        for path in sorted((root / "meta/episodes").glob("*/*.parquet"))
    ]
    combined = pd.concat(frames, ignore_index=True)
    return {
        int(row.episode_index): int(row.length)
        for row in combined.itertuples(index=False)
    }


def numeric_audit(root: Path) -> dict:
    total = 0
    finite = True
    per_episode = Counter()
    for path in sorted((root / "data").glob("*/*.parquet")):
        frame = pd.read_parquet(
            path, columns=["episode_index", "observation.state", "action"]
        )
        state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(frame["action"].to_numpy()).astype(np.float32)
        finite = finite and bool(np.isfinite(state).all() and np.isfinite(action).all())
        total += len(frame)
        per_episode.update(int(value) for value in frame["episode_index"])
    return {
        "frames": total,
        "finite_state_and_action": finite,
        "episode_frame_counts": {
            str(key): per_episode[key] for key in sorted(per_episode)
        },
    }


def loader_audit(repo_id: str, root: Path) -> tuple[LeRobotDataset, dict]:
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    lengths = episode_lengths(root)
    starts = []
    offset = 0
    for episode in sorted(lengths):
        starts.append(offset)
        offset += lengths[episode]
    sample_indices = sorted(set(starts + [offset - 1]))
    decoded = []
    for index in sample_indices:
        sample = dataset[index]
        state = sample["observation.state"]
        action = sample["action"]
        front = sample["observation.images.front"]
        if tuple(state.shape) != (6,) or state.dtype != torch.float32:
            raise RuntimeError(f"Bad state at index {index}: {state.shape} {state.dtype}")
        if tuple(action.shape) != (6,) or action.dtype != torch.float32:
            raise RuntimeError(f"Bad action at index {index}: {action.shape} {action.dtype}")
        if tuple(front.shape) != (3, 480, 640) or front.dtype != torch.float32:
            raise RuntimeError(f"Bad front image at index {index}: {front.shape} {front.dtype}")
        if not bool(
            torch.isfinite(state).all()
            and torch.isfinite(action).all()
            and torch.isfinite(front).all()
        ):
            raise RuntimeError(f"Non-finite loader sample at index {index}")
        decoded.append(index)
    return dataset, {
        "status": "pass",
        "repo_id": repo_id,
        "root": str(root),
        "episodes": dataset.meta.total_episodes,
        "frames": len(dataset),
        "fps": dataset.meta.fps,
        "decoded_sample_indices": decoded,
        "state_shape_dtype": ["6", "float32"],
        "action_shape_dtype": ["6", "float32"],
        "front_shape_dtype": ["3x480x640", "float32"],
        "tasks": load_tasks(root),
    }


def base_joint_names(names: list[str]) -> list[str]:
    return [name.removesuffix(".pos") for name in names]


def source_audit() -> tuple[dict, LeRobotDataset, LeRobotDataset]:
    real_freeze = json.loads(REAL_FREEZE.read_text())
    real_subset = json.loads(REAL_SUBSET.read_text())
    sim_freeze = json.loads(SIM_FREEZE.read_text())

    real_sha, real_files, real_bytes, _ = real_tree_identity(REAL_ROOT)
    if real_sha != REAL_TREE_SHA256 or real_sha != real_freeze["tree_sha256"]:
        raise RuntimeError(f"Real tree hash mismatch: {real_sha}")
    real_subset_payload = {
        key: value
        for key, value in real_subset.items()
        if key not in {"subset_sha256", "created_at_utc"}
    }
    real_subset_sha = canonical_sha(real_subset_payload)
    if real_subset_sha != REAL_SUBSET_SHA256:
        raise RuntimeError(f"Real subset hash mismatch: {real_subset_sha}")

    sim_sha, sim_files, sim_bytes, _ = sim_tree_identity(SIM_SESSION_ROOT)
    if sim_sha != SIM_TREE_SHA256 or sim_sha != sim_freeze["raw_tree_sha256"]:
        raise RuntimeError(f"Sim tree hash mismatch: {sim_sha}")
    sim_subset_sha = sha256_file(SIM_SUBSET)
    if sim_subset_sha != SIM_SUBSET_SHA256:
        raise RuntimeError(f"Sim subset hash mismatch: {sim_subset_sha}")

    real_dataset, real_loader = loader_audit(REAL_REPO_ID, REAL_ROOT)
    sim_dataset, sim_loader = loader_audit(SIM_REPO_ID, SIM_ROOT)
    real_info = real_dataset.meta.info
    sim_info = sim_dataset.meta.info
    real_features = real_dataset.meta.features
    sim_features = sim_dataset.meta.features

    required = {
        "observation.state",
        "observation.images.front",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    if set(real_features) != required or set(sim_features) != required:
        raise RuntimeError("Source feature keys differ from the required front-only schema")
    for key in required:
        if real_features[key]["shape"] != sim_features[key]["shape"]:
            raise RuntimeError(f"Shape mismatch for {key}")
        if real_features[key]["dtype"] != sim_features[key]["dtype"]:
            raise RuntimeError(f"Dtype mismatch for {key}")
    real_names = real_features["observation.state"]["names"]
    sim_names = sim_features["observation.state"]["names"]
    if base_joint_names(real_names) != base_joint_names(sim_names):
        raise RuntimeError("Semantic joint order mismatch")
    if real_info.fps != 20 or sim_info.fps != 20:
        raise RuntimeError("Both sources must be 20 FPS")

    real_stats = json.loads((REAL_ROOT / "meta/stats.json").read_text())
    sim_stats = json.loads((SIM_ROOT / "meta/stats.json").read_text())
    distributions = {}
    for key in ("observation.state", "action"):
        distributions[key] = {
            "real": {
                stat: real_stats[key][stat] for stat in ("min", "max", "mean", "std")
            },
            "simulation": {
                stat: sim_stats[key][stat] for stat in ("min", "max", "mean", "std")
            },
            "absolute_mean_delta": np.abs(
                np.asarray(real_stats[key]["mean"])
                - np.asarray(sim_stats[key]["mean"])
            ).tolist(),
        }

    audit = {
        "schema": "task1_picklift_combined48_source_compatibility_audit_v1",
        "plan_id": PLAN_ID,
        "status": "pass_with_explicit_derived_metadata_normalization",
        "source_identity": {
            "real": {
                "tree_sha256": real_sha,
                "subset_sha256": real_subset_sha,
                "file_count": real_files,
                "total_bytes": real_bytes,
            },
            "simulation": {
                "tree_sha256": sim_sha,
                "subset_sha256": sim_subset_sha,
                "file_count": sim_files,
                "total_bytes": sim_bytes,
            },
        },
        "official_lerobot_0_6_1_loader": {
            "real": real_loader,
            "simulation": sim_loader,
        },
        "numeric_audit": {
            "real": numeric_audit(REAL_ROOT),
            "simulation": numeric_audit(SIM_ROOT),
        },
        "compatible_contract": {
            "fps": 20,
            "front_rgb": "640x480",
            "front_only": True,
            "state_and_action_dtype_shape": "float32[6]",
            "joint_order": CANONICAL_JOINT_NAMES,
            "state_units": "arm degrees and gripper range_0_100",
            "action_units": "arm degrees and gripper range_0_100",
            "task_semantics": "same PickLift red-cube grasp and >=5 cm stable-lift task",
        },
        "source_metadata_differences": {
            "joint_names": {
                "real": real_names,
                "simulation": sim_names,
                "normalization": "append .pos to Real names in derived metadata only",
            },
            "robot_type": {
                "real": real_info.robot_type,
                "simulation": sim_info.robot_type,
                "normalization": "canonical derived robot_type=so101",
            },
            "task_text": {
                "real": real_loader["tasks"],
                "simulation": sim_loader["tasks"],
                "normalization": CANONICAL_TASK,
            },
            "video_codec": {
                "real": real_features["observation.images.front"]["info"][
                    "video.codec"
                ],
                "simulation": sim_features["observation.images.front"]["info"][
                    "video.codec"
                ],
                "handling": "copy source video files separately; no concatenation or re-encode",
            },
        },
        "distribution_differences_recorded_not_blocking": distributions,
    }
    return audit, real_dataset, sim_dataset


def normalize_staging_metadata(root: Path) -> None:
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = "so101"
    for key in ("observation.state", "action"):
        info["features"][key]["names"] = CANONICAL_JOINT_NAMES
    video_info = info["features"]["observation.images.front"]["info"]
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
    for video_path in sorted(
        (root / "videos/observation.images.front").glob("*/*.mp4")
    ):
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                image = frame.to_ndarray(format="rgb24")
                if image.shape != (480, 640, 3):
                    raise RuntimeError(
                        f"Unexpected decoded shape {image.shape} in {video_path}"
                    )
                for channel in range(3):
                    histogram[channel] += np.bincount(
                        image[:, :, channel].reshape(-1), minlength=256
                    ).astype(np.uint64)
                decoded_frames += 1
    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"Decoded {decoded_frames} frames, expected {expected_frames}"
        )
    values = np.arange(256, dtype=np.float64)
    count = int(histogram[0].sum())
    mean_255 = (histogram * values).sum(axis=1) / count
    second_255 = (histogram * values**2).sum(axis=1) / count
    std_255 = np.sqrt(np.maximum(second_255 - mean_255**2, 0.0))

    def channel_shape(items: np.ndarray) -> list:
        return [[[float(item)]] for item in items]

    def quantile(probability: float) -> np.ndarray:
        target = int(np.floor(probability * (count - 1)))
        cumulative = np.cumsum(histogram, axis=1)
        return np.asarray(
            [np.searchsorted(cumulative[channel], target + 1) for channel in range(3)],
            dtype=np.float64,
        )

    minimum = np.asarray(
        [np.flatnonzero(histogram[channel])[0] for channel in range(3)],
        dtype=np.float64,
    )
    maximum = np.asarray(
        [np.flatnonzero(histogram[channel])[-1] for channel in range(3)],
        dtype=np.float64,
    )
    return {
        "min": channel_shape(minimum / 255.0),
        "max": channel_shape(maximum / 255.0),
        "mean": channel_shape(mean_255 / 255.0),
        "std": channel_shape(std_255 / 255.0),
        "count": [count],
        "q01": channel_shape(quantile(0.01) / 255.0),
        "q10": channel_shape(quantile(0.10) / 255.0),
        "q50": channel_shape(quantile(0.50) / 255.0),
        "q90": channel_shape(quantile(0.90) / 255.0),
        "q99": channel_shape(quantile(0.99) / 255.0),
    }


def provenance_rows(
    real_lengths: dict[int, int],
    sim_lengths: dict[int, int],
) -> list[dict]:
    real_subset = json.loads(REAL_SUBSET.read_text())
    sim_subset = json.loads(SIM_SUBSET.read_text())
    sim_episodes = {
        int(row["episode_index"]): row for row in sim_subset["episodes"]
    }
    rows = []
    for episode in sorted(real_lengths):
        rows.append(
            {
                "derived_episode_index": episode,
                "frame_count": real_lengths[episode],
                "source_domain": "real",
                "source_episode_index": episode,
                "source_repo_id": REAL_REPO_ID,
                "source_root": str(REAL_ROOT),
                "source_subset_id": real_subset["subset_id"],
                "source_subset_sha256": REAL_SUBSET_SHA256,
                "source_tree_sha256": REAL_TREE_SHA256,
                "source_episode_evidence_sha256": real_subset[
                    "episode_provenance_sha256"
                ][f"{episode:06d}"],
                "source_task_text": load_tasks(REAL_ROOT)[0],
            }
        )
    for episode in sorted(sim_lengths):
        evidence = sim_episodes[episode]
        rows.append(
            {
                "derived_episode_index": 24 + episode,
                "frame_count": sim_lengths[episode],
                "source_domain": "remote_mujoco_human_quest",
                "source_episode_index": episode,
                "source_episode_id": evidence["episode_id"],
                "source_attempt_id": evidence["attempt_id"],
                "source_repo_id": SIM_REPO_ID,
                "source_root": str(SIM_ROOT),
                "source_subset_id": (
                    "task1_picklift_quest_remote_mujoco_formal24_20260729_r1_"
                    "accepted_v1"
                ),
                "source_subset_sha256": SIM_SUBSET_SHA256,
                "source_tree_sha256": SIM_TREE_SHA256,
                "source_episode_evidence_sha256": evidence[
                    "sidecar_manifest_sha256"
                ],
                "source_standard_video_sha256": sha256_file(
                    SIM_ROOT
                    / "videos/observation.images.front/chunk-000"
                    / f"file-{episode:03d}.mp4"
                ),
                "source_standard_data_sha256": sha256_file(
                    SIM_ROOT / "data/chunk-000" / f"file-{episode:03d}.parquet"
                ),
                "source_task_text": load_tasks(SIM_ROOT)[0],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen Task1 Real24 + Quest-Sim24 Dataset v3 view."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    audit, _, _ = source_audit()
    if args.audit_only:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return
    if args.output_root.exists() or args.evidence_root.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing derived dataset or evidence root"
        )

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = args.output_root.parent
    staging = Path(
        tempfile.mkdtemp(prefix=f".{DERIVED_DATASET_ID}_staging_", dir=staging_parent)
    )
    try:
        real_stage = staging / "real"
        sim_stage = staging / "simulation"
        shutil.copytree(REAL_ROOT, real_stage)
        shutil.copytree(SIM_ROOT, sim_stage)
        normalize_staging_metadata(real_stage)
        normalize_staging_metadata(sim_stage)

        staged_real = LeRobotDataset(REAL_REPO_ID, root=real_stage)
        staged_sim = LeRobotDataset(SIM_REPO_ID, root=sim_stage)
        merged = merge_datasets(
            [staged_real, staged_sim],
            output_repo_id=DERIVED_REPO_ID,
            output_dir=args.output_root,
            concatenate_videos=False,
            concatenate_data=False,
        )
        recompute_stats(merged, skip_image_video=True)
        image_stats = exact_video_stats(
            args.output_root, expected_frames=3790 + 4176
        )
        stats_path = args.output_root / "meta/stats.json"
        stats = json.loads(stats_path.read_text())
        stats["observation.images.front"] = image_stats
        write_json(stats_path, stats)

        real_lengths = episode_lengths(REAL_ROOT)
        sim_lengths = episode_lengths(SIM_ROOT)
        rows = provenance_rows(real_lengths, sim_lengths)
        provenance_root = args.output_root / "provenance"
        provenance_root.mkdir(parents=True, exist_ok=True)
        with (provenance_root / "source_episodes.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        domain_frames = Counter()
        domain_episodes = Counter()
        for row in rows:
            domain_frames[row["source_domain"]] += row["frame_count"]
            domain_episodes[row["source_domain"]] += 1
        provenance_manifest = {
            "schema": "task1_picklift_combined48_provenance_v1",
            "plan_id": PLAN_ID,
            "dataset_id": DERIVED_DATASET_ID,
            "repo_id": DERIVED_REPO_ID,
            "source_episode_map": "source_episodes.jsonl",
            "episodes": len(rows),
            "frames": sum(row["frame_count"] for row in rows),
            "domain_episode_counts": dict(sorted(domain_episodes.items())),
            "domain_frame_counts": dict(sorted(domain_frames.items())),
            "sampling": (
                "standard frame sampling; every accepted source episode exactly "
                "once; no duplication, weighting, upsampling, or downsampling"
            ),
            "metadata_normalization": audit["source_metadata_differences"],
            "high_resolution_sidecars_copied": False,
            "raw_sources_modified": False,
        }
        write_json(provenance_root / "manifest.json", provenance_manifest)

        derived, loader = loader_audit(DERIVED_REPO_ID, args.output_root)
        if derived.meta.total_episodes != 48 or len(derived) != 7966:
            raise RuntimeError("Derived dataset does not have 48 episodes / 7966 frames")
        if set(domain_episodes.values()) != {24}:
            raise RuntimeError("Derived domain episode counts are not 24 + 24")
        if any("highres" in path.as_posix() for path in args.output_root.rglob("*")):
            raise RuntimeError("High-resolution sidecar unexpectedly entered training view")

        args.evidence_root.mkdir(parents=True)
        write_json(args.evidence_root / "source_compatibility_audit.json", audit)
        derived_manifest = {
            "schema": "task1_picklift_combined48_derived_manifest_v1",
            "plan_id": PLAN_ID,
            "dataset_id": DERIVED_DATASET_ID,
            "repo_id": DERIVED_REPO_ID,
            "root": str(args.output_root),
            "episodes": derived.meta.total_episodes,
            "frames": len(derived),
            "fps": derived.meta.fps,
            "domain_episode_counts": dict(sorted(domain_episodes.items())),
            "domain_frame_counts": dict(sorted(domain_frames.items())),
            "simulation_frame_fraction": domain_frames[
                "remote_mujoco_human_quest"
            ]
            / len(derived),
            "official_loader": loader,
            "features": derived.meta.features,
            "normalization_statistics": {
                "numeric": "recomputed from the derived data parquet files",
                "visual": {
                    "stats": image_stats,
                    "decoded_frames": 7966,
                    "method": (
                        "exact uint8 histogram over every decoded training-video pixel"
                    ),
                },
                "use_imagenet_stats": False,
            },
            "source_tree_sha256": {
                "real": REAL_TREE_SHA256,
                "simulation": SIM_TREE_SHA256,
            },
            "source_subset_sha256": {
                "real": REAL_SUBSET_SHA256,
                "simulation": SIM_SUBSET_SHA256,
            },
            "source_episodes_exactly_once": True,
            "high_resolution_sidecars_copied": False,
            "raw_sources_modified": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(args.evidence_root / "derived_manifest.json", derived_manifest)

        tree_sha, file_count, total_bytes, inventory = derived_tree_identity(
            args.output_root
        )
        (args.evidence_root / "tree_inventory.sha256").write_bytes(inventory)
        freeze = {
            "schema": "task1_picklift_combined48_freeze_v1",
            "freeze_id": DERIVED_FREEZE_ID,
            "dataset_id": DERIVED_DATASET_ID,
            "repo_id": DERIVED_REPO_ID,
            "root": str(args.output_root),
            "tree_sha256": tree_sha,
            "tree_algorithm": (
                "SHA-256 of exact UTF-8/LF tree_inventory.sha256 bytes; each "
                "lexicographically path-sorted line is "
                "'<file_sha256><two spaces><decimal_bytes><two spaces>"
                "<POSIX_relative_path>\\n'"
            ),
            "file_count": file_count,
            "total_bytes": total_bytes,
            "episodes": 48,
            "frames": 7966,
            "status": "frozen_official_loader_pass",
            "frozen_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(args.evidence_root / "freeze_manifest.json", freeze)
        result = {
            "status": "pass",
            "derived_manifest": derived_manifest,
            "freeze": freeze,
            "evidence_root": str(args.evidence_root),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
