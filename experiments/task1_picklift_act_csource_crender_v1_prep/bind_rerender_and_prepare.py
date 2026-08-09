from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets import LeRobotDataset, MatchedTwoStreamSampler
from lerobot.datasets.dataset_tools import merge_datasets

REPO = Path("/home/ubuntu24/Teleop/lerobot")
HERE = REPO / "experiments/task1_picklift_act_csource_crender_v1_prep"
OLD_EXP = REPO / "experiments/task1_picklift_act_additive_three_model_200k_v1"
REAL24 = Path("/home/ubuntu24/Teleop/artifacts/derived/task1_picklift_real24_budget_extension_v1/accepted")
SOURCE_SIM = Path("/home/ubuntu24/SO101QuestLocalSim-data/postcollection/task1-localsim48-gridphase0-v2-s01-finalization-v1/derived/sim24_gap/dataset")
DATASET_OUT = Path("/home/ubuntu24/Teleop/artifacts/datasets/task1_picklift_act_csource_crender_v1/combined48_v1")
TRAIN_OUT = Path("/home/ubuntu24/Teleop/artifacts/training/task1_picklift_act_csource_crender_v1")
EVIDENCE = Path("/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_act_csource_crender_v1/binding_v1")
CANONICAL_JOINT_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]
CANONICAL_TASK = "Pick up the red cube."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path, algorithm: str) -> str:
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if algorithm == "legacy_sha_rows":
        rows = [f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files]
        return hashlib.sha256("".join(rows).encode()).hexdigest()
    if algorithm == "length_prefixed_v1":
        digest = hashlib.sha256()
        for path in files:
            rel = path.relative_to(root).as_posix().encode()
            digest.update(len(rel).to_bytes(8, "big"))
            digest.update(rel)
            digest.update(path.stat().st_size.to_bytes(8, "big"))
            digest.update(bytes.fromhex(sha256(path)))
        return digest.hexdigest()
    if algorithm == "materializer_v1":
        digest = hashlib.sha256()
        for path in files:
            rel = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(rel)
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256(path)))
            digest.update(b"\n")
        return digest.hexdigest()
    raise ValueError(f"unsupported tree hash algorithm: {algorithm}")


def parquet_rows(root: Path) -> pd.DataFrame:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise RuntimeError(f"no parquet data in {root}")
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def array_bytes(series: pd.Series) -> bytes:
    return np.asarray(np.stack(series.to_numpy()), dtype=np.float32).tobytes(order="C")


def compare_nonvisual_rows(source: Path, rerender: Path) -> dict:
    left = parquet_rows(source)
    right = parquet_rows(rerender)
    required = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    for column in required:
        if column not in left or column not in right:
            raise RuntimeError(f"missing comparison column: {column}")
    if len(left) != 1933 or len(right) != 1933:
        raise RuntimeError("frame count mismatch")
    scalar_equal = {}
    for column in ("episode_index", "frame_index", "timestamp"):
        scalar_equal[column] = np.array_equal(left[column].to_numpy(), right[column].to_numpy())
        if not scalar_equal[column]:
            raise RuntimeError(f"row identity mismatch: {column}")
    hashes = {}
    for column in ("observation.state", "action"):
        left_bytes, right_bytes = array_bytes(left[column]), array_bytes(right[column])
        hashes[column] = {
            "source_sha256_float32le": hashlib.sha256(left_bytes).hexdigest(),
            "rerender_sha256_float32le": hashlib.sha256(right_bytes).hexdigest(),
            "exact_equal": left_bytes == right_bytes,
        }
        if left_bytes != right_bytes:
            raise RuntimeError(f"nonvisual value mismatch: {column}")
    return {"frames": len(left), "scalar_columns_exact": scalar_equal, "tensor_hashes": hashes}


def official_loader(root: Path, repo_id: str) -> LeRobotDataset:
    ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    if ds.meta.total_episodes != 24 or len(ds) != 1933 or ds.meta.fps != 20:
        raise RuntimeError("rerender loader identity mismatch")
    for index in (0, len(ds) - 1):
        sample = ds[index]
        if sample["observation.state"].shape != (6,) or sample["action"].shape != (6,):
            raise RuntimeError("state/action feature mismatch")
        if sample["observation.images.front"].shape != (3, 480, 640):
            raise RuntimeError("front feature mismatch")
        if not all(torch.isfinite(sample[key]).all() for key in ("observation.state", "action", "observation.images.front")):
            raise RuntimeError("non-finite loader sample")
    return ds


def normalize_aggregate_shadow(root: Path) -> None:
    """Normalize metadata only, matching the already-validated old-C materializer."""
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = "so101"
    for key in ("observation.state", "action"):
        info["features"][key]["names"] = CANONICAL_JOINT_NAMES
    video_info = info["features"]["observation.images.front"].get("info", {})
    video_info["video.codec"] = None
    video_info["video.pix_fmt"] = None
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
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


def sampler_digest(ds: LeRobotDataset, groups: dict[str, list[int]]) -> dict:
    sampler = MatchedTwoStreamSampler(
        ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"],
        ds.meta.episodes["episode_index"], groups, 8, 1000,
        episode_indices_to_use=ds.episodes, drop_n_last_frames=0, seed=1000,
        absolute_to_relative_idx=ds.absolute_to_relative_idx,
    )
    digests = [hashlib.sha256(), hashlib.sha256(), hashlib.sha256()]
    for _ in range(200):
        order = list(sampler)
        for offset in range(0, len(order), 8):
            batch = order[offset : offset + 8]
            for index in batch: digests[0].update(struct.pack("<q", index))
            for index in batch[:4]: digests[1].update(struct.pack("<q", index))
            for index in batch[4:]: digests[2].update(struct.pack("<q", index))
    return {"full": digests[0].hexdigest(), "real24": digests[1].hexdigest(), "sim": digests[2].hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerender-root", type=Path, required=True)
    parser.add_argument("--rerender-repo-id", required=True)
    parser.add_argument("--rerender-tree-sha256", required=True)
    parser.add_argument(
        "--tree-hash-algorithm",
        choices=("legacy_sha_rows", "length_prefixed_v1", "materializer_v1"),
        required=True,
    )
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--accepted-manifest-sha256", required=True)
    parser.add_argument("--accepted-status", required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--source-mapping-sha256", required=True)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    if not args.rerender_root.is_dir() or not args.accepted_manifest.is_file() or not args.source_mapping.is_file():
        raise RuntimeError("final rerender handoff paths are incomplete")
    if tree_hash(args.rerender_root, args.tree_hash_algorithm) != args.rerender_tree_sha256:
        raise RuntimeError("rerender tree hash mismatch")
    if sha256(args.accepted_manifest) != args.accepted_manifest_sha256:
        raise RuntimeError("accepted manifest hash mismatch")
    if sha256(args.source_mapping) != args.source_mapping_sha256:
        raise RuntimeError("source mapping hash mismatch")
    manifest = json.loads(args.accepted_manifest.read_text())
    if manifest.get("status") != args.accepted_status:
        raise RuntimeError("rerender manifest status mismatch")
    rerender = official_loader(args.rerender_root, args.rerender_repo_id)
    comparison = compare_nonvisual_rows(SOURCE_SIM, args.rerender_root)
    binding = {
        "status": "rerender_identity_validated_not_materialized" if not args.materialize else "rerender_identity_validated",
        "rerender": {
            "root": str(args.rerender_root), "repo_id": args.rerender_repo_id,
            "tree_sha256": args.rerender_tree_sha256, "tree_hash_algorithm": args.tree_hash_algorithm,
            "manifest": str(args.accepted_manifest), "manifest_sha256": args.accepted_manifest_sha256,
            "mapping": str(args.source_mapping), "mapping_sha256": args.source_mapping_sha256,
            "accepted_status": args.accepted_status,
        },
        "nonvisual_exact_comparison": comparison,
        "training_started": False,
        "hardware_accessed": False,
    }
    if not args.materialize:
        print(json.dumps(binding, indent=2, sort_keys=True))
        return
    if DATASET_OUT.exists() or EVIDENCE.exists():
        raise FileExistsError("materialized output/evidence already exists")
    real = LeRobotDataset("local/task1_picklift_real24_budget_extension_v1_accepted", root=REAL24, video_backend="pyav")
    # aggregate_datasets reloads metadata from disk and requires one
    # robot_type. Build an experiment-local metadata shadow: metadata is
    # copied, large immutable data/video files are hard-linked, and only the
    # shadow info.json robot_type is harmonized to the Real24 deployment
    # identity. The rerender source remains untouched.
    source_robot_type = rerender.meta.robot_type
    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rerender_meta_shadow_", dir=DATASET_OUT.parent) as tmp:
        real_shadow_root = Path(tmp) / "real"
        sim_shadow_root = Path(tmp) / "simulation"

        def shadow_copy(src: str, dst: str) -> str:
            source = Path(src)
            if "meta" in source.parts:
                return shutil.copy2(src, dst)
            Path(dst).hardlink_to(source)
            return dst

        shutil.copytree(REAL24, real_shadow_root, copy_function=shadow_copy)
        shutil.copytree(args.rerender_root, sim_shadow_root, copy_function=shadow_copy)
        normalize_aggregate_shadow(real_shadow_root)
        normalize_aggregate_shadow(sim_shadow_root)
        real_shadow = LeRobotDataset("local/real24_aggregate_shadow", root=real_shadow_root, video_backend="pyav")
        sim_shadow = LeRobotDataset(args.rerender_repo_id + "_aggregate_shadow", root=sim_shadow_root, video_backend="pyav")
        merge_datasets([real_shadow, sim_shadow], "local/task1_picklift_real24_localsim24gap_rerender_additive_v1", DATASET_OUT,
                       concatenate_videos=False, concatenate_data=False)
    shutil.copy2(REAL24 / "meta/stats.json", DATASET_OUT / "meta/stats.json")
    combined = LeRobotDataset("local/task1_picklift_real24_localsim24gap_rerender_additive_v1", root=DATASET_OUT, video_backend="pyav")
    if combined.meta.total_episodes != 48 or len(combined) != 6196:
        raise RuntimeError("combined Dataset identity mismatch")
    reference = json.loads((HERE / "old_c_sampling_reference.json").read_text())["sampler"]
    groups = {"real24": list(range(24)), "source_b": list(range(24, 48))}
    digest = sampler_digest(combined, groups)
    expected = {"full": reference["full_index_stream_sha256_int64le"], "real24": reference["real24_index_stream_sha256_int64le"], "sim": reference["sim_index_stream_sha256_int64le"]}
    if digest != expected:
        raise RuntimeError(f"sampling stream mismatch: {digest} != {expected}")
    old = json.loads((OLD_EXP / "configs/r24_localsim24gap_full.json").read_text())
    formal = json.loads(json.dumps(old))
    formal["dataset"]["root"] = str(DATASET_OUT)
    formal["dataset"]["repo_id"] = "local/task1_picklift_real24_localsim24gap_rerender_additive_v1"
    formal["output_dir"] = str(TRAIN_OUT / "full_200k")
    formal["job_name"] = "task1_picklift_act_c_render_200k_v1_seed1000_step200000"
    smoke = json.loads(json.dumps(formal))
    smoke["steps"] = 500
    smoke["save_freq"] = 500
    smoke["output_dir"] = str(TRAIN_OUT / "smoke_500")
    smoke["job_name"] += "_smoke500"
    configs = HERE / "bound_configs"
    configs.mkdir(parents=True)
    (configs / "c_render_full_200k.json").write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    (configs / "c_render_smoke_500.json").write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n")
    binding.update({
        "status": "bound_ready_for_authorized_smoke_then_fresh_200k",
        "derived_dataset": {"root": str(DATASET_OUT), "episodes": 48, "frames": 6196},
        "aggregate_metadata_harmonization": {
            "source_rerender_robot_type": source_robot_type,
            "derived_robot_type": real.meta.robot_type,
            "scope": "in_memory_aggregate_metadata_only",
            "immutable_source_mutated": False,
        },
        "sampling_digest": digest,
        "sampling_exactly_matches_old_c": True,
        "configs": {
            "smoke": {"path": str(configs / "c_render_smoke_500.json"), "sha256": sha256(configs / "c_render_smoke_500.json")},
            "formal": {"path": str(configs / "c_render_full_200k.json"), "sha256": sha256(configs / "c_render_full_200k.json")},
        },
    })
    EVIDENCE.mkdir(parents=True)
    (EVIDENCE / "binding_result.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    print(json.dumps(binding, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
