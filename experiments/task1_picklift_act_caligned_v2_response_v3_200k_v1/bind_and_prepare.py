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
HERE = REPO / "experiments/task1_picklift_act_caligned_v2_response_v3_200k_v1"
OLD_EXP = REPO / "experiments/task1_picklift_act_additive_three_model_200k_v1"
CONTRACT = HERE / "preparation_contract.json"
OLD_CONFIG = OLD_EXP / "configs/r24_localsim24gap_full.json"
REAL24 = Path("/home/ubuntu24/Teleop/artifacts/derived/task1_picklift_real24_budget_extension_v1/accepted")
DATASET_OUT = Path(
    "/home/ubuntu24/Teleop/artifacts/datasets/"
    "task1_picklift_act_caligned_v2_response_v3_200k_v1/combined48_v1"
)
TRAIN_OUT = Path(
    "/home/ubuntu24/Teleop/artifacts/training/"
    "task1_picklift_act_caligned_v2_response_v3_200k_v1"
)
EVIDENCE = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/"
    "task1_picklift_act_caligned_v2_response_v3_200k_v1/binding_v1"
)
COMBINED_REPO_ID = "local/task1_picklift_real24_localsim24gap_response_v3_real_appearance_additive_v1"
EXPECTED_FINALIZATION_STATUS = "aligned_gap24_dataset_ready_for_authorized_act200k_training"
EXPECTED_REAL_STREAM = "f392d7b148905d90467a2565229df92d33e7805e8037a48eeca02c6d31730c53"
CANONICAL_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CANONICAL_TASK = "Pick up the red cube."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_tree(root: Path) -> str:
    rows = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def length_prefixed_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def derived_tree(root: Path) -> tuple[str, int, int]:
    rows: list[str] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        rows.append(f"{sha256(path)}  {size}  {path.relative_to(root).as_posix()}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest(), len(rows), total_bytes


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parquet_rows(root: Path) -> pd.DataFrame:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise RuntimeError(f"no parquet rows in {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def tensor_bytes(series: pd.Series) -> bytes:
    return np.asarray(np.stack(series.to_numpy()), dtype=np.float32).tobytes(order="C")


def compare_new_source_to_derived(source: Path, derived: Path) -> dict:
    """Validate the new trajectory mapping; never compare against historical Sim."""
    left = parquet_rows(source)
    right = parquet_rows(derived)
    if len(left) != 3023 or len(right) != 3023:
        raise RuntimeError("new response-v3 source/derived frame count mismatch")
    scalar_columns = [
        column
        for column in ("episode_index", "frame_index", "index", "task_index", "timestamp")
        if column in left.columns and column in right.columns
    ]
    if set(scalar_columns) != {"episode_index", "frame_index", "index", "task_index", "timestamp"}:
        raise RuntimeError("new response-v3 source/derived scalar columns incomplete")
    scalar_exact: dict[str, bool] = {}
    for column in scalar_columns:
        scalar_exact[column] = np.array_equal(left[column].to_numpy(), right[column].to_numpy())
        if not scalar_exact[column]:
            raise RuntimeError(f"source-to-derived row identity mismatch: {column}")
    tensors: dict[str, dict[str, object]] = {}
    for column in ("observation.state", "action"):
        if column not in left.columns or column not in right.columns:
            raise RuntimeError(f"missing source-to-derived tensor column: {column}")
        left_bytes = tensor_bytes(left[column])
        right_bytes = tensor_bytes(right[column])
        tensors[column] = {
            "source_sha256_float32le": hashlib.sha256(left_bytes).hexdigest(),
            "derived_sha256_float32le": hashlib.sha256(right_bytes).hexdigest(),
            "exact_equal": left_bytes == right_bytes,
        }
        if left_bytes != right_bytes:
            raise RuntimeError(f"source-to-derived nonvisual mismatch: {column}")
    return {
        "comparison": "new_response_v3_human_source_to_real_appearance_derived",
        "historical_sim_compared": False,
        "frames": len(left),
        "scalar_columns_exact": scalar_exact,
        "tensor_hashes": tensors,
    }


def validate_info(root: Path) -> dict:
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("total_episodes") != 24 or info.get("total_frames") != 3023 or info.get("fps") != 20:
        raise RuntimeError("Sim metadata count/fps mismatch")
    features = info["features"]
    for key in ("observation.state", "action"):
        feature = features[key]
        if feature.get("dtype") != "float32" or feature.get("shape") != [6]:
            raise RuntimeError(f"{key} dtype/shape mismatch")
        if feature.get("names") != CANONICAL_JOINT_NAMES:
            raise RuntimeError(f"{key} joint order mismatch")
    image = features["observation.images.front"]
    if image.get("shape") != [480, 640, 3] or image.get("dtype") != "video":
        raise RuntimeError("front camera metadata mismatch")
    return {
        "joint_order": CANONICAL_JOINT_NAMES,
        "state_action_dtype_shape": "float32[6]",
        "front_hwc": [480, 640, 3],
        "fps": 20,
    }


def official_loader(root: Path, repo_id: str, episodes: int, frames: int, *, scan_all: bool) -> LeRobotDataset:
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    if dataset.meta.total_episodes != episodes or len(dataset) != frames or dataset.meta.fps != 20:
        raise RuntimeError(f"official loader identity mismatch: {root}")
    indices = range(frames) if scan_all else sorted({0, frames // 2, frames - 1})
    checked = 0
    for index in indices:
        sample = dataset[index]
        if sample["observation.state"].shape != (6,) or sample["action"].shape != (6,):
            raise RuntimeError(f"state/action shape mismatch at {index}")
        if sample["observation.images.front"].shape != (3, 480, 640):
            raise RuntimeError(f"front shape mismatch at {index}")
        if not all(
            torch.isfinite(sample[key]).all()
            for key in ("observation.state", "action", "observation.images.front")
        ):
            raise RuntimeError(f"non-finite official loader sample at {index}")
        checked += 1
    dataset._binding_loader_checked_samples = checked
    return dataset


def normalize_aggregate_shadow(root: Path) -> None:
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


def sampler_digest(dataset: LeRobotDataset) -> dict:
    groups = {"real24": list(range(24)), "source_b": list(range(24, 48))}
    sampler = MatchedTwoStreamSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        dataset.meta.episodes["episode_index"],
        groups,
        8,
        1000,
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=0,
        seed=1000,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    full, real, sim = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    counts = {"real24": 0, "source_b": 0}
    for _ in range(200):
        order = list(sampler)
        if len(order) != 8000:
            raise RuntimeError("matched sampler epoch length mismatch")
        for offset in range(0, len(order), 8):
            batch = order[offset : offset + 8]
            if len(batch) != 8:
                raise RuntimeError("incomplete matched sampler batch")
            for index in batch:
                full.update(struct.pack("<q", index))
            for index in batch[:4]:
                real.update(struct.pack("<q", index))
            for index in batch[4:]:
                sim.update(struct.pack("<q", index))
            counts["real24"] += 4
            counts["source_b"] += 4
    result = {
        "full_index_stream_sha256_int64le": full.hexdigest(),
        "real24_index_stream_sha256_int64le": real.hexdigest(),
        "sim24_index_stream_sha256_int64le": sim.hexdigest(),
        "sample_slots": counts,
        "batch_composition": "4 Real24 + 4 response-v3 Sim24",
    }
    if result["real24_index_stream_sha256_int64le"] != EXPECTED_REAL_STREAM:
        raise RuntimeError("Real24 sampling stream does not byte-match old C")
    if counts != {"real24": 800000, "source_b": 800000}:
        raise RuntimeError("formal sampler exposure mismatch")
    return result


def changed_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in left.keys() | right.keys():
            child = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths |= changed_paths(left[key], right[key], child)
        return paths
    return set() if left == right else {prefix}


def make_configs() -> dict:
    old = json.loads(OLD_CONFIG.read_text())
    formal = json.loads(json.dumps(old))
    formal["dataset"]["root"] = str(DATASET_OUT)
    formal["dataset"]["repo_id"] = COMBINED_REPO_ID
    formal["output_dir"] = str(TRAIN_OUT / "full_200k")
    formal["job_name"] = "task1_picklift_act_caligned_v2_response_v3_200k_v1_seed1000_step200000"
    allowed = {"dataset.root", "dataset.repo_id", "output_dir", "job_name"}
    actual = changed_paths(old, formal)
    if actual != allowed:
        raise RuntimeError(f"formal config differs from old C outside allowed fields: {sorted(actual)}")
    smoke = json.loads(json.dumps(formal))
    smoke["steps"] = 500
    smoke["save_freq"] = 500
    smoke["output_dir"] = str(TRAIN_OUT / "smoke_500")
    smoke["job_name"] += "_smoke500"
    configs = HERE / "bound_configs"
    if configs.exists():
        raise FileExistsError(configs)
    configs.mkdir(parents=True)
    formal_path = configs / "c_aligned_v2_full_200k.json"
    smoke_path = configs / "c_aligned_v2_smoke_500.json"
    write_json(formal_path, formal)
    write_json(smoke_path, smoke)
    return {
        "formal": {"path": str(formal_path), "sha256": sha256(formal_path), "allowed_diff_from_old_c": sorted(allowed)},
        "smoke": {"path": str(smoke_path), "sha256": sha256(smoke_path)},
    }


def validate_finalization(path: Path, expected_sha: str) -> tuple[dict, Path, str, Path, Path, dict]:
    contract = json.loads(CONTRACT.read_text())
    if expected_sha != contract["research_authority"]["finalization_manifest_sha256"]:
        raise RuntimeError("requested finalization SHA is not the frozen contract")
    if sha256(path) != expected_sha:
        raise RuntimeError("research finalization manifest hash mismatch")
    finalization = json.loads(path.read_text())
    if finalization.get("status") != EXPECTED_FINALIZATION_STATUS:
        raise RuntimeError("new response-v3 Dataset is not accepted for training")
    training = finalization["training_input"]
    sim_contract = contract["sim24"]
    exact = {
        "root": str(training["root"]),
        "repo_id": training["repo_id"],
        "episodes": training["episode_count"],
        "frames": training["frame_count"],
        "tree": training["length_prefixed_tree_sha256"],
    }
    expected = {
        "root": sim_contract["root"],
        "repo_id": sim_contract["repo_id"],
        "episodes": sim_contract["episodes"],
        "frames": sim_contract["frames"],
        "tree": sim_contract["tree_sha256"],
    }
    if exact != expected:
        raise RuntimeError(f"finalization training input drift: {exact} != {expected}")
    if finalization["collection_identity"]["membership_sha256"] != sim_contract["membership_sha256"]:
        raise RuntimeError("new human membership SHA mismatch")
    runtime = finalization["product_runtime"]
    if runtime["camera_profile_id"] != sim_contract["camera_profile_id"]:
        raise RuntimeError("camera-v8 identity mismatch")
    if runtime["response_profile_id"] != sim_contract["response_profile_id"]:
        raise RuntimeError("response-v3 identity mismatch")
    if runtime["response_profile_sha256"] != sim_contract["response_profile_sha256"]:
        raise RuntimeError("response-v3 profile hash mismatch")
    if runtime["action_semantics"] != "official_sent_real_unit_command_before_internal_actuator_response":
        raise RuntimeError("Sim action semantics mismatch")
    boundaries = finalization["historical_and_claim_boundaries"]
    if boundaries["unity_source_rgb_is_training_input"] is not False:
        raise RuntimeError("Unity source RGB is ineligible for training")
    materialization = Path(training["materialization_manifest"])
    if sha256(materialization) != sim_contract["materialization_manifest_sha256"]:
        raise RuntimeError("materialization manifest hash mismatch")
    materialization_payload = json.loads(materialization.read_text())
    mapping = materialization.parent / materialization_payload["output"]["source_to_derived_frame_mapping_relative"]
    if sha256(mapping) != sim_contract["source_to_derived_mapping_sha256"]:
        raise RuntimeError("source-to-derived mapping hash mismatch")
    source = Path(finalization["source_lerobot_v3"]["root"])
    sim_root = Path(training["root"])
    return finalization, sim_root, training["repo_id"], source, mapping, materialization_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalization-manifest", type=Path, required=True)
    parser.add_argument("--finalization-manifest-sha256", required=True)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--sample-loader-check", action="store_true", help="check 3 frames instead of all; tests only")
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text())
    finalization, sim_root, sim_repo_id, source_root, mapping, materialization = validate_finalization(
        args.finalization_manifest, args.finalization_manifest_sha256
    )
    if legacy_tree(REAL24) != contract["real24"]["tree_sha256"]:
        raise RuntimeError("frozen Real24 tree mismatch")
    if sha256(REAL24 / "meta/stats.json") != contract["real24"]["dataset_stats_json_sha256"]:
        raise RuntimeError("frozen Real24 Dataset stats.json mismatch")
    if length_prefixed_tree(sim_root) != contract["sim24"]["tree_sha256"]:
        raise RuntimeError("response-v3 derived Sim24 tree mismatch")
    validate_info(sim_root)
    sim = official_loader(sim_root, sim_repo_id, 24, 3023, scan_all=not args.sample_loader_check)
    comparison = compare_new_source_to_derived(source_root, sim_root)
    membership = materialization["source"]["membership"]
    if len(membership) != 24 or len({row["plan_item_id"] for row in membership}) != 24:
        raise RuntimeError("new human source membership is not 24 unique episodes")
    result: dict[str, object] = {
        "schema": "task1_picklift_act_caligned_v2_response_v3_binding_result_v1",
        "status": "input_identity_validated_not_materialized",
        "finalization": {
            "path": str(args.finalization_manifest),
            "sha256": args.finalization_manifest_sha256,
            "status": finalization["status"],
        },
        "sim24": {
            "root": str(sim_root),
            "repo_id": sim_repo_id,
            "episodes": 24,
            "frames": 3023,
            "tree_sha256": contract["sim24"]["tree_sha256"],
            "official_loader_checked_samples": sim._binding_loader_checked_samples,
            "mapping": str(mapping),
            "mapping_sha256": sha256(mapping),
            "membership_count": 24,
            "membership_sha256": finalization["collection_identity"]["membership_sha256"],
        },
        "new_source_to_derived_exactness": comparison,
        "historical_sim_state_action_comparison_performed": False,
        "training_started": False,
        "hardware_accessed": False,
    }
    if not args.materialize:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if DATASET_OUT.exists() or EVIDENCE.exists() or (HERE / "bound_configs").exists():
        raise FileExistsError("new experiment materialization/config/evidence identity already exists")
    real = official_loader(REAL24, contract["real24"]["repo_id"], 24, 4263, scan_all=False)
    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task1_caligned_v2_bind_", dir=DATASET_OUT.parent) as temporary:
        temporary_root = Path(temporary)
        real_shadow_root = temporary_root / "real_shadow"
        sim_shadow_root = temporary_root / "sim_shadow"

        def shadow_copy(src: str, dst: str) -> str:
            source = Path(src)
            if "meta" in source.parts:
                return shutil.copy2(src, dst)
            Path(dst).hardlink_to(source)
            return dst

        shutil.copytree(REAL24, real_shadow_root, copy_function=shadow_copy)
        shutil.copytree(sim_root, sim_shadow_root, copy_function=shadow_copy)
        normalize_aggregate_shadow(real_shadow_root)
        normalize_aggregate_shadow(sim_shadow_root)
        real_shadow = LeRobotDataset("local/task1_caligned_v2_real_shadow", root=real_shadow_root, video_backend="pyav")
        sim_shadow = LeRobotDataset("local/task1_caligned_v2_sim_shadow", root=sim_shadow_root, video_backend="pyav")
        merged_root = temporary_root / "combined"
        merge_datasets(
            [real_shadow, sim_shadow],
            COMBINED_REPO_ID,
            merged_root,
            concatenate_videos=False,
            concatenate_data=False,
        )
        shutil.copy2(REAL24 / "meta/stats.json", merged_root / "meta/stats.json")
        combined = LeRobotDataset(COMBINED_REPO_ID, root=merged_root, video_backend="pyav")
        if combined.meta.total_episodes != 48 or len(combined) != 7286 or combined.meta.fps != 20:
            raise RuntimeError("combined Real24 + response-v3 Sim24 identity mismatch")
        digest = sampler_digest(combined)
        merged_root.rename(DATASET_OUT)
    combined_tree, combined_files, combined_bytes = derived_tree(DATASET_OUT)
    configs = make_configs()
    result.update(
        {
            "status": "bound_ready_for_authorized_smoke_then_fresh_200k",
            "combined_dataset": {
                "root": str(DATASET_OUT),
                "repo_id": COMBINED_REPO_ID,
                "episodes": 48,
                "frames": 7286,
                "tree_sha256": combined_tree,
                "file_count": combined_files,
                "bytes": combined_bytes,
                "stats_sha256": sha256(DATASET_OUT / "meta/stats.json"),
            },
            "sampling": {
                **digest,
                "real24_stream_exactly_matches_old_c": True,
                "historical_sim_stream_equality_required": False,
            },
            "configs": configs,
        }
    )
    EVIDENCE.mkdir(parents=True)
    write_json(EVIDENCE / "binding_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
