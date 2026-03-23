import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from huggingface_hub import HfApi, create_repo

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import (
        get_hf_features_from_features,
        load_info,
        to_parquet_with_hf_images,
        write_info,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


WRIST_ORIENTATION_PATTERN = re.compile(r"^(left_hand|right_hand)\.orientation\.d([1-6])$", re.IGNORECASE)


def _load_info(root: Path) -> dict[str, Any]:
    return load_info(root)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _is_wrist_orientation_name(name: str) -> bool:
    return WRIST_ORIENTATION_PATTERN.match(_normalize_name(name)) is not None


def _update_info_for_wrist_orientations(info: dict[str, Any]) -> tuple[dict[str, Any], list[int], list[str]]:
    if "observation.state" not in info.get("features", {}):
        raise ValueError("'observation.state' not found in info.json")

    obs_state_info = info["features"]["observation.state"]
    original_names = obs_state_info.get("names", [])

    indices_to_remove = [i for i, name in enumerate(original_names) if _is_wrist_orientation_name(str(name))]
    if not indices_to_remove:
        raise ValueError(
            "No wrist orientation observation keys found. Expected keys like "
            "left_hand.orientation.d1..d6 and right_hand.orientation.d1..d6."
        )

    indices_to_keep = [i for i in range(len(original_names)) if i not in indices_to_remove]
    filtered_names = [original_names[i] for i in indices_to_keep]

    info["features"]["observation.state"]["names"] = filtered_names
    info["features"]["observation.state"]["shape"] = [len(filtered_names)]

    removed_names = [original_names[i] for i in indices_to_remove]
    return info, indices_to_keep, removed_names


def _filter_stats(stats: dict[str, Any], indices_to_keep: list[int], original_len: int) -> dict[str, Any]:
    if "observation.state" not in stats:
        return stats

    obs_stats = stats["observation.state"]
    for stat_key, stat_val in list(obs_stats.items()):
        if isinstance(stat_val, list) and len(stat_val) == original_len:
            obs_stats[stat_key] = [stat_val[i] for i in indices_to_keep]

    return stats


def _ensure_list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim > 0 else [value.item()]
    return [value]


def _normalize_list_columns(df: pd.DataFrame, features: dict[str, Any]) -> pd.DataFrame:
    for key, spec in features.items():
        if key not in df.columns:
            continue

        shape = spec.get("shape")
        if shape is None:
            continue

        if shape == (1,):
            df[key] = df[key].apply(lambda v: _ensure_list_value(v)[0])

    return df


def _verify_no_wrist_orientations_in_info(info: dict[str, Any]) -> None:
    names = info.get("features", {}).get("observation.state", {}).get("names", [])
    leftovers = [name for name in names if _is_wrist_orientation_name(str(name))]
    if leftovers:
        raise ValueError(f"Wrist orientation keys still present after update: {leftovers}")


def remove_wrists_orientation_observations(
    old_repo_id: str,
    new_repo_id: str,
    local_dir: str,
    revision: str | None = "main",
) -> None:
    print(f"🚀 Loading source dataset: {old_repo_id}")
    if revision is not None:
        print(f"🔖 Using revision: {revision}")

    dataset = LeRobotDataset(old_repo_id, force_cache_sync=True, revision=revision)
    root = Path(dataset.root)

    info = _load_info(root)
    original_info = copy.deepcopy(info)
    original_len = len(info["features"]["observation.state"].get("names", []))

    original_action_spec = copy.deepcopy(info["features"].get("action"))

    info, indices_to_keep, removed_names = _update_info_for_wrist_orientations(info)

    print(f"🗑️ Removing {len(removed_names)} wrist-orientation dims from observation.state")
    print(f"   Removed names: {removed_names}")

    output_path = Path(local_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    meta_src = root / "meta"
    if meta_src.exists():
        shutil.copytree(meta_src, output_path / "meta", dirs_exist_ok=True)

    videos_src = root / "videos"
    if videos_src.exists():
        shutil.copytree(videos_src, output_path / "videos", dirs_exist_ok=True)

    for fname in [".gitattributes", "README.md"]:
        src = root / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)

    data_src = root / "data"
    parquet_files = sorted(data_src.glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_src}")

    write_features = get_hf_features_from_features(info["features"])

    for src_path in parquet_files:
        df = pd.read_parquet(src_path).reset_index(drop=True)
        if "observation.state" not in df.columns:
            raise ValueError(f"Column 'observation.state' not found in {src_path}")

        df["observation.state"] = df["observation.state"].apply(lambda row: [row[i] for i in indices_to_keep])
        df = _normalize_list_columns(df, info["features"])

        rel_path = src_path.relative_to(root)
        dst_path = output_path / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        to_parquet_with_hf_images(df, dst_path, features=write_features)

    written_parquets = sorted((output_path / "data").glob("*/*.parquet"))
    if not written_parquets:
        raise FileNotFoundError(f"No parquet files written in {output_path / 'data'}")

    write_info(info, output_path)
    _verify_no_wrist_orientations_in_info(info)

    if info["features"].get("action") != original_action_spec:
        raise ValueError("Action feature spec changed unexpectedly; expected action to be preserved exactly.")

    stats_src = root / "meta" / "stats.json"
    if stats_src.exists():
        with open(stats_src, "r") as f:
            stats = json.load(f)
        stats = _filter_stats(stats, indices_to_keep, original_len)
        with open(output_path / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

    print("✅ Verification passed: wrist orientation observation keys removed")
    print("✅ Verification passed: action feature spec preserved")

    print(f"☁️ Uploading dataset to HF: {new_repo_id}")
    api = HfApi()
    create_repo(repo_id=new_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_path),
        repo_id=new_repo_id,
        repo_type="dataset",
    )

    print(f"\n✅ Dataset uploaded: https://huggingface.co/datasets/{new_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_repo", type=str, default="steb6/ROBOARENA_ONLY_YAW_FINAL")
    parser.add_argument("--new_repo", type=str, default="steb6/ROBOARENA_ONLY_YAW_FINAL-nowrist")
    parser.add_argument("--local_dir", type=str, default="data/clean_temp/ROBOARENA_ONLY_YAW_FINAL-nowrist")
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="HF revision/tag/commit to load (e.g. 'main')",
    )

    args = parser.parse_args()

    if args.new_repo == args.old_repo:
        args.new_repo = args.old_repo + "-nowrist"

    remove_wrists_orientation_observations(
        old_repo_id=args.old_repo,
        new_repo_id=args.new_repo,
        local_dir=args.local_dir,
        revision=args.revision,
    )
