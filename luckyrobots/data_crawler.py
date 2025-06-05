import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_DIRS = ["meta", "data", "videos"]
REQUIRED_META_FILES = ["dataset.json"]


def check_root_structure(root):
    print(f"\n[Root Structure Check] {root}")
    missing = []
    for d in REQUIRED_DIRS:
        if not (Path(root) / d).is_dir():
            print(f"  MISSING: {d}/")
            missing.append(d)
        else:
            print(f"  FOUND:   {d}/")
    return missing


def check_meta_files(meta_dir):
    print(f"\n[Meta Files Check] {meta_dir}")
    missing = []
    for f in REQUIRED_META_FILES:
        if not (Path(meta_dir) / f).is_file():
            print(f"  MISSING: {f}")
            missing.append(f)
        else:
            print(f"  FOUND:   {f}")
    # Optionally, parse and print summary
    for f in REQUIRED_META_FILES:
        fpath = Path(meta_dir) / f
        if fpath.is_file():
            try:
                with open(fpath) as fp:
                    data = json.load(fp)
                print(f"    {f} keys: {list(data.keys())}")
            except Exception as e:
                print(f"    ERROR reading {f}: {e}")
    return missing


def traverse_and_summarize(root):
    print("\n[Directory Traversal & Summary]")
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            rel = "(root)"
        print(f"  {rel}/: {len(dirnames)} dirs, {len(filenames)} files")
        # Optionally, print a few files
        for f in filenames[:3]:
            print(f"    - {f}")


def find_parquet_files(root):
    parquet_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".parquet"):
                parquet_files.append(os.path.join(dirpath, f))
    return parquet_files


def analyze_parquet(file_path):
    print(f"\n[Parquet Analysis] {file_path}")
    try:
        df = pd.read_parquet(file_path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  dtypes: {df.dtypes.to_dict()}")
        print(f"  Head:\n{df.head(2)}")
        print(f"  Describe:\n{df.describe(include='all').transpose()}")
        nans = df.isna().sum().sum()
        if nans > 0:
            print(f"  WARNING: {nans} NaN values found!")
        if df.duplicated().any():
            print("  WARNING: Duplicated rows found!")
        if df.shape[0] == 0:
            print("  WARNING: No rows!")
    except Exception as e:
        print(f"  ERROR reading parquet: {e}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def minimal_print(msg):
    print(msg)


def main(dataset_path):
    summary = defaultdict(list)
    minimal_print(f"Analyzing dataset at: {dataset_path}")
    # 1. Check root structure
    missing_dirs = check_root_structure(dataset_path)
    if missing_dirs:
        minimal_print(f"ERROR: Missing required directories: {missing_dirs}. Exiting.")
        return
    meta_dir = Path(dataset_path) / "meta"
    info_json_path = meta_dir / "info.json"
    if not info_json_path.is_file():
        minimal_print(f"ERROR: Missing required file: {info_json_path}. Exiting.")
        return
    info = load_json(info_json_path)
    features = info["features"]
    # 2. Validate meta/ files
    check_meta_files(meta_dir)
    # 3. Traverse and summarize
    traverse_and_summarize(dataset_path)
    # 4. Parquet analysis
    parquet_files = find_parquet_files(dataset_path)
    if parquet_files:
        print(f"\nFound {len(parquet_files)} parquet files.")
        for pf in parquet_files:
            analyze_parquet(pf)
    else:
        print("\nNo parquet files found.")
    # 5. Load meta info
    # --- Print detected image features and their properties ---
    image_features = []
    minimal_print("\nDetected image features:")
    for fname, fdef in features.items():
        if fdef.get("dtype") == "video":
            image_features.append(fname)
            minimal_print(f"  - {fname}: shape={fdef.get('shape')}, info={fdef.get('info', {})}")
            # Warn if important metadata is missing
            info_dict = fdef.get("info", {})
            for key in ["video.codec", "video.height", "video.width", "video.channels"]:
                if key not in info_dict:
                    minimal_print(f"    WARNING: {fname} missing '{key}' in info")
    if not image_features:
        minimal_print("  (No image/video features detected)")

    # --- OG format check ---
    lerobot21_format_ok = True
    parquet_image_issues = []
    video_file_issues = []
    # 1. Check that no image features are present in Parquet columns
    data_dir = Path(dataset_path) / "data/chunk-000"
    parquet_files = list(data_dir.glob("*.parquet"))
    for pf in parquet_files[:1]:  # Just check the first file for schema
        try:
            df = pd.read_parquet(pf)
            for img_feat in image_features:
                if img_feat in df.columns:
                    parquet_image_issues.append(
                        f"Image feature '{img_feat}' found in Parquet columns of {pf.name}"
                    )
                    lerobot21_format_ok = False
        except Exception as e:
            summary["read_errors"].append(f"OG check: {pf.name}: {e}")
    # 2. Check that video files exist for each image feature and episode
    for img_feat in image_features:
        for ep_idx in range(info.get("total_episodes", 0)):
            chunk = ep_idx // info.get("chunks_size", 1000)
            video_path = Path(dataset_path) / f"videos/chunk-{chunk:03d}/{img_feat}/episode_{ep_idx:06d}.mp4"
            if not video_path.is_file():
                video_file_issues.append(f"Missing video file: {video_path.relative_to(dataset_path)}")
                lerobot21_format_ok = False
    if parquet_image_issues:
        summary["lerobot21_schema"].extend(parquet_image_issues)
    if video_file_issues:
        summary["lerobot21_schema"].extend(video_file_issues)
    # --- End OG format check ---

    # 6. Load episode stats and metadata
    stats = {e["episode_index"]: e["stats"] for e in load_jsonl(meta_dir / "episodes_stats.jsonl")}
    episodes = {e["episode_index"]: e for e in load_jsonl(meta_dir / "episodes.jsonl")}
    tasks = {t["task_index"]: t["task"] for t in load_jsonl(meta_dir / "tasks.jsonl")}
    # 7. Prepare expected columns and dtypes
    # Only require non-image features in Parquet files
    expected_cols = {k for k, v in features.items() if v.get("dtype") != "video"}
    expected_dtypes = {k: v["dtype"] for k, v in features.items()}
    # 8. Check all expected parquet files exist
    data_dir = Path(dataset_path) / "data/chunk-000"
    parquet_files = list(data_dir.glob("*.parquet"))
    found_indices = set()
    for pf in parquet_files:
        idx = int(pf.stem.split("_")[-1])
        found_indices.add(idx)
    missing_indices = set(episodes.keys()) - found_indices
    if missing_indices:
        summary["missing_files"].append(f"Missing parquet files for episodes: {sorted(missing_indices)}")
    # 9. Check for duplicate episode indices
    if len(episodes) != len(set(episodes.keys())):
        summary["duplicates"].append("Duplicate episode_index in episodes.jsonl")
    # 10. Check referential integrity for tasks
    for e in episodes.values():
        for t in e["tasks"]:
            if t not in tasks.values():
                summary["referential"].append(
                    f"Task '{t}' in episode {e['episode_index']} not found in tasks.jsonl"
                )
    # 11. Per-episode checks
    for idx, ep in episodes.items():
        pf = data_dir / f"episode_{idx:06d}.parquet"
        if not pf.exists():
            continue
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            summary["read_errors"].append(f"Episode {idx}: {e}")
            continue
        # Schema check
        missing_cols = expected_cols - set(df.columns)
        extra_cols = set(df.columns) - expected_cols
        if missing_cols:
            summary["schema"].append(f"Episode {idx}: missing columns {missing_cols}")
        if extra_cols:
            summary["schema"].append(f"Episode {idx}: extra columns {extra_cols}")
        # Dtype check
        for col in expected_cols & set(df.columns):
            expected = expected_dtypes[col]
            # Determine if this is a vector-valued column
            is_vector = False
            if (
                "shape" in features[col]
                and isinstance(features[col]["shape"], list)
                and features[col]["shape"]
                and features[col]["shape"][0] > 1
            ):
                is_vector = True
            actual = str(df[col].dtype)
            if is_vector:
                # Accept 'object' dtype for vector columns (OG format)
                continue
            else:
                # Only check dtype for scalar columns
                if expected == "float32" and not actual.startswith("float"):
                    summary["dtype"].append(f"Episode {idx}: {col} dtype {actual} != {expected}")
                if expected == "int64" and not actual.startswith("int"):
                    summary["dtype"].append(f"Episode {idx}: {col} dtype {actual} != {expected}")
        # Length check
        if len(df) != ep["length"]:
            summary["length"].append(f"Episode {idx}: length {len(df)} != {ep['length']}")
        # Value range, NaN/Inf, outlier checks
        ep_stats = stats.get(idx, {})
        for col in expected_cols & set(df.columns):
            if col not in ep_stats:
                continue
            col_min, col_max = np.array(ep_stats[col]["min"]), np.array(ep_stats[col]["max"])
            values = df[col].values
            # Handle scalar vs array-valued columns
            try:
                if values.ndim == 1 and col_min.shape == ():
                    # Scalar feature
                    if np.any(values < col_min) or np.any(values > col_max):
                        summary["range"].append(f"Episode {idx}: {col} out of min/max range")
                elif values.ndim == 1 and col_min.shape != ():
                    # Vector feature stored as object/array
                    stacked = np.stack(values)
                    if np.any(stacked < col_min) or np.any(stacked > col_max):
                        summary["range"].append(f"Episode {idx}: {col} out of min/max range (vector)")
                else:
                    summary["shape"].append(f"Episode {idx}: {col} unexpected value shape {values.shape}")
            except Exception as e:
                summary["range"].append(f"Episode {idx}: {col} range check error: {e}")
            # NaN/Inf check
            try:
                if values.ndim == 1 and col_min.shape == ():
                    if np.isnan(values).any() or np.isinf(values).any():
                        summary["nan"].append(f"Episode {idx}: {col} contains NaN/Inf")
                elif values.ndim == 1 and col_min.shape != ():
                    stacked = np.stack(values)
                    if np.isnan(stacked).any() or np.isinf(stacked).any():
                        summary["nan"].append(f"Episode {idx}: {col} contains NaN/Inf (vector)")
            except Exception as e:
                summary["nan"].append(f"Episode {idx}: {col} NaN/Inf check error: {e}")
            # Outlier check (mean ± 3*std)
            try:
                mean, std = np.array(ep_stats[col]["mean"]), np.array(ep_stats[col]["std"])
                if values.ndim == 1 and mean.shape == ():
                    if np.any(values < mean - 3 * std) or np.any(values > mean + 3 * std):
                        summary["outlier"].append(f"Episode {idx}: {col} has outliers (beyond 3*std)")
                elif values.ndim == 1 and mean.shape != ():
                    stacked = np.stack(values)
                    if np.any(stacked < mean - 3 * std) or np.any(stacked > mean + 3 * std):
                        summary["outlier"].append(f"Episode {idx}: {col} has outliers (vector)")
            except Exception as e:
                summary["outlier"].append(f"Episode {idx}: {col} outlier check error: {e}")
            # Image normalization
            if features[col]["dtype"] == "video":
                try:
                    if values.ndim == 1:
                        stacked = np.stack(values)
                        if np.any(stacked < 0) or np.any(stacked > 1):
                            summary["image"].append(f"Episode {idx}: {col} image values not in [0,1]")
                except Exception as e:
                    summary["image"].append(f"Episode {idx}: {col} image normalization check error: {e}")
        # Episode index check
        if df["episode_index"].nunique() != 1 or df["episode_index"].iloc[0] != idx:
            summary["referential"].append(f"Episode {idx}: episode_index column mismatch")
    # --- Final summary ---
    minimal_print("\n===== DATA INTEGRITY SUMMARY =====")
    if lerobot21_format_ok:
        minimal_print("LeRobot 2.1 FORMAT: ✅ (matches ideal LeRobot 2.1 schema)")
    else:
        minimal_print("LeRobot 2.1 FORMAT: ❌ (deviates from ideal LeRobot 2.1 schema)")
    total_episodes = len(episodes)
    total_files = len(parquet_files)
    error_categories = [k for k in summary if k not in ["range", "outlier"]]
    warning_categories = ["range", "outlier"]
    errors = sum(len(summary[k]) for k in error_categories)
    warnings = sum(len(summary[k]) for k in warning_categories)
    # Print detailed issues and warnings with intuitive names and explanations
    category_names = {
        "lerobot21_schema": "LeRobot 2.1 Format Violations",
        "schema": "Schema Mismatches",
        "dtype": "Data Type Issues",
        "length": "Episode Length Mismatches",
        "nan": "NaN/Inf Issues",
        "range": "Value Range Warnings",
        "outlier": "Statistical Outlier Warnings",
        "referential": "Referential Integrity Issues",
        "duplicates": "Duplicate Indices",
        "missing_files": "Missing Parquet Files",
        "read_errors": "File Read Errors",
        "image": "Image Normalization Issues",
        "shape": "Shape Issues",
    }
    category_explanations = {
        "lerobot21_schema": "These issues indicate the dataset does not follow the LeRobot 2.1 format: image features should not be in Parquet, and all referenced video files must exist.",
        "schema": "Schema mismatches mean required columns are missing or extra columns are present in Parquet files. This usually means the data was not exported with the correct schema as defined in info.json.",
        "dtype": "Data type issues mean a column has a different dtype than expected (e.g., int instead of float). For vector columns, object dtype is allowed.",
        "length": "Episode length mismatches mean the number of rows in a Parquet file does not match the expected length from metadata.",
        "nan": "NaN/Inf issues mean some values are missing or infinite, which can break downstream processing.",
        "range": "Value range warnings mean some values are outside the expected min/max from stats. This may indicate outliers or data corruption.",
        "outlier": "Statistical outlier warnings mean values are far from the mean (beyond 3*std). These may be valid but should be reviewed.",
        "referential": "Referential integrity issues mean indices or references do not match across files (e.g., missing tasks or episode indices).",
        "duplicates": "Duplicate indices mean the same episode or task index appears more than once in metadata.",
        "missing_files": "Missing Parquet files mean some expected episode files are not present.",
        "read_errors": "File read errors mean a file could not be opened or parsed.",
        "image": "Image normalization issues mean image values are not in the expected [0,1] range.",
        "shape": "Shape issues mean the data shape does not match the expected shape from info.json.",
    }
    for k, v in summary.items():
        if v:
            display_name = category_names.get(k, k.upper())
            explanation = category_explanations.get(k, "")
            if k in warning_categories:
                minimal_print(f"⚠️  {display_name}: {len(v)} warnings")
            else:
                minimal_print(f"❌ {display_name}: {len(v)} errors")
            if explanation:
                minimal_print(f"    → {explanation}")
            for msg in v[:5]:  # Show up to 5 per category
                minimal_print(f"  - {msg}")
            if len(v) > 5:
                minimal_print(f"  ...and {len(v) - 5} more")
    # Print summary table
    minimal_print("\n----- SUMMARY TABLE -----")
    minimal_print(f"{'-' * 35}-|{'-' * 6}")
    for k in error_categories + warning_categories:
        count = len(summary.get(k, []))
        if count > 0:
            display_name = category_names.get(k, k.upper())
            minimal_print(f"{display_name:<35} | {count:>5}")
    minimal_print(f"{'Total Episodes':<35} | {total_episodes:>5}")
    minimal_print(f"{'Parquet Files Found':<35} | {total_files:>5}")
    minimal_print(f"{'LeRobot 2.1 Format':<35} | {'YES' if lerobot21_format_ok else 'NO':>5}")
    minimal_print(f"{'Total Errors':<35} | {errors:>5}")
    minimal_print(f"{'Total Warnings':<35} | {warnings:>5}")
    minimal_print(f"{'-' * 45}")
    if errors == 0:
        minimal_print("✅ All checks passed. Dataset integrity: OK.")
    else:
        minimal_print("❌ Some checks failed. Please review the issues above.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_crawler.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
