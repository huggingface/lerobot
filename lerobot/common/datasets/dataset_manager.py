# lerobot_dataset_manager.py
# Copyright 2024-2025 The HuggingFace Inc. team and contributors.
# Licensed under the Apache-2.0 license.

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd  # type: ignore

# --- Constants ---
PAD = 6  # Padding for episode numbers (e.g., 000032)
CHUNK_NAME_DEFAULT = "chunk-000"  # Default chunk name, primarily for CLI convenience
MERGE_NUM_KEYS = ["total_episodes", "total_frames", "total_videos"]  # For merge_info
DELETE_STEM_RE = re.compile(r"^episode_(\d{6})$")
DELETE_PATCH_KEYS = {"episode_index", "index"}  # For delete _patch


class DatasetManager:
    """
    Manages Lerobot datasets, allowing operations like merging and deleting episodes.
    """

    # ─────────────────────────────────── Static Utilities ────────────────────────────────── #
    @staticmethod
    def _extract_idx_from_name(name_with_index: str, default_pad: int = PAD) -> int:
        """Extracts a numerical index from a filename stem or full name."""
        num_re = re.compile(r"(\d+)(?=\.parquet$|\.mp4$|$)")
        match = num_re.search(name_with_index)
        if not match:
            raise ValueError(f"Impossible de trouver un index numérique dans {name_with_index}")
        return int(match.group(1))

    @staticmethod
    def _natural_sort_paths(paths: Iterable[Path]) -> List[Path]:
        """Sorts an iterable of Path objects naturally based on extracted numerical indices."""
        return sorted(paths, key=lambda p: DatasetManager._extract_idx_from_name(p.name))

    @staticmethod
    def safe_mkdir(path: Path) -> None:
        """Creates a directory if it doesn't exist, including parent directories."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_jsonl(path: Path) -> List[Dict]:
        """Reads a JSONL file and returns a list of dictionaries."""
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def write_jsonl(objs: Iterable[Dict], path: Path) -> None:
        """Writes an iterable of dictionaries to a JSONL file."""
        with path.open("w") as f:
            for o in objs:
                f.write(json.dumps(o, separators=(",", ":")) + "\n")

    # --- Utilities for MERGE ---
    @staticmethod
    def _shift_any_positive_recursive(obj: Any, offset: int) -> Any:
        """Recursively shifts integer values in a JSON-like object by a positive offset (for merging)."""
        if isinstance(obj, int):
            return obj + offset
        if isinstance(obj, list):
            return [DatasetManager._shift_any_positive_recursive(x, offset) for x in obj]
        if isinstance(obj, dict):
            return {k: DatasetManager._shift_any_positive_recursive(v, offset) for k, v in obj.items()}
        return obj

    # --- Utilities for DELETE ---
    @staticmethod
    def _ep_id_from_stem(name: str) -> Optional[int]:
        m = DELETE_STEM_RE.match(name)
        return int(m.group(1)) if m else None

    @staticmethod
    def _add_offset_for_delete(val: Any, off: int):
        if isinstance(val, int):
            return val + off
        if isinstance(val, list):
            return [DatasetManager._add_offset_for_delete(x, off) for x in val]
        return val

    @staticmethod
    def _patch_indices_recursive_negative(obj: Any, off: int):
        """Recursively shifts specific integer values in a JSON-like object by a negative offset (for deleting)."""
        if isinstance(obj, dict):
            return {
                k: (
                    DatasetManager._add_offset_for_delete(v, off)
                    if k in DELETE_PATCH_KEYS
                    else DatasetManager._patch_indices_recursive_negative(v, off)
                )
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [DatasetManager._patch_indices_recursive_negative(x, off) for x in obj]
        return obj

    # ─────────────────────────────────── MERGE Operation ─────────────────────────────────── #

    def merge_datasets(
        self, dataset_paths_str: str, output_dir: Path, chunk_name: str, verbose: bool = False
    ):
        """
        Merges multiple Lerobot datasets into a new output directory.
        """
        dataset_paths = [Path(p.strip()) for p in dataset_paths_str.strip().split() if p.strip()]
        if not dataset_paths:
            print("No dataset paths provided for merging.")
            return

        if verbose:
            print(f"Starting merge operation. Output directory: {output_dir}")

        data_dst_dir = output_dir / "data" / chunk_name
        meta_dst_dir = output_dir / "meta"
        video_dst_chunk_root = output_dir / "videos" / chunk_name

        self.safe_mkdir(data_dst_dir)
        self.safe_mkdir(meta_dst_dir)
        self.safe_mkdir(video_dst_chunk_root.parent)
        self.safe_mkdir(video_dst_chunk_root)

        cumulative_episode_offset_parquets = 0
        cumulative_frame_offset_parquets = 0
        total_parquets_processed_overall = 0
        actual_episode_counts_per_dataset = []

        if verbose:
            print("--- Processing Parquet Files and Determining Episode Counts ---")
        for i, dataset_path in enumerate(dataset_paths):
            if verbose:
                print(f"Processing dataset {i + 1}/{len(dataset_paths)}: {dataset_path}")
            current_dataset_frames = 0
            info_path = dataset_path / "meta" / "info.json"
            if info_path.exists():
                try:
                    info_data = json.loads(info_path.read_text())
                    current_dataset_frames = info_data.get("total_frames", 0)
                    if not isinstance(current_dataset_frames, int):
                        current_dataset_frames = 0
                except json.JSONDecodeError:
                    current_dataset_frames = 0

            processed_eps = self._copy_parquet_and_update_indices_for_merge(
                dataset_path,
                data_dst_dir,
                chunk_name,
                cumulative_episode_offset_parquets,
                cumulative_frame_offset_parquets,
                verbose,
            )
            if verbose:
                print(f"  Processed {processed_eps} Parquet episode files from {dataset_path}.")
            total_parquets_processed_overall += processed_eps
            actual_episode_counts_per_dataset.append(processed_eps)
            cumulative_episode_offset_parquets += processed_eps
            cumulative_frame_offset_parquets += current_dataset_frames

        if verbose:
            print("\n--- Processing Metadata Files ---")
        self._merge_all_meta_files(dataset_paths, meta_dst_dir, actual_episode_counts_per_dataset, verbose)

        if verbose:
            print("\n--- Processing Video Files ---")
        self._copy_all_videos_for_merge(
            dataset_paths, video_dst_chunk_root, chunk_name, actual_episode_counts_per_dataset, verbose
        )

        final_info_path = meta_dst_dir / "info.json"
        final_total_episodes = "N/A"
        if final_info_path.exists():
            try:
                final_info = json.loads(final_info_path.read_text())
                final_total_episodes = final_info.get("total_episodes", "N/A")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not parse {final_info_path} to get final total episodes: {e}")

        print(
            "\n✅ Merge finished!\n"
            f"  • Total Parquet files processed: {total_parquets_processed_overall}\n"
            f"  • Total episodes in merged dataset (from info.json): {final_total_episodes}\n"
            f"  • Output directory: {output_dir}"
        )

    def _copy_parquet_and_update_indices_for_merge(
        self,
        src_root: Path,
        dst_data_dir: Path,
        chunk_name: str,
        episode_idx_offset: int,
        frame_idx_offset: int,
        verbose: bool,
    ) -> int:
        src_chunk_dir = src_root / "data" / chunk_name
        if not src_chunk_dir.exists():
            if verbose:
                print(f"Source chunk directory not found: {src_chunk_dir}")
            return 0
        src_files = self._natural_sort_paths(src_chunk_dir.glob("episode_*.parquet"))
        if not src_files:
            if verbose:
                print(f"No Parquet files found in {src_chunk_dir}")
            return 0

        count_processed = 0
        for src_file_path in src_files:
            original_episode_idx = self._extract_idx_from_name(src_file_path.name)
            new_episode_global_idx = original_episode_idx + episode_idx_offset
            dst_file_path = dst_data_dir / f"episode_{new_episode_global_idx:0{PAD}d}.parquet"
            try:
                df = pd.read_parquet(src_file_path)
                if "episode_index" in df.columns:
                    df["episode_index"] = new_episode_global_idx
                if "index" in df.columns:
                    df["index"] = df["index"] + frame_idx_offset
                if "frame_index" in df.columns:
                    df["frame_index"] = df["frame_index"] + frame_idx_offset
                df.to_parquet(dst_file_path)
                count_processed += 1
            except Exception as e:
                print(f"Error processing Parquet {src_file_path} to {dst_file_path}: {e}")
                if dst_file_path.exists():
                    dst_file_path.unlink(missing_ok=True)
        return count_processed

    def _merge_all_meta_files(
        self, dataset_paths: List[Path], meta_dst_dir: Path, actual_episode_counts: List[int], verbose: bool
    ):
        current_meta_episode_offset = 0
        ep_stats_out = meta_dst_dir / "episodes_stats.jsonl"
        ep_out = meta_dst_dir / "episodes.jsonl"
        tasks_out = meta_dst_dir / "tasks.jsonl"
        info_out = meta_dst_dir / "info.json"

        for p in [ep_stats_out, ep_out, tasks_out, info_out]:
            p.unlink(missing_ok=True)

        for i, dataset_path in enumerate(dataset_paths):
            src_meta_dir = dataset_path / "meta"
            eps_in_this_ds_for_meta = actual_episode_counts[i]
            if not src_meta_dir.exists():
                if verbose:
                    print(f"  Warning: Meta dir {src_meta_dir} not found.")
                current_meta_episode_offset += eps_in_this_ds_for_meta
                continue

            # Merge episodes_stats.jsonl
            src_ep_stats = src_meta_dir / "episodes_stats.jsonl"
            if src_ep_stats.exists():
                base_data = self.read_jsonl(ep_stats_out) if ep_stats_out.exists() else []
                new_data = self.read_jsonl(src_ep_stats)
                for r_new in new_data:
                    r_new["episode_index"] += current_meta_episode_offset
                    if "index" in r_new:  # often frame indices
                        idx_val = r_new["index"]
                        r_new["index"] = (
                            [x + current_meta_episode_offset for x in idx_val]
                            if isinstance(idx_val, list)
                            else idx_val + current_meta_episode_offset
                        )
                    if "stats" in r_new:  # Shift indices inside stats dictionary
                        r_new["stats"] = self._shift_any_positive_recursive(
                            r_new["stats"], current_meta_episode_offset
                        )
                self.write_jsonl(base_data + new_data, ep_stats_out)

            # Merge episodes.jsonl
            src_ep = src_meta_dir / "episodes.jsonl"
            if src_ep.exists():
                base_data = self.read_jsonl(ep_out) if ep_out.exists() else []
                new_data = self.read_jsonl(src_ep)
                for r_new in new_data:  # Similar shifting as episodes_stats
                    r_new["episode_index"] += current_meta_episode_offset
                    if "index" in r_new:
                        idx_val = r_new["index"]
                        r_new["index"] = (
                            [x + current_meta_episode_offset for x in idx_val]
                            if isinstance(idx_val, list)
                            else idx_val + current_meta_episode_offset
                        )
                self.write_jsonl(base_data + new_data, ep_out)

            # Merge tasks.jsonl
            src_tasks = src_meta_dir / "tasks.jsonl"
            if src_tasks.exists():
                base_tasks = self.read_jsonl(tasks_out) if tasks_out.exists() else []
                new_tasks = self.read_jsonl(src_tasks)
                existing_task_names = {r["task"]: r["task_index"] for r in base_tasks}
                next_idx = max(existing_task_names.values()) + 1 if existing_task_names else 0
                for r_new in new_tasks:
                    if r_new["task"] not in existing_task_names:
                        base_tasks.append({"task": r_new["task"], "task_index": next_idx})
                        existing_task_names[r_new["task"]] = next_idx
                        next_idx += 1
                self.write_jsonl(sorted(base_tasks, key=lambda x: x["task_index"]), tasks_out)

            # Merge info.json
            src_info = src_meta_dir / "info.json"
            if src_info.exists():
                d_base = json.loads(info_out.read_text()) if info_out.exists() else {}
                d_new = json.loads(src_info.read_text())
                merged_info = d_base.copy()
                for k in MERGE_NUM_KEYS:
                    merged_info[k] = merged_info.get(k, 0) + d_new.get(k, 0)
                for k, v in d_new.items():
                    if k not in MERGE_NUM_KEYS and k != "splits" or k == "splits" and not d_base:
                        merged_info[k] = v
                total_eps = merged_info.get("total_episodes", 0)
                merged_info.setdefault("splits", {})["train"] = f"0:{total_eps - 1 if total_eps > 0 else 0}"
                info_out.write_text(json.dumps(merged_info, indent=2))

            current_meta_episode_offset += eps_in_this_ds_for_meta

    def _copy_all_videos_for_merge(
        self,
        dataset_paths: List[Path],
        video_dst_chunk_root: Path,
        chunk_name: str,
        actual_episode_counts: List[int],
        verbose: bool,
    ):
        current_video_start_idx = 0
        for i, dataset_path in enumerate(dataset_paths):
            src_video_root = dataset_path / "videos" / chunk_name
            eps_in_this_ds = actual_episode_counts[i]
            if not src_video_root.exists():
                if verbose:
                    print(f"  Video source dir not found: {src_video_root}")
                current_video_start_idx += eps_in_this_ds
                continue

            cam_dirs = sorted(p for p in src_video_root.iterdir() if p.is_dir())
            if not cam_dirs:  # Videos directly under chunk root
                vids_in_chunk = self._natural_sort_paths(src_video_root.glob("episode_*.mp4"))
                for src_vid in vids_in_chunk:
                    dst_idx = self._extract_idx_from_name(src_vid.name) + current_video_start_idx
                    shutil.copy2(src_vid, video_dst_chunk_root / f"episode_{dst_idx:0{PAD}d}.mp4")
            else:  # Videos in camera subdirectories
                for cam_dir_path in cam_dirs:
                    dst_cam_path = video_dst_chunk_root / cam_dir_path.name
                    self.safe_mkdir(dst_cam_path)
                    vids = self._natural_sort_paths(cam_dir_path.glob("episode_*.mp4"))
                    for src_vid_path in vids:
                        dst_idx = self._extract_idx_from_name(src_vid_path.name) + current_video_start_idx
                        shutil.copy2(src_vid_path, dst_cam_path / f"episode_{dst_idx:0{PAD}d}.mp4")
            if verbose:
                print(f"  Copied videos from {dataset_path} with offset {current_video_start_idx}")
            current_video_start_idx += eps_in_this_ds

    # ─────────────────────────────────── DELETE Operation ─────────────────────────────────── #

    def delete_episode_from_dataset(
        self, ds_dir: Path, ep_id_to_delete: int, chunk_name: str, verbose: bool = False
    ):
        """
        Deletes a specific episode from a dataset and renumbers subsequent episodes.
        Modifies the dataset in-place.
        """
        ds_dir = ds_dir.resolve()
        if not ds_dir.is_dir():
            print(f"Error: Dataset directory not found: {ds_dir}")
            return

        if verbose:
            print(f"Starting delete operation for episode {ep_id_to_delete} in dataset: {ds_dir}")
            print(f"Target chunk: {chunk_name}")

        tgt_stem_to_delete = f"episode_{ep_id_to_delete:0{PAD}d}"
        frames_removed_count = 0
        videos_removed_count = 0  # This will count 1 if any video for the episode is deleted.

        # --- 1. Delete physical files of the target episode ---
        data_chunk_dir = ds_dir / "data" / chunk_name
        tgt_parquet = data_chunk_dir / f"{tgt_stem_to_delete}.parquet"
        if tgt_parquet.exists():
            if verbose:
                print(f"  Deleting Parquet: {tgt_parquet}")
            try:
                df = pd.read_parquet(tgt_parquet)
                frames_removed_count = len(df)
            except Exception as e:
                if verbose:
                    print(f"    Could not read parquet {tgt_parquet} to count frames: {e}")
            tgt_parquet.unlink()

        video_chunk_dir = ds_dir / "videos" / chunk_name
        deleted_any_video_for_ep = False
        if video_chunk_dir.exists():
            camera_subdirs = [d for d in video_chunk_dir.iterdir() if d.is_dir()]
            if camera_subdirs:
                for cam_subdir in camera_subdirs:
                    tgt_video_file = cam_subdir / f"{tgt_stem_to_delete}.mp4"
                    if tgt_video_file.exists():
                        if verbose:
                            print(f"  Deleting video: {tgt_video_file}")
                        tgt_video_file.unlink()
                        deleted_any_video_for_ep = True
            else:
                tgt_video_file = video_chunk_dir / f"{tgt_stem_to_delete}.mp4"
                if tgt_video_file.exists():
                    if verbose:
                        print(f"  Deleting video: {tgt_video_file}")
                    tgt_video_file.unlink()
                    deleted_any_video_for_ep = True
        if deleted_any_video_for_ep:
            videos_removed_count = 1  # Count as 1 episode's worth of videos removed

        images_root_dir = ds_dir / "images"
        if images_root_dir.exists():
            for cam_obs_dir in images_root_dir.iterdir():
                if cam_obs_dir.is_dir():
                    tgt_image_dir = cam_obs_dir / tgt_stem_to_delete
                    if tgt_image_dir.exists() and tgt_image_dir.is_dir():
                        if verbose:
                            print(f"  Deleting image directory: {tgt_image_dir}")
                        shutil.rmtree(tgt_image_dir)

        # --- 2. Shift higher episode files and directories, and patch their content ---
        if data_chunk_dir.exists():
            all_parquets = sorted(
                [
                    p
                    for p in data_chunk_dir.glob("episode_*.parquet")
                    if self._ep_id_from_stem(p.stem) is not None
                ],
                key=lambda p: self._ep_id_from_stem(p.stem),  # type: ignore
            )
            for p_file in all_parquets:
                ep_idx = self._ep_id_from_stem(p_file.stem)
                if ep_idx is not None and ep_idx > ep_id_to_delete:
                    new_idx = ep_idx - 1
                    new_stem = f"episode_{new_idx:0{PAD}d}"
                    new_file_path = p_file.with_name(new_stem + p_file.suffix)
                    if verbose:
                        print(f"  Renaming data {p_file.name} -> {new_file_path.name}")
                    shutil.move(str(p_file), new_file_path)
                    self._patch_parquet_for_delete(new_file_path, -1, verbose)

        if video_chunk_dir.exists():
            video_paths_to_process = []
            camera_subdirs_for_rename = [d for d in video_chunk_dir.iterdir() if d.is_dir()]  # Re-check
            if camera_subdirs_for_rename:
                for cam_subdir in camera_subdirs_for_rename:
                    video_paths_to_process.extend(list(cam_subdir.glob("episode_*.mp4")))
            else:
                video_paths_to_process.extend(list(video_chunk_dir.glob("episode_*.mp4")))

            sorted_video_paths = sorted(
                [p for p in video_paths_to_process if self._ep_id_from_stem(p.stem) is not None],
                key=lambda p: self._ep_id_from_stem(p.stem),  # type: ignore
            )
            for v_file in sorted_video_paths:
                ep_idx = self._ep_id_from_stem(v_file.stem)
                if ep_idx is not None and ep_idx > ep_id_to_delete:
                    new_idx = ep_idx - 1
                    new_stem = f"episode_{new_idx:0{PAD}d}"
                    new_file_path = v_file.with_name(new_stem + v_file.suffix)
                    if verbose:
                        print(f"  Renaming video {v_file.name} -> {new_file_path.name} in {v_file.parent}")
                    shutil.move(str(v_file), new_file_path)

        if images_root_dir.exists():
            for cam_obs_dir in images_root_dir.iterdir():
                if cam_obs_dir.is_dir():
                    image_dirs_to_process = sorted(
                        [
                            d
                            for d in cam_obs_dir.glob("episode_*")
                            if d.is_dir() and self._ep_id_from_stem(d.name) is not None
                        ],
                        key=lambda d: self._ep_id_from_stem(d.name),  # type: ignore
                    )
                    for img_dir in image_dirs_to_process:
                        ep_idx = self._ep_id_from_stem(img_dir.name)
                        if ep_idx is not None and ep_idx > ep_id_to_delete:
                            new_idx = ep_idx - 1
                            new_name = f"episode_{new_idx:0{PAD}d}"
                            new_dir_path = img_dir.with_name(new_name)
                            if verbose:
                                print(
                                    f"  Renaming image dir {img_dir.name} -> {new_dir_path.name} in {img_dir.parent}"
                                )
                            shutil.move(str(img_dir), new_dir_path)

        # --- 3. Update JSONL metadata files (meta/*.jsonl) ---
        meta_dir = ds_dir / "meta"
        for name in ["episodes_stats.jsonl", "episodes.jsonl", "episodes.json"]:
            path = meta_dir / name
            if path.exists():
                if verbose:
                    print(f"  Updating metadata file: {path.name}")
                self._rewrite_json_or_jsonl_for_delete(path, ep_id_to_delete, -1, verbose)

        # --- 4. Update meta/info.json counts ---
        info_path = meta_dir / "info.json"
        if info_path.exists():
            if verbose:
                print(f"  Updating global metadata: {info_path.name}")
            try:
                meta_info = json.loads(info_path.read_text())
                if isinstance(meta_info.get("total_episodes"), int):
                    meta_info["total_episodes"] = max(0, meta_info["total_episodes"] - 1)
                if isinstance(meta_info.get("total_frames"), int):
                    meta_info["total_frames"] = max(0, meta_info["total_frames"] - frames_removed_count)
                if videos_removed_count > 0 and isinstance(meta_info.get("total_videos"), int):
                    meta_info["total_videos"] = max(0, meta_info["total_videos"] - videos_removed_count)

                if isinstance(meta_info.get("splits"), dict) and isinstance(
                    meta_info["splits"].get("train"), str
                ):
                    start_str, _, end_str = meta_info["splits"]["train"].partition(":")
                    try:
                        start_idx = int(start_str)
                        new_end_idx = int(end_str) - 1
                        if meta_info["total_episodes"] == 0:
                            meta_info["splits"]["train"] = "0:0"  # Or some other indicator of empty
                        elif start_idx <= new_end_idx:
                            meta_info["splits"]["train"] = f"{start_idx}:{new_end_idx}"
                        else:
                            meta_info["splits"]["train"] = f"{start_idx}:{start_idx}"

                    except ValueError:
                        pass
                info_path.write_text(json.dumps(meta_info, indent=2))
            except Exception as e:
                print(f"    Error updating {info_path.name}: {e}")

        print(f"✅ Episode {ep_id_to_delete} deleted and dataset renumbered in {ds_dir}")

    def _patch_parquet_for_delete(self, path: Path, off: int, verbose: bool) -> int:
        """Patches 'episode_index' in a Parquet file by an offset."""
        try:
            df = pd.read_parquet(path)
            nrows = len(df)
            if "episode_index" in df.columns and pd.api.types.is_integer_dtype(df["episode_index"]):
                df["episode_index"] += off  # e.g., off = -1
                df.to_parquet(path, index=False)
            return nrows
        except Exception as e:
            if verbose:
                print(f"    Could not patch Parquet {path}: {e}")
            return 0

    def _rewrite_json_or_jsonl_for_delete(
        self, path: Path, ep_id_to_remove: int, offset_for_shifting: int, verbose: bool
    ):
        """
        Rewrites a JSON or JSONL file:
        - Removes entries matching ep_id_to_remove.
        - Shifts 'episode_index' (and other keys in DELETE_PATCH_KEYS) for entries with episode_index > ep_id_to_remove.
        """
        try:
            content = path.read_text().strip()
            if not content:  # File is empty
                if verbose:
                    print(f"    {path.name} is empty, skipping.")
                return
        except Exception as e:
            if verbose:
                print(f"    Could not read {path.name}: {e}")
            return

        is_json_list_format = content.startswith("[") and content.endswith("]")
        new_data_list_for_output = []  # Stores processed objects for JSON list or strings for JSONL

        original_data_list_parsed = []
        if is_json_list_format:
            try:
                original_data_list_parsed = json.loads(content)
                if not isinstance(original_data_list_parsed, list):  # Should be a list
                    if verbose:
                        print(
                            f"    Content of {path.name} looks like JSON list but is not a list. Processing as JSONL."
                        )
                    is_json_list_format = False  # Fallback
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"    Invalid JSON in {path.name}: {e}. Attempting line-by-line (JSONL).")
                is_json_list_format = False  # Fallback to JSONL

        items_to_process = original_data_list_parsed if is_json_list_format else content.splitlines()

        for item in items_to_process:
            obj_to_process = None
            is_decodable_json = False
            raw_line_if_not_json = item  # For JSONL, item is a line string

            if is_json_list_format:  # item is already a Python object (dict, list, etc.)
                obj_to_process = item
                is_decodable_json = True
            elif isinstance(item, str) and item.strip():  # For JSONL, item is a line string
                try:
                    obj_to_process = json.loads(item)
                    is_decodable_json = True
                except json.JSONDecodeError:
                    is_decodable_json = False  # Keep raw_line_if_not_json

            if is_decodable_json and isinstance(obj_to_process, dict):
                current_ep_idx = obj_to_process.get("episode_index")
                if current_ep_idx == ep_id_to_remove:
                    if verbose:
                        print(f"      Removing episode {ep_id_to_remove} entry from {path.name}")
                    continue  # Skip this episode
                if isinstance(current_ep_idx, int) and current_ep_idx > ep_id_to_remove:
                    obj_to_process = self._patch_indices_recursive_negative(
                        obj_to_process, offset_for_shifting
                    )

            # Add to output list
            if is_json_list_format:
                new_data_list_for_output.append(obj_to_process)
            else:  # JSONL
                if is_decodable_json:  # If it was JSON, serialize it back
                    new_data_list_for_output.append(json.dumps(obj_to_process, separators=(",", ":")))
                else:  # If it was not JSON (e.g. malformed line), keep original line
                    new_data_list_for_output.append(raw_line_if_not_json)

        # Write back to file
        if is_json_list_format:
            path.write_text(json.dumps(new_data_list_for_output, indent=2) + "\n")
        else:  # JSONL
            path.write_text("\n".join(new_data_list_for_output) + ("\n" if new_data_list_for_output else ""))

        if verbose:
            print(f"    {path.name} updated.")
