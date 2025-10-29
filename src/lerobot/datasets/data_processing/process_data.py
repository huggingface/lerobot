import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from moviepy import VideoFileClip

from lerobot.datasets.video_utils import concatenate_video_files, get_video_info


def parse_arguments():
    """Parse command line arguments for the video processing script."""
    parser = argparse.ArgumentParser(
        description="Process and merge clipped robot dataset videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        default="ywu67/record-test30",
        help="Source HuggingFace repository ID",
    )

    parser.add_argument(
        "--clip_second",
        type=float,
        default=10.0,
        help="Duration in seconds to clip from each episode",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="clipped_dataset",
        help="Directory to save clipped dataset files",
    )

    parser.add_argument(
        "--refined_dir",
        type=str,
        default="refined_dataset",
        help="Directory to save refined/merged dataset files",
    )

    parser.add_argument(
        "--new_repo_id",
        type=str,
        default="ywu67/record-test-new",
        help="New HuggingFace repository ID to save processed dataset",
    )

    parser.add_argument(
        "--process_videos",
        action="store_true",
        help="Enable video downloading and processing (default: False, only process parquet data)",
    )

    parser.add_argument(
        "--local_data_path",
        type=str,
        default=None,
        help="Path to local dataset directory containing data, meta, and video folders. If provided, will use local data instead of downloading from HuggingFace",
    )

    parser.add_argument(
        "--verify_clipping",
        action="store_true",
        help="Enable verification of clipping results after processing (default: False)",
    )

    parser.add_argument(
        "--delete_episodes",
        type=str,
        default=None,
        help="Comma-separated list of episode indices to delete (e.g., '0,2,5' or '1-3,7'). Use with --delete_episodes_only flag.",
    )

    parser.add_argument(
        "--delete_episodes_only",
        action="store_true",
        help="Only perform episode deletion without other processing. Requires --delete_episodes.",
    )

    parser.add_argument(
        "--clean_up_local",
        action="store_true",
        help="Clean up video files from local refined dataset after successful upload (default: False)",
    )

    return parser.parse_args()


def parse_episode_indices(episode_string):
    """
    Parse episode indices from a string format.

    Args:
        episode_string: String containing episode indices like "0,2,5" or "1-3,7,10-12"

    Returns:
        List of episode indices
    """
    if not episode_string:
        return []

    indices = []
    parts = episode_string.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Handle range like "1-3"
            try:
                start, end = map(int, part.split("-"))
                indices.extend(range(start, end + 1))
            except ValueError:
                raise ValueError(
                    f"Invalid episode range format: '{part}'. Expected format like '1-3'"
                )
        else:
            # Handle single episode index
            try:
                indices.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid episode index: '{part}'. Expected integer.")

    return sorted(list(set(indices)))  # Remove duplicates and sort


def load_local_data(local_data_path):
    """Load dataset and metadata from local files instead of HuggingFace."""
    local_path = Path(local_data_path)
    # Assert that the local data path exists
    assert local_path.exists(), f"Local data path does not exist: {local_data_path}"
    assert local_path.is_dir(), f"Local data path is not a directory: {local_data_path}"
    # Assert that required directories exist
    required_dirs = ["data", "meta"]
    for dir_name in required_dirs:
        dir_path = local_path / dir_name
        assert dir_path.exists(), f"Required directory not found: {dir_path}"
        assert dir_path.is_dir(), f"Required path is not a directory: {dir_path}"
    # Check for videos directory (optional warning since it might not be needed if process_videos=False)
    videos_dir = local_path / "videos"
    if not videos_dir.exists():
        print(f"Warning: Videos directory not found: {videos_dir} (will only affect video processing)")
    # Assert that metadata file exists
    meta_path = local_path / "meta" / "info.json"
    assert meta_path.exists(), f"Required metadata file not found: {meta_path}"
    assert meta_path.is_file(), f"Metadata path is not a file: {meta_path}"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    # Assert that data directory contains dataset files
    data_dir = local_path / "data"
    # Try to find parquet files first
    parquet_files = list(data_dir.rglob("*.parquet"))
    if parquet_files:
        # Load parquet files
        all_dataframes = []
        for parquet_file in sorted(parquet_files):
            assert parquet_file.exists(), f"Parquet file not found: {parquet_file}"
            df = pd.read_parquet(parquet_file)
            all_dataframes.append(df)
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Loaded {len(parquet_files)} parquet files with {len(combined_df)} total rows")
    else:
        # Try to load from episodes.jsonl (original format)
        episodes_file = local_path / "episodes.jsonl"
        if episodes_file.exists():
            assert (episodes_file.is_file()), f"Episodes path is not a file: {episodes_file}"
            episodes_data = []
            with open(episodes_file, "r") as f:
                for line in f:
                    episodes_data.append(json.loads(line))
            combined_df = pd.DataFrame(episodes_data)
            print(f"Loaded episodes.jsonl with {len(combined_df)} total rows")
        else:
            # Check for other possible dataset files
            task_file = local_path / "tasks.jsonl"
            episode_stats_file = local_path / "episode_stats.jsonl"
            available_files = []
            if task_file.exists():
                available_files.append("tasks.jsonl")
            if episode_stats_file.exists():
                available_files.append("episode_stats.jsonl")
            if available_files:
                print(f"Found other dataset files: {available_files}, but no main dataset file (episodes.jsonl or parquet files)")
            assert False, f"No dataset files found in {data_dir}. Expected parquet files or episodes.jsonl"

    # Assert that dataset is not empty
    assert len(combined_df) > 0, f"Dataset is empty. Loaded dataframe has 0 rows"    
    return meta, combined_df


def get_video_duration_ffprobe(video_path):
    """Get video duration using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        return None


def verify_clipping_results(args, target_clip_duration):
    """
    Verify that episodes have been properly clipped to the target duration.
    Randomly samples at most 10 episodes to check (or all if less than 10).
    
    Args:
        args: Command line arguments containing paths
        target_clip_duration: Expected duration per episode in seconds
    """
    refined_dir = Path(args.refined_dir)
    
    # Check if refined dataset exists
    dataset_file = refined_dir / "dataset" / "data-00000-of-00001.arrow"
    meta_file = refined_dir / "meta" / "info.json"
    
    if not dataset_file.exists():
        print(f"❌ Dataset file not found: {dataset_file}")
        return False
        
    if not meta_file.exists():
        print(f"❌ Metadata file not found: {meta_file}")
        return False
    
    try:
        # Load metadata
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        # Load dataset
        dataset = Dataset.from_file(str(dataset_file))
        df = dataset.to_pandas()
        
        print(f"✓ Loaded dataset with {len(df)} total frames")
        
        # Get all episode indices
        episode_indices = sorted(df['episode_index'].unique())
        total_episodes = len(episode_indices)

        print(f"✓ Found {total_episodes} episodes")

        # Sample episodes to check (max 10 or all if less than 10)
        max_episodes_to_check = min(10, total_episodes)
        if total_episodes <= 10:
            episodes_to_check = episode_indices
            print(f"✓ Checking all {total_episodes} episodes")
        else:
            episodes_to_check = sorted(random.sample(episode_indices, max_episodes_to_check))
            print(f"✓ Randomly sampling {max_episodes_to_check} episodes to check: {episodes_to_check}")
        
        # Verify episode durations in dataset
        fps = meta.get("fps", 30)
        all_passed = True
        
        for ep_idx in episodes_to_check:
            episode_data = df[df['episode_index'] == ep_idx]
            frame_count = len(episode_data)
            
            if len(episode_data) > 0:
                min_timestamp = episode_data['timestamp'].min()
                max_timestamp = episode_data['timestamp'].max()
                duration = max_timestamp - min_timestamp
                expected_frames = int(target_clip_duration * fps)
                
                # Check if duration is close to target (within 5% tolerance)
                duration_ok = abs(duration - target_clip_duration) <= (target_clip_duration * 0.05)
                frames_ok = abs(frame_count - expected_frames) <= (expected_frames * 0.05)
                
                status = "✅" if (duration_ok and frames_ok) else "❌"
                print(f"  Episode {ep_idx}: {frame_count} frames, {duration:.2f}s {status}")
                
                if not (duration_ok and frames_ok):
                    all_passed = False
                    print(f"    Expected: ~{expected_frames} frames, ~{target_clip_duration}s")
            else:
                print(f"  Episode {ep_idx}: No data found ❌")
                all_passed = False
        
        # Verify video files if they exist
        cameras = ["left", "top"]
        video_dir = refined_dir / "videos"
        
        if not video_dir.exists():
            # Check in clipped_dataset instead
            video_dir = Path(args.output_dir) / "videos"
            if video_dir.exists():
                print(f"✓ Checking videos in clipped dataset: {video_dir}")
            else:
                print("ℹ️ No video files found to verify")
                video_dir = None
        else:
            print(f"✓ Checking videos in refined dataset: {video_dir}")
        
        if video_dir and video_dir.exists():
            for camera in cameras:
                camera_dir = video_dir / camera / "chunk-000"
                if camera_dir.exists():
                    video_files = sorted(camera_dir.glob("file-*.mp4"))
                    print(f"\n  {camera.upper()} camera:")
                    
                    for video_file in video_files:
                        if video_file.name.replace('.mp4', '').split('-')[-1].isdigit():
                            # Extract episode number from filename
                            ep_num = int(video_file.name.replace('.mp4', '').split('-')[-1])
                            if ep_num in episodes_to_check:
                                duration = get_video_duration_ffprobe(video_file)
                                if duration is not None:
                                    duration_ok = abs(duration - target_clip_duration) <= (target_clip_duration * 0.05)
                                    status = "✅" if duration_ok else "❌"
                                    print(f"    Episode {ep_num}: {duration:.2f}s {status}")
                                    if not duration_ok:
                                        all_passed = False
                                else:
                                    print(f"    Episode {ep_num}: Could not read duration ❌")
                                    all_passed = False
                else:
                    print(f"  {camera.upper()} camera: Directory not found")
        
        # Summary
        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 VERIFICATION PASSED: All checked episodes are properly clipped!")
        else:
            print("⚠️  VERIFICATION FAILED: Some episodes may not be properly clipped!")
        
        # Print metadata info
        if "clipping_info" in meta:
            clipping_info = meta["clipping_info"]
            print(f"\nCLIPPING METADATA:")
            print(f"  Original duration: {clipping_info.get('original_duration_seconds', 'N/A'):.1f}s")
            print(f"  Target clip duration: {clipping_info.get('clip_duration_seconds', 'N/A')}s")
            print(f"  Frames per episode: {clipping_info.get('frames_per_episode_clipped', 'N/A')}")
            print(f"  Processing date: {clipping_info.get('processing_date', 'N/A')}")
        
        print("=" * 60)
        return all_passed
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False


def clip_split_hf_videos(args, df, fps):
    """Process and clip the dataset episodes."""
    total_frames = 0
    clipped_groups = []
    cameras = ["left", "top"]

    for ep_idx, group in df.groupby("episode_index"):
        clipped_group = group[group["timestamp"] < args.clip_second].copy()

        # Update frame_index and index (per file now)
        num_frames = len(clipped_group)
        clipped_group["frame_index"] = range(num_frames)
        clipped_group["index"] = range(num_frames)
        # Save individual Parquet for each episode
        parquet_path = os.path.join(
            args.output_dir, "data", "chunk-000", f"file-{ep_idx:03d}.parquet"
        )
        clipped_group.to_parquet(parquet_path)
        clipped_groups.append(clipped_group)
        total_frames += num_frames

    print(f"total frames = {total_frames}")

    # Only process videos if flag is enabled
    if args.process_videos:
        print("Processing videos...")
        for camera in cameras:
            if args.local_data_path:
                # Use local video files
                local_video_path = Path(args.local_data_path) / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-000.mp4"
                if not local_video_path.exists():
                    print(f"Warning: Local video not found: {local_video_path}, skipping {camera} camera")
                    continue
                local_path = str(local_video_path)
                print(f"Using local video: {local_path}")
            else:
                # Download the single original video (file-000.mp4)
                video_filename = f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
                local_path = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=video_filename,
                    repo_type="dataset"
                )
                print(f"Downloaded video from HF: {local_path}")
            
            full_video = VideoFileClip(local_path)

            for ep_idx, group in df.groupby("episode_index"):
                # Compute global start frame for the episode
                start_frame = int(group["index"].min())

                # Number of frames to clip (based on timestamp <=10)
                clipped_group = group[group["timestamp"] < args.clip_second]
                num_clipped_frames = len(clipped_group)

                # Compute start and end times in the original video
                start_time = start_frame / fps
                end_time = (start_frame + num_clipped_frames) / fps
                print(f"ep_idx={ep_idx}, start_time={start_time}, end_time={end_time}")

                clip = full_video.subclipped(start_time, end_time)

                # Save clipped video for this episode
                output_video_path = os.path.join(args.output_dir, f"videos/{camera}/chunk-000/file-{ep_idx:03d}.mp4")
                clip.write_videofile(output_video_path, codec="libx264", audio=False)
                print(f"Clipped and saved {camera} video for episode {ep_idx} to {output_video_path}")

            full_video.close()
    else:
        print("Video processing skipped (process_videos=False)")


def merge_clipped_videos_to_hub(args, meta, fps):
    """
    Merge all clipped videos into one large video while preserving metadata,
    then upload to new repository
    """
    print(f"Process videos flag is set to: {args.process_videos}")

    # Create refined dataset directory
    os.makedirs(os.path.join(args.refined_dir, "data", "chunk-000"), exist_ok=True)
    os.makedirs(
        os.path.join(args.refined_dir, "videos", "left", "chunk-000"), exist_ok=True
    )
    os.makedirs(
        os.path.join(args.refined_dir, "videos", "top", "chunk-000"), exist_ok=True
    )
    os.makedirs(os.path.join(args.refined_dir, "meta"), exist_ok=True)

    # Merge parquet data from all episodes
    all_clipped_data = []
    total_merged_frames = 0

    # Read all clipped parquet files and combine them
    clipped_data_dir = Path(args.output_dir) / "data" / "chunk-000"
    parquet_files = sorted(clipped_data_dir.glob("file-*.parquet"))

    for parquet_file in parquet_files:
        df_episode = pd.read_parquet(parquet_file)
        # Update frame_index and index to be continuous across all episodes
        df_episode["frame_index"] = range(total_merged_frames, total_merged_frames + len(df_episode))
        df_episode["index"] = range(total_merged_frames, total_merged_frames + len(df_episode))
        all_clipped_data.append(df_episode)
        total_merged_frames += len(df_episode)

    # Combine all episode data into single dataframe
    merged_df = pd.concat(all_clipped_data, ignore_index=True)

    # Save merged parquet data
    merged_parquet_path = os.path.join(args.refined_dir, "data", "chunk-000", "file-000.parquet")
    merged_df.to_parquet(merged_parquet_path)
    print(f"Saved merged parquet with {len(merged_df)} frames to {merged_parquet_path}")

    # Merge videos for each camera (only if videos were processed)
    if args.process_videos:
        cameras = ["left", "top"]
        for camera in cameras:
            # Get all clipped video files for this camera
            clipped_video_dir = Path(args.output_dir) / "videos" / camera / "chunk-000"

            # Check if directory exists first
            if not clipped_video_dir.exists():
                continue

            video_files = sorted(clipped_video_dir.glob("file-*.mp4"))

            if not video_files:
                continue

            print(f"Found {len(video_files)} video files for {camera}: {[f.name for f in video_files]}")

            # Convert to absolute paths for concatenate_video_files
            absolute_video_files = [video_file.resolve() for video_file in video_files]

            # Output path for merged video with -new postfix
            merged_video_path = Path(args.refined_dir) / f"videos/{camera}/chunk-000/file-000-new.mp4"
            merged_video_path = merged_video_path.resolve()

            print(f"merged_video_path={merged_video_path}")

            # Use the concatenate_video_files function from video_utils
            concatenate_video_files(absolute_video_files, merged_video_path, overwrite=True)
            print(f"Merged {len(absolute_video_files)} videos into {merged_video_path}")

    else:
        print("Video merging skipped (process_videos=False)")

    # Create updated metadata
    print("Creating updated metadata...")
    updated_meta = meta.copy()

    # Update frame counts in metadata
    updated_meta["total_frames"] = total_merged_frames
    updated_meta["num_episodes"] = len(
        parquet_files
    )  # Number of original episodes that were clipped

    # Calculate new duration
    merged_duration_seconds = total_merged_frames / fps
    updated_meta["total_seconds"] = merged_duration_seconds

    # Add information about the clipping process
    duration_seconds = meta["total_frames"] / fps
    updated_meta["clipping_info"] = {
        "original_total_frames": meta["total_frames"],
        "original_duration_seconds": duration_seconds,
        "clip_duration_seconds": args.clip_second,
        "frames_per_episode_clipped": args.clip_second * fps,
        "processing_date": pd.Timestamp.now().isoformat(),
    }
    return merged_df, updated_meta, merged_parquet_path


def upload_to_hf(args, merged_df, updated_meta, merged_parquet_path):
    """Upload the processed dataset to HuggingFace Hub."""
    # Save updated metadata
    updated_meta_path = os.path.join(args.refined_dir, "meta", "info.json")
    with open(updated_meta_path, "w") as f:
        json.dump(updated_meta, f, indent=2)
    print(f"Saved updated metadata to {updated_meta_path}")
    dataset = Dataset.from_pandas(merged_df)

    # Save dataset to disk in HuggingFace format
    dataset_path = os.path.join(args.refined_dir, "dataset")
    dataset.save_to_disk(dataset_path)

    # Upload to HuggingFace Hub
    print("Uploading to HuggingFace Hub...")
    api = HfApi()
    cameras = ["left", "top"]

    try:
        # Create new repository for the merged dataset
        repo_id = args.new_repo_id

        # Create the repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False
            )
            print(f"Created/verified repository: {repo_id}")
        except Exception as e:
            print(f"Repository creation note: {e}")

        # Upload metadata
        api.upload_file(
            path_or_fileobj=updated_meta_path,
            path_in_repo="meta/info.json",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Upload merged parquet data
        api.upload_file(
            path_or_fileobj=merged_parquet_path,
            path_in_repo="data/chunk-000/file-000.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Upload merged videos (only if they exist)
        if args.process_videos:
            for camera in cameras:
                merged_video_path = (
                    Path(args.refined_dir)
                    / f"videos/{camera}/chunk-000/file-000-new.mp4"
                )
                if merged_video_path.exists():
                    api.upload_file(
                        path_or_fileobj=str(merged_video_path),
                        path_in_repo=f"videos/observation.images.{camera}/chunk-000/file-000-new.mp4",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                    print(f"Uploaded merged {camera} video")
        else:
            print("Video upload skipped (process_videos=False)")

        # Upload dataset files
        api.upload_folder(
            folder_path=dataset_path,
            path_in_repo="dataset",
            repo_id=repo_id,
            repo_type="dataset",
        )

        print(f"Successfully uploaded refined dataset to {repo_id}")

        # Clean up video files after successful upload (based on clean_up_local flag)
        if args.process_videos and args.clean_up_local:
            print("Cleaning up video files from local refined_dataset...")
            videos_dir = Path(args.refined_dir) / "videos"
            if videos_dir.exists():
                shutil.rmtree(videos_dir)
                print("✓ Removed all video files from refined_dataset/videos")
            else:
                print("No videos directory found to clean up")
        elif args.process_videos and not args.clean_up_local:
            print("Video files kept in local refined_dataset (clean_up_local=False)")
        else:
            print("No video cleanup needed (process_videos=False)")

    except Exception as e:
        print(f"Error uploading to HuggingFace Hub: {e}")
        print("Dataset files are saved locally in 'refined_dataset' directory")
        if args.clean_up_local:
            print("Video files NOT removed due to upload error")
        else:
            print("Video files kept locally (clean_up_local=False)")

    return args.refined_dir


def delete_episodes_from_dataset(args, episode_indices_to_delete):
    """
    Delete specified episodes from the dataset including videos, data, and metadata.

    Args:
        args: Command line arguments containing paths and settings
        episode_indices_to_delete: List of episode indices to delete

    Returns:
        dict: Summary of deletion results
    """
    if not episode_indices_to_delete:
        print("No episodes specified for deletion.")
        return {"deleted_episodes": [], "status": "no_episodes"}

    episode_indices_to_delete = sorted(
        set(episode_indices_to_delete)
    )  # Remove duplicates and sort
    print(f"Episodes to delete: {episode_indices_to_delete}")

    # Load dataset information
    if args.local_data_path:
        print("Loading data from local files...")
        meta, df = load_local_data(args.local_data_path)
        data_path = Path(args.local_data_path)
    else:
        print("Loading data from HuggingFace...")
        # Download metadata from HuggingFace
        meta_src = hf_hub_download(
            repo_id=args.repo_id, filename="meta/info.json", repo_type="dataset"
        )
        with open(meta_src, "r") as f:
            meta = json.load(f)

        # Load the dataset from HuggingFace
        ds = load_dataset(args.repo_id)
        df = pd.DataFrame(ds["train"])
        data_path = Path(args.output_dir)  # Use output directory for processing

    # Get available episode indices
    available_episodes = sorted(df["episode_index"].unique())
    print(f"Available episodes: {available_episodes}")

    # Validate episode indices to delete
    invalid_episodes = [
        ep for ep in episode_indices_to_delete if ep not in available_episodes
    ]
    if invalid_episodes:
        print(
            f"Warning: Episodes {invalid_episodes} not found in dataset. Skipping them."
        )
        episode_indices_to_delete = [
            ep for ep in episode_indices_to_delete if ep in available_episodes
        ]

    if not episode_indices_to_delete:
        print("No valid episodes found for deletion.")
        return {"deleted_episodes": [], "status": "no_valid_episodes"}

    print(f"Valid episodes to delete: {episode_indices_to_delete}")

    deletion_summary = {
        "deleted_episodes": episode_indices_to_delete,
        "deleted_data_files": [],
        "deleted_video_files": [],
        "original_episodes": len(available_episodes),
        "remaining_episodes": len(available_episodes) - len(episode_indices_to_delete),
        "status": "success",
    }

    try:
        # 1. Backup and delete data files
        print("\n1. Processing data files...")
        remaining_df_list = []

        for ep_idx in available_episodes:
            if ep_idx in episode_indices_to_delete:
                # Delete episode data
                episode_data = df[df["episode_index"] == ep_idx]
                if len(episode_data) > 0:
                    # Backup data before deletion
                    deletion_summary["deleted_data_files"].append(
                        f"episode_{ep_idx:03d}"
                    )
                else:
                    print(f"  Episode {ep_idx}: No data found")
            else:
                # Keep episode data
                episode_data = df[df["episode_index"] == ep_idx].copy()
                remaining_df_list.append(episode_data)

        # 2. Delete video files
        print("\n2. Processing video files...")
        cameras = ["left", "top"]

        for camera in cameras:
            camera_video_dir = data_path / "videos" / camera / "chunk-000"
            if camera_video_dir.exists():
                for ep_idx in episode_indices_to_delete:
                    video_file = camera_video_dir / f"file-{ep_idx:03d}.mp4"
                    if video_file.exists():
                        # Delete video file
                        video_file.unlink()
                        print(f"  Deleted {camera} video for episode {ep_idx}")
                        deletion_summary["deleted_video_files"].append(
                            f"episode_{ep_idx:03d}_{camera}.mp4"
                        )
                    else:
                        print(f"  Episode {ep_idx}: {camera} video not found")
            else:
                print(f"  {camera} camera directory not found: {camera_video_dir}")

        # 3. Reindex remaining episodes to be continuous
        print("\n3. Reindexing remaining episodes...")
        if remaining_df_list:
            # Combine remaining data
            combined_df = pd.concat(remaining_df_list, ignore_index=True)

            # Create mapping from old episode indices to new ones
            old_episode_indices = sorted(combined_df["episode_index"].unique())
            episode_mapping = {
                old_idx: new_idx for new_idx, old_idx in enumerate(old_episode_indices)
            }
            print(f"  Episode mapping: {episode_mapping}")

            # Update episode indices in the dataframe
            combined_df["episode_index"] = combined_df["episode_index"].map(
                episode_mapping
            )

            # Update frame indices to be continuous
            combined_df["frame_index"] = range(len(combined_df))
            combined_df["index"] = range(len(combined_df))

            # Save updated data
            updated_data_dir = data_path / "data" / "chunk-000"
            updated_data_dir.mkdir(parents=True, exist_ok=True)

            # Save as single file or split by episode
            if args.refined_dir:
                # Save to refined directory as single merged file
                refined_data_dir = Path(args.refined_dir) / "data" / "chunk-000"
                refined_data_dir.mkdir(parents=True, exist_ok=True)
                merged_file = refined_data_dir / "file-000.parquet"
                combined_df.to_parquet(merged_file)
                print(f"  Saved merged data to: {merged_file}")
            else:
                # Save split by new episode indices
                for new_ep_idx in sorted(combined_df["episode_index"].unique()):
                    episode_data = combined_df[
                        combined_df["episode_index"] == new_ep_idx
                    ]
                    # Reset frame indices within episode
                    episode_data = episode_data.copy()
                    episode_data["frame_index"] = range(len(episode_data))

                    output_file = updated_data_dir / f"file-{new_ep_idx:03d}.parquet"
                    episode_data.to_parquet(output_file)
                    print(f"  Saved episode {new_ep_idx} data to: {output_file}")

        # 4. Rename video files to match new episode indices
        print("\n4. Renaming video files...")
        if remaining_df_list:
            old_episode_indices = sorted(
                [ep for ep in available_episodes if ep not in episode_indices_to_delete]
            )

            for camera in cameras:
                camera_video_dir = data_path / "videos" / camera / "chunk-000"
                if camera_video_dir.exists():
                    # Create temporary directory for renaming
                    temp_video_dir = camera_video_dir / "temp_rename"
                    temp_video_dir.mkdir(exist_ok=True)

                    # Move files to temporary directory with new names
                    for new_idx, old_idx in enumerate(old_episode_indices):
                        old_video_file = camera_video_dir / f"file-{old_idx:03d}.mp4"
                        if old_video_file.exists():
                            temp_video_file = temp_video_dir / f"file-{new_idx:03d}.mp4"
                            shutil.move(old_video_file, temp_video_file)
                            print(
                                f"  Renamed {camera} video: episode {old_idx} -> {new_idx}"
                            )

                    # Move files back to original directory
                    for video_file in temp_video_dir.glob("*.mp4"):
                        shutil.move(video_file, camera_video_dir / video_file.name)

                    # Remove temporary directory
                    temp_video_dir.rmdir()

        # 5. Update metadata
        print("\n5. Updating metadata...")
        original_total_frames = meta.get("total_frames", 0)
        original_total_episodes = meta.get("total_episodes", 0)

        # Calculate new totals
        if remaining_df_list:
            new_total_frames = len(combined_df)
            new_total_episodes = len(combined_df["episode_index"].unique())
        else:
            new_total_frames = 0
            new_total_episodes = 0

        deleted_frames = original_total_frames - new_total_frames

        # Update metadata
        meta["total_frames"] = new_total_frames
        meta["total_episodes"] = new_total_episodes

        if "splits" in meta:
            meta["splits"] = {"train": f"0:{new_total_episodes}"}

        # Add deletion information to metadata
        meta["deletion_info"] = {
            "deleted_episodes": episode_indices_to_delete,
            "original_total_episodes": original_total_episodes,
            "original_total_frames": original_total_frames,
            "deleted_frames": deleted_frames,
            "deletion_date": pd.Timestamp.now().isoformat(),
        }

        # Calculate new duration
        fps = meta.get("fps", 30)
        new_duration_seconds = new_total_frames / fps
        meta["total_seconds"] = new_duration_seconds

        # Save updated metadata
        meta_dir = (Path(args.refined_dir) if args.refined_dir else data_path) / "meta"
        meta_dir.mkdir(exist_ok=True)
        updated_meta_path = meta_dir / "info.json"

        with open(updated_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Updated metadata saved to: {updated_meta_path}")
        print(f"  Original episodes: {original_total_episodes}")
        print(f"  Remaining episodes: {new_total_episodes}")
        print(f"  Original frames: {original_total_frames}")
        print(f"  Remaining frames: {new_total_frames}")
        print(f"  Deleted frames: {deleted_frames}")

        deletion_summary.update(
            {
                "original_frames": original_total_frames,
                "remaining_frames": new_total_frames,
                "deleted_frames": deleted_frames,
            }
        )

    except Exception as e:
        print(f"❌ Error during episode deletion: {e}")
        print("Check backup directory for any backed up data.")
        deletion_summary["status"] = "error"
        deletion_summary["error"] = str(e)
        return deletion_summary

    # Summary
    print("DELETION COMPLETE")
    print(f"✅ Successfully deleted {len(episode_indices_to_delete)} episodes")
    print(
        f"📊 Updated dataset: {new_total_episodes} episodes, {new_total_frames} frames"
    )
    return deletion_summary


def main():
    """Main function to process the dataset with command line arguments."""
    args = parse_arguments()
    
    # Track whether episode deletion was performed
    episodes_deleted = False

    # Handle episode deletion
    if args.delete_episodes or args.delete_episodes_only:
        if not args.delete_episodes:
            print(
                "Error: --delete_episodes_only requires --delete_episodes to specify which episodes to delete"
            )
            return

        try:
            episode_indices_to_delete = parse_episode_indices(args.delete_episodes)
        except ValueError as e:
            print(f"Error parsing episode indices: {e}")
            return

        print(f"Episodes to delete: {episode_indices_to_delete}")

        # Perform episode deletion
        deletion_result = delete_episodes_from_dataset(args, episode_indices_to_delete)

        if deletion_result["status"] == "success":
            print(f"\n✅ Episode deletion completed successfully!")
            episodes_deleted = True
        else:
            print(
                f"\n❌ Episode deletion failed: {deletion_result.get('error', 'Unknown error')}"
            )

        # If delete_episodes_only is set, exit after deletion
        if args.delete_episodes_only:
            return

        print("\nContinuing with regular dataset processing...")

    print(f"  Source repository: {args.repo_id}")
    print(f"  Clip duration: {args.clip_second} seconds")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Refined directory: {args.refined_dir}")
    print(f"  New repository: {args.new_repo_id}")
    print(f"  Process videos: {args.process_videos}")
    print(f"  Clean up local videos: {args.clean_up_local}")
    if args.local_data_path:
        print(f"  Using local data from: {args.local_data_path}")

    # Load dataset based on whether episodes were deleted
    if episodes_deleted:
        # Load the updated dataset from refined_dir after episode deletion
        print("Loading updated dataset after episode deletion...")
        refined_meta_path = Path(args.refined_dir) / "meta" / "info.json"
        refined_data_path = Path(args.refined_dir) / "data" / "chunk-000" / "file-000.parquet"
        
        if refined_meta_path.exists() and refined_data_path.exists():
            with open(refined_meta_path, "r") as f:
                meta = json.load(f)
            df = pd.read_parquet(refined_data_path)
            print(f"Loaded updated dataset: {len(df)} frames, {len(df['episode_index'].unique())} episodes")
        else:
            print(f"Error: Could not find updated dataset files after deletion")
            return
    elif args.local_data_path:
        # Load from local data
        meta, df = load_local_data(args.local_data_path)
    else:
        # Download and load metadata from HuggingFace
        meta_src = hf_hub_download(
            repo_id=args.repo_id, filename="meta/info.json", repo_type="dataset"
        )
        with open(meta_src, "r") as f:
            meta = json.load(f)

        # Load the dataset from HuggingFace
        ds = load_dataset(args.repo_id)
        df = pd.DataFrame(ds["train"])  # Convert train split to DataFrame

    # FPS from metadata
    fps = meta["fps"]
    print(f"Original FPS from metadata: {fps}")

    # Total frames from metadata
    total_frames = meta["total_frames"]
    print(f"Original total frames from metadata: {total_frames}")

    duration_seconds = total_frames / fps
    print(f"Total duration: {duration_seconds:.1f} seconds")
    num_episodes = df["episode_index"].nunique()
    print(f"Number of episodes from Parquet: {num_episodes}")
    print(f"final output video should be {args.clip_second * num_episodes}")
    print(f"final num frame should be {int(args.clip_second * num_episodes / duration_seconds * total_frames)}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data", "chunk-000"), exist_ok=True)
    os.makedirs( os.path.join(args.output_dir, "videos", "left", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "videos", "top", "chunk-000"), exist_ok=True)

    # Create refined directory
    os.makedirs(args.refined_dir, exist_ok=True)

    # Process the clipped data first
    clip_split_hf_videos(args, df, fps)

    # Then merge the results
    merged_df, updated_meta, merged_parquet_path = merge_clipped_videos_to_hub(args, meta, fps)

    # Upload to HuggingFace
    upload_to_hf(args, merged_df, updated_meta, merged_parquet_path)

    # Verify clipping results if requested
    if args.verify_clipping:
        print(f"\nRunning clipping verification...")
        verification_passed = verify_clipping_results(args, args.clip_second)
        if verification_passed:
            print("✅ Clipping verification completed successfully!")
        else:
            print("⚠️ Clipping verification found issues. Please check the output above.")


if __name__ == "__main__":
    main()
