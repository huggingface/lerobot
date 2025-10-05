import argparse
import json
import os
import shutil
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--repo_id", 
        type=str, 
        default="ywu67/record-test30",
        help="Source HuggingFace repository ID"
    )

    parser.add_argument(
        "--clip_second", 
        type=float, 
        default=10.0,
        help="Duration in seconds to clip from each episode"
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="clipped_dataset",
        help="Directory to save clipped dataset files"
    )

    parser.add_argument(
        "--refined_dir", 
        type=str, 
        default="refined_dataset",
        help="Directory to save refined/merged dataset files"
    )

    parser.add_argument(
        "--new_repo_id", 
        type=str, 
        default="ywu67/record-test-new",
        help="New HuggingFace repository ID to save processed dataset"
    )

    parser.add_argument(
        "--process_videos", 
        action="store_true",
        help="Enable video downloading and processing (default: False, only process parquet data)"
    )

    return parser.parse_args()


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
        parquet_path = os.path.join(args.output_dir, "data", "chunk-000", f"file-{ep_idx:03d}.parquet")
        clipped_group.to_parquet(parquet_path)
        clipped_groups.append(clipped_group)
        total_frames += num_frames

    print(f"total frames = {total_frames}")

    # Only process videos if flag is enabled
    if args.process_videos:
        print("Processing videos...")
        for camera in cameras:
            # Download the single original video (file-000.mp4)
            video_filename = f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
            local_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=video_filename,
                repo_type="dataset"
            )
            full_video = VideoFileClip(local_path)

            for ep_idx, group in df.groupby("episode_index"):
                # Compute global start frame for the episode
                start_frame = int(group["index"].min())

                # Number of frames to clip (based on timestamp <=10)
                clipped_group = group[group["timestamp"] < args.clip_second]
                num_clipped_frames = len(clipped_group)
                print(num_clipped_frames)

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
    print("Starting video merging process...")
    print(f"Process videos flag is set to: {args.process_videos}")

    # Create refined dataset directory
    os.makedirs(os.path.join(args.refined_dir, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(args.refined_dir, "videos", "left", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(args.refined_dir, "videos", "top", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(args.refined_dir, "meta"), exist_ok=True)

    # Merge parquet data from all episodes
    print("Merging parquet data...")
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
            print(f"Merging {camera} camera videos...")

            # Get all clipped video files for this camera
            clipped_video_dir = Path(args.output_dir) / "videos" / camera / "chunk-000"

            # Check if directory exists first
            if not clipped_video_dir.exists():
                print(f"Video directory does not exist for camera {camera}: {clipped_video_dir}")
                continue

            video_files = sorted(clipped_video_dir.glob("file-*.mp4"))

            if not video_files:
                print(f"No video files found for camera {camera} in {clipped_video_dir}")
                continue

            print(f"Found {len(video_files)} video files for {camera}: {[f.name for f in video_files]}")

            # Convert to absolute paths for concatenate_video_files
            absolute_video_files = [video_file.resolve() for video_file in video_files]

            # Output path for merged video with -new postfix
            merged_video_path = Path(args.refined_dir) / f"videos/{camera}/chunk-000/file-000-new.mp4"
            merged_video_path = merged_video_path.resolve()

            print(f"merged_video_path={merged_video_path}")
            print(f"absolute_video_files={absolute_video_files}")

            # Use the concatenate_video_files function from video_utils
            concatenate_video_files(absolute_video_files, merged_video_path, overwrite=True)
            print(f"Merged {len(absolute_video_files)} videos into {merged_video_path}")

            # Verify merged video info
            video_info = get_video_info(merged_video_path)
            print(f"Merged {camera} video info: {video_info.get('video.fps')} fps, "
                  f"{video_info.get('video.width')}x{video_info.get('video.height')}")
    else:
        print("Video merging skipped (process_videos=False)")

    # Create updated metadata
    print("Creating updated metadata...")
    updated_meta = meta.copy()

    # Update frame counts in metadata
    updated_meta["total_frames"] = total_merged_frames
    updated_meta["num_episodes"] = len(parquet_files)  # Number of original episodes that were clipped

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
        "processing_date": pd.Timestamp.now().isoformat()
    }
    return merged_df, updated_meta, merged_parquet_path


def upload_to_hf(args, merged_df, updated_meta, merged_parquet_path):
    """Upload the processed dataset to HuggingFace Hub."""
    # Save updated metadata
    updated_meta_path = os.path.join(args.refined_dir, "meta", "info.json")
    with open(updated_meta_path, "w") as f:
        json.dump(updated_meta, f, indent=2)
    print(f"Saved updated metadata to {updated_meta_path}")

    # Create dataset info for HuggingFace
    print("Preparing dataset for upload...")

    # Create the HuggingFace dataset from merged parquet
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
                repo_id=repo_id,
                repo_type="dataset",
                exist_ok=True,
                private=False
            )
            print(f"Created/verified repository: {repo_id}")
        except Exception as e:
            print(f"Repository creation note: {e}")

        # Upload metadata
        api.upload_file(
            path_or_fileobj=updated_meta_path,
            path_in_repo="meta/info.json",
            repo_id=repo_id,
            repo_type="dataset"
        )

        # Upload merged parquet data
        api.upload_file(
            path_or_fileobj=merged_parquet_path,
            path_in_repo="data/chunk-000/file-000.parquet",
            repo_id=repo_id,
            repo_type="dataset"
        )

        # Upload merged videos (only if they exist)
        if args.process_videos:
            for camera in cameras:
                merged_video_path = Path(args.refined_dir) / f"videos/{camera}/chunk-000/file-000-new.mp4"
                if merged_video_path.exists():
                    api.upload_file(
                        path_or_fileobj=str(merged_video_path),
                        path_in_repo=f"videos/observation.images.{camera}/chunk-000/file-000-new.mp4",
                        repo_id=repo_id,
                        repo_type="dataset"
                    )
                    print(f"Uploaded merged {camera} video")
        else:
            print("Video upload skipped (process_videos=False)")

        # Upload dataset files
        api.upload_folder(
            folder_path=dataset_path,
            path_in_repo="dataset",
            repo_id=repo_id,
            repo_type="dataset"
        )

        print(f"Successfully uploaded refined dataset to {repo_id}")

        # Clean up video files after successful upload
        if args.process_videos:
            print("Cleaning up video files from local refined_dataset...")
            videos_dir = Path(args.refined_dir) / "videos"
            if videos_dir.exists():
                shutil.rmtree(videos_dir)
                print("✓ Removed all video files from refined_dataset/videos")
            else:
                print("No videos directory found to clean up")
        else:
            print("No video cleanup needed (process_videos=False)")

    except Exception as e:
        print(f"Error uploading to HuggingFace Hub: {e}")
        print("Dataset files are saved locally in 'refined_dataset' directory")
        print("Video files NOT removed due to upload error")

    return args.refined_dir


def main():
    """Main function to process the dataset with command line arguments."""
    args = parse_arguments()
    print(f"Processing dataset with the following settings:")
    print(f"  Source repository: {args.repo_id}")
    print(f"  Clip duration: {args.clip_second} seconds")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Refined directory: {args.refined_dir}")
    print(f"  New repository: {args.new_repo_id}")
    print(f"  Process videos: {args.process_videos}")

    # Download and load metadata
    meta_src = hf_hub_download(
        repo_id=args.repo_id,
        filename="meta/info.json",
        repo_type="dataset"
    )
    with open(meta_src, "r") as f:
        meta = json.load(f)

    # FPS from metadata
    fps = meta["fps"]
    print(f"Original FPS from metadata: {fps}")

    # Total frames from metadata
    total_frames = meta["total_frames"]
    print(f"Original total frames from metadata: {total_frames}")

    duration_seconds = total_frames / fps
    print(f"Total duration: {duration_seconds:.1f} seconds")

    # Load the dataset and clip the Parquet data
    ds = load_dataset(args.repo_id)
    df = pd.DataFrame(ds["train"])  # Convert train split to DataFrame
    num_episodes = df["episode_index"].nunique()
    print(f"Number of episodes from Parquet: {num_episodes}")
    print(f"final output video should be {args.clip_second * num_episodes}")
    print(f"final num frame should be {int(args.clip_second * num_episodes / duration_seconds * total_frames)}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "videos", "left", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "videos", "top", "chunk-000"), exist_ok=True)

    # Create refined directory
    os.makedirs(args.refined_dir, exist_ok=True)

    print("All keys (columns) in the groups:")
    # each episode index: ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']

    # Process the clipped data first
    clip_split_hf_videos(args, df, fps)

    # Then merge the results
    merged_df, updated_meta, merged_parquet_path = merge_clipped_videos_to_hub(args, meta, fps)

    # Upload to HuggingFace
    upload_to_hf(args, merged_df, updated_meta, merged_parquet_path)


if __name__ == "__main__":
    main()
