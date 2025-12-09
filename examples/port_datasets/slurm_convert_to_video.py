#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parallelize video conversion using SLURM and Datatrove.

This script converts an image-based LeRobot dataset to video format by distributing 
episodes across multiple workers for parallel processing.

Example usage:
    python slurm_convert_to_video.py \
        --repo-id lerobot \
        --root /fsx/jade_choghari/libero/ \
        --output-dir /fsx/jade_choghari/libero_video \
        --output-repo-id lerobot_video \
        --workers 100 \
        --partition cpu_partition \
        --cpus-per-task 8 \
        --logs-dir ./logs
"""

import argparse
import logging
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep


class ConvertEpisodesToVideo(PipelineStep):
    """Pipeline step that converts episodes to videos in parallel."""

    def __init__(
        self,
        repo_id: str,
        root: str | None,
        output_dir: str,
        output_repo_id: str | None,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        g: int = 2,
        crf: int = 30,
        fast_decode: int = 0,
        num_image_workers: int = 4,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = root
        self.output_dir = Path(output_dir)
        self.output_repo_id = output_repo_id or f"{repo_id}_video"
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.g = g
        self.crf = crf
        self.fast_decode = fast_decode
        self.num_image_workers = num_image_workers

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        """Process a shard of episodes."""
        from datasets.utils.tqdm import disable_progress_bars

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.scripts.lerobot_edit_dataset import (
            encode_episode_videos,
        )
        from lerobot.utils.utils import init_logging
        import logging
        init_logging()
        disable_progress_bars()

        logging.info(f"Worker {rank}/{world_size} starting video conversion")

        # Load source dataset
        dataset = LeRobotDataset(self.repo_id, root=self.root)
        total_episodes = dataset.meta.total_episodes

        # Determine which episodes this worker processes
        episodes_per_worker = total_episodes // world_size
        remainder = total_episodes % world_size

        if rank < remainder:
            # First 'remainder' workers get one extra episode
            start_ep = rank * (episodes_per_worker + 1)
            end_ep = start_ep + episodes_per_worker + 1
        else:
            start_ep = remainder * (episodes_per_worker + 1) + (rank - remainder) * episodes_per_worker
            end_ep = start_ep + episodes_per_worker

        episode_indices = list(range(start_ep, end_ep))

        logging.info(
            f"Worker {rank} processing episodes {start_ep} to {end_ep-1} ({len(episode_indices)} episodes)"
        )

        if len(episode_indices) == 0:
            logging.info(f"Worker {rank} has no episodes to process")
            return

        # Create shard-specific output directory
        import shutil
        shard_output_dir = self.output_dir / f"shard_{rank:04d}"
        
        # Remove existing directory to avoid conflicts with LeRobotDatasetMetadata.create
        if shard_output_dir.exists():
            logging.warning(f"Shard directory {shard_output_dir} already exists, removing it")
            shutil.rmtree(shard_output_dir)

        # Import conversion function
        from lerobot.scripts.lerobot_edit_dataset import convert_dataset_to_videos

        logging.info(
            f"Worker {rank} converting {len(episode_indices)} episodes with codec {self.vcodec}, CRF {self.crf}"
        )

        # Convert this shard's episodes with remapped indices
        # We need to remap episode indices to start from 0 for proper file structure
        self._convert_shard_to_videos(
            dataset=dataset,
            shard_output_dir=shard_output_dir,
            shard_repo_id=f"{self.output_repo_id}_shard_{rank:04d}",
            episode_indices=episode_indices,
        )

        logging.info(f"Worker {rank} completed successfully")

    def _convert_shard_to_videos(
        self,
        dataset,
        shard_output_dir,
        shard_repo_id,
        episode_indices,
    ):
        """Convert a shard's episodes to videos with proper index remapping."""
        import shutil
        import pandas as pd
        from tqdm import tqdm

        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.datasets.utils import write_info, write_stats, write_tasks
        from lerobot.datasets.video_utils import encode_video_frames, get_video_info
        from lerobot.scripts.lerobot_edit_dataset import (
            save_episode_images_for_video,
            _copy_data_without_images,
        )
        from lerobot.utils.constants import OBS_IMAGE

        # Check that it's an image dataset
        if len(dataset.meta.video_keys) > 0:
            raise ValueError(
                f"This operation is for image datasets only. Video dataset provided: {dataset.repo_id}"
            )

        # Get all image keys
        hf_dataset = dataset.hf_dataset.with_format(None)
        img_keys = [key for key in hf_dataset.features if key.startswith(OBS_IMAGE)]

        if len(img_keys) == 0:
            raise ValueError(f"No image keys found in dataset {dataset.repo_id}")

        logging.info(f"Converting {len(episode_indices)} episodes with {len(img_keys)} cameras")

        # Create new features dict, converting image features to video features
        new_features = {}
        for key, value in dataset.meta.features.items():
            if key not in img_keys:
                new_features[key] = value
            else:
                # Convert image key to video format
                new_features[key] = value.copy()
                new_features[key]["dtype"] = "video"

        # Create new metadata for video dataset
        new_meta = LeRobotDatasetMetadata.create(
            repo_id=shard_repo_id,
            fps=dataset.meta.fps,
            features=new_features,
            robot_type=dataset.meta.robot_type,
            root=shard_output_dir,
            use_videos=True,
            chunks_size=dataset.meta.chunks_size,
            data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
            video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
        )

        # Create temporary directory for image extraction
        temp_dir = shard_output_dir / "temp_images"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Process each episode with REMAPPED indices (0, 1, 2, ...)
        all_episode_metadata = []
        fps = int(dataset.fps)

        try:
            for new_ep_idx, orig_ep_idx in enumerate(tqdm(episode_indices, desc="Converting episodes")):
                # Get episode metadata from source using ORIGINAL index
                src_episode = dataset.meta.episodes[orig_ep_idx]
                episode_length = src_episode["length"]
                episode_duration = episode_length / dataset.fps

                video_metadata = {}

                # Encode videos for each camera
                for img_key in img_keys:
                    # Save images temporarily using ORIGINAL episode index
                    imgs_dir = temp_dir / f"episode_{orig_ep_idx:06d}" / img_key
                    save_episode_images_for_video(
                        dataset, imgs_dir, img_key, orig_ep_idx, self.num_image_workers
                    )

                    # Determine chunk and file indices using NEW (remapped) episode index
                    chunk_idx = new_ep_idx // new_meta.chunks_size
                    file_idx = new_ep_idx % new_meta.chunks_size

                    # Create video path in the new dataset structure
                    video_path = new_meta.root / new_meta.video_path.format(
                        video_key=img_key, chunk_index=chunk_idx, file_index=file_idx
                    )
                    video_path.parent.mkdir(parents=True, exist_ok=True)

                    # Encode video
                    encode_video_frames(
                        imgs_dir=imgs_dir,
                        video_path=video_path,
                        fps=fps,
                        vcodec=self.vcodec,
                        pix_fmt=self.pix_fmt,
                        g=self.g,
                        crf=self.crf,
                        fast_decode=self.fast_decode,
                        overwrite=True,
                    )

                    # Clean up temporary images
                    shutil.rmtree(imgs_dir)

                    # Store video metadata
                    video_metadata[img_key] = {
                        f"videos/{img_key}/chunk_index": chunk_idx,
                        f"videos/{img_key}/file_index": file_idx,
                        f"videos/{img_key}/from_timestamp": 0.0,
                        f"videos/{img_key}/to_timestamp": episode_duration,
                    }

                # Build episode metadata using NEW index
                episode_meta = {
                    "episode_index": new_ep_idx,
                    "length": episode_length,
                    "dataset_from_index": new_ep_idx * episode_length,
                    "dataset_to_index": (new_ep_idx + 1) * episode_length,
                }

                # Add video metadata
                for img_key in img_keys:
                    episode_meta.update(video_metadata[img_key])

                # Add data chunk/file info
                if "data/chunk_index" in src_episode:
                    episode_meta["data/chunk_index"] = src_episode["data/chunk_index"]
                    episode_meta["data/file_index"] = src_episode["data/file_index"]

                all_episode_metadata.append(episode_meta)

            # Copy and transform data files (removing image columns)
            _copy_data_without_images(dataset, new_meta, episode_indices, img_keys)

            # Save episode metadata
            episodes_df = pd.DataFrame(all_episode_metadata)
            episodes_path = new_meta.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
            episodes_path.parent.mkdir(parents=True, exist_ok=True)
            episodes_df.to_parquet(episodes_path, index=False)

            # Update metadata info
            new_meta.info["total_episodes"] = len(episode_indices)
            new_meta.info["total_frames"] = sum(ep["length"] for ep in all_episode_metadata)
            new_meta.info["total_tasks"] = dataset.meta.total_tasks
            new_meta.info["splits"] = {"train": f"0:{len(episode_indices)}"}

            # Update video info for all image keys using the actual first video file
            for img_key in img_keys:
                if not new_meta.features[img_key].get("info", None):
                    # Use the first actually created video file
                    chunk_idx = all_episode_metadata[0][f"videos/{img_key}/chunk_index"]
                    file_idx = all_episode_metadata[0][f"videos/{img_key}/file_index"]
                    video_path = new_meta.root / new_meta.video_path.format(
                        video_key=img_key, chunk_index=chunk_idx, file_index=file_idx
                    )
                    new_meta.info["features"][img_key]["info"] = get_video_info(video_path)

            write_info(new_meta.info, new_meta.root)

            # Copy stats and tasks
            if dataset.meta.stats is not None:
                # Remove image stats
                new_stats = {k: v for k, v in dataset.meta.stats.items() if k not in img_keys}
                write_stats(new_stats, new_meta.root)

            if dataset.meta.tasks is not None:
                write_tasks(dataset.meta.tasks, new_meta.root)

        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


def make_convert_executor(
    repo_id,
    root,
    output_dir,
    output_repo_id,
    vcodec,
    pix_fmt,
    g,
    crf,
    fast_decode,
    num_image_workers,
    job_name,
    logs_dir,
    workers,
    partition,
    cpus_per_task,
    mem_per_cpu,
    time_limit,
    slurm=True,
):
    """Create executor for parallel video conversion."""
    kwargs = {
        "pipeline": [
            ConvertEpisodesToVideo(
                repo_id=repo_id,
                root=root,
                output_dir=output_dir,
                output_repo_id=output_repo_id,
                vcodec=vcodec,
                pix_fmt=pix_fmt,
                g=g,
                crf=crf,
                fast_decode=fast_decode,
                num_image_workers=num_image_workers,
            ),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": workers,  # Number of parallel tasks = number of workers
                "workers": workers,
                "time": time_limit,
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        executor = SlurmPipelineExecutor(**kwargs)
    else:
        kwargs.update(
            {
                "tasks": workers,
                "workers": 1,  # Local mode: sequential
            }
        )
        executor = LocalPipelineExecutor(**kwargs)

    return executor


def main():
    parser = argparse.ArgumentParser(
        description="Parallelize video conversion across SLURM workers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output paths
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the source image dataset",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing the source dataset (default: HF cache)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the video dataset",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Repository ID for output dataset (default: <repo_id>_video)",
    )

    # Video encoding parameters
    parser.add_argument(
        "--vcodec",
        type=str,
        default="libsvtav1",
        help="Video codec",
    )
    parser.add_argument(
        "--pix-fmt",
        type=str,
        default="yuv420p",
        help="Pixel format",
    )
    parser.add_argument(
        "--g",
        type=int,
        default=2,
        help="GOP size (group of pictures)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=30,
        help="Constant rate factor (quality)",
    )
    parser.add_argument(
        "--fast-decode",
        type=int,
        default=0,
        help="Fast decode tuning",
    )
    parser.add_argument(
        "--num-image-workers",
        type=int,
        default=4,
        help="Number of threads per worker for saving images",
    )

    # SLURM parameters
    parser.add_argument(
        "--logs-dir",
        type=Path,
        required=True,
        help="Path to logs directory for datatrove",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="convert_to_video",
        help="Job name for SLURM",
    )
    parser.add_argument(
        "--slurm",
        type=int,
        default=1,
        help="Launch over SLURM (1) or locally (0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Number of parallel workers (each processes different episodes)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        required=True,
        help="SLURM partition (use CPU partition)",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=8,
        help="Number of CPUs per SLURM task",
    )
    parser.add_argument(
        "--mem-per-cpu",
        type=str,
        default="4G",
        help="Memory per CPU",
    )
    parser.add_argument(
        "--time-limit",
        type=str,
        default="08:00:00",
        help="Time limit for SLURM job",
    )

    args = parser.parse_args()

    # Convert slurm flag to boolean
    slurm = args.slurm == 1

    # Create and run executor
    executor = make_convert_executor(
        repo_id=args.repo_id,
        root=args.root,
        output_dir=args.output_dir,
        output_repo_id=args.output_repo_id,
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
        g=args.g,
        crf=args.crf,
        fast_decode=args.fast_decode,
        num_image_workers=args.num_image_workers,
        job_name=args.job_name,
        logs_dir=args.logs_dir,
        workers=args.workers,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu=args.mem_per_cpu,
        time_limit=args.time_limit,
        slurm=slurm,
    )

    logging.info(f"Starting video conversion with {args.workers} workers")
    executor.run()
    logging.info("All workers submitted/completed")


if __name__ == "__main__":
    main()


