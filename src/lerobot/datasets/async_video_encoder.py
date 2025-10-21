#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Asynchronous video encoding utilities.

This module provides an AsyncVideoEncoder class that handles video encoding
operations in the background to avoid blocking the main recording thread.
It encodes videos to temporary files, which are then finalized by the main
LeRobotDataset class.
"""

import contextlib
import logging
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .gpu_video_encoder import GPUVideoEncoder, create_gpu_encoder_config
from .video_utils import encode_video_frames


@dataclass
class EncodingTask:
    """Represents a video encoding task."""

    episode_index: int
    video_keys: list[str]
    fps: int
    root_path: Path  # Used to find the raw images
    temp_dir: Path  # A dedicated temporary directory for this episode's output
    priority: int = 0  # Higher number = higher priority

    def __lt__(self, other):
        """Sort by priority (higher first), then by episode index."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.episode_index < other.episode_index


@dataclass
class EncodingResult:
    """
    Result of a video encoding operation.
    Contains the path to the temporary video file.
    """

    episode_index: int
    # A dictionary mapping video_key to the path of its temporary encoded video
    temp_video_paths: dict[str, Path]
    success: bool
    duration: float
    error_message: str | None = None


class AsyncVideoEncoder:
    """
    Asynchronous video encoder that processes encoding tasks in the background.

    This class uses a thread pool to encode videos to temporary files without
    blocking the main recording thread.
    """

    def __init__(
        self,
        num_workers: int = 2,
        max_queue_size: int = 100,
        enable_logging: bool = True,
        gpu_video_encoding: bool = False,
        gpu_encoder_config: dict[str, Any] | None = None,
        vcodec: str = "h264",
    ):
        """
        Initialize the async video encoder.

        Args:
            num_workers: Number of worker threads for encoding
            max_queue_size: Maximum number of tasks in the queue
            enable_logging: Whether to enable detailed logging
        """
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.enable_logging = enable_logging
        self.vcodec = vcodec

        # Task queue (priority queue)
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)

        # Results storage
        self.results: list[EncodingResult] = []
        self.results_lock = threading.Lock()

        # Worker thread pool
        self.executor: ThreadPoolExecutor | None = None
        self.worker_thread: threading.Thread | None = None

        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()

        # GPU encoding support
        self.gpu_video_encoding = gpu_video_encoding
        self.gpu_encoder = None

        if self.gpu_video_encoding:
            config = gpu_encoder_config or {}
            gpu_config = create_gpu_encoder_config(
                encoder_type=config.get("encoder_type", "auto"),
                codec=config.get("codec", "h264"),
                preset=config.get("preset", "fast"),
                quality=config.get("quality", 23),
                bitrate=config.get("bitrate"),
                gpu_id=config.get("gpu_id", 0),
                enable_logging=enable_logging,
            )
            self.gpu_encoder = GPUVideoEncoder(gpu_config)

            if self.enable_logging:
                encoder_info = self.gpu_encoder.get_encoder_info()
                logging.info(
                    f"GPU encoder initialized: {encoder_info['selected_encoder']} {encoder_info['selected_codec']}"
                )

        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_encoding_time": 0.0,
            "average_encoding_time": 0.0,
        }
        self.stats_lock = threading.Lock()

        if self.enable_logging:
            logging.info(f"AsyncVideoEncoder initialized with {num_workers} workers")

    def start(self) -> None:
        """Start the async video encoder."""
        if self.running:
            logging.warning("AsyncVideoEncoder is already running")
            return

        self.running = True
        self.shutdown_event.clear()

        # Create thread pool
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="VideoEncoder")

        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self._worker_loop, name="AsyncVideoEncoder-Worker", daemon=True
        )
        self.worker_thread.start()

        if self.enable_logging:
            logging.info("AsyncVideoEncoder started")

    def stop(self, wait: bool = True, timeout: float | None = None) -> None:
        """
        Stop the async video encoder.

        Args:
            wait: Whether to wait for pending tasks to complete
            timeout: Maximum time to wait for completion
        """
        if not self.running:
            return

        if self.enable_logging:
            logging.info("Stopping AsyncVideoEncoder...")

        # Signal shutdown
        self.shutdown_event.set()
        self.running = False

        contextlib.suppress(queue.Full)
        # Use a sentinel to unblock the queue.get() call
        self.task_queue.put_nowait((float("-inf"), None))

        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)

        if self.executor:
            self.executor.shutdown(wait=wait)

        if self.enable_logging:
            logging.info("AsyncVideoEncoder stopped")

    def submit_encoding_task(
        self,
        episode_index: int,
        video_keys: list[str],
        fps: int,
        root_path: Path,
        temp_dir: Path,
        priority: int = 0,
    ) -> bool:
        """
        Submit a video encoding task to the queue.

        Args:
            episode_index: Index of the episode to encode
            video_keys: List of video keys to encode
            fps: Frames per second for the video
            root_path: Root path for the dataset
            temp_dir: Directory where the video file should be stored
            priority: Task priority (higher = more important)

        Returns:
            True if task was successfully submitted, False otherwise
        """
        if not self.running:
            logging.error("AsyncVideoEncoder is not running")
            return False

        task = EncodingTask(
            episode_index=episode_index,
            video_keys=video_keys,
            fps=fps,
            root_path=root_path,
            temp_dir=temp_dir,
            priority=priority,
        )

        try:
            # Use negative priority for max-heap behavior (higher priority first)
            self.task_queue.put_nowait((-priority, task))

            with self.stats_lock:
                self.stats["tasks_submitted"] += 1

            if self.enable_logging:
                logging.debug(f"Submitted encoding task for episode {episode_index}")

            return True

        except queue.Full:
            logging.error(f"Encoding task queue is full. Episode {episode_index} not submitted.")
            return False

    def get_completed_tasks(self, clear: bool = True) -> list[EncodingResult]:
        """
        Get completed encoding results.

        Args:
            clear: Whether to clear the results after returning them

        Returns:
            List of encoding results
        """
        with self.results_lock:
            results_copy = list(self.results)
            if clear:
                self.results.clear()
            return results_copy

    def get_stats(self) -> dict[str, Any]:
        """Get encoding statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
            if stats["tasks_completed"] > 0:
                stats["average_encoding_time"] = stats["total_encoding_time"] / stats["tasks_completed"]
            return stats

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """
        Wait for all submitted tasks to complete.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()

        while True:
            # Check if queue is empty and no tasks are running
            if self.task_queue.empty():
                with self.stats_lock:
                    if self.stats["tasks_submitted"] == self.stats["tasks_completed"]:
                        return True

            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                return False

            time.sleep(0.1)

    def _worker_loop(self) -> None:
        """Main worker loop that processes encoding tasks."""
        if self.enable_logging:
            logging.info("AsyncVideoEncoder worker started")

        while self.running and not self.shutdown_event.is_set():
            try:
                # Blocks until a task is available or timeout occurs
                _, task = self.task_queue.get(timeout=1.0)
                if task is None:  # Sentinel value
                    break
                self._process_encoding_task(task)
            except queue.Empty:
                continue  # Go back to waiting

        if self.enable_logging:
            logging.info("AsyncVideoEncoder worker stopped")

    def _process_encoding_task(self, task: EncodingTask) -> None:
        """Process a single encoding task."""
        start_time = time.time()
        success = True
        error_message = None
        temp_video_paths = {}

        if self.enable_logging:
            logging.info(f"Processing encoding task for episode {task.episode_index}")

        try:
            for video_key in task.video_keys:
                # The worker's ONLY job is to create a temporary video file.
                temp_video_path = self._encode_to_temp_video(task, video_key)
                temp_video_paths[video_key] = temp_video_path
        except Exception as e:
            # TODO it would probably be preferable to not catch this and allow it to print a trace.
            success = False
            error_message = str(e)
            logging.error(f"Failed to encode episode {task.episode_index}: {e}", exc_info=True)
            # Clean up partially created temp files for this task
            shutil.rmtree(task.temp_dir, ignore_errors=True)

        duration = time.time() - start_time
        result = EncodingResult(
            episode_index=task.episode_index,
            temp_video_paths=temp_video_paths,
            success=success,
            duration=duration,
            error_message=error_message,
        )

        with self.results_lock:
            self.results.append(result)

        # Update statistics
        with self.stats_lock:
            self.stats["tasks_completed"] += 1
            if success:
                self.stats["total_encoding_time"] += duration
            else:
                self.stats["tasks_failed"] += 1

        # After successful encoding, the raw image directory is no longer needed.
        # The finalizer will clean up the temp video file later.
        if success:
            for video_key in task.video_keys:
                img_dir = self._get_image_dir(task.root_path, task.episode_index, video_key)
                shutil.rmtree(img_dir, ignore_errors=True)

        if self.enable_logging:
            status = "completed" if success else "failed"
            logging.info(f"Encoding task for episode {task.episode_index} {status} in {duration:.2f}s")

    def _wait_for_images(self, img_dir: Path, timeout_s: int = 10):
        """
        Wait for the last frame in an image directory to be fully written, as indicated by whether PIL can decode it.

        the async image writer (separate from the async video encoder) may still be writing to the image folder for
        an episode when the video encoding task begins.
        the image writer has a wait function, but not a function to wait only on images in a certain directory.
        therefore in order to wait only until the images in this directory are complete, we can periodically try to decode the last frame.
        as long as decoding fails, wait another half second, up to a reasonable timeout of ten seconds.
        """
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout_s:
            try:
                # Find the last frame by sorting the filenames
                files = sorted(img_dir.glob("*.png"))
                if not files:
                    # No images yet, wait a bit
                    time.sleep(0.5)
                    continue
                last_frame_path = files[-1]

                # Try to open the image. If it's still being written it will appear truncated to PIL
                with Image.open(last_frame_path) as img:
                    img.load()  # Try to fully decode the PNG
                return

            except OSError:
                if self.enable_logging:
                    logging.debug(f"Waiting for images in {img_dir} to be fully written...")
                time.sleep(0.5)

        raise TimeoutError(f"Timed out after {timeout_s}s waiting for images in {img_dir}.")

    def _get_image_dir(self, root_path: Path, episode_index: int, image_key: str) -> Path:
        """Helper to construct the path to the raw images for an episode."""
        # matches the return value of `_get_image_file_dir` in LeRobotDataset without reading episode metadata
        return root_path / "images" / image_key / f"episode-{episode_index:06d}"

    def _encode_to_temp_video(self, task: EncodingTask, video_key: str) -> Path:
        """Encode a single video to a temporary file."""
        img_dir = self._get_image_dir(task.root_path, task.episode_index, video_key)

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # wait for all images in the image dir to be completely written
        self._wait_for_images(img_dir)

        # Output to a unique file inside the task's dedicated temporary directory
        output_path = task.temp_dir / f"{video_key}.mp4"

        # Encode the video using GPU or CPU
        if self.gpu_video_encoding and self.gpu_encoder:
            # Use GPU encoding with timeout
            try:
                success = self.gpu_encoder.encode_video(
                    input_dir=img_dir, output_path=output_path, fps=task.fps
                )

                if not success:
                    if self.enable_logging:
                        logging.warning(
                            f"GPU encoding failed for {output_path}, falling back to CPU encoding"
                        )
                    # Fallback to CPU encoding
                    encode_video_frames(
                        imgs_dir=img_dir,
                        video_path=output_path,
                        fps=task.fps,
                        overwrite=True,
                        vcodec=self.vcodec,
                    )
            except Exception as e:
                if self.enable_logging:
                    logging.error(f"GPU encoding error for {output_path}: {e}, falling back to CPU encoding")
                # Fallback to CPU encoding
                encode_video_frames(
                    imgs_dir=img_dir, video_path=output_path, fps=task.fps, overwrite=True, vcodec=self.vcodec
                )
        else:
            # Use CPU encoding
            encode_video_frames(
                imgs_dir=img_dir, video_path=output_path, fps=task.fps, overwrite=True, vcodec=self.vcodec
            )

        if self.enable_logging:
            logging.debug(f"Encoded temporary video: {output_path}")

        return output_path

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop(wait=True)
