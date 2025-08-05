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
"""

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gpu_video_encoder import GPUVideoEncoder, create_gpu_encoder_config
from .video_utils import encode_video_frames


@dataclass
class EncodingTask:
    """Represents a video encoding task."""

    episode_index: int
    video_keys: list[str]
    fps: int
    root_path: Path
    priority: int = 0  # Higher number = higher priority

    def __lt__(self, other):
        """Sort by priority (higher first), then by episode index."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.episode_index < other.episode_index


@dataclass
class EncodingResult:
    """Result of a video encoding operation."""

    episode_index: int
    video_keys: list[str]
    success: bool
    duration: float
    error_message: str | None = None


class AsyncVideoEncoder:
    """
    Asynchronous video encoder that processes encoding tasks in the background.

    This class uses a thread pool to encode videos without blocking the main
    recording thread. It maintains a priority queue to ensure important episodes
    are encoded first.
    """

    def __init__(
        self,
        num_workers: int = 2,
        max_queue_size: int = 100,
        enable_logging: bool = True,
        gpu_encoding: bool = False,
        gpu_encoder_config: dict[str, Any] | None = None,
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
        self.gpu_encoding = gpu_encoding
        self.gpu_encoder = None

        if self.gpu_encoding:
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

        # Submit a sentinel task to wake up the worker
        try:
            self.task_queue.put_nowait((0, None))  # Sentinel task
        except queue.Full:
            pass

        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)

        # Shutdown executor
        if self.executor:
            if wait:
                self.executor.shutdown(wait=True)
            else:
                self.executor.shutdown(wait=False)

        if self.enable_logging:
            logging.info("AsyncVideoEncoder stopped")

    def submit_encoding_task(
        self, episode_index: int, video_keys: list[str], fps: int, root_path: Path, priority: int = 0
    ) -> bool:
        """
        Submit a video encoding task to the queue.

        Args:
            episode_index: Index of the episode to encode
            video_keys: List of video keys to encode
            fps: Frames per second for the video
            root_path: Root path for the dataset
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

    def get_results(self, clear: bool = False) -> list[EncodingResult]:
        """
        Get completed encoding results.

        Args:
            clear: Whether to clear the results after returning them

        Returns:
            List of encoding results
        """
        with self.results_lock:
            if clear:
                results = self.results.copy()
                self.results.clear()
                return results
            else:
                return self.results.copy()

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
            if timeout is not None:
                if time.time() - start_time > timeout:
                    return False

            time.sleep(0.1)

    def _worker_loop(self) -> None:
        """Main worker loop that processes encoding tasks."""
        if self.enable_logging:
            logging.info("AsyncVideoEncoder worker started")

        while self.running and not self.shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                try:
                    priority, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Check for sentinel task
                if task is None:
                    break

                # Process the encoding task
                self._process_encoding_task(task)

            except Exception as e:
                logging.error(f"Error in AsyncVideoEncoder worker: {e}")

        if self.enable_logging:
            logging.info("AsyncVideoEncoder worker stopped")

    def _process_encoding_task(self, task: EncodingTask) -> None:
        """Process a single encoding task."""
        start_time = time.time()
        success = True
        error_message = None

        if self.enable_logging:
            logging.info(f"Processing encoding task for episode {task.episode_index}")

        try:
            # Encode each video key
            for video_key in task.video_keys:
                self._encode_single_video(task, video_key)

        except Exception as e:
            success = False
            error_message = str(e)
            logging.error(f"Failed to encode episode {task.episode_index}: {e}")

        # Record result
        duration = time.time() - start_time
        result = EncodingResult(
            episode_index=task.episode_index,
            video_keys=task.video_keys,
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

        if self.enable_logging:
            status = "completed" if success else "failed"
            logging.info(f"Encoding task for episode {task.episode_index} {status} in {duration:.2f}s")

    def _encode_single_video(self, task: EncodingTask, video_key: str) -> None:
        """Encode a single video file."""
        # Construct paths
        video_path = (
            task.root_path
            / "videos"
            / f"chunk-{task.episode_index // 1000:03d}"
            / video_key
            / f"episode_{task.episode_index:06d}.mp4"
        )
        img_dir = task.root_path / "images" / video_key / f"episode_{task.episode_index:06d}"

        # Ensure video directory exists
        video_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if video already exists
        if video_path.exists():
            if self.enable_logging:
                logging.debug(f"Video {video_path} already exists, skipping")
            return

        # Encode the video using GPU or CPU
        if self.gpu_encoding and self.gpu_encoder:
            # Use GPU encoding with timeout
            try:
                success = self.gpu_encoder.encode_video(
                    input_dir=img_dir, output_path=video_path, fps=task.fps
                )

                if not success:
                    if self.enable_logging:
                        logging.warning(f"GPU encoding failed for {video_path}, falling back to CPU encoding")
                    # Fallback to CPU encoding
                    encode_video_frames(imgs_dir=img_dir, video_path=video_path, fps=task.fps, overwrite=True)
            except Exception as e:
                if self.enable_logging:
                    logging.error(f"GPU encoding error for {video_path}: {e}, falling back to CPU encoding")
                # Fallback to CPU encoding
                encode_video_frames(imgs_dir=img_dir, video_path=video_path, fps=task.fps, overwrite=True)
        else:
            # Use CPU encoding
            encode_video_frames(imgs_dir=img_dir, video_path=video_path, fps=task.fps, overwrite=True)

        if self.enable_logging:
            logging.debug(f"Encoded video: {video_path}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop(wait=True)
