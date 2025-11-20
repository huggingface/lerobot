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
Asynchronous video encoding utilities.

This module provides an AsyncVideoEncoder class that handles video encoding
operations in the background to avoid blocking the main recording thread.
It encodes videos to temporary files, which are then finalized by the main
LeRobotDataset class.
"""

import logging
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
from typing import Any

from lerobot.datasets.video_utils import encode_video_frames


@total_ordering
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

    def __eq__(self, other):
        return self.priority == other.priority and self.episode_index == other.episode_index


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
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        g: int | None = 2,
        crf: int | None = 30,
        fast_decode: int = 0,
    ):
        """
        Initialize the async video encoder.

        Args: see encode_video_frames
        """
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.g = g
        self.crf = crf
        self.fast_decode = fast_decode

        # Task queue (priority queue)
        self.task_queue = queue.PriorityQueue()

        # Results storage
        self.results: list[EncodingResult] = []
        self.results_lock = threading.Lock()

        # Worker thread
        self.worker_thread: threading.Thread | None = None

        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()

        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_encoding_time": 0.0,
            "average_encoding_time": 0.0,
        }
        self.stats_lock = threading.Lock()

        logging.info("AsyncVideoEncoder initialized")

    def start(self) -> None:
        """Start the async video encoder."""
        if self.running:
            logging.warning("AsyncVideoEncoder is already running")
            return

        self.running = True
        self.shutdown_event.clear()

        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self._worker_loop, name="AsyncVideoEncoder-Worker", daemon=True
        )
        self.worker_thread.start()

        logging.info("AsyncVideoEncoder started")

    def stop(self, wait: bool = True, timeout: float | None = None) -> None:
        """
        Stop the async video encoder.

        Args:
            timeout: Maximum time to wait for completion.
        """
        if not self.running:
            return

        logging.info("Stopping AsyncVideoEncoder...")

        # Signal shutdown. Worker loop will continue consuming queue until it is empty
        self.shutdown_event.set()
        self.running = False

        # Wait for worker thread to finish
        if not wait:
            self.timeout = 0
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)

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

        # Use negative priority for max-heap behavior (higher priority first)
        self.task_queue.put(task)

        with self.stats_lock:
            self.stats["tasks_submitted"] += 1

        logging.debug(f"Submitted encoding task for episode {episode_index}")

        return True

    def get_completed_tasks(self, clear: bool = True) -> list[EncodingResult]:
        """
        Get completed encoding results.

        Args:
            clear: Whether to clear the results after returning them

        Returns:
            List of encoding results
        """
        with self.results_lock:
            results_copy = self.results.copy()
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
        logging.info("AsyncVideoEncoder worker started")

        while True:
            try:
                # Blocks until a task is available or timeout occurs
                task = self.task_queue.get(timeout=1.0)
                self._process_encoding_task(task)
            except queue.Empty:
                if self.shutdown_event.is_set():
                    # We have finished processing all tasks and this event being set tells us that no more will be equeued.
                    break
                continue  # Go back to waiting

        logging.info("AsyncVideoEncoder worker stopped")

    def _process_encoding_task(self, task: EncodingTask) -> None:
        """Process a single encoding task."""
        start_time = time.time()
        success = True
        error_message = None
        temp_video_paths = {}

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

        status = "completed" if success else "failed"
        logging.info(f"Encoding task for episode {task.episode_index} {status} in {duration:.2f}s")

    def _get_image_dir(self, root_path: Path, episode_index: int, image_key: str) -> Path:
        """Helper to construct the path to the raw images for an episode."""
        # matches the return value of `_get_image_file_dir` in LeRobotDataset without reading episode metadata
        return root_path / "images" / image_key / f"episode-{episode_index:06d}"

    def _encode_to_temp_video(self, task: EncodingTask, video_key: str) -> Path:
        """
        Encode a single video to a temporary file.
        it is assumed that LerobotDataset called _wait_image_writer before enqueuing this task.
        """
        img_dir = self._get_image_dir(task.root_path, task.episode_index, video_key)

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Output to a unique file inside the task's dedicated temporary directory
        output_path = task.temp_dir / f"{video_key}.mp4"

        encode_video_frames(
            imgs_dir=img_dir,
            video_path=output_path,
            fps=task.fps,
            overwrite=False,  # there should be nothing in this temp dir
            vcodec=self.vcodec,
            pix_fmt=self.pix_fmt,
            g=self.g,
            crf=self.crf,
            fast_decode=self.fast_decode,
        )

        logging.debug(f"Encoded temporary video: {output_path}")
        return output_path
