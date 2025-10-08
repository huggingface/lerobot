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

import logging
import queue
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .video_utils import encode_video_frames


@dataclass
class EncodingTask:
    """Represents a video encoding task."""

    episode_index: int
    video_keys: list[str]
    fps: int
    root_path: Path  # Used to find the raw images
    temp_dir: Path  # A dedicated temporary directory for this episode's output
    priority: int = 0

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
        # TODO(hf-team): GPU encoding is a potential future enhancement.
        # For now, the logic is simplified to focus on the core async pattern.
        # gpu_encoding: bool = False,
        # gpu_encoder_config: dict[str, Any] | None = None,
    ):
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.enable_logging = enable_logging
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.results: list[EncodingResult] = []
        self.results_lock = threading.Lock()
        self.executor: ThreadPoolExecutor | None = None
        self.worker_thread: threading.Thread | None = None
        self.running = False
        self.shutdown_event = threading.Event()
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_encoding_time": 0.0,
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
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="VideoEncoder")
        self.worker_thread = threading.Thread(target=self._worker_loop, name="AsyncVideoEncoder-Worker", daemon=True)
        self.worker_thread.start()

        if self.enable_logging:
            logging.info("AsyncVideoEncoder started")

    def stop(self, wait: bool = True, timeout: float | None = None) -> None:
        """Stop the async video encoder."""
        if not self.running:
            return

        if self.enable_logging:
            logging.info("Stopping AsyncVideoEncoder...")

        self.shutdown_event.set()
        self.running = False

        try:
            # Use a sentinel to unblock the queue.get() call
            self.task_queue.put_nowait((float("-inf"), None))
        except queue.Full:
            pass

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)

        if self.executor:
            self.executor.shutdown(wait=wait)

        if self.enable_logging:
            logging.info("AsyncVideoEncoder stopped")

    def submit_encoding_task(
        self, episode_index: int, video_keys: list[str], fps: int, root_path: Path, temp_dir: Path, priority: int = 0
    ) -> bool:
        """Submit a video encoding task to the queue."""
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
        """Get completed encoding results."""
        with self.results_lock:
            results_copy = list(self.results)
            if clear:
                self.results.clear()
            return results_copy

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Wait for all submitted tasks to complete."""
        start_time = time.time()
        while True:
            with self.stats_lock:
                if self.stats["tasks_submitted"] == self.stats["tasks_completed"]:
                    return True

            if timeout is not None and (time.time() - start_time) > timeout:
                logging.warning("Timeout waiting for async encoding completion.")
                return False

            time.sleep(0.1)

    def _worker_loop(self) -> None:
        """Main worker loop that processes encoding tasks."""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Blocks until a task is available or timeout occurs
                _, task = self.task_queue.get(timeout=1.0)
                if task is None:  # Sentinel value
                    break
                self._process_encoding_task(task)
            except queue.Empty:
                continue  # Go back to waiting
            except Exception as e:
                logging.error(f"Error in AsyncVideoEncoder worker loop: {e}", exc_info=True)

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

        with self.stats_lock:
            self.stats["tasks_completed"] += 1
            if not success:
                self.stats["tasks_failed"] += 1
            else:
                self.stats["total_encoding_time"] += duration
        
        # After successful encoding, the raw image directory is no longer needed.
        # The finalizer will clean up the temp video file later.
        if success:
            for video_key in task.video_keys:
                img_dir = self._get_image_dir(task.root_path, task.episode_index, video_key)
                shutil.rmtree(img_dir, ignore_errors=True)


    def _get_image_dir(self, root_path: Path, episode_index: int, image_key: str) -> Path:
        """Helper to construct the path to the raw images for an episode."""
        # This logic should match `_get_image_file_dir` in LeRobotDataset
        return root_path / "images" / image_key / f"episode-{episode_index:06d}"

    def _encode_to_temp_video(self, task: EncodingTask, video_key: str) -> Path:
        """Encode a single video to a temporary file."""
        img_dir = self._get_image_dir(task.root_path, task.episode_index, video_key)
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Output to a unique file inside the task's dedicated temporary directory
        output_path = task.temp_dir / f"{video_key}.mp4"

        # Using the existing CPU-based encoder from video_utils
        encode_video_frames(imgs_dir=img_dir, video_path=output_path, fps=task.fps, overwrite=True)

        if self.enable_logging:
            logging.debug(f"Encoded temporary video: {output_path}")

        return output_path
