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

# The following section is commented out and can be used for future implementation or as a TODO.
# TODO: Implement parallel camera reading utilities here if needed in the future.

"""
Parallel camera reading utility for LeRobot.
Reduces camera read latency by executing async_read operations concurrently.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("performance")


class ParallelCameraReader:
    """
    Reads from multiple cameras in parallel using ThreadPoolExecutor.

    This class provides a significant performance improvement over sequential
    camera reading by executing all camera async_read operations concurrently.

    Performance characteristics:
    - Sequential reading: N cameras × read_time (15-45ms for 3 cameras)
    - Parallel reading: max(read_time) across cameras (3-5ms for 3 cameras)

    Thread safety:
    - Each camera's async_read is already thread-safe
    - ThreadPoolExecutor manages thread lifecycle
    - No shared state between reads

    Example:
        >>> reader = ParallelCameraReader()
        >>> cameras = {"cam1": camera1, "cam2": camera2}
        >>> frames = reader.read_cameras_parallel(cameras)
        >>> print(frames["cam1"].shape)  # (480, 640, 3)
    """

    def __init__(self, max_workers: int | None = None, persistent_executor: bool = True):
        """
        Initialize parallel camera reader.

        Args:
            max_workers: Maximum number of threads. None = number of cameras.
                        Recommended: min(num_cameras, 8) to avoid thread overhead.
            persistent_executor: If True, reuse executor across calls for better
                               performance (saves ~5ms per call). If False, create
                               new executor each time (safer but slower).
        """
        self.max_workers = max_workers
        self.persistent_executor = persistent_executor
        self._executor = None
        self._stats = {
            "total_reads": 0,
            "failed_reads": 0,
            "total_time_ms": 0.0,
            "max_time_ms": 0.0,
        }
        self._last_log_time = time.perf_counter()
        self._log_interval = 5.0  # Log every 5 seconds
        self._read_count = 0

        if persistent_executor:
            # Pre-create executor to avoid startup overhead
            self._executor = ThreadPoolExecutor(max_workers=max_workers or 8)

    def read_cameras_parallel(
        self,
        cameras: dict[str, Any],
        timeout_ms: int = 1000,
        with_depth: bool = True,
        return_partial: bool = True,
    ) -> dict[str, Any]:
        """
        Read all cameras in parallel with optimized thread execution.

        Args:
            cameras: Dictionary mapping camera_name -> camera_object.
                    Each camera must have async_read() or async_read_all() method.
            timeout_ms: Maximum time to wait for each camera in milliseconds.
                       Individual camera timeouts, not total operation timeout.
            with_depth: If True, read depth data from cameras that support it
                       using async_read_all(). If False, only read color.
            return_partial: If True, return successful reads even if some fail.
                          If False, raise exception if any camera fails.

        Returns:
            Dictionary with camera data:
            - For regular cameras: {camera_name: frame_array}
            - For depth cameras: {camera_name: color_array, camera_name_depth: depth_array}
            - For failed cameras (if return_partial=True): {camera_name: None}

        Raises:
            RuntimeError: If return_partial=False and any camera read fails.
            ValueError: If cameras dict is empty.

        Example:
            >>> # Basic usage
            >>> frames = reader.read_cameras_parallel(cameras)

            >>> # Without depth for faster reads
            >>> frames = reader.read_cameras_parallel(cameras, with_depth=False)

            >>> # Strict mode - fail if any camera fails
            >>> frames = reader.read_cameras_parallel(cameras, return_partial=False)
        """
        if not cameras:
            raise ValueError("No cameras provided for parallel reading")

        start_time = time.perf_counter()
        obs_dict = {}
        failed_cameras = []

        # Determine executor strategy
        if self.persistent_executor and self._executor:
            executor = self._executor
            should_shutdown = False
        else:
            # Create temporary executor sized for current camera count
            num_workers = self.max_workers or min(len(cameras), 8)
            executor = ThreadPoolExecutor(max_workers=num_workers)
            should_shutdown = True

        try:
            # Submit all camera reads in parallel
            # This is the key optimization - all reads start simultaneously
            futures = {}
            submit_time = time.perf_counter()
            submit_timings = {}  # Track individual submission times

            for cam_key, cam in cameras.items():
                cam_submit_start = time.perf_counter()
                
                # Determine which read method to use based on camera capabilities
                if (
                    with_depth
                    and hasattr(cam, "use_depth")
                    and cam.use_depth
                    and hasattr(cam, "async_read_all")
                ):
                    # Camera supports depth - use async_read_all
                    futures[cam_key] = (executor.submit(cam.async_read_all, timeout_ms), "depth")
                else:
                    # Regular camera or depth disabled - use async_read
                    futures[cam_key] = (executor.submit(cam.async_read, timeout_ms), "color")
                
                submit_timings[cam_key] = (time.perf_counter() - cam_submit_start) * 1000

            submit_duration_ms = (time.perf_counter() - submit_time) * 1000
            if submit_duration_ms > 1.0:  # Warn if submission takes too long
                perf_logger.warning(
                    f"Camera read submission took {submit_duration_ms:.1f}ms for {len(cameras)} cameras"
                )

            # Collect results with individual timeout handling
            collect_time = time.perf_counter()
            individual_timings = {}
            wait_timings = {}  # Time waiting for hardware
            process_timings = {}  # Time processing results

            for cam_key, (future, read_type) in futures.items():
                cam_start = time.perf_counter()
                try:
                    # Convert timeout from ms to seconds for future.result()
                    timeout_s = timeout_ms / 1000.0

                    # Check if future is already done (no wait needed)
                    if future.done():
                        result = future.result()
                        wait_timings[cam_key] = 0.0  # No wait, already completed
                    else:
                        # Get result with timeout (this includes hardware wait time)
                        result = future.result(timeout=timeout_s)
                        wait_timings[cam_key] = (time.perf_counter() - cam_start) * 1000
                    
                    process_start = time.perf_counter()

                    # Process result based on read type (this is software overhead)
                    if read_type == "depth" and isinstance(result, dict):
                        # async_read_all returns dict with "color" and possibly "depth_rgb"
                        obs_dict[cam_key] = result.get("color")

                        # Add depth frame with _depth suffix if present
                        if "depth_rgb" in result and result["depth_rgb"] is not None:
                            obs_dict[f"{cam_key}_depth"] = result["depth_rgb"]
                    else:
                        # Regular async_read returns the frame directly
                        obs_dict[cam_key] = result

                    process_timings[cam_key] = (time.perf_counter() - process_start) * 1000
                    individual_timings[cam_key] = (time.perf_counter() - cam_start) * 1000

                except FutureTimeoutError:
                    perf_logger.warning(f"Camera {cam_key} read timeout after {timeout_ms}ms")
                    failed_cameras.append(cam_key)
                    if return_partial:
                        obs_dict[cam_key] = None

                except Exception as e:
                    perf_logger.warning(f"Camera {cam_key} read failed: {e}")
                    failed_cameras.append(cam_key)
                    if return_partial:
                        obs_dict[cam_key] = None

            collect_duration_ms = (time.perf_counter() - collect_time) * 1000

        finally:
            # Clean up temporary executor if created
            if should_shutdown:
                executor.shutdown(wait=False)

        # Calculate and log statistics
        total_duration_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(total_duration_ms, len(failed_cameras))
        self._read_count += 1

        # Only log periodically (every 5 seconds) to reduce overhead
        current_time = time.perf_counter()
        should_log = (current_time - self._last_log_time) >= self._log_interval

        if should_log and wait_timings:
            # Find bottleneck camera (longest wait time)
            bottleneck_cam = max(wait_timings, key=wait_timings.get)
            bottleneck_wait = wait_timings[bottleneck_cam]

            # Calculate actual parallel efficiency
            max_wait = max(wait_timings.values())
            total_process = sum(process_timings.values())
            overhead = total_duration_ms - max_wait - total_process
            
            # Show per-camera timing breakdown
            cam_breakdown = []
            for cam_key in sorted(wait_timings.keys()):
                wait = wait_timings.get(cam_key, 0)
                proc = process_timings.get(cam_key, 0)
                total = wait + proc
                # Mark the bottleneck camera
                mark = "*" if cam_key == bottleneck_cam else ""
                cam_breakdown.append(f"{cam_key}{mark}:{total:.1f}ms")

            # Build compact timing string showing key metrics
            timing_str = " | ".join(cam_breakdown)
            
            # Calculate average stats over the interval
            avg_time = self._stats["total_time_ms"] / self._stats["total_reads"] if self._stats["total_reads"] > 0 else 0

            perf_logger.info(
                f"Camera Stats (5s avg): {avg_time:.1f}ms | Last: {total_duration_ms:.1f}ms | "
                f"Breakdown: [{timing_str}] | Reads: {self._read_count}"
            )
            
            self._last_log_time = current_time
            self._read_count = 0

        # Handle failures in strict mode
        if not return_partial and failed_cameras:
            raise RuntimeError(f"Failed to read from cameras: {', '.join(failed_cameras)}")

        return obs_dict

    def _update_stats(self, duration_ms: float, num_failed: int):
        """Update internal statistics for monitoring."""
        self._stats["total_reads"] += 1
        self._stats["failed_reads"] += num_failed
        self._stats["total_time_ms"] += duration_ms
        self._stats["max_time_ms"] = max(self._stats["max_time_ms"], duration_ms)

    def get_stats(self) -> dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with:
            - total_reads: Total number of read operations
            - failed_reads: Total number of failed camera reads
            - avg_time_ms: Average read time in milliseconds
            - max_time_ms: Maximum read time observed
        """
        stats = self._stats.copy()
        if stats["total_reads"] > 0:
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_reads"]
        else:
            stats["avg_time_ms"] = 0.0
        return stats

    def reset_stats(self):
        """Reset performance statistics."""
        self._stats = {
            "total_reads": 0,
            "failed_reads": 0,
            "total_time_ms": 0.0,
            "max_time_ms": 0.0,
        }

    def __del__(self):
        """Clean up persistent executor on deletion."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
