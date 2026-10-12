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

import math
from multiprocessing import Lock, Value, shared_memory
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray


class SharedArray:
    """
    A ring buffer of rows backed by a `shared_memory` block, so that one process can append rows
    while another reads them.

    - `write` appends rows. When the buffer is full, the oldest unread rows are overwritten and
      counted in `dropped`.
    - `read` returns the unread rows in order (and optionally marks them as read).
    - `read_latest` returns the most recent rows, whether they were read before or not.

    Each write can carry a timestamp: the time at which the last written row was produced. Reads
    return the timestamp of the last returned row along with the data, under the same lock.

    Compared to a `multiprocessing.Queue`, reading or flushing the whole buffer is a single copy,
    but the capacity is fixed at initialization.

    Example:
        _Main_process_
        shared_array = SharedArray(shape=(10, 2), dtype=np.dtype("float32"))
        local_array = shared_array.get_local_array()
        shared_array.write(local_array, np.array([[1, 2], [3, 4]]), timestamp=time.perf_counter())

        _Child_process_
        local_array = shared_array.get_local_array()
        data, timestamp = shared_array.read(local_array, flush=True)
    """

    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike):
        """
        Initialize a SharedArray.

        Args:
            shape: The shape of the buffer. The first dimension is the capacity in rows.
            dtype: The dtype of the buffer.
        """
        if len(shape) == 0 or shape[0] <= 0:
            raise ValueError(f"The first dimension of `shape` must be a positive capacity, got {shape}.")

        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.capacity = self.shape[0]
        self.shared_memory = shared_memory.SharedMemory(
            create=True, size=max(1, int(np.prod(self.shape)) * self.dtype.itemsize)
        )
        self.lock = Lock()
        # Total number of rows ever written / consumed since the last reset. Only their difference
        # and their value modulo the capacity matter, so they never need to wrap.
        self._written = Value("q", 0, lock=False)
        self._consumed = Value("q", 0, lock=False)
        self._dropped = Value("q", 0, lock=False)
        self._timestamp = Value("d", math.nan, lock=False)

    def get_local_array(self) -> NDArray[Any]:
        """
        Get a process-local view of the shared buffer.

        Returns:
            A numpy array backed by the shared memory block.
        """
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shared_memory.buf)

    def delete(self) -> None:
        """
        Release and unlink the shared memory block.
        """
        self.shared_memory.close()
        self.shared_memory.unlink()

    @property
    def dropped(self) -> int:
        """Number of rows overwritten before being read since the last reset."""
        with self.lock:
            return int(self._dropped.value)

    def write(self, local_array: NDArray[Any], data: NDArray[Any], timestamp: float = math.nan) -> None:
        """
        Append rows to the buffer, overwriting the oldest unread rows if it is full.

        Args:
            local_array: The process-local view of the buffer to write to.
            data: The rows to append. Its trailing dimensions must match the buffer's.
            timestamp: The time at which the last row of `data` was produced.
        """
        n = len(data)
        if n == 0:
            return

        with self.lock:
            if n > self.capacity:
                # Only the most recent rows fit. The skipped ones are counted as dropped below.
                self._written.value += n - self.capacity
                data = data[-self.capacity :]
                n = self.capacity

            start = self._written.value % self.capacity
            first = min(n, self.capacity - start)
            local_array[start : start + first] = data[:first]
            local_array[: n - first] = data[first:]

            self._written.value += n
            overflow = self._written.value - self._consumed.value - self.capacity
            if overflow > 0:
                self._dropped.value += overflow
                self._consumed.value += overflow
            self._timestamp.value = timestamp

    def _copy_rows(self, local_array: NDArray[Any], start: int, end: int) -> NDArray[Any]:
        """Copy rows `start:end` (absolute row counts) out of the ring, in order."""
        n = end - start
        begin = start % self.capacity
        first = min(n, self.capacity - begin)
        return np.concatenate((local_array[begin : begin + first], local_array[: n - first]))

    def read(self, local_array: NDArray[Any], flush: bool = True) -> tuple[NDArray[Any], float]:
        """
        Read the rows written since the last flush.

        Args:
            local_array: The process-local view of the buffer to read from.
            flush: Whether to mark the returned rows as read.

        Returns:
            The unread rows in write order, and the timestamp of the last written row (NaN if
            nothing was written since the last reset).
        """
        with self.lock:
            data = self._copy_rows(local_array, self._consumed.value, self._written.value)
            if flush:
                self._consumed.value = self._written.value
            return data, self._timestamp.value

    def read_latest(self, local_array: NDArray[Any], num_rows: int) -> tuple[NDArray[Any], float]:
        """
        Read the most recent rows, whether they were already read or not. Does not flush.

        Args:
            local_array: The process-local view of the buffer to read from.
            num_rows: The number of rows to return. Fewer are returned if fewer were written since
                the last reset. Must not exceed the capacity.

        Returns:
            The most recent rows in write order, and the timestamp of the last one.
        """
        if not 0 <= num_rows <= self.capacity:
            raise ValueError(f"`num_rows` must be in [0, {self.capacity}], got {num_rows}.")

        with self.lock:
            n = min(num_rows, self._written.value)
            data = self._copy_rows(local_array, self._written.value - n, self._written.value)
            return data, self._timestamp.value

    def reset(self) -> None:
        """
        Discard all rows and clear the timestamp and the dropped counter.
        """
        with self.lock:
            self._written.value = 0
            self._consumed.value = 0
            self._dropped.value = 0
            self._timestamp.value = math.nan
