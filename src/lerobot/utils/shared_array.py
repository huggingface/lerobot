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

from multiprocessing import Lock, Value, shared_memory

import numpy as np


class SharedArray:
    """
    A SharedArray is a numpy array shared between multiple processes in a shared_memory object.
    - Data is written to the array using the `write` method, which appends data to the array.
    - Data is read from the array (and eventually flushed) using the `read` method, which copies the _whole_ array.

    SharedArray offers quasi-instantaneous array-wide read and flush capabilities in comparison to Queues, but has a limited size defined at initialization.

    Example:
    _Main_process_
    shared_array = SharedArray(shape=(10, 10), dtype=np.dtype("float32"))
    local_array = shared_array.get_local_array()
    shared_array.write(local_array, np.array([[1, 2, 3], [4, 5, 6]]))

    _Child_process_
    local_array = shared_array.get_local_array()
    data = shared_array.read(local_array, flush=True)
    """

    def __init__(self, shape: tuple[int], dtype: np.dtype | str):
        """
        Initialize a SharedArray.

        Args:
            shape: The shape of the shared array.
            dtype: The dtype of the shared array.
        """
        self.shape = shape
        self.dtype = dtype

        self.shared_memory = shared_memory.SharedMemory(
            create=True, size=np.prod(shape) * np.dtype(dtype).itemsize
        )
        self.read_index = Value("i", 0)
        self.lock = Lock()

    def get_local_array(self) -> np.ndarray:
        """
        Get a process-local instance of the shared array.

        Returns:
            A process-local instance of the shared array.
        """
        return np.ndarray(self.shape, dtype=np.dtype(self.dtype), buffer=self.shared_memory.buf)

    def delete(self):
        """
        Delete the shared array.
        """
        self.shared_memory.close()
        self.shared_memory.unlink()

    def write(self, local_array: np.ndarray, data: np.ndarray):
        """
        Write data to the shared array.

        Args:
            local_array: The process-local instance of the shared array to write to.
            data: The data to write to the shared array.
        """
        with self.lock:
            local_array[self.read_index.value : self.read_index.value + len(data)] = data
            self.read_index.value += len(data)

    def read(self, local_array: np.ndarray, flush: bool = True) -> np.ndarray:
        """
        Read data from the shared array.

        Args:
            local_array: The process-local instance of the shared array to read from.
            flush: Whether to flush the shared array after reading.
        """
        with self.lock:
            data = np.copy(local_array[: self.read_index.value])
            if flush:
                self.read_index.value = 0
            return data

    def reset(self):
        """
        Reset the read index to 0.
        """
        with self.lock:
            self.read_index.value = 0
