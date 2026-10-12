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
from multiprocessing import Barrier, Event, Process, Queue

import numpy as np
import pytest

from lerobot.utils.shared_array import SharedArray


@pytest.fixture
def shared_array():
    array = SharedArray(shape=(8, 2), dtype=np.float32)
    yield array
    array.delete()


def rows(start: int, stop: int) -> np.ndarray:
    """Rows whose values encode their index, to check ordering."""
    return np.stack([np.arange(start, stop), np.arange(start, stop)], axis=1).astype(np.float32)


# Basic functionality


def test_creation(shared_array):
    assert shared_array.shape == (8, 2)
    assert shared_array.dtype == np.float32
    assert shared_array.capacity == 8
    assert shared_array.dropped == 0

    local_array = shared_array.get_local_array()
    assert isinstance(local_array, np.ndarray)
    assert local_array.shape == (8, 2)
    assert local_array.dtype == np.float32


@pytest.mark.parametrize("shape", [(), (0, 2)])
def test_creation_invalid_shape(shape):
    with pytest.raises(ValueError):
        SharedArray(shape=shape, dtype=np.float32)


def test_read_empty(shared_array):
    local_array = shared_array.get_local_array()

    data, timestamp = shared_array.read(local_array)
    assert data.shape == (0, 2)
    assert math.isnan(timestamp)

    data, timestamp = shared_array.read_latest(local_array, 4)
    assert data.shape == (0, 2)
    assert math.isnan(timestamp)


def test_write_and_read(shared_array):
    local_array = shared_array.get_local_array()

    shared_array.write(local_array, rows(0, 2), timestamp=1.0)
    shared_array.write(local_array, rows(2, 3), timestamp=2.0)

    data, timestamp = shared_array.read(local_array, flush=False)
    np.testing.assert_array_equal(data, rows(0, 3))
    assert timestamp == 2.0

    # Without flush, the same rows are returned again
    data, _ = shared_array.read(local_array, flush=True)
    np.testing.assert_array_equal(data, rows(0, 3))

    data, _ = shared_array.read(local_array)
    assert len(data) == 0
    shared_array.write(local_array, rows(3, 5), timestamp=3.0)
    data, timestamp = shared_array.read(local_array)
    np.testing.assert_array_equal(data, rows(3, 5))
    assert timestamp == 3.0


def test_write_empty_is_noop(shared_array):
    local_array = shared_array.get_local_array()
    shared_array.write(local_array, rows(0, 0), timestamp=1.0)

    data, timestamp = shared_array.read(local_array)
    assert len(data) == 0
    assert math.isnan(timestamp)


def test_read_returns_a_copy(shared_array):
    local_array = shared_array.get_local_array()
    shared_array.write(local_array, rows(0, 2))

    data, _ = shared_array.read(local_array, flush=False)
    data[:] = -1

    data, _ = shared_array.read(local_array)
    np.testing.assert_array_equal(data, rows(0, 2))


# Ring buffer behavior


def test_wrap_around(shared_array):
    local_array = shared_array.get_local_array()

    shared_array.write(local_array, rows(0, 6))
    data, _ = shared_array.read(local_array)
    np.testing.assert_array_equal(data, rows(0, 6))

    # This write wraps around the end of the buffer
    shared_array.write(local_array, rows(6, 12))
    data, _ = shared_array.read(local_array)
    np.testing.assert_array_equal(data, rows(6, 12))
    assert shared_array.dropped == 0


def test_overflow_drops_oldest_rows(shared_array):
    local_array = shared_array.get_local_array()

    shared_array.write(local_array, rows(0, 6))
    shared_array.write(local_array, rows(6, 11))

    data, _ = shared_array.read(local_array)
    np.testing.assert_array_equal(data, rows(3, 11))
    assert shared_array.dropped == 3


def test_write_larger_than_capacity(shared_array):
    local_array = shared_array.get_local_array()

    shared_array.write(local_array, rows(0, 2))
    shared_array.write(local_array, rows(2, 20), timestamp=5.0)

    data, timestamp = shared_array.read(local_array)
    np.testing.assert_array_equal(data, rows(12, 20))
    assert timestamp == 5.0
    # The 2 unread rows and the 10 rows that did not fit
    assert shared_array.dropped == 12


def test_read_latest(shared_array):
    local_array = shared_array.get_local_array()

    shared_array.write(local_array, rows(0, 3), timestamp=1.0)
    data, timestamp = shared_array.read_latest(local_array, 5)
    np.testing.assert_array_equal(data, rows(0, 3))
    assert timestamp == 1.0

    # read_latest does not flush, and still returns rows that were already read
    shared_array.read(local_array)
    shared_array.write(local_array, rows(3, 10), timestamp=2.0)
    data, timestamp = shared_array.read_latest(local_array, 5)
    np.testing.assert_array_equal(data, rows(5, 10))
    assert timestamp == 2.0
    data, _ = shared_array.read(local_array)
    np.testing.assert_array_equal(data, rows(3, 10))

    data, _ = shared_array.read_latest(local_array, 8)
    np.testing.assert_array_equal(data, rows(2, 10))


@pytest.mark.parametrize("num_rows", [-1, 9])
def test_read_latest_invalid_num_rows(shared_array, num_rows):
    with pytest.raises(ValueError):
        shared_array.read_latest(shared_array.get_local_array(), num_rows)


def test_read_latest_does_not_count_as_dropped(shared_array):
    local_array = shared_array.get_local_array()
    shared_array.write(local_array, rows(0, 4))
    shared_array.read_latest(local_array, 4)
    shared_array.write(local_array, rows(4, 10))

    # The first rows were never consumed by `read`, so they were dropped
    assert shared_array.dropped == 2


def test_reset(shared_array):
    local_array = shared_array.get_local_array()
    shared_array.write(local_array, rows(0, 20), timestamp=1.0)

    shared_array.reset()

    data, timestamp = shared_array.read(local_array)
    assert len(data) == 0
    assert math.isnan(timestamp)
    assert shared_array.dropped == 0
    assert len(shared_array.read_latest(local_array, 8)[0]) == 0

    shared_array.write(local_array, rows(0, 2))
    data, _ = shared_array.read(local_array)
    np.testing.assert_array_equal(data, rows(0, 2))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.int32])
def test_dtypes(dtype):
    shared_array = SharedArray(shape=(4, 3), dtype=dtype)
    try:
        local_array = shared_array.get_local_array()
        data = np.arange(6).reshape(2, 3).astype(dtype)
        shared_array.write(local_array, data)
        read_data, _ = shared_array.read(local_array)
        assert read_data.dtype == dtype
        np.testing.assert_array_equal(read_data, data)
    finally:
        shared_array.delete()


# Multi-process behavior


def _writer(shared_array, barrier, start, stop, chunk_size):
    local_array = shared_array.get_local_array()
    barrier.wait()
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        shared_array.write(local_array, rows(chunk_start, chunk_stop), timestamp=float(chunk_stop))


def _reader(shared_array, barrier, done_event, result_queue):
    local_array = shared_array.get_local_array()
    barrier.wait()
    chunks = []
    while True:
        done = done_event.is_set()
        data, _ = shared_array.read(local_array)
        chunks.append(data)
        if done:
            break
    result_queue.put(np.concatenate(chunks))


def test_single_writer_single_reader_across_processes():
    """Rows written by one process are read by another one, in order, without loss."""
    num_rows = 2000
    # Large enough for the reader to never fall behind by more than the capacity
    shared_array = SharedArray(shape=(num_rows, 2), dtype=np.float32)
    barrier = Barrier(2)
    done_event = Event()
    result_queue = Queue()

    try:
        reader = Process(target=_reader, args=(shared_array, barrier, done_event, result_queue))
        writer = Process(target=_writer, args=(shared_array, barrier, 0, num_rows, 7))
        reader.start()
        writer.start()
        writer.join(timeout=10)
        done_event.set()
        result = result_queue.get(timeout=10)
        reader.join(timeout=10)

        np.testing.assert_array_equal(result, rows(0, num_rows))
        assert shared_array.dropped == 0
    finally:
        shared_array.delete()


def test_concurrent_writers_do_not_lose_rows():
    """Writes from several processes never interleave inside a chunk, and no row is lost."""
    num_writers, rows_per_writer, chunk_size = 3, 300, 5
    shared_array = SharedArray(shape=(num_writers * rows_per_writer, 2), dtype=np.float32)
    barrier = Barrier(num_writers)

    try:
        writers = [
            Process(
                target=_writer,
                args=(shared_array, barrier, i * rows_per_writer, (i + 1) * rows_per_writer, chunk_size),
            )
            for i in range(num_writers)
        ]
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join(timeout=10)

        data, _ = shared_array.read(shared_array.get_local_array())
        assert len(data) == num_writers * rows_per_writer
        np.testing.assert_array_equal(np.sort(data[:, 0]), np.arange(num_writers * rows_per_writer))
        # Each chunk was written atomically
        for chunk in data.reshape(-1, chunk_size, 2):
            np.testing.assert_array_equal(np.diff(chunk[:, 0]), np.ones(chunk_size - 1))
    finally:
        shared_array.delete()
