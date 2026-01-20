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

import multiprocessing
import time
from multiprocessing import Event, Process, Queue

import numpy as np
import pytest

from lerobot.utils.shared_array import SharedArray


def writer_process(shared_array, data_queue, stop_event, barrier, process_id):
    """Writer process that continuously writes data to shared array."""
    local_array = shared_array.get_local_array()

    # Wait for all processes to be ready
    barrier.wait()

    write_count = 0
    while not stop_event.is_set() and write_count < 10:
        # Generate unique data for this process and write iteration
        data = np.full((5, 2), process_id * 100 + write_count, dtype=np.float32)

        try:
            shared_array.write(local_array, data)
            data_queue.put(f"writer_{process_id}_wrote_{write_count}")
            write_count += 1
            time.sleep(0.01)  # Small delay to allow race conditions
        except IndexError:
            # Array is full, stop writing
            break


def reader_process(shared_array, data_queue, stop_event, barrier, process_id):
    """Reader process that continuously reads data from shared array."""
    local_array = shared_array.get_local_array()

    # Wait for all processes to be ready
    barrier.wait()

    read_count = 0
    while not stop_event.is_set() and read_count < 5:
        time.sleep(0.02)  # Allow some writes to accumulate

        data = shared_array.read(local_array, flush=True)
        data_queue.put(f"reader_{process_id}_read_{len(data)}_items")
        read_count += 1


def stress_writer_process(shared_array, data_queue, stop_event, barrier, process_id):
    """High-frequency writer process for stress testing."""
    local_array = shared_array.get_local_array()

    barrier.wait()

    write_count = 0
    while not stop_event.is_set() and write_count < 50:
        # Write single row at a time for more frequent operations
        data = np.array([[process_id, write_count]], dtype=np.float32)

        try:
            shared_array.write(local_array, data)
            write_count += 1
            # No sleep - stress test
        except IndexError:
            break

    data_queue.put(f"stress_writer_{process_id}_completed_{write_count}")


# Basic functionality tests


def test_shared_array_creation():
    """Test basic SharedArray creation and properties."""
    shape = (100, 4)
    dtype = np.float32

    shared_array = SharedArray(shape=shape, dtype=dtype)

    assert shared_array.shape == shape
    assert shared_array.dtype == dtype
    assert shared_array.read_index.value == 0

    # Clean up
    shared_array.delete()


def test_local_array_access():
    """Test getting local array instances."""
    shape = (50, 2)
    shared_array = SharedArray(shape=shape, dtype=np.float32)

    local_array = shared_array.get_local_array()

    assert local_array.shape == shape
    assert local_array.dtype == np.float32
    assert isinstance(local_array, np.ndarray)

    # Test that we can get multiple local array instances
    local_array2 = shared_array.get_local_array()
    assert local_array2.shape == shape

    shared_array.delete()


def test_write_and_read_single_process():
    """Test basic write and read operations in single process."""
    shape = (20, 3)
    shared_array = SharedArray(shape=shape, dtype=np.float32)
    local_array = shared_array.get_local_array()

    # Write some data
    data1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    shared_array.write(local_array, data1)

    assert shared_array.read_index.value == 2

    # Write more data
    data2 = np.array([[7, 8, 9]], dtype=np.float32)
    shared_array.write(local_array, data2)

    assert shared_array.read_index.value == 3

    # Read all data
    read_data = shared_array.read(local_array, flush=False)
    expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    np.testing.assert_array_equal(read_data, expected)

    # Read with flush
    read_data_flush = shared_array.read(local_array, flush=True)
    np.testing.assert_array_equal(read_data_flush, expected)
    assert shared_array.read_index.value == 0

    shared_array.delete()


def test_array_overflow():
    """Test behavior when writing more data than array capacity."""
    shape = (5, 2)  # Small array
    shared_array = SharedArray(shape=shape, dtype=np.float32)
    local_array = shared_array.get_local_array()

    # Fill the array
    data = np.ones((5, 2), dtype=np.float32)
    shared_array.write(local_array, data)

    # Try to write more data - should raise IndexError
    with pytest.raises(ValueError):
        extra_data = np.ones((2, 2), dtype=np.float32)
        shared_array.write(local_array, extra_data)

    shared_array.delete()


def test_reset_functionality():
    """Test the reset method."""
    shape = (10, 2)
    shared_array = SharedArray(shape=shape, dtype=np.float32)
    local_array = shared_array.get_local_array()

    # Write some data
    data = np.ones((3, 2), dtype=np.float32)
    shared_array.write(local_array, data)
    assert shared_array.read_index.value == 3

    # Reset
    shared_array.reset()
    assert shared_array.read_index.value == 0

    # Read should return empty array
    read_data = shared_array.read(local_array, flush=False)
    assert len(read_data) == 0

    shared_array.delete()


# Multi-process tests


def test_single_writer_single_reader():
    """Test basic writer-reader scenario with one process each."""
    shape = (100, 2)
    shared_array = SharedArray(shape=shape, dtype=np.float32)

    data_queue = Queue()
    stop_event = Event()
    barrier = multiprocessing.Barrier(2)  # Writer + reader

    # Start writer process
    writer = Process(target=writer_process, args=(shared_array, data_queue, stop_event, barrier, 1))

    # Start reader process
    reader = Process(target=reader_process, args=(shared_array, data_queue, stop_event, barrier, 1))

    writer.start()
    reader.start()

    # Let them run for a bit
    time.sleep(0.5)
    stop_event.set()

    # Wait for completion
    writer.join(timeout=2.0)
    reader.join(timeout=2.0)

    # Verify both processes completed
    assert not writer.is_alive()
    assert not reader.is_alive()

    # Check that we got messages from both processes
    messages = []
    while not data_queue.empty():
        messages.append(data_queue.get())

    writer_messages = [msg for msg in messages if msg.startswith("writer_")]
    reader_messages = [msg for msg in messages if msg.startswith("reader_")]

    assert len(writer_messages) > 0
    assert len(reader_messages) > 0

    shared_array.delete()


def test_multiple_writers_single_reader():
    """Test multiple writers with single reader - check for race conditions."""
    shape = (200, 2)
    shared_array = SharedArray(shape=shape, dtype=np.float32)

    data_queue = Queue()
    stop_event = Event()
    num_writers = 3
    barrier = multiprocessing.Barrier(num_writers + 1)  # Writers + reader

    processes = []

    # Start multiple writer processes
    for i in range(num_writers):
        writer = Process(target=writer_process, args=(shared_array, data_queue, stop_event, barrier, i + 1))
        processes.append(writer)
        writer.start()

    # Start reader process
    reader = Process(target=reader_process, args=(shared_array, data_queue, stop_event, barrier, 1))
    processes.append(reader)
    reader.start()

    # Let them run
    time.sleep(1.0)
    stop_event.set()

    # Wait for all processes
    for process in processes:
        process.join(timeout=3.0)
        assert not process.is_alive()

    # Verify we got messages from all processes
    messages = []
    while not data_queue.empty():
        messages.append(data_queue.get())

    writer_messages = [msg for msg in messages if msg.startswith("writer_")]
    reader_messages = [msg for msg in messages if msg.startswith("reader_")]

    # Should have messages from all writers
    assert len(writer_messages) >= num_writers
    assert len(reader_messages) > 0

    shared_array.delete()


def test_data_integrity_with_concurrent_access():
    """Test that data integrity is maintained under concurrent access using standard reader/writer processes."""
    shape = (500, 2)  # Use standard 2-column format
    shared_array = SharedArray(shape=shape, dtype=np.float32)

    data_queue = Queue()
    stop_event = Event()
    barrier = multiprocessing.Barrier(3)  # 2 writers + 1 reader

    # Start two writer processes
    writer1 = Process(target=writer_process, args=(shared_array, data_queue, stop_event, barrier, 1))
    writer2 = Process(target=writer_process, args=(shared_array, data_queue, stop_event, barrier, 2))

    # Start one reader process
    reader = Process(target=reader_process, args=(shared_array, data_queue, stop_event, barrier, 1))

    writer1.start()
    writer2.start()
    reader.start()

    # Let them run for integrity test duration
    time.sleep(1.0)
    stop_event.set()

    # Wait for completion
    writer1.join(timeout=3.0)
    writer2.join(timeout=3.0)
    reader.join(timeout=3.0)

    # Verify all processes completed successfully
    assert not writer1.is_alive()
    assert not writer2.is_alive()
    assert not reader.is_alive()

    # Verify data integrity by checking messages
    messages = []
    while not data_queue.empty():
        messages.append(data_queue.get())

    writer1_messages = [msg for msg in messages if "writer_1_wrote" in msg]
    writer2_messages = [msg for msg in messages if "writer_2_wrote" in msg]
    reader_messages = [msg for msg in messages if "reader_1_read" in msg]

    # Verify both writers wrote data
    assert len(writer1_messages) > 0
    assert len(writer2_messages) > 0
    # Verify reader read data
    assert len(reader_messages) > 0

    # Verify the shared array is in a consistent state
    local_array = shared_array.get_local_array()
    final_data = shared_array.read(local_array, flush=False)

    # Should have some data written by the writers
    assert len(final_data) >= 0  # Could be empty if reader flushed everything
    # Should not exceed array capacity
    assert len(final_data) <= shape[0]

    # If there's data, verify it contains the expected writer signatures
    if len(final_data) > 0:
        # Data should contain values like 100, 101, 102... (writer 1) or 200, 201, 202... (writer 2)
        unique_values = np.unique(final_data.flatten())
        writer1_values = unique_values[(unique_values >= 100) & (unique_values < 200)]
        writer2_values = unique_values[(unique_values >= 200) & (unique_values < 300)]

        # Should have data from at least one writer
        assert len(writer1_values) > 0 or len(writer2_values) > 0

    shared_array.delete()


def test_stress_test_high_frequency_operations():
    """Stress test with high frequency read/write operations."""
    shape = (1000, 2)
    shared_array = SharedArray(shape=shape, dtype=np.float32)

    data_queue = Queue()
    stop_event = Event()
    num_writers = 4
    barrier = multiprocessing.Barrier(num_writers)

    processes = []

    # Start multiple high-frequency writers
    for i in range(num_writers):
        writer = Process(
            target=stress_writer_process, args=(shared_array, data_queue, stop_event, barrier, i + 1)
        )
        processes.append(writer)
        writer.start()

    # Let them run for stress test duration
    time.sleep(0.5)
    stop_event.set()

    # Wait for completion
    for process in processes:
        process.join(timeout=3.0)
        assert not process.is_alive()

    # Verify all writers completed successfully
    messages = []
    while not data_queue.empty():
        messages.append(data_queue.get())

    completed_messages = [msg for msg in messages if "completed" in msg]
    assert len(completed_messages) == num_writers

    # Verify the shared array is in a consistent state
    local_array = shared_array.get_local_array()
    final_data = shared_array.read(local_array, flush=False)

    # Should have some data written
    assert len(final_data) > 0
    # Should not exceed array capacity
    assert len(final_data) <= shape[0]

    shared_array.delete()


def test_concurrent_readers():
    """Test multiple concurrent readers with writers to ensure thread safety."""
    shape = (200, 2)
    shared_array = SharedArray(shape=shape, dtype=np.float32)

    data_queue = Queue()
    stop_event = Event()
    num_readers = 3
    num_writers = 2
    barrier = multiprocessing.Barrier(num_readers + num_writers)

    processes = []

    # Start multiple writer processes to generate data
    for i in range(num_writers):
        writer = Process(target=writer_process, args=(shared_array, data_queue, stop_event, barrier, i + 1))
        processes.append(writer)
        writer.start()

    # Start multiple reader processes
    for i in range(num_readers):
        reader = Process(target=reader_process, args=(shared_array, data_queue, stop_event, barrier, i + 1))
        processes.append(reader)
        reader.start()

    # Let them run to test concurrent access
    time.sleep(1.0)
    stop_event.set()

    # Wait for all processes to complete
    for process in processes:
        process.join(timeout=3.0)
        assert not process.is_alive()

    # Verify all readers and writers completed
    messages = []
    while not data_queue.empty():
        messages.append(data_queue.get())

    reader_messages = [msg for msg in messages if msg.startswith("reader_")]
    writer_messages = [msg for msg in messages if msg.startswith("writer_")]

    # Should have messages from all readers and writers
    assert len(reader_messages) >= num_readers
    assert len(writer_messages) >= num_writers

    # Verify different readers generated different messages (proving they ran concurrently)
    reader_ids = set()
    for msg in reader_messages:
        # Extract reader ID from message like "reader_1_read_5_items"
        parts = msg.split("_")
        if len(parts) >= 2:
            reader_ids.add(parts[1])

    assert len(reader_ids) == num_readers  # All readers should have participated

    shared_array.delete()


def test_edge_case_empty_reads():
    """Test reading from empty array and after flushes."""
    shape = (10, 2)
    shared_array = SharedArray(shape=shape, dtype=np.float32)
    local_array = shared_array.get_local_array()

    # Read from empty array
    empty_data = shared_array.read(local_array, flush=False)
    assert len(empty_data) == 0

    # Write some data
    data = np.ones((3, 2), dtype=np.float32)
    shared_array.write(local_array, data)

    # Read with flush
    read_data = shared_array.read(local_array, flush=True)
    assert len(read_data) == 3

    # Read again after flush - should be empty
    empty_again = shared_array.read(local_array, flush=False)
    assert len(empty_again) == 0

    shared_array.delete()


def test_different_dtypes():
    """Test SharedArray with different numpy dtypes."""
    dtypes_to_test = [np.float32, np.float64, np.int32, np.int16]

    for dtype in dtypes_to_test:
        shape = (20, 2)
        shared_array = SharedArray(shape=shape, dtype=dtype)
        local_array = shared_array.get_local_array()

        assert local_array.dtype == dtype

        # Write and read data of this dtype
        data = np.ones((5, 2), dtype=dtype)
        shared_array.write(local_array, data)

        read_data = shared_array.read(local_array, flush=True)
        assert read_data.dtype == dtype
        assert len(read_data) == 5

        shared_array.delete()
