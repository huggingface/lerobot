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

import io
from multiprocessing import Event, Queue

from lerobot.common.transport import services_pb2
from lerobot.common.transport.utils import (
    CHUNK_SIZE,
    bytes_buffer_size,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
)


def test_bytes_buffer_size_empty_buffer():
    """Test with an empty buffer."""
    buffer = io.BytesIO()
    assert bytes_buffer_size(buffer) == 0
    # Ensure position is reset to beginning
    assert buffer.tell() == 0


def test_bytes_buffer_size_small_buffer():
    """Test with a small buffer."""
    buffer = io.BytesIO(b"Hello, World!")
    assert bytes_buffer_size(buffer) == 13
    assert buffer.tell() == 0


def test_bytes_buffer_size_large_buffer():
    """Test with a large buffer."""
    data = b"x" * (CHUNK_SIZE * 2 + 1000)
    buffer = io.BytesIO(data)
    assert bytes_buffer_size(buffer) == len(data)
    assert buffer.tell() == 0


def test_send_bytes_in_chunks_empty_data():
    """Test sending empty data."""
    message_class = services_pb2.InteractionMessage
    chunks = list(send_bytes_in_chunks(b"", message_class))
    assert len(chunks) == 0


def test_single_chunk_small_data():
    """Test data that fits in a single chunk."""
    data = b"Some data"
    message_class = services_pb2.InteractionMessage
    chunks = list(send_bytes_in_chunks(data, message_class))

    assert len(chunks) == 1
    assert chunks[0].data == b"Some data"
    assert chunks[0].transfer_state == services_pb2.TransferState.TRANSFER_END


def test_not_silent_mode():
    """Test not silent mode."""
    data = b"Some data"
    message_class = services_pb2.InteractionMessage
    chunks = list(send_bytes_in_chunks(data, message_class, silent=False))
    assert len(chunks) == 1
    assert chunks[0].data == b"Some data"


def test_send_bytes_in_chunks_large_data():
    """Test sending large data."""
    data = b"x" * (CHUNK_SIZE * 2 + 1000)
    message_class = services_pb2.InteractionMessage
    chunks = list(send_bytes_in_chunks(data, message_class))
    assert len(chunks) == 3
    assert chunks[0].data == b"x" * CHUNK_SIZE
    assert chunks[0].transfer_state == services_pb2.TransferState.TRANSFER_BEGIN
    assert chunks[1].data == b"x" * CHUNK_SIZE
    assert chunks[1].transfer_state == services_pb2.TransferState.TRANSFER_MIDDLE
    assert chunks[2].data == b"x" * 1000
    assert chunks[2].transfer_state == services_pb2.TransferState.TRANSFER_END


def test_send_bytes_in_chunks_large_data_with_exact_chunk_size():
    """Test sending large data with exact chunk size."""
    data = b"x" * CHUNK_SIZE
    message_class = services_pb2.InteractionMessage
    chunks = list(send_bytes_in_chunks(data, message_class))
    assert len(chunks) == 1
    assert chunks[0].data == data
    assert chunks[0].transfer_state == services_pb2.TransferState.TRANSFER_END


def test_receive_bytes_in_chunks_empty_data():
    """Test receiving empty data."""
    queue = Queue()
    shutdown_event = Event()

    # Empty iterator
    receive_bytes_in_chunks(iter([]), queue, shutdown_event)

    assert queue.empty()


def test_receive_bytes_in_chunks_single_chunk():
    """Test receiving a single chunk message."""
    queue = Queue()
    shutdown_event = Event()

    data = b"Single chunk data"
    chunks = [
        services_pb2.InteractionMessage(data=data, transfer_state=services_pb2.TransferState.TRANSFER_END)
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    assert queue.get(timeout=0.01) == data
    assert queue.empty()


def test_receive_bytes_in_chunks_single_not_end_chunk():
    """Test receiving a single chunk message."""
    queue = Queue()
    shutdown_event = Event()

    data = b"Single chunk data"
    chunks = [
        services_pb2.InteractionMessage(data=data, transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE)
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    assert queue.empty()


def test_receive_bytes_in_chunks_multiple_chunks():
    """Test receiving a multi-chunk message."""
    queue = Queue()
    shutdown_event = Event()

    chunks = [
        services_pb2.InteractionMessage(
            data=b"First ", transfer_state=services_pb2.TransferState.TRANSFER_BEGIN
        ),
        services_pb2.InteractionMessage(
            data=b"Middle ", transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE
        ),
        services_pb2.InteractionMessage(data=b"Last", transfer_state=services_pb2.TransferState.TRANSFER_END),
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    assert queue.get(timeout=0.01) == b"First Middle Last"
    assert queue.empty()


def test_receive_bytes_in_chunks_multiple_messages():
    """Test receiving multiple complete messages in sequence."""
    queue = Queue()
    shutdown_event = Event()

    chunks = [
        # First message - single chunk
        services_pb2.InteractionMessage(
            data=b"Message1", transfer_state=services_pb2.TransferState.TRANSFER_END
        ),
        # Second message - multi chunk
        services_pb2.InteractionMessage(
            data=b"Start2 ", transfer_state=services_pb2.TransferState.TRANSFER_BEGIN
        ),
        services_pb2.InteractionMessage(
            data=b"Middle2 ", transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE
        ),
        services_pb2.InteractionMessage(data=b"End2", transfer_state=services_pb2.TransferState.TRANSFER_END),
        # Third message - single chunk
        services_pb2.InteractionMessage(
            data=b"Message3", transfer_state=services_pb2.TransferState.TRANSFER_END
        ),
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    # Should have three messages in queue
    assert queue.get(timeout=0.01) == b"Message1"
    assert queue.get(timeout=0.01) == b"Start2 Middle2 End2"
    assert queue.get(timeout=0.01) == b"Message3"
    assert queue.empty()


def test_receive_bytes_in_chunks_shutdown_during_receive():
    """Test that shutdown event stops receiving mid-stream."""
    queue = Queue()
    shutdown_event = Event()
    shutdown_event.set()

    chunks = [
        services_pb2.InteractionMessage(
            data=b"First ", transfer_state=services_pb2.TransferState.TRANSFER_BEGIN
        ),
        services_pb2.InteractionMessage(
            data=b"Middle ", transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE
        ),
        services_pb2.InteractionMessage(data=b"Last", transfer_state=services_pb2.TransferState.TRANSFER_END),
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    assert queue.empty()


def test_receive_bytes_in_chunks_only_begin_chunk():
    """Test receiving only a BEGIN chunk without END."""
    queue = Queue()
    shutdown_event = Event()

    chunks = [
        services_pb2.InteractionMessage(
            data=b"Start", transfer_state=services_pb2.TransferState.TRANSFER_BEGIN
        ),
        # No END chunk
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    assert queue.empty()


def test_receive_bytes_in_chunks_missing_begin():
    """Test receiving chunks starting with MIDDLE instead of BEGIN."""
    queue = Queue()
    shutdown_event = Event()

    chunks = [
        # Missing BEGIN
        services_pb2.InteractionMessage(
            data=b"Middle", transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE
        ),
        services_pb2.InteractionMessage(data=b"End", transfer_state=services_pb2.TransferState.TRANSFER_END),
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    # The implementation continues from where it is, so we should get partial data
    assert queue.get(timeout=0.01) == b"MiddleEnd"
    assert queue.empty()
