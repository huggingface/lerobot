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

    assert not queue.empty()
    assert queue.get() == data


# def test_receive_bytes_in_chunks_multiple_chunks():
#     """Test receiving a multi-chunk message."""
#     queue = Queue()
#     shutdown_event = Event()

#     chunks = [
#         create_chunk(b"First ", services_pb2.TransferState.TRANSFER_BEGIN),
#         create_chunk(b"Middle ", services_pb2.TransferState.TRANSFER_MIDDLE),
#         create_chunk(b"Last", services_pb2.TransferState.TRANSFER_END),
#     ]

#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

#     assert not queue.empty()
#     assert queue.get() == b"First Middle Last"
#     assert queue.empty()


# def test_receive_bytes_in_chunks_large_data():
#     """Test receiving large data split into multiple chunks."""
#     queue = Queue()
#     shutdown_event = Event()

#     # Create chunks that simulate what send_bytes_in_chunks would produce
#     chunks = [
#         create_chunk(b"x" * CHUNK_SIZE, services_pb2.TransferState.TRANSFER_BEGIN),
#         create_chunk(b"x" * CHUNK_SIZE, services_pb2.TransferState.TRANSFER_MIDDLE),
#         create_chunk(b"x" * 1000, services_pb2.TransferState.TRANSFER_END),
#     ]

#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

#     assert not queue.empty()
#     received_data = queue.get()
#     assert received_data == b"x" * (CHUNK_SIZE * 2 + 1000)
#     assert queue.empty()


# def test_receive_bytes_in_chunks_exact_chunk_size():
#     """Test receiving data that is exactly one chunk size."""
#     queue = Queue()
#     shutdown_event = Event()

#     data = b"y" * CHUNK_SIZE
#     chunks = [
#         create_chunk(data, services_pb2.TransferState.TRANSFER_END)
#     ]

#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

#     assert not queue.empty()
#     assert queue.get() == data
#     assert queue.empty()


# def test_receive_bytes_in_chunks_multiple_messages():
#     """Test receiving multiple complete messages in sequence."""
#     queue = Queue()
#     shutdown_event = Event()

#     chunks = [
#         # First message - single chunk
#         create_chunk(b"Message1", services_pb2.TransferState.TRANSFER_END),
#         # Second message - multi chunk
#         create_chunk(b"Start2 ", services_pb2.TransferState.TRANSFER_BEGIN),
#         create_chunk(b"Middle2 ", services_pb2.TransferState.TRANSFER_MIDDLE),
#         create_chunk(b"End2", services_pb2.TransferState.TRANSFER_END),
#         # Third message - single chunk
#         create_chunk(b"Message3", services_pb2.TransferState.TRANSFER_END),
#     ]

#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

#     # Should have three messages in queue
#     assert queue.get() == b"Message1"
#     assert queue.get() == b"Start2 Middle2 End2"
#     assert queue.get() == b"Message3"
#     assert queue.empty()


# def test_receive_bytes_in_chunks_many_middle_chunks():
#     """Test receiving a message with many middle chunks."""
#     queue = Queue()
#     shutdown_event = Event()

#     chunks = [
#         create_chunk(b"Start", services_pb2.TransferState.TRANSFER_BEGIN),
#     ]
#     # Add many middle chunks
#     for i in range(100):
#         chunks.append(
#             create_chunk(f" Part{i}".encode(), services_pb2.TransferState.TRANSFER_MIDDLE)
#         )
#     chunks.append(
#         create_chunk(b" End", services_pb2.TransferState.TRANSFER_END)
#     )

#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

#     result = queue.get()
#     expected = b"Start" + b"".join(f" Part{i}".encode() for i in range(100)) + b" End"
#     assert result == expected
#     assert queue.empty()


# def test_receive_bytes_in_chunks_shutdown_during_receive():
#     """Test that shutdown event stops receiving mid-stream."""
#     queue = Queue()
#     shutdown_event = Event()

#     def delayed_shutdown():
#         time.sleep(0.1)
#         shutdown_event.set()

#     # Create a generator that yields chunks slowly
#     def chunk_generator():
#         yield create_chunk(b"Data", services_pb2.TransferState.TRANSFER_BEGIN)
#         while True:
#             time.sleep(0.05)
#             if shutdown_event.is_set():
#                 break
#             yield create_chunk(b"More", services_pb2.TransferState.TRANSFER_MIDDLE)

#     shutdown_thread = threading.Thread(target=delayed_shutdown)
#     shutdown_thread.start()

#     receive_bytes_in_chunks(chunk_generator(), queue, shutdown_event, "TEST")

#     shutdown_thread.join()

#     # Should have stopped without completing message
#     assert queue.empty()


# def test_receive_bytes_in_chunks_immediate_shutdown():
#     """Test receiving with shutdown event already set."""
#     queue = Queue()
#     shutdown_event = Event()
#     shutdown_event.set()  # Set before starting

#     chunks = [
#         create_chunk(b"Data", services_pb2.TransferState.TRANSFER_BEGIN),
#         create_chunk(b"More", services_pb2.TransferState.TRANSFER_END),
#     ]

#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

#     # Should stop immediately without processing any chunks
#     assert queue.empty()


# def test_receive_bytes_in_chunks_only_begin_chunk():
#     """Test receiving only a BEGIN chunk without END."""
#     queue = Queue()
#     shutdown_event = Event()

#     chunks = [
#         create_chunk(b"Start", services_pb2.TransferState.TRANSFER_BEGIN),
#         # No END chunk
#     ]

#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

#     # Should not have completed message
#     assert queue.empty()


# def test_receive_bytes_in_chunks_only_middle_chunks():
#     """Test receiving only MIDDLE chunks without BEGIN or END."""
#     queue = Queue()
#     shutdown_event = Event()

#     chunks = [
#         create_chunk(b"Middle1", services_pb2.TransferState.TRANSFER_MIDDLE),
#         create_chunk(b"Middle2", services_pb2.TransferState.TRANSFER_MIDDLE),
#     ]

#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

#     # Should not have any complete messages
#     assert queue.empty()

#     # def test_send_receive_integration(self):
#     #     """Test sending and receiving data through chunks."""
#     #     # Create test data
#     #     state_dict = {
#     #         "model.weight": torch.randn(100, 100),
#     #         "model.bias": torch.randn(100),
#     #     }

# def test_receive_bytes_in_chunks_missing_begin():
#     """Test receiving chunks starting with MIDDLE instead of BEGIN."""
#     queue = Queue()
#     shutdown_event = Event()

#     chunks = [
#         # Missing BEGIN
#         create_chunk(b"Middle", services_pb2.TransferState.TRANSFER_MIDDLE),
#         create_chunk(b"End", services_pb2.TransferState.TRANSFER_END),
#     ]

#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

#     # The implementation continues from where it is, so we should get partial data
#     assert not queue.empty()
#     assert queue.get() == b"MiddleEnd"


# def test_receive_bytes_in_chunks_interleaved_messages():
#     """Test receiving a new BEGIN before previous message completes."""
#     queue = Queue()
#     shutdown_event = Event()

#     chunks = [
#         create_chunk(b"Start1", services_pb2.TransferState.TRANSFER_BEGIN),
#         create_chunk(b"Middle1", services_pb2.TransferState.TRANSFER_MIDDLE),
#         # New message starts before first one ends
#         create_chunk(b"Start2", services_pb2.TransferState.TRANSFER_BEGIN),
#         create_chunk(b"End2", services_pb2.TransferState.TRANSFER_END),
#     ]

#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

#     # Should only have the second complete message
#     assert not queue.empty()
#     assert queue.get() == b"Start2End2"
#     assert queue.empty()
