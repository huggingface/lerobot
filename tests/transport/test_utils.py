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
from pickle import UnpicklingError

import pytest
import torch

from lerobot.common.transport import services_pb2
from lerobot.common.transport.utils import (
    CHUNK_SIZE,
    bytes_buffer_size,
    bytes_to_state_dict,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    state_to_bytes,
)
from tests.utils import require_cuda


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


# Tests for state_to_bytes and bytes_to_state_dict
def test_state_to_bytes_empty_dict():
    """Test converting empty state dict to bytes."""
    state_dict = {}
    data = state_to_bytes(state_dict)
    reconstructed = bytes_to_state_dict(data)
    assert reconstructed == state_dict


def test_bytes_to_state_dict_empty_data():
    """Test converting empty data to state dict."""
    with pytest.raises(EOFError):
        bytes_to_state_dict(b"")


def test_state_to_bytes_simple_dict():
    """Test converting simple state dict to bytes."""
    state_dict = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(1, 10),
        "layer2.bias": torch.randn(1),
    }

    data = state_to_bytes(state_dict)
    assert isinstance(data, bytes)
    assert len(data) > 0

    reconstructed = bytes_to_state_dict(data)

    assert len(reconstructed) == len(state_dict)
    for key in state_dict:
        assert key in reconstructed
        assert torch.allclose(state_dict[key], reconstructed[key])


def test_state_to_bytes_various_dtypes():
    """Test converting state dict with various tensor dtypes."""
    state_dict = {
        "float32": torch.randn(5, 5),
        "float64": torch.randn(3, 3).double(),
        "int32": torch.randint(0, 100, (4, 4), dtype=torch.int32),
        "int64": torch.randint(0, 100, (2, 2), dtype=torch.int64),
        "bool": torch.tensor([True, False, True]),
        "uint8": torch.randint(0, 255, (3, 3), dtype=torch.uint8),
    }

    data = state_to_bytes(state_dict)
    reconstructed = bytes_to_state_dict(data)

    for key in state_dict:
        assert reconstructed[key].dtype == state_dict[key].dtype
        if state_dict[key].dtype == torch.bool:
            assert torch.equal(state_dict[key], reconstructed[key])
        else:
            assert torch.allclose(state_dict[key], reconstructed[key])


def test_bytes_to_state_dict_invalid_data():
    """Test bytes_to_state_dict with invalid data."""
    with pytest.raises(UnpicklingError):
        bytes_to_state_dict(b"This is not a valid torch save file")


@require_cuda
def test_state_to_bytes_various_dtypes_cuda():
    """Test converting state dict with various tensor dtypes."""
    state_dict = {
        "float32": torch.randn(5, 5).cuda(),
        "float64": torch.randn(3, 3).double().cuda(),
        "int32": torch.randint(0, 100, (4, 4), dtype=torch.int32).cuda(),
        "int64": torch.randint(0, 100, (2, 2), dtype=torch.int64).cuda(),
        "bool": torch.tensor([True, False, True]),
        "uint8": torch.randint(0, 255, (3, 3), dtype=torch.uint8),
    }

    data = state_to_bytes(state_dict)
    reconstructed = bytes_to_state_dict(data)

    for key in state_dict:
        assert reconstructed[key].dtype == state_dict[key].dtype
        if state_dict[key].dtype == torch.bool:
            assert torch.equal(state_dict[key], reconstructed[key])
        else:
            assert torch.allclose(state_dict[key], reconstructed[key])


# # Tests for python_object_to_bytes and bytes_to_python_object
# def test_python_object_to_bytes_none():
#     """Test converting None to bytes."""
#     obj = None
#     data = python_object_to_bytes(obj)
#     reconstructed = bytes_to_python_object(data)
#     assert reconstructed is None


# def test_python_object_to_bytes_simple_types():
#     """Test converting simple Python types."""
#     test_objects = [
#         42,
#         -123,
#         3.14159,
#         -2.71828,
#         "Hello, World!",
#         "Unicode: 你好世界 🌍",
#         True,
#         False,
#         b"byte string",
#     ]

#     for obj in test_objects:
#         data = python_object_to_bytes(obj)
#         assert isinstance(data, bytes)
#         reconstructed = bytes_to_python_object(data)
#         assert reconstructed == obj
#         assert type(reconstructed) == type(obj)


# def test_python_object_to_bytes_collections():
#     """Test converting Python collections."""
#     test_objects = [
#         [],
#         [1, 2, 3, 4, 5],
#         [1, "two", 3.0, True, None],
#         {},
#         {"key": "value", "number": 123, "nested": {"a": 1}},
#         (),
#         (1, 2, 3),
#         {1, 2, 3, 4, 5},
#         frozenset([1, 2, 3]),
#     ]

#     for obj in test_objects:
#         data = python_object_to_bytes(obj)
#         reconstructed = bytes_to_python_object(data)
#         assert reconstructed == obj
#         assert type(reconstructed) == type(obj)


# def test_python_object_to_bytes_nested_structures():
#     """Test converting deeply nested structures."""
#     obj = {
#         'list': [1, [2, [3, [4, [5]]]]],
#         'dict': {'a': {'b': {'c': {'d': {'e': 'deep'}}}}},
#         'mixed': {
#             'numbers': [1, 2.5, -3],
#             'strings': ["a", "b", "c"],
#             'bools': [True, False, None],
#             'nested_list': [[1, 2], [3, 4], [5, 6]],
#             'nested_dict': {'x': {'y': {'z': 'end'}}},
#         }
#     }

#     data = python_object_to_bytes(obj)
#     reconstructed = bytes_to_python_object(data)
#     assert reconstructed == obj


# def test_python_object_to_bytes_with_tensors():
#     """Test converting objects containing PyTorch tensors."""
#     obj = {
#         'tensor': torch.randn(5, 5),
#         'list_with_tensor': [1, 2, torch.randn(3, 3), "string"],
#         'nested': {
#             'tensor1': torch.randn(2, 2),
#             'tensor2': torch.tensor([1, 2, 3]),
#         }
#     }

#     data = python_object_to_bytes(obj)
#     reconstructed = bytes_to_python_object(data)

#     assert torch.allclose(obj['tensor'], reconstructed['tensor'])
#     assert reconstructed['list_with_tensor'][0] == 1
#     assert reconstructed['list_with_tensor'][3] == "string"
#     assert torch.allclose(obj['list_with_tensor'][2], reconstructed['list_with_tensor'][2])
#     assert torch.allclose(obj['nested']['tensor1'], reconstructed['nested']['tensor1'])
#     assert torch.equal(obj['nested']['tensor2'], reconstructed['nested']['tensor2'])


# def test_python_object_to_bytes_custom_class():
#     """Test converting custom class instances."""
#     class TestClass:
#         def __init__(self, value, name):
#             self.value = value
#             self.name = name
#             self.data = [1, 2, 3]

#         def __eq__(self, other):
#             return (self.value == other.value and
#                    self.name == other.name and
#                    self.data == other.data)

#     obj = TestClass(42, "test")
#     data = python_object_to_bytes(obj)
#     reconstructed = bytes_to_python_object(data)

#     assert reconstructed.value == obj.value
#     assert reconstructed.name == obj.name
#     assert reconstructed.data == obj.data


# def test_python_object_to_bytes_large_object():
#     """Test converting very large objects."""
#     # Create a large object with many elements
#     obj = {
#         f'key_{i}': list(range(100)) for i in range(100)
#     }
#     obj['large_string'] = 'x' * 1000000  # 1MB string

#     data = python_object_to_bytes(obj)
#     reconstructed = bytes_to_python_object(data)

#     assert len(reconstructed) == len(obj)
#     assert reconstructed['key_50'] == list(range(100))
#     assert reconstructed['large_string'] == obj['large_string']


# def test_bytes_to_python_object_invalid_data():
#     """Test bytes_to_python_object with invalid pickle data."""
#     with pytest.raises(Exception):
#         bytes_to_python_object(b"This is not valid pickle data")

#     with pytest.raises(Exception):
#         bytes_to_python_object(b"")


# # Tests for transitions_to_bytes and bytes_to_transitions
# def test_transitions_to_bytes_empty_list():
#     """Test converting empty transitions list."""
#     transitions = []
#     data = transitions_to_bytes(transitions)
#     reconstructed = bytes_to_transitions(data)
#     assert reconstructed == transitions
#     assert isinstance(reconstructed, list)


# def test_transitions_to_bytes_single_transition():
#     """Test converting a single transition."""
#     transition = Transition(
#         observation={'image': torch.randn(3, 64, 64), 'state': torch.randn(10)},
#         action=torch.randn(5),
#         reward=torch.tensor(1.5),
#         done=torch.tensor(False),
#         next_observation={'image': torch.randn(3, 64, 64), 'state': torch.randn(10)},
#     )

#     transitions = [transition]
#     data = transitions_to_bytes(transitions)
#     reconstructed = bytes_to_transitions(data)

#     assert len(reconstructed) == 1
#     assert_transitions_equal(transitions[0], reconstructed[0])


# def test_transitions_to_bytes_multiple_transitions():
#     """Test converting multiple transitions."""
#     transitions = []
#     for i in range(5):
#         transition = Transition(
#             observation={'data': torch.randn(10)},
#             action=torch.randn(3),
#             reward=torch.tensor(float(i)),
#             done=torch.tensor(i == 4),  # Last one is done
#             next_observation={'data': torch.randn(10)},
#         )
#         transitions.append(transition)

#     data = transitions_to_bytes(transitions)
#     reconstructed = bytes_to_transitions(data)

#     assert len(reconstructed) == len(transitions)
#     for original, reconstructed_item in zip(transitions, reconstructed):
#         assert_transitions_equal(original, reconstructed_item)


# def test_transitions_to_bytes_complex_observations():
#     """Test converting transitions with complex observation structures."""
#     transitions = []

#     # Various observation structures
#     observation_types = [
#         {'image': torch.randn(3, 128, 128), 'depth': torch.randn(1, 128, 128), 'state': torch.randn(7)},
#         {'lidar': torch.randn(360), 'imu': torch.randn(6), 'joint_pos': torch.randn(7)},
#         {'rgb': torch.randn(3, 64, 64), 'segmentation': torch.randint(0, 10, (64, 64))},
#     ]

#     for i, obs_type in enumerate(observation_types):
#         transition = Transition(
#             observation=obs_type.copy(),
#             action=torch.randn(4),
#             reward=torch.tensor(float(i)),
#             done=torch.tensor(False),
#             next_observation=obs_type.copy(),
#         )
#         transitions.append(transition)

#     data = transitions_to_bytes(transitions)
#     reconstructed = bytes_to_transitions(data)

#     assert len(reconstructed) == len(transitions)
#     for original, reconstructed_item in zip(transitions, reconstructed):
#         assert_transitions_equal(original, reconstructed_item)


# def test_transitions_to_bytes_large_batch():
#     """Test converting a large batch of transitions."""
#     transitions = []
#     for i in range(100):
#         transition = Transition(
#             observation={'state': torch.randn(20)},
#             action=torch.randn(6),
#             reward=torch.tensor(float(i % 10) / 10),
#             done=torch.tensor(i % 20 == 0),
#             next_observation={'state': torch.randn(20)},
#         )
#         transitions.append(transition)

#     data = transitions_to_bytes(transitions)
#     reconstructed = bytes_to_transitions(data)

#     assert len(reconstructed) == 100
#     # Spot check some transitions
#     for i in [0, 25, 50, 75, 99]:
#         assert_transitions_equal(transitions[i], reconstructed[i])


# def test_bytes_to_transitions_invalid_data():
#     """Test bytes_to_transitions with invalid data."""
#     with pytest.raises(Exception):
#         bytes_to_transitions(b"This is not valid transition data")

#     with pytest.raises(Exception):
#         bytes_to_transitions(b"")


# def test_transitions_with_different_dtypes():
#     """Test transitions with various tensor dtypes."""
#     transition = Transition(
#         observation={
#             'float32': torch.randn(5),
#             'int64': torch.randint(0, 100, (3,)),
#             'bool': torch.tensor([True, False, True]),
#         },
#         action=torch.randn(2, dtype=torch.float64),
#         reward=torch.tensor(1.0, dtype=torch.float32),
#         done=torch.tensor(True),
#         next_observation={
#             'float32': torch.randn(5),
#             'int64': torch.randint(0, 100, (3,)),
#             'bool': torch.tensor([False, True, False]),
#         },
#     )

#     transitions = [transition]
#     data = transitions_to_bytes(transitions)
#     reconstructed = bytes_to_transitions(data)

#     assert len(reconstructed) == 1

#     # Check dtypes are preserved
#     assert reconstructed[0].observation['float32'].dtype == torch.float32
#     assert reconstructed[0].observation['int64'].dtype == torch.int64
#     assert reconstructed[0].observation['bool'].dtype == torch.bool
#     assert reconstructed[0].action.dtype == torch.float64
#     assert reconstructed[0].reward.dtype == torch.float32
#     assert reconstructed[0].done.dtype == torch.bool


# # Helper function for transition comparison
# def assert_transitions_equal(t1: Transition, t2: Transition):
#     """Helper to assert two transitions are equal."""
#     # Check observations
#     assert set(t1.observation.keys()) == set(t2.observation.keys())
#     for key in t1.observation:
#         if t1.observation[key].dtype == torch.bool:
#             assert torch.equal(t1.observation[key], t2.observation[key])
#         else:
#             assert torch.allclose(t1.observation[key], t2.observation[key])

#     # Check actions
#     if t1.action.dtype == torch.bool:
#         assert torch.equal(t1.action, t2.action)
#     else:
#         assert torch.allclose(t1.action, t2.action)

#     # Check rewards
#     assert torch.allclose(t1.reward, t2.reward)

#     # Check done flags
#     assert torch.equal(t1.done, t2.done)

#     # Check next observations
#     assert set(t1.next_observation.keys()) == set(t2.next_observation.keys())
#     for key in t1.next_observation:
#         if t1.next_observation[key].dtype == torch.bool:
#             assert torch.equal(t1.next_observation[key], t2.next_observation[key])
#         else:
#             assert torch.allclose(t1.next_observation[key], t2.next_observation[key])


# # Integration tests
# def test_state_dict_through_chunks():
#     """Test sending state dict through chunking system."""
#     state_dict = {
#         'model.layer1.weight': torch.randn(100, 50),
#         'model.layer1.bias': torch.randn(100),
#         'model.layer2.weight': torch.randn(10, 100),
#         'model.layer2.bias': torch.randn(10),
#     }

#     # Convert to bytes
#     data = state_to_bytes(state_dict)

#     # Send through chunks
#     chunks = list(send_bytes_in_chunks(data, services_pb2.InteractionMessage))

#     # Simulate receiving
#     queue = Queue()
#     shutdown_event = Event()
#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

#     # Get received data and convert back
#     received_data = queue.get(timeout=0.1)
#     reconstructed = bytes_to_state_dict(received_data)

#     # Verify
#     assert len(reconstructed) == len(state_dict)
#     for key in state_dict:
#         assert torch.allclose(state_dict[key], reconstructed[key])


# def test_transitions_through_chunks():
#     """Test sending transitions through chunking system."""
#     transitions = []
#     for i in range(10):
#         transition = Transition(
#             observation={'image': torch.randn(3, 32, 32), 'state': torch.randn(5)},
#             action=torch.randn(3),
#             reward=torch.tensor(float(i)),
#             done=torch.tensor(i == 9),
#             next_observation={'image': torch.randn(3, 32, 32), 'state': torch.randn(5)},
#         )
#         transitions.append(transition)

#     # Convert to bytes
#     data = transitions_to_bytes(transitions)

#     # Send through chunks
#     chunks = list(send_bytes_in_chunks(data, services_pb2.Transition))

#     # Simulate receiving
#     queue = Queue()
#     shutdown_event = Event()
#     receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

#     # Get received data and convert back
#     received_data = queue.get(timeout=0.1)
#     reconstructed = bytes_to_transitions(received_data)

#     # Verify
#     assert len(reconstructed) == len(transitions)
#     for original, reconstructed_item in zip(transitions, reconstructed):
#         assert_transitions_equal(original, reconstructed_item)
