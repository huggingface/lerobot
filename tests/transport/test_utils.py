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

from lerobot.common.transport.utils import (
    CHUNK_SIZE,
    bytes_buffer_size,
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

    # class TestSendBytesInChunks:
    #     """Test cases for send_bytes_in_chunks function."""

    #     @pytest.fixture
    #     def mock_message_class(self):
    #         """Create a mock message class for testing."""
    #         return Mock()

    #     def test_empty_data(self, mock_message_class):
    #         """Test sending empty data."""
    #         chunks = list(send_bytes_in_chunks(b"", mock_message_class))
    #         assert len(chunks) == 0

    #     def test_single_chunk_small_data(self, mock_message_class):
    #         """Test data that fits in a single chunk."""
    #         data = b"Small data"
    #         chunks = list(send_bytes_in_chunks(data, mock_message_class))

    #         assert len(chunks) == 1
    #         assert chunks[0] == mock_message_class.return_value

    #         # Check the mock was called correctly
    #         mock_message_class.assert_called_once_with(
    #             transfer_state=services_pb2.TransferState.TRANSFER_END,
    #             data=data
    #         )

    #     def test_exact_chunk_size_data(self, mock_message_class):
    #         """Test data that is exactly CHUNK_SIZE."""
    #         data = b"x" * CHUNK_SIZE
    #         chunks = list(send_bytes_in_chunks(data, mock_message_class))

    #         assert len(chunks) == 1
    #         mock_message_class.assert_called_with(
    #             transfer_state=services_pb2.TransferState.TRANSFER_END,
    #             data=data
    #         )

    #     def test_multiple_chunks(self, mock_message_class):
    #         """Test data that requires multiple chunks."""
    #         # Create data that will require 3 chunks
    #         data = b"x" * (CHUNK_SIZE * 2 + 1000)
    #         chunks = list(send_bytes_in_chunks(data, mock_message_class))

    #         assert len(chunks) == 3

    #         # Verify transfer states
    #         calls = mock_message_class.call_args_list
    #         assert calls[0][1]['transfer_state'] == services_pb2.TransferState.TRANSFER_BEGIN
    #         assert calls[1][1]['transfer_state'] == services_pb2.TransferState.TRANSFER_MIDDLE
    #         assert calls[2][1]['transfer_state'] == services_pb2.TransferState.TRANSFER_END

    #         # Verify data integrity
    #         reconstructed = b"".join(call[1]['data'] for call in calls)
    #         assert reconstructed == data

    #     def test_logging_silent_mode(self, mock_message_class, caplog):
    #         """Test logging in silent mode."""
    #         with caplog.at_level(logging.DEBUG):
    #             data = b"Test data"
    #             list(send_bytes_in_chunks(data, mock_message_class, log_prefix="TEST", silent=True))

    #             # Should use debug level
    #             assert any("TEST" in record.message for record in caplog.records)
    #             assert all(record.levelno == logging.DEBUG for record in caplog.records if "TEST" in record.message)

    #     def test_logging_verbose_mode(self, mock_message_class, caplog):
    #         """Test logging in verbose mode."""
    #         with caplog.at_level(logging.INFO):
    #             data = b"Test data"
    #             list(send_bytes_in_chunks(data, mock_message_class, log_prefix="TEST", silent=False))

    #             # Should use info level
    #             assert any("TEST" in record.message for record in caplog.records)
    #             assert any(record.levelno == logging.INFO for record in caplog.records if "TEST" in record.message)

    # class TestReceiveBytesInChunks:
    #     """Test cases for receive_bytes_in_chunks function."""

    #     @pytest.fixture
    #     def queue(self):
    #         """Create a queue for testing."""
    #         return Queue()

    #     @pytest.fixture
    #     def shutdown_event(self):
    #         """Create a shutdown event for testing."""
    #         return Event()

    #     def create_chunk(self, data: bytes, transfer_state: int) -> Mock:
    #         """Helper to create a chunk with data and transfer state."""
    #         chunk = Mock()
    #         chunk.data = data
    #         chunk.transfer_state = transfer_state
    #         return chunk

    #     def test_single_chunk_message(self, queue, shutdown_event):
    #         """Test receiving a single chunk message."""
    #         data = b"Single chunk data"
    #         chunks = [
    #             self.create_chunk(data, services_pb2.TransferState.TRANSFER_END)
    #         ]

    #         receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

    #         assert not queue.empty()
    #         assert queue.get() == data

    #     def test_multiple_chunks_message(self, queue, shutdown_event):
    #         """Test receiving a multi-chunk message."""
    #         chunks = [
    #             self.create_chunk(b"First ", services_pb2.TransferState.TRANSFER_BEGIN),
    #             self.create_chunk(b"Middle ", services_pb2.TransferState.TRANSFER_MIDDLE),
    #             self.create_chunk(b"Last", services_pb2.TransferState.TRANSFER_END),
    #         ]

    #         receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

    #         assert not queue.empty()
    #         assert queue.get() == b"First Middle Last"

    #     def test_multiple_messages(self, queue, shutdown_event):
    #         """Test receiving multiple complete messages."""
    #         chunks = [
    #             # First message
    #             self.create_chunk(b"Message1", services_pb2.TransferState.TRANSFER_END),
    #             # Second message
    #             self.create_chunk(b"Start2 ", services_pb2.TransferState.TRANSFER_BEGIN),
    #             self.create_chunk(b"End2", services_pb2.TransferState.TRANSFER_END),
    #         ]

    #         receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

    #         # Should have two messages in queue
    #         assert queue.get() == b"Message1"
    #         assert queue.get() == b"Start2 End2"
    #         assert queue.empty()

    #     def test_shutdown_during_receive(self, queue, shutdown_event):
    #         """Test shutdown event stops receiving."""
    #         def delayed_shutdown():
    #             time.sleep(0.1)
    #             shutdown_event.set()

    #         # Create an infinite iterator
    #         def chunk_generator():
    #             yield self.create_chunk(b"Data", services_pb2.TransferState.TRANSFER_BEGIN)
    #             while True:
    #                 time.sleep(0.05)
    #                 if shutdown_event.is_set():
    #                     break
    #                 yield self.create_chunk(b"More", services_pb2.TransferState.TRANSFER_MIDDLE)

    #         shutdown_thread = threading.Thread(target=delayed_shutdown)
    #         shutdown_thread.start()

    #         receive_bytes_in_chunks(chunk_generator(), queue, shutdown_event, "TEST")

    #         shutdown_thread.join()

    #         # Should have stopped without completing message
    #         assert queue.empty()

    #     def test_empty_iterator(self, queue, shutdown_event):
    #         """Test with empty iterator."""
    #         receive_bytes_in_chunks(iter([]), queue, shutdown_event, "TEST")
    #         assert queue.empty()

    #     def test_multiple_middle_chunks(self, queue, shutdown_event):
    #         """Test with many middle chunks."""
    #         chunks = [
    #             self.create_chunk(b"Start", services_pb2.TransferState.TRANSFER_BEGIN),
    #         ]
    #         # Add many middle chunks
    #         for i in range(10):
    #             chunks.append(
    #                 self.create_chunk(f" Part{i}".encode(), services_pb2.TransferState.TRANSFER_MIDDLE)
    #             )
    #         chunks.append(
    #             self.create_chunk(b" End", services_pb2.TransferState.TRANSFER_END)
    #         )

    #         receive_bytes_in_chunks(iter(chunks), queue, shutdown_event, "TEST")

    #         result = queue.get()
    #         expected = b"Start" + b"".join(f" Part{i}".encode() for i in range(10)) + b" End"
    #         assert result == expected

    # class TestStateDictConversion:
    #     """Test cases for state_to_bytes and bytes_to_state_dict functions."""

    #     def test_empty_state_dict(self):
    #         """Test conversion of empty state dict."""
    #         state_dict = {}
    #         data = state_to_bytes(state_dict)
    #         reconstructed = bytes_to_state_dict(data)
    #         assert reconstructed == state_dict

    #     def test_simple_state_dict(self):
    #         """Test conversion of simple state dict."""
    #         state_dict = {
    #             'layer1.weight': torch.randn(10, 5),
    #             'layer1.bias': torch.randn(10),
    #             'layer2.weight': torch.randn(1, 10),
    #             'layer2.bias': torch.randn(1),
    #         }

    #         data = state_to_bytes(state_dict)
    #         reconstructed = bytes_to_state_dict(data)

    #         assert len(reconstructed) == len(state_dict)
    #         for key in state_dict:
    #             assert key in reconstructed
    #             assert torch.allclose(state_dict[key], reconstructed[key])

    #     def test_large_state_dict(self):
    #         """Test conversion of large state dict."""
    #         state_dict = {
    #             f'layer{i}.weight': torch.randn(100, 100)
    #             for i in range(50)
    #         }

    #         data = state_to_bytes(state_dict)
    #         reconstructed = bytes_to_state_dict(data)

    #         assert len(reconstructed) == len(state_dict)
    #         for key in state_dict:
    #             assert torch.allclose(state_dict[key], reconstructed[key])

    #     def test_various_tensor_types(self):
    #         """Test with various tensor types and shapes."""
    #         state_dict = {
    #             'float32': torch.randn(5, 5),
    #             'float64': torch.randn(3, 3).double(),
    #             'int32': torch.randint(0, 100, (4, 4)),
    #             'bool': torch.tensor([True, False, True]),
    #             '1d': torch.randn(100),
    #             '3d': torch.randn(10, 20, 30),
    #             '4d': torch.randn(2, 3, 4, 5),
    #             'scalar': torch.tensor(42.0),
    #         }

    #         data = state_to_bytes(state_dict)
    #         reconstructed = bytes_to_state_dict(data)

    #         for key in state_dict:
    #             if state_dict[key].dtype == torch.bool:
    #                 assert torch.equal(state_dict[key], reconstructed[key])
    #             else:
    #                 assert torch.allclose(state_dict[key], reconstructed[key])

    #     def test_corrupted_data(self):
    #         """Test with corrupted data."""
    #         with pytest.raises(Exception):
    #             bytes_to_state_dict(b"This is not a valid torch save")

    # class TestPythonObjectConversion:
    #     """Test cases for python_object_to_bytes and bytes_to_python_object functions."""

    #     def test_none_object(self):
    #         """Test conversion of None."""
    #         obj = None
    #         data = python_object_to_bytes(obj)
    #         reconstructed = bytes_to_python_object(data)
    #         assert reconstructed is None

    #     def test_simple_types(self):
    #         """Test conversion of simple Python types."""
    #         test_objects = [
    #             42,
    #             3.14,
    #             "Hello, World!",
    #             True,
    #             False,
    #             [1, 2, 3],
    #             (4, 5, 6),
    #             {"key": "value", "number": 123},
    #             {1, 2, 3, 4, 5},
    #         ]

    #         for obj in test_objects:
    #             data = python_object_to_bytes(obj)
    #             reconstructed = bytes_to_python_object(data)
    #             assert reconstructed == obj
    #             assert type(reconstructed) == type(obj)

    #     def test_nested_structures(self):
    #         """Test conversion of nested data structures."""
    #         obj = {
    #             'list': [1, [2, 3], {'nested': True}],
    #             'dict': {'a': 1, 'b': {'c': 2}},
    #             'tuple': (1, (2, 3), [4, 5]),
    #             'mixed': {
    #                 'tensors': torch.randn(3, 3),
    #                 'strings': ["a", "b", "c"],
    #                 'numbers': [1, 2.5, -3],
    #             }
    #         }

    #         data = python_object_to_bytes(obj)
    #         reconstructed = bytes_to_python_object(data)

    #         assert reconstructed['list'] == obj['list']
    #         assert reconstructed['dict'] == obj['dict']
    #         assert reconstructed['tuple'] == obj['tuple']
    #         assert torch.allclose(reconstructed['mixed']['tensors'], obj['mixed']['tensors'])

    #     def test_custom_class(self):
    #         """Test conversion of custom class instances."""
    #         class TestClass:
    #             def __init__(self, value):
    #                 self.value = value
    #                 self.tensor = torch.randn(2, 2)

    #             def __eq__(self, other):
    #                 return (self.value == other.value and
    #                         torch.allclose(self.tensor, other.tensor))

    #         obj = TestClass(42)
    #         data = python_object_to_bytes(obj)
    #         reconstructed = bytes_to_python_object(data)

    #         assert reconstructed.value == obj.value
    #         assert torch.allclose(reconstructed.tensor, obj.tensor)

    #     def test_large_object(self):
    #         """Test conversion of large objects."""
    #         obj = {
    #             f'key_{i}': list(range(1000))
    #             for i in range(100)
    #         }

    #         data = python_object_to_bytes(obj)
    #         reconstructed = bytes_to_python_object(data)
    #         assert reconstructed == obj

    #     def test_corrupted_pickle_data(self):
    #         """Test with corrupted pickle data."""
    #         with pytest.raises(Exception):
    #             bytes_to_python_object(b"Not valid pickle data")

    # class TestTransitionConversion:
    #     """Test cases for transitions_to_bytes and bytes_to_transitions functions."""

    #     def create_transition(self, index: int) -> Transition:
    #         """Helper to create a test transition."""
    #         return Transition(
    #             observation={'image': torch.randn(3, 64, 64), 'state': torch.randn(10)},
    #             action=torch.randn(5),
    #             reward=torch.tensor(float(index)),
    #             done=torch.tensor(index % 2 == 0),
    #             next_observation={'image': torch.randn(3, 64, 64), 'state': torch.randn(10)},
    #         )

    #     def test_empty_transitions_list(self):
    #         """Test conversion of empty transitions list."""
    #         transitions = []
    #         data = transitions_to_bytes(transitions)
    #         reconstructed = bytes_to_transitions(data)
    #         assert reconstructed == transitions

    #     def test_single_transition(self):
    #         """Test conversion of single transition."""
    #         transitions = [self.create_transition(0)]
    #         data = transitions_to_bytes(transitions)
    #         reconstructed = bytes_to_transitions(data)

    #         assert len(reconstructed) == 1
    #         self.assert_transitions_equal(transitions[0], reconstructed[0])

    #     def test_multiple_transitions(self):
    #         """Test conversion of multiple transitions."""
    #         transitions = [self.create_transition(i) for i in range(10)]
    #         data = transitions_to_bytes(transitions)
    #         reconstructed = bytes_to_transitions(data)

    #         assert len(reconstructed) == len(transitions)
    #         for original, reconstructed_item in zip(transitions, reconstructed):
    #             self.assert_transitions_equal(original, reconstructed_item)

    #     def test_large_transitions_list(self):
    #         """Test conversion of large transitions list."""
    #         transitions = [self.create_transition(i) for i in range(100)]
    #         data = transitions_to_bytes(transitions)
    #         reconstructed = bytes_to_transitions(data)

    #         assert len(reconstructed) == len(transitions)
    #         # Spot check a few
    #         for i in [0, 50, 99]:
    #             self.assert_transitions_equal(transitions[i], reconstructed[i])

    #     def test_transitions_with_different_observation_shapes(self):
    #         """Test transitions with varying observation shapes."""
    #         transitions = [
    #             Transition(
    #                 observation={'data': torch.randn(*shape)},
    #                 action=torch.randn(3),
    #                 reward=torch.tensor(1.0),
    #                 done=torch.tensor(False),
    #                 next_observation={'data': torch.randn(*shape)},
    #             )
    #             for shape in [(10,), (5, 5), (2, 3, 4)]
    #         ]

    #         data = transitions_to_bytes(transitions)
    #         reconstructed = bytes_to_transitions(data)

    #         assert len(reconstructed) == len(transitions)
    #         for original, reconstructed_item in zip(transitions, reconstructed):
    #             assert original.observation['data'].shape == reconstructed_item.observation['data'].shape

    #     def test_corrupted_transition_data(self):
    #         """Test with corrupted transition data."""
    #         with pytest.raises(Exception):
    #             bytes_to_transitions(b"Invalid transition data")

    #     def assert_transitions_equal(self, t1: Transition, t2: Transition):
    #         """Helper to assert two transitions are equal."""
    #         # Check observations
    #         for key in t1.observation:
    #             assert key in t2.observation
    #             assert torch.allclose(t1.observation[key], t2.observation[key])

    #         # Check actions
    #         assert torch.allclose(t1.action, t2.action)

    #         # Check rewards
    #         assert torch.allclose(t1.reward, t2.reward)

    #         # Check done flags
    #         assert torch.equal(t1.done, t2.done)

    #         # Check next observations
    #         for key in t1.next_observation:
    #             assert key in t2.next_observation
    #             assert torch.allclose(t1.next_observation[key], t2.next_observation[key])

    # class TestIntegration:
    # """Integration tests combining multiple functions."""

    # def test_send_receive_integration(self):
    #     """Test sending and receiving data through chunks."""
    #     # Create test data
    #     state_dict = {
    #         "model.weight": torch.randn(100, 100),
    #         "model.bias": torch.randn(100),
    #     }

    #     # Convert to bytes
    #     data = state_to_bytes(state_dict)

    #     # Mock message class
    #     mock_message_class = Mock()

    #     # Send through chunks
    #     chunks = list(send_bytes_in_chunks(data, mock_message_class))

    #     # Prepare chunks for receiving (extract data from mock calls)
    #     receive_chunks = []
    #     for call in mock_message_class.call_args_list:
    #         chunk = Mock()
    #         chunk.data = call[1]["data"]
    #         chunk.transfer_state = call[1]["transfer_state"]
    #         receive_chunks.append(chunk)

    #     # Receive chunks
    #     queue = Queue()
    #     shutdown_event = Event()
    #     receive_bytes_in_chunks(iter(receive_chunks), queue, shutdown_event)

    #     # Verify data integrity
    #     received_data = queue.get()
    #     reconstructed_state_dict = bytes_to_state_dict(received_data)

    #     assert len(reconstructed_state_dict) == len(state_dict)
    #     for key in state_dict:
    #         assert torch.allclose(state_dict[key], reconstructed_state_dict[key])

    # def test_large_data_chunking(self):
    #     """Test chunking with data larger than multiple chunk sizes."""
    #     # Create large tensor
    #     large_tensor = torch.randn(2000, 2000)  # ~16MB
    #     state_dict = {"large": large_tensor}

    #     data = state_to_bytes(state_dict)

    #     # Should require multiple chunks
    #     chunks = list(send_bytes_in_chunks(data, Mock()))
    #     assert len(chunks) > 1

    #     # Verify total size matches
    #     total_size = sum(len(call[1]["data"]) for call in chunks[0].call_args_list)
    #     assert total_size == len(data)
