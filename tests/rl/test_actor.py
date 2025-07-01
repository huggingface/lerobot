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

from concurrent import futures
from unittest.mock import patch

import pytest
import torch
from torch.multiprocessing import Event, Queue

from lerobot.utils.transition import Transition
from tests.utils import require_package


def create_learner_service_stub():
    import grpc

    from lerobot.transport import services_pb2, services_pb2_grpc

    class MockLearnerService(services_pb2_grpc.LearnerServiceServicer):
        def __init__(self):
            self.ready_call_count = 0
            self.should_fail = False

        def Ready(self, request, context):  # noqa: N802
            self.ready_call_count += 1
            if self.should_fail:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service unavailable")
                raise grpc.RpcError("Service unavailable")
            return services_pb2.Empty()

    """Fixture to start a LearnerService gRPC server and provide a connected stub."""

    servicer = MockLearnerService()

    # Create a gRPC server and add our servicer to it.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_LearnerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")  # bind to a free port chosen by OS
    server.start()  # start the server (non-blocking call):contentReference[oaicite:1]{index=1}

    # Create a client channel and stub connected to the server's port.
    channel = grpc.insecure_channel(f"localhost:{port}")
    return services_pb2_grpc.LearnerServiceStub(channel), servicer, channel, server


def close_service_stub(channel, server):
    channel.close()
    server.stop(None)


@require_package("grpc")
def test_establish_learner_connection_success():
    from lerobot.scripts.rl.actor import establish_learner_connection

    """Test successful connection establishment."""
    stub, _servicer, channel, server = create_learner_service_stub()

    shutdown_event = Event()

    # Test successful connection
    result = establish_learner_connection(stub, shutdown_event, attempts=5)

    assert result is True

    close_service_stub(channel, server)


@require_package("grpc")
def test_establish_learner_connection_failure():
    from lerobot.scripts.rl.actor import establish_learner_connection

    """Test connection failure."""
    stub, servicer, channel, server = create_learner_service_stub()
    servicer.should_fail = True

    shutdown_event = Event()

    # Test failed connection
    with patch("time.sleep"):  # Speed up the test
        result = establish_learner_connection(stub, shutdown_event, attempts=2)

    assert result is False

    close_service_stub(channel, server)


@require_package("grpc")
def test_push_transitions_to_transport_queue():
    from lerobot.scripts.rl.actor import push_transitions_to_transport_queue
    from lerobot.transport.utils import bytes_to_transitions
    from tests.transport.test_transport_utils import assert_transitions_equal

    """Test pushing transitions to transport queue."""
    # Create mock transitions
    transitions = []
    for i in range(3):
        transition = Transition(
            state={"observation": torch.randn(3, 64, 64), "state": torch.randn(10)},
            action=torch.randn(5),
            reward=torch.tensor(1.0 + i),
            done=torch.tensor(False),
            truncated=torch.tensor(False),
            next_state={"observation": torch.randn(3, 64, 64), "state": torch.randn(10)},
            complementary_info={"step": torch.tensor(i)},
        )
        transitions.append(transition)

    transitions_queue = Queue()

    # Test pushing transitions
    push_transitions_to_transport_queue(transitions, transitions_queue)

    # Verify the data can be retrieved
    serialized_data = transitions_queue.get()
    assert isinstance(serialized_data, bytes)
    deserialized_transitions = bytes_to_transitions(serialized_data)
    assert len(deserialized_transitions) == len(transitions)
    for i, deserialized_transition in enumerate(deserialized_transitions):
        assert_transitions_equal(deserialized_transition, transitions[i])


@require_package("grpc")
@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_transitions_stream():
    from lerobot.scripts.rl.actor import transitions_stream

    """Test transitions stream functionality."""
    shutdown_event = Event()
    transitions_queue = Queue()

    # Add test data to queue
    test_data = [b"transition_data_1", b"transition_data_2", b"transition_data_3"]
    for data in test_data:
        transitions_queue.put(data)

    # Collect streamed data
    streamed_data = []
    stream_generator = transitions_stream(shutdown_event, transitions_queue, 0.1)

    # Process a few items
    for i, message in enumerate(stream_generator):
        streamed_data.append(message)
        if i >= len(test_data) - 1:
            shutdown_event.set()
            break

    # Verify we got messages
    assert len(streamed_data) == len(test_data)
    assert streamed_data[0].data == b"transition_data_1"
    assert streamed_data[1].data == b"transition_data_2"
    assert streamed_data[2].data == b"transition_data_3"


@require_package("grpc")
@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_interactions_stream():
    from lerobot.scripts.rl.actor import interactions_stream
    from lerobot.transport.utils import bytes_to_python_object, python_object_to_bytes

    """Test interactions stream functionality."""
    shutdown_event = Event()
    interactions_queue = Queue()

    # Create test interaction data (similar structure to what would be sent)
    test_interactions = [
        {"episode_reward": 10.5, "step": 1, "policy_fps": 30.2},
        {"episode_reward": 15.2, "step": 2, "policy_fps": 28.7},
        {"episode_reward": 8.7, "step": 3, "policy_fps": 29.1},
    ]

    # Serialize the interaction data as it would be in practice
    test_data = [
        interactions_queue.put(python_object_to_bytes(interaction)) for interaction in test_interactions
    ]

    # Collect streamed data
    streamed_data = []
    stream_generator = interactions_stream(shutdown_event, interactions_queue, 0.1)

    # Process the items
    for i, message in enumerate(stream_generator):
        streamed_data.append(message)
        if i >= len(test_data) - 1:
            shutdown_event.set()
            break

    # Verify we got messages
    assert len(streamed_data) == len(test_data)

    # Verify the messages can be deserialized back to original data
    for i, message in enumerate(streamed_data):
        deserialized_interaction = bytes_to_python_object(message.data)
        assert deserialized_interaction == test_interactions[i]
