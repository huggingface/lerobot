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
from multiprocessing import Event, Queue

import grpc
import pytest

# Import the generated classes and service implementation from our project.
from lerobot.common.transport import services_pb2, services_pb2_grpc  # generated from .proto
from lerobot.scripts.rl.learner_service import LearnerService  # our gRPC servicer class


@pytest.fixture(scope="function")
def learner_service_stub():
    shutdown_event = Event()
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    seconds_between_pushes = 1
    client, channel, server = create_learner_service_stub(
        shutdown_event, parameters_queue, transitions_queue, interactions_queue, seconds_between_pushes
    )

    yield client  # provide the stub to the test function

    close_learner_service_stub(channel, server)


def create_learner_service_stub(
    shutdown_event: Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
    seconds_between_pushes: int,
):
    """Fixture to start a LearnerService gRPC server and provide a connected stub."""

    servicer = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=seconds_between_pushes,
        transition_queue=transitions_queue,
        interaction_message_queue=interactions_queue,
    )

    # Create a gRPC server and add our servicer to it.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_LearnerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")  # bind to a free port chosen by OS
    server.start()  # start the server (non-blocking call):contentReference[oaicite:1]{index=1}

    # Create a client channel and stub connected to the server's port.
    channel = grpc.insecure_channel(f"localhost:{port}")
    return services_pb2_grpc.LearnerServiceStub(channel), channel, server


def close_learner_service_stub(channel: grpc.Channel, server: grpc.Server):
    channel.close()
    server.stop(None)


@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_ready_method(learner_service_stub):
    """Test the ready method of the UserService."""
    request = services_pb2.Empty()
    response = learner_service_stub.Ready(request)
    assert response == services_pb2.Empty()


@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_send_interactions():
    shutdown_event = Event()

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    seconds_between_pushes = 1
    client, channel, server = create_learner_service_stub(
        shutdown_event, parameters_queue, transitions_queue, interactions_queue, seconds_between_pushes
    )

    list_of_interaction_messages = [
        services_pb2.InteractionMessage(transfer_state=services_pb2.TransferState.TRANSFER_BEGIN, data=b"1"),
        services_pb2.InteractionMessage(transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE, data=b"2"),
        services_pb2.InteractionMessage(transfer_state=services_pb2.TransferState.TRANSFER_END, data=b"3"),
        services_pb2.InteractionMessage(transfer_state=services_pb2.TransferState.TRANSFER_END, data=b"4"),
        services_pb2.InteractionMessage(transfer_state=services_pb2.TransferState.TRANSFER_END, data=b"5"),
        services_pb2.InteractionMessage(transfer_state=services_pb2.TransferState.TRANSFER_BEGIN, data=b"6"),
        services_pb2.InteractionMessage(transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE, data=b"7"),
        services_pb2.InteractionMessage(transfer_state=services_pb2.TransferState.TRANSFER_END, data=b"8"),
    ]

    def mock_intercations_stream():
        yield from list_of_interaction_messages

        return services_pb2.Empty()

    response = client.SendInteractions(mock_intercations_stream())
    assert response == services_pb2.Empty()

    close_learner_service_stub(channel, server)

    # Extract the data from the interactions queue
    interactions = []
    while not interactions_queue.empty():
        interactions.append(interactions_queue.get())

    assert interactions == [b"123", b"4", b"5", b"678"]


@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_send_transitions():
    """Test the SendTransitions method with various transition data."""
    shutdown_event = Event()
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    seconds_between_pushes = 1

    client, channel, server = create_learner_service_stub(
        shutdown_event, parameters_queue, transitions_queue, interactions_queue, seconds_between_pushes
    )

    # Create test transition messages
    list_of_transition_messages = [
        services_pb2.Transition(
            transfer_state=services_pb2.TransferState.TRANSFER_BEGIN, data=b"transition_1"
        ),
        services_pb2.Transition(
            transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE, data=b"transition_2"
        ),
        services_pb2.Transition(transfer_state=services_pb2.TransferState.TRANSFER_END, data=b"transition_3"),
        services_pb2.Transition(transfer_state=services_pb2.TransferState.TRANSFER_BEGIN, data=b"batch_1"),
        services_pb2.Transition(transfer_state=services_pb2.TransferState.TRANSFER_END, data=b"batch_2"),
    ]

    def mock_transitions_stream():
        yield from list_of_transition_messages

    response = client.SendTransitions(mock_transitions_stream())
    assert response == services_pb2.Empty()

    close_learner_service_stub(channel, server)

    # Extract the data from the transitions queue
    transitions = []
    while not transitions_queue.empty():
        transitions.append(transitions_queue.get())

    # Should have assembled the chunked data
    assert transitions == [b"transition_1transition_2transition_3", b"batch_1batch_2"]


@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_send_transitions_empty_stream():
    """Test SendTransitions with empty stream."""
    shutdown_event = Event()
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    seconds_between_pushes = 1

    client, channel, server = create_learner_service_stub(
        shutdown_event, parameters_queue, transitions_queue, interactions_queue, seconds_between_pushes
    )

    def empty_stream():
        return iter([])

    response = client.SendTransitions(empty_stream())
    assert response == services_pb2.Empty()

    close_learner_service_stub(channel, server)

    # Queue should remain empty
    assert transitions_queue.empty()


@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_stream_parameters():
    """Test the StreamParameters method."""
    shutdown_event = Event()
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    seconds_between_pushes = 0.1  # Short delay for testing

    client, channel, server = create_learner_service_stub(
        shutdown_event, parameters_queue, transitions_queue, interactions_queue, seconds_between_pushes
    )

    # Add test parameters to the queue
    test_params = [b"param_batch_1", b"param_batch_2", b"param_batch_3"]
    for param in test_params:
        parameters_queue.put(param)

    # Start streaming parameters
    request = services_pb2.Empty()
    stream = client.StreamParameters(request)

    # Collect streamed parameters
    received_params = []

    for response in stream:
        received_params.append(response.data)

        if len(test_params) == len(received_params):
            break

    shutdown_event.set()
    close_learner_service_stub(channel, server)

    assert received_params == test_params


@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_stream_parameters_with_shutdown():
    """Test StreamParameters handles shutdown gracefully."""
    shutdown_event = Event()
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    seconds_between_pushes = 0.1

    client, channel, server = create_learner_service_stub(
        shutdown_event, parameters_queue, transitions_queue, interactions_queue, seconds_between_pushes
    )

    test_params = [b"param_batch_1", b"stop", b"param_batch_3", b"param_batch_4"]

    # Add one parameter batch
    for param in test_params:
        parameters_queue.put(param)

    # Start streaming
    request = services_pb2.Empty()
    stream = client.StreamParameters(request)

    # Collect streamed parameters
    received_params = []

    for response in stream:
        received_params.append(response.data)

        if response.data == b"stop":
            shutdown_event.set()

    close_learner_service_stub(channel, server)

    assert received_params == [b"param_batch_1", b"stop"]
