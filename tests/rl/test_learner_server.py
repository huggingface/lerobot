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
    stub, channel, server = create_learner_service_stub(
        shutdown_event, parameters_queue, transitions_queue, interactions_queue, seconds_between_pushes
    )

    yield stub  # provide the stub to the test function

    # **Teardown Phase**: close channel and stop server.
    channel.close()
    server.stop(None)  # stop the server; None for immediate shutdown (no grace period)


def create_learner_service_stub(
    shutdown_event: Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
    seconds_between_pushes: int,
):
    """Fixture to start a LearnerService gRPC server and provide a connected stub."""

    servicer = LearnerService(
        shutdown_event, parameters_queue, transitions_queue, interactions_queue, seconds_between_pushes
    )

    # Create a gRPC server and add our servicer to it.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_LearnerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")  # bind to a free port chosen by OS
    server.start()  # start the server (non-blocking call):contentReference[oaicite:1]{index=1}

    # Create a client channel and stub connected to the server's port.
    channel = grpc.insecure_channel(f"localhost:{port}")
    return services_pb2_grpc.LearnerServiceStub(channel), channel, server


def test_ready_method(learner_service_stub):
    """Test the ready method of the UserService."""
    request = services_pb2.Empty()
    response = learner_service_stub.Ready(request)
    assert response == services_pb2.Empty()
