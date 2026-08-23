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
import threading
import time
from concurrent import futures
from multiprocessing import Event, Queue

import pytest

from tests.utils import skip_if_package_missing  # our gRPC servicer class


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


@skip_if_package_missing("grpcio", "grpc")
def create_learner_service_stub(
    shutdown_event: Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
    seconds_between_pushes: int,
    queue_get_timeout: float = 0.1,
):
    import grpc

    from lerobot.rl.learner_service import LearnerService
    from lerobot.transport import services_pb2_grpc  # generated from .proto

    """Fixture to start a LearnerService gRPC server and provide a connected stub."""

    servicer = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=seconds_between_pushes,
        transition_queue=transitions_queue,
        interaction_message_queue=interactions_queue,
        queue_get_timeout=queue_get_timeout,
    )

    # Create a gRPC server and add our servicer to it.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_LearnerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")  # bind to a free port chosen by OS
    server.start()  # start the server (non-blocking call):contentReference[oaicite:1]{index=1}

    # Create a client channel and stub connected to the server's port.
    channel = grpc.insecure_channel(f"localhost:{port}")
    return services_pb2_grpc.LearnerServiceStub(channel), channel, server


@skip_if_package_missing("grpcio", "grpc")
def close_learner_service_stub(channel, server):
    channel.close()
    server.stop(None)


@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_ready_method(learner_service_stub):
    from lerobot.transport import services_pb2

    """Test the ready method of the UserService."""
    request = services_pb2.Empty()
    response = learner_service_stub.Ready(request)
    assert response == services_pb2.Empty()


@skip_if_package_missing("grpcio", "grpc")
@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_send_interactions():
    from lerobot.transport import services_pb2

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

    def mock_interactions_stream():
        yield from list_of_interaction_messages

        return services_pb2.Empty()

    response = client.SendInteractions(mock_interactions_stream())
    assert response == services_pb2.Empty()

    close_learner_service_stub(channel, server)

    # Extract the data from the interactions queue
    interactions = []
    while not interactions_queue.empty():
        interactions.append(interactions_queue.get())

    assert interactions == [b"123", b"4", b"5", b"678"]


@skip_if_package_missing("grpcio", "grpc")
@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_send_transitions():
    from lerobot.transport import services_pb2

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


@skip_if_package_missing("grpcio", "grpc")
@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_send_transitions_empty_stream():
    from lerobot.transport import services_pb2

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


@skip_if_package_missing("grpcio", "grpc")
@pytest.mark.timeout(10)  # force cross-platform watchdog
def test_stream_parameters():
    import time

    from lerobot.transport import services_pb2

    """Test the StreamParameters method."""
    shutdown_event = Event()
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    seconds_between_pushes = 0.2  # Short delay for testing

    client, channel, server = create_learner_service_stub(
        shutdown_event, parameters_queue, transitions_queue, interactions_queue, seconds_between_pushes
    )

    # Add test parameters to the queue
    test_params = [b"param_batch_1", b"param_batch_2"]
    for param in test_params:
        parameters_queue.put(param)

    # Start streaming parameters
    request = services_pb2.Empty()
    stream = client.StreamParameters(request)

    # Collect streamed parameters and timestamps
    received_params = []
    timestamps = []

    for response in stream:
        received_params.append(response.data)
        timestamps.append(time.time())

        # We should receive one last item
        break

    parameters_queue.put(b"param_batch_3")

    for response in stream:
        received_params.append(response.data)
        timestamps.append(time.time())

        # We should receive only one item
        break

    shutdown_event.set()
    close_learner_service_stub(channel, server)

    assert received_params == [b"param_batch_2", b"param_batch_3"]

    # Check the time difference between the two sends
    time_diff = timestamps[1] - timestamps[0]
    # Check if the time difference is close to the expected push frequency
    assert time_diff == pytest.approx(seconds_between_pushes, abs=0.1)


@skip_if_package_missing("grpcio", "grpc")
@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_stream_parameters_with_shutdown():
    from lerobot.transport import services_pb2

    """Test StreamParameters handles shutdown gracefully."""
    shutdown_event = Event()
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    seconds_between_pushes = 0.1
    queue_get_timeout = 0.001

    client, channel, server = create_learner_service_stub(
        shutdown_event,
        parameters_queue,
        transitions_queue,
        interactions_queue,
        seconds_between_pushes,
        queue_get_timeout=queue_get_timeout,
    )

    test_params = [b"param_batch_1", b"stop", b"param_batch_3", b"param_batch_4"]

    # create a thread that will put the parameters in the queue
    def producer():
        for param in test_params:
            parameters_queue.put(param)
            time.sleep(0.1)

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    # Start streaming
    request = services_pb2.Empty()
    stream = client.StreamParameters(request)

    # Collect streamed parameters
    received_params = []

    for response in stream:
        received_params.append(response.data)

        if response.data == b"stop":
            shutdown_event.set()

    producer_thread.join()
    close_learner_service_stub(channel, server)

    assert received_params == [b"param_batch_1", b"stop"]


@skip_if_package_missing("grpcio", "grpc")
@pytest.mark.timeout(3)  # force cross-platform watchdog
def test_stream_parameters_waits_and_retries_on_empty_queue():
    import threading
    import time

    from lerobot.transport import services_pb2

    """Test that StreamParameters waits and retries when the queue is empty."""
    shutdown_event = Event()
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    seconds_between_pushes = 0.05
    queue_get_timeout = 0.01

    client, channel, server = create_learner_service_stub(
        shutdown_event,
        parameters_queue,
        transitions_queue,
        interactions_queue,
        seconds_between_pushes,
        queue_get_timeout=queue_get_timeout,
    )

    request = services_pb2.Empty()
    stream = client.StreamParameters(request)

    received_params = []

    def producer():
        # Let the consumer start and find an empty queue.
        # It will wait `seconds_between_pushes` (0.05s), then `get` will timeout after `queue_get_timeout` (0.01s).
        # Total time for the first empty loop is > 0.06s. We wait a bit longer to be safe.
        time.sleep(0.06)
        parameters_queue.put(b"param_after_wait")
        time.sleep(0.05)
        parameters_queue.put(b"param_after_wait_2")

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    # The consumer will block here until the producer sends an item.
    for response in stream:
        received_params.append(response.data)
        if response.data == b"param_after_wait_2":
            break  # We only need one item for this test.

    shutdown_event.set()
    producer_thread.join()
    close_learner_service_stub(channel, server)

    assert received_params == [b"param_after_wait", b"param_after_wait_2"]


@skip_if_package_missing("grpcio", "grpc")
@pytest.mark.timeout(25)  # force cross-platform watchdog
def test_establish_learner_connection_retries_when_workers_saturated():
    """A replacement actor must not hang forever at Ready() when the learner's gRPC
    thread pool is fully occupied by one already-connected actor's long-lived RPCs
    (StreamParameters, SendTransitions, SendInteractions) -- exactly MAX_WORKERS calls.

    This reproduces #3979: without a deadline on stub.Ready(), establish_learner_connection's
    retry loop never gets a chance to retry because the call never returns.
    """
    import contextlib
    import threading
    from concurrent import futures

    import grpc

    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    from lerobot.rl.actor import establish_learner_connection
    from lerobot.rl.learner_service import MAX_WORKERS, LearnerService
    from lerobot.transport import services_pb2, services_pb2_grpc

    shutdown_event = Event()
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    servicer = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=1,
        transition_queue=transitions_queue,
        interaction_message_queue=interactions_queue,
    )

    # Mirror production exactly: learner.py builds the server's executor with MAX_WORKERS
    # threads and nothing more -- there is no headroom above one actor's RPC count.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    services_pb2_grpc.add_LearnerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = services_pb2_grpc.LearnerServiceStub(channel)

    # Occupy all MAX_WORKERS=3 worker threads with the same three long-lived RPCs one real
    # actor opens. None of them ever finishes on its own -- exactly like a lingering/half-open
    # connection from a previous actor.
    stream_call = stub.StreamParameters(services_pb2.Empty())

    def _drain_stream():
        try:
            for _ in stream_call:
                pass
        except grpc.RpcError:
            pass

    threading.Thread(target=_drain_stream, daemon=True).start()

    release_gate = threading.Event()  # never set during the saturated phase

    def _blocking_client_iterator():
        release_gate.wait()
        return
        yield  # pragma: no cover - unreachable; keeps this a generator

    def _call_and_ignore_cancellation(stub_method):
        # Expected once the channel is closed during test cleanup.
        with contextlib.suppress(grpc.RpcError):
            stub_method(_blocking_client_iterator())

    threading.Thread(target=_call_and_ignore_cancellation, args=(stub.SendTransitions,), daemon=True).start()
    threading.Thread(target=_call_and_ignore_cancellation, args=(stub.SendInteractions,), daemon=True).start()

    # Give the three RPCs a moment to actually reach the server and occupy their workers.
    time.sleep(0.5)

    # A raw Ready() call with an explicit deadline must fail fast with DEADLINE_EXCEEDED
    # instead of hanging -- proves the pool is genuinely starved, not just slow.
    with pytest.raises(grpc.RpcError) as exc_info:
        stub.Ready(services_pb2.Empty(), timeout=2)
    assert exc_info.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED

    # The actual regression: establish_learner_connection's retry loop must return in
    # bounded time instead of hanging forever on the first Ready() call.
    start = time.time()
    connected = establish_learner_connection(stub, Event(), attempts=2)
    elapsed = time.time() - start

    release_gate.set()
    shutdown_event.set()
    channel.close()
    server.stop(None)

    assert connected is False
    # 2 attempts * (READY_RPC_TIMEOUT_S deadline + 2s retry sleep) plus slack.
    assert elapsed < 20


@skip_if_package_missing("grpcio", "grpc")
@pytest.mark.timeout(5)  # force cross-platform watchdog
def test_stream_parameters_stops_when_context_goes_inactive():
    """StreamParameters must release its worker when the peer disconnects, even if the
    parameters queue never gets new data and shutdown_event is never set. Before the fix,
    the loop only checked shutdown_event, so a dead/half-open peer held the worker forever.
    """
    from lerobot.rl.learner_service import LearnerService
    from lerobot.transport import services_pb2

    class _FakeContext:
        """Stand-in for grpc.ServicerContext exposing only is_active(), controlled by the test."""

        def __init__(self):
            self._active = True

        def is_active(self):
            return self._active

        def disconnect(self):
            self._active = False

    shutdown_event = Event()
    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    servicer = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=0.05,
        transition_queue=transitions_queue,
        interaction_message_queue=interactions_queue,
        queue_get_timeout=0.01,
    )

    context = _FakeContext()
    stream = servicer.StreamParameters(services_pb2.Empty(), context)

    # Simulate the client disconnecting / cancelling before any parameters were ever pushed.
    context.disconnect()

    # Without checking context.is_active(), the generator loops forever on the empty queue
    # (shutdown_event is never set); with the fix it must stop as soon as it notices.
    with pytest.raises(StopIteration):
        next(stream)

    shutdown_event.set()  # cleanup safety net in case the generator is still alive
