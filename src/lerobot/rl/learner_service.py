# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import logging
import time
from multiprocessing import Event, Queue
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _grpc_available

from .queue import get_last_item_from_queue

if TYPE_CHECKING or _grpc_available:
    import grpc

    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import receive_bytes_in_chunks, send_bytes_in_chunks

    _ServicerBase = services_pb2_grpc.LearnerServiceServicer
else:
    grpc = None
    services_pb2 = None
    services_pb2_grpc = None
    receive_bytes_in_chunks = None
    send_bytes_in_chunks = None
    _ServicerBase = object

MAX_WORKERS = 3  # Stream parameters, send transitions and interactions
SHUTDOWN_TIMEOUT = 10


class LearnerService(_ServicerBase):
    """Implementation of the LearnerService gRPC service.

    Sends policy parameters to the actor and receives transitions and interactions from it; see
    `transport.proto` for the gRPC service definition.

    Args:
        shutdown_event (`Event`): Set to stop `StreamParameters`'s push loop.
        parameters_queue (`Queue`): Queue of serialized policy weights, drained and streamed to
            the actor by `StreamParameters`.
        seconds_between_pushes (`float`): Minimum interval between successive parameter pushes.
        transition_queue (`Queue`): Queue filled by `SendTransitions` with received transitions.
        interaction_message_queue (`Queue`): Queue filled by `SendInteractions` with received
            interaction messages.
        queue_get_timeout (`float`, *optional*, defaults to 0.001): Timeout used when polling
            `parameters_queue`.
    """

    def __init__(
        self,
        shutdown_event: Event,  # type: ignore
        parameters_queue: Queue,
        seconds_between_pushes: float,
        transition_queue: Queue,
        interaction_message_queue: Queue,
        queue_get_timeout: float = 0.001,
    ):
        self.shutdown_event = shutdown_event
        self.parameters_queue = parameters_queue
        self.seconds_between_pushes = seconds_between_pushes
        self.transition_queue = transition_queue
        self.interaction_message_queue = interaction_message_queue
        self.queue_get_timeout = queue_get_timeout

    def StreamParameters(  # noqa: N802
        self, request: "services_pb2.Empty", context: "grpc.ServicerContext"
    ):
        """GRPC server-streaming RPC: push the latest policy parameters to the actor.

        Runs until `shutdown_event` is set, pushing at most once every `seconds_between_pushes`.

        Args:
            request (`services_pb2.Empty`): Unused; required by the gRPC service signature.
            context (`grpc.ServicerContext`): gRPC call context.

        Yields:
            Chunks of a `services_pb2.Parameters` message, produced by `send_bytes_in_chunks`.
        """
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to stream parameters from the Actor")

        last_push_time = 0

        while not self.shutdown_event.is_set():
            time_since_last_push = time.time() - last_push_time
            if time_since_last_push < self.seconds_between_pushes:
                self.shutdown_event.wait(self.seconds_between_pushes - time_since_last_push)
                # Continue, because we could receive a shutdown event,
                # and it's checked in the while loop
                continue

            logging.info("[LEARNER] Push parameters to the Actor")
            buffer = get_last_item_from_queue(
                self.parameters_queue, block=True, timeout=self.queue_get_timeout
            )

            if buffer is None:
                continue

            yield from send_bytes_in_chunks(
                buffer,
                services_pb2.Parameters,
                log_prefix="[LEARNER] Sending parameters",
                silent=True,
            )

            last_push_time = time.time()
            logging.info("[LEARNER] Parameters sent")

        logging.info("[LEARNER] Stream parameters finished")
        return services_pb2.Empty()

    def SendTransitions(self, request_iterator, _context: "grpc.ServicerContext"):  # noqa: N802
        """GRPC client-streaming RPC: receive transition chunks from the actor into `transition_queue`.

        Args:
            request_iterator: Stream of `services_pb2.Transition` chunks sent by the actor's
                `transitions_stream`.
            _context (`grpc.ServicerContext`): gRPC call context.

        Returns:
            services_pb2.Empty: Acknowledgement sent once the actor closes the stream.
        """
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to receive transitions from the Actor")

        receive_bytes_in_chunks(
            request_iterator,
            self.transition_queue,
            self.shutdown_event,
            log_prefix="[LEARNER] transitions",
        )

        logging.debug("[LEARNER] Finished receiving transitions")
        return services_pb2.Empty()

    def SendInteractions(self, request_iterator, _context: "grpc.ServicerContext"):  # noqa: N802
        """GRPC client-streaming RPC: receive interaction-message chunks into `interaction_message_queue`.

        Args:
            request_iterator: Stream of `services_pb2.InteractionMessage` chunks sent by the actor's
                `interactions_stream`.
            _context (`grpc.ServicerContext`): gRPC call context.

        Returns:
            services_pb2.Empty: Acknowledgement sent once the actor closes the stream.
        """
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to receive interactions from the Actor")

        receive_bytes_in_chunks(
            request_iterator,
            self.interaction_message_queue,
            self.shutdown_event,
            log_prefix="[LEARNER] interactions",
        )

        logging.debug("[LEARNER] Finished receiving interactions")
        return services_pb2.Empty()

    def Ready(self, request: "services_pb2.Empty", context: "grpc.ServicerContext"):  # noqa: N802
        """GRPC health check: returns immediately, confirming the learner server is up."""
        return services_pb2.Empty()
