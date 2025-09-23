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

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import receive_bytes_in_chunks, send_bytes_in_chunks
from lerobot.utils.queue import get_last_item_from_queue

MAX_WORKERS = 3  # Stream parameters, send transitions and interactions
SHUTDOWN_TIMEOUT = 10


class LearnerService(services_pb2_grpc.LearnerServiceServicer):
    """
    Implementation of the LearnerService gRPC service
    This service is used to send parameters to the Actor and receive transitions and interactions from the Actor
    check transport.proto for the gRPC service definition
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

    def StreamParameters(self, request, context):  # noqa: N802
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

    def SendTransitions(self, request_iterator, _context):  # noqa: N802
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

    def SendInteractions(self, request_iterator, _context):  # noqa: N802
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

    def Ready(self, request, context):  # noqa: N802
        return services_pb2.Empty()
