import hilserl_pb2  # type: ignore
import hilserl_pb2_grpc  # type: ignore
import torch
import logging
import io
import pickle
from multiprocessing import Event, Queue
from queue import Empty

from lerobot.scripts.server.buffer import (
    bytes_buffer_size,
    state_to_bytes,
)


MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB
CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_WORKERS = 10
STUTDOWN_TIMEOUT = 10


class LearnerService(hilserl_pb2_grpc.LearnerServiceServicer):
    def __init__(
        self,
        shutdown_event: Event,
        parameters_queue: Queue,
        seconds_between_pushes: float,
        transition_queue: Queue,
        interaction_message_queue: Queue,
    ):
        self.shutdown_event = shutdown_event
        self.parameters_queue = parameters_queue
        self.seconds_between_pushes = seconds_between_pushes
        self.transition_queue = transition_queue
        self.interaction_message_queue = interaction_message_queue

    def _get_policy_state(self):
        # Get initial parameters
        params_dict = self.parameters_queue.get()

        # Drain queue and keep only the most recent parameters
        try:
            while True:
                params_dict = self.parameters_queue.get_nowait()
        except Empty:
            pass

        return params_dict

    def _send_bytes(self, buffer: bytes):
        size_in_bytes = bytes_buffer_size(buffer)

        sent_bytes = 0

        logging.info(f"Model state size {size_in_bytes/1024/1024} MB with")

        while sent_bytes < size_in_bytes:
            transfer_state = hilserl_pb2.TransferState.TRANSFER_MIDDLE

            if sent_bytes + CHUNK_SIZE >= size_in_bytes:
                transfer_state = hilserl_pb2.TransferState.TRANSFER_END
            elif sent_bytes == 0:
                transfer_state = hilserl_pb2.TransferState.TRANSFER_BEGIN

            size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
            chunk = buffer.read(size_to_read)

            yield hilserl_pb2.Parameters(
                transfer_state=transfer_state, parameter_bytes=chunk
            )
            sent_bytes += size_to_read
            logging.info(
                f"[Learner] Sent {sent_bytes}/{size_in_bytes} bytes with state {transfer_state}"
            )

        logging.info(f"[LEARNER] Published {sent_bytes/1024/1024} MB to the Actor")

    def StreamParameters(self, request, context):
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to stream parameters from the Actor")

        while not self.shutdown_event.is_set():
            logging.debug("[LEARNER] Push parameters to the Actor")
            state_dict = self._get_policy_state()

            with state_to_bytes(state_dict) as buffer:
                yield from self._send_bytes(buffer)

            self.shutdown_event.wait(self.seconds_between_pushes)

    def ReceiveTransitions(self, request_iterator, context):
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to receive transitions from the Actor")

        for request in request_iterator:
            logging.debug("[LEARNER] Received request")
            if request.HasField("transition"):
                buffer = io.BytesIO(request.transition.transition_bytes)
                transition = torch.load(buffer)
                self.transition_queue.put(transition)
            if request.HasField("interaction_message"):
                content = pickle.loads(
                    request.interaction_message.interaction_message_bytes
                )
                self.interaction_message_queue.put(content)

    def Ready(self, request, context):
        return hilserl_pb2.Empty()
