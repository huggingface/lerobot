import hilserl_pb2  # type: ignore
import hilserl_pb2_grpc  # type: ignore
import logging
from multiprocessing import Event, Queue
from queue import Empty

from lerobot.scripts.server.buffer import (
    state_to_bytes,
)
from lerobot.scripts.server.network_utils import receive_bytes_in_chunks
from lerobot.scripts.server.network_utils import send_bytes_in_chunks


MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB
CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_WORKERS = 3  # Stream parameters, send transitions and interactions
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

    def StreamParameters(self, request, context):
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to stream parameters from the Actor")

        while not self.shutdown_event.is_set():
            logging.debug("[LEARNER] Push parameters to the Actor")
            state_dict = self._get_policy_state()

            with state_to_bytes(state_dict) as buffer:
                yield from send_bytes_in_chunks(buffer, hilserl_pb2.Parameters)

            self.shutdown_event.wait(self.seconds_between_pushes)

    def SendTransitions(self, request_iterator, _context):
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to receive transitions from the Actor")

        receive_bytes_in_chunks(
            request_iterator, self.transition_queue, self.shutdown_event
        )

        logging.debug("[LEARNER] Finished receiving transitions")
        return hilserl_pb2.Empty()

    def SendInteractions(self, request_iterator, _context):
        # TODO: authorize the request
        logging.info(
            "[LEARNER] Received request to receive interactions from the Actor"
        )

        receive_bytes_in_chunks(
            request_iterator, self.interaction_message_queue, self.shutdown_event
        )

        logging.debug("[LEARNER] Finished receiving interactions")
        return hilserl_pb2.Empty()

    def Ready(self, request, context):
        return hilserl_pb2.Empty()
