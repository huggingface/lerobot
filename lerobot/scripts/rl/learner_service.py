import logging
from multiprocessing import Event, Queue

from lerobot.common.transport import services_pb2, services_pb2_grpc
from lerobot.common.transport.utils import receive_bytes_in_chunks, send_bytes_in_chunks

MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB
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
    ):
        self.shutdown_event = shutdown_event
        self.parameters_queue = parameters_queue
        self.seconds_between_pushes = seconds_between_pushes
        self.transition_queue = transition_queue
        self.interaction_message_queue = interaction_message_queue

    def StreamParameters(self, request, context):  # noqa: N802
        # TODO: authorize the request
        logging.info("[LEARNER] Received request to stream parameters from the Actor")

        while not self.shutdown_event.is_set():
            logging.info("[LEARNER] Push parameters to the Actor")
            buffer = self.parameters_queue.get()

            yield from send_bytes_in_chunks(
                buffer,
                services_pb2.Parameters,
                log_prefix="[LEARNER] Sending parameters",
                silent=True,
            )

            logging.info("[LEARNER] Parameters sent")

            self.shutdown_event.wait(self.seconds_between_pushes)

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
