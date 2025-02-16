import hilserl_pb2  # type: ignore
import hilserl_pb2_grpc  # type: ignore
import torch
from torch import nn
from threading import Lock, Event
import logging
import queue
import io
import pickle

from lerobot.scripts.server.buffer import (
    move_state_dict_to_device,
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
        policy: nn.Module,
        policy_lock: Lock,
        seconds_between_pushes: float,
        transition_queue: queue.Queue,
        interaction_message_queue: queue.Queue,
    ):
        self.shutdown_event = shutdown_event
        self.policy = policy
        self.policy_lock = policy_lock
        self.seconds_between_pushes = seconds_between_pushes
        self.transition_queue = transition_queue
        self.interaction_message_queue = interaction_message_queue

    def _get_policy_state(self):
        with self.policy_lock:
            params_dict = self.policy.actor.state_dict()
            if self.policy.config.vision_encoder_name is not None:
                if self.policy.config.freeze_vision_encoder:
                    params_dict: dict[str, torch.Tensor] = {
                        k: v
                        for k, v in params_dict.items()
                        if not k.startswith("encoder.")
                    }
                else:
                    raise NotImplementedError(
                        "Vision encoder is not frozen, we need to send the full model over the network which requires chunking the model."
                    )

        return move_state_dict_to_device(params_dict, device="cpu")

    def _send_bytes(self, buffer: bytes):
        size_in_bytes = bytes_buffer_size(buffer)

        sent_bytes = 0

        logging.info(f"Model state size {size_in_bytes/1024/1024} MB with")

        while sent_bytes < size_in_bytes:
            transfer_state = hilserl_pb2.TransferState.TRANSFER_MIDDLE

            if sent_bytes == 0:
                transfer_state = hilserl_pb2.TransferState.TRANSFER_BEGIN
            elif sent_bytes + CHUNK_SIZE >= size_in_bytes:
                transfer_state = hilserl_pb2.TransferState.TRANSFER_END

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
