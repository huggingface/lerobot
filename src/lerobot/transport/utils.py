#!/usr/bin/env python

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

import io
import json
import logging
import pickle  # nosec B403: Safe usage for internal serialization only
from multiprocessing.synchronize import Event as MpEvent
from queue import Queue
from typing import Any

import torch

from lerobot.transport import services_pb2
from lerobot.utils.transition import Transition

# FIX for protobuf: Assign the enum to a variable and ignore the type error once
TransferState = services_pb2.TransferState  # type: ignore[attr-defined]

CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB


def bytes_buffer_size(buffer: io.BytesIO) -> int:
    buffer.seek(0, io.SEEK_END)
    result = buffer.tell()
    buffer.seek(0)
    return result


def send_bytes_in_chunks(buffer: bytes, message_class: Any, log_prefix: str = "", silent: bool = True):
    bytes_buffer: io.BytesIO = io.BytesIO(buffer)
    size_in_bytes = bytes_buffer_size(bytes_buffer)

    sent_bytes = 0

    logging_method = logging.info if not silent else logging.debug

    logging_method(f"{log_prefix} Buffer size {size_in_bytes / 1024 / 1024} MB with")

    while sent_bytes < size_in_bytes:
        transfer_state = TransferState.TRANSFER_MIDDLE

        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = TransferState.TRANSFER_BEGIN

        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        chunk = bytes_buffer.read(size_to_read)

        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read
        logging_method(f"{log_prefix} Sent {sent_bytes}/{size_in_bytes} bytes with state {transfer_state}")

    logging_method(f"{log_prefix} Published {sent_bytes / 1024 / 1024} MB")


def receive_bytes_in_chunks(iterator, queue: Queue | None, shutdown_event: MpEvent, log_prefix: str = ""):
    bytes_buffer = io.BytesIO()
    step = 0

    logging.info(f"{log_prefix} Starting receiver")
    for item in iterator:
        logging.debug(f"{log_prefix} Received item")
        if shutdown_event.is_set():
            logging.info(f"{log_prefix} Shutting down receiver")
            return

        if item.transfer_state == TransferState.TRANSFER_BEGIN:
            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step 0")
            step = 0
        elif item.transfer_state == TransferState.TRANSFER_MIDDLE:
            bytes_buffer.write(item.data)
            step += 1
            logging.debug(f"{log_prefix} Received data at step {step}")
        elif item.transfer_state == TransferState.TRANSFER_END:
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step end size {bytes_buffer_size(bytes_buffer)}")

            if queue is not None:
                queue.put(bytes_buffer.getvalue())
            else:
                return bytes_buffer.getvalue()

            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            step = 0

            logging.debug(f"{log_prefix} Queue updated")
        else:
            logging.warning(f"{log_prefix} Received unknown transfer state {item.transfer_state}")
            raise ValueError(f"Received unknown transfer state {item.transfer_state}")


def state_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Convert model state dict to flat array for transmission"""
    bytes_buffer = io.BytesIO()

    torch.save(state_dict, bytes_buffer)

    return bytes_buffer.getvalue()


def bytes_to_state_dict(buffer: bytes) -> dict[str, torch.Tensor]:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    return torch.load(bytes_buffer, weights_only=True)


def python_object_to_bytes(python_object: Any) -> bytes:
    return pickle.dumps(python_object)


def bytes_to_python_object(buffer: bytes) -> Any:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    obj = pickle.load(bytes_buffer)  # nosec B301: Safe usage of pickle.load
    # Add validation checks here
    return obj


def bytes_to_transitions(buffer: bytes) -> list[Transition]:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    transitions = torch.load(bytes_buffer, weights_only=True)
    return transitions


def transitions_to_bytes(transitions: list[Transition]) -> bytes:
    bytes_buffer = io.BytesIO()
    torch.save(transitions, bytes_buffer)
    return bytes_buffer.getvalue()


def grpc_channel_options(
    max_receive_message_length: int = MAX_MESSAGE_SIZE,
    max_send_message_length: int = MAX_MESSAGE_SIZE,
    enable_retries: bool = True,
    initial_backoff: str = "0.1s",
    max_attempts: int = 5,
    backoff_multiplier: float = 2,
    max_backoff: str = "2s",
):
    service_config = {
        "methodConfig": [
            {
                "name": [{}],  # Applies to ALL methods in ALL services
                "retryPolicy": {
                    "maxAttempts": max_attempts,  # Max retries (total attempts = 5)
                    "initialBackoff": initial_backoff,  # First retry after 0.1s
                    "maxBackoff": max_backoff,  # Max wait time between retries
                    "backoffMultiplier": backoff_multiplier,  # Exponential backoff factor
                    "retryableStatusCodes": [
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                    ],  # Retries on network failures
                },
            }
        ]
    }

    service_config_json = json.dumps(service_config)

    retries_option = 1 if enable_retries else 0

    return [
        ("grpc.max_receive_message_length", max_receive_message_length),
        ("grpc.max_send_message_length", max_send_message_length),
        ("grpc.enable_retries", retries_option),
        ("grpc.service_config", service_config_json),
    ]
