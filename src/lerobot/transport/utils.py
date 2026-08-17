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

from lerobot.utils.transition import Transition

from . import services_pb2

# FIX for protobuf: Assign the enum to a variable and ignore the type error once
TransferState = services_pb2.TransferState  # type: ignore[attr-defined]

CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB


def bytes_buffer_size(buffer: io.BytesIO) -> int:
    """Return `buffer`'s total size in bytes, restoring its read position to the start.

    Args:
        buffer (`io.BytesIO`): Buffer to measure.

    Returns:
        `int`: Total size, in bytes.
    """
    buffer.seek(0, io.SEEK_END)
    result = buffer.tell()
    buffer.seek(0)
    return result


def send_bytes_in_chunks(buffer: bytes, message_class: Any, log_prefix: str = "", silent: bool = True):
    """Split `buffer` into `CHUNK_SIZE` pieces and yield them as gRPC transfer messages.

    Each yielded message carries a `transfer_state` (`TRANSFER_BEGIN`/`TRANSFER_MIDDLE`/
    `TRANSFER_END`) so the receiver (see `receive_bytes_in_chunks`) can reassemble the buffer.

    Args:
        buffer (`bytes`): Data to send.
        message_class (`Any`): gRPC message class to wrap each chunk in (must accept
            `transfer_state` and `data` kwargs).
        log_prefix (`str`, *optional*, defaults to `""`): Prefix prepended to progress log lines.
        silent (`bool`, *optional*, defaults to `True`): Whether to log progress at `DEBUG` level
            instead of `INFO`.

    Yields:
        An instance of `message_class` for each chunk.
    """
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
    """Reassemble chunks yielded by `send_bytes_in_chunks` into the original bytes.

    Args:
        iterator (`Iterable`): Iterable of gRPC transfer messages produced by `send_bytes_in_chunks`,
            each with `transfer_state` and `data`.
        queue (`Queue | None`): When set, each fully reassembled buffer is put on this queue
            instead of being returned, and the function keeps reading further transfers.
        shutdown_event (`multiprocessing.synchronize.Event`): Checked before processing each item;
            returns immediately when set.
        log_prefix (`str`, *optional*, defaults to `""`): Prefix prepended to log lines.

    Returns:
        `bytes | None`: The reassembled buffer, when `queue` is `None`. Returns `None` (implicitly)
            if `shutdown_event` fires or `iterator` is exhausted mid-transfer.

    Raises:
        ValueError: If a message carries an unrecognized `transfer_state`.
    """
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
    """Serialize a model state dict for transmission over gRPC.

    Args:
        state_dict (`dict[str, torch.Tensor]`): State dict to serialize.

    Returns:
        `bytes`: The serialized state dict.
    """
    bytes_buffer = io.BytesIO()

    torch.save(state_dict, bytes_buffer)

    return bytes_buffer.getvalue()


def bytes_to_state_dict(buffer: bytes) -> dict[str, torch.Tensor]:
    """Deserialize a model state dict produced by `state_to_bytes`.

    Args:
        buffer (`bytes`): Serialized state dict.

    Returns:
        `dict[str, torch.Tensor]`: The deserialized state dict.
    """
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    return torch.load(bytes_buffer, weights_only=True)


def python_object_to_bytes(python_object: Any) -> bytes:
    """Pickle an arbitrary Python object for transmission over gRPC.

    Args:
        python_object (`Any`): Object to serialize.

    Returns:
        `bytes`: The pickled object.
    """
    return pickle.dumps(python_object)


def bytes_to_python_object(buffer: bytes) -> Any:
    """Unpickle an object produced by `python_object_to_bytes`.

    Args:
        buffer (`bytes`): Pickled object.

    Returns:
        `Any`: The deserialized object.
    """
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    obj = pickle.load(bytes_buffer)  # nosec B301: Safe usage of pickle.load
    # Add validation checks here
    return obj


def bytes_to_transitions(buffer: bytes) -> list[Transition]:
    """Deserialize a list of `Transition`s produced by `transitions_to_bytes`.

    Args:
        buffer (`bytes`): Serialized transitions.

    Returns:
        `list[Transition]`: The deserialized transitions.
    """
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    transitions = torch.load(bytes_buffer, weights_only=True)
    return transitions


def transitions_to_bytes(transitions: list[Transition]) -> bytes:
    """Serialize a list of `Transition`s for transmission over gRPC.

    Args:
        transitions (`list[Transition]`): Transitions to serialize.

    Returns:
        `bytes`: The serialized transitions.
    """
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
    """Build gRPC channel options with message-size limits and a retry policy.

    Args:
        max_receive_message_length (`int`, *optional*, defaults to 4194304): Maximum
            message size, in bytes, the channel can receive.
        max_send_message_length (`int`, *optional*, defaults to 4194304): Maximum
            message size, in bytes, the channel can send.
        enable_retries (`bool`, *optional*, defaults to `True`): Whether to enable gRPC's built-in
            retry policy.
        initial_backoff (`str`, *optional*, defaults to `"0.1s"`): Delay before the first retry.
        max_attempts (`int`, *optional*, defaults to 5): Maximum total attempts (including the
            initial call).
        backoff_multiplier (`float`, *optional*, defaults to 2): Exponential backoff multiplier
            applied between retries.
        max_backoff (`str`, *optional*, defaults to `"2s"`): Maximum delay between retries.

    Returns:
        `list[tuple[str, Any]]`: Channel options suitable for `grpc.insecure_channel(..., options=...)`.
    """
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
