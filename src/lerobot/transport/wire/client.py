# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Common bounded client exchanges; service lifecycles and retries stay with callers."""

from collections.abc import Callable
from typing import Any

from .codec import decode_message, encode_message, peek_envelope
from .protocol import (
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
    correlate_reply,
    raise_peer_error,
    validate_reply,
)


def query_reply(
    transport: Any,
    key: str,
    request: Envelope,
    timeout: float,
    expected: MessageType,
    *,
    encoder: Callable[[Envelope], bytes] = encode_message,
    check_instance: bool = True,
    check_session: bool = True,
    cancelled: Callable[[], bool] | None = None,
    **query_options: Any,
) -> Envelope:
    """Issue one query without retries, requiring one correlated, validated response."""
    if cancelled is not None:
        query_options["cancelled"] = cancelled
    replies = transport.query(key, encoder(request), timeout, **query_options)
    if len(replies) != 1:
        raise TimeoutError("Expected one reply within its deadline")
    response = decode_message(replies[0])
    validate_reply(response, request, expected, check_instance=check_instance, check_session=check_session)
    return response


def decode_reply(payload: bytes, request: Envelope, *, discard_stale: bool = False) -> Envelope | None:
    """Correlate before decoding large bodies; optionally discard obsolete channel traffic."""
    try:
        correlate_reply(peek_envelope(payload), request)
    except ProtocolError as exc:
        if discard_stale and exc.code is ErrorCode.STALE:
            return None
        raise
    response = decode_message(payload)
    raise_peer_error(response)
    return response
