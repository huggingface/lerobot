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

"""Versioned message envelopes, a size-limited codec and request/reply helpers shared by LeRobot services."""

from .client import decode_reply, query_reply
from .codec import CodecLimits, RGBImage, decode_message, encode_message, peek_envelope
from .envelope import (
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
    correlate_reply,
    validate_reply,
    validate_segment,
)

__all__ = [
    "CodecLimits",
    "Envelope",
    "ErrorCode",
    "MessageType",
    "ProtocolError",
    "RGBImage",
    "correlate_reply",
    "decode_message",
    "decode_reply",
    "encode_message",
    "peek_envelope",
    "query_reply",
    "validate_reply",
    "validate_segment",
]
