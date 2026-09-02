#!/usr/bin/env python

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

"""Dependency-light helpers for the semantic message contract shared by policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def semantic_message_content_text(content: Any) -> str:
    """Return the text payload from plain or HF-style multimodal message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, str | bytes):
        return "\n".join(
            str(block["text"])
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text" and "text" in block
        )
    if content is None:
        return ""
    return str(content)


def normalize_semantic_messages(
    messages: Any,
    *,
    policy_name: str,
    batch_size: int | None = None,
) -> list[list[Mapping[str, Any]]]:
    """Normalize one conversation or a batch of conversations and validate its shape."""
    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes) or not messages:
        raise ValueError(f"{policy_name} text generation requires preprocessed `messages`.")

    if isinstance(messages[0], Mapping):
        conversations = [list(messages)]
    else:
        conversations = []
        for conversation in messages:
            if not isinstance(conversation, Sequence) or isinstance(conversation, str | bytes):
                raise TypeError(f"{policy_name} semantic messages must be conversations of mappings.")
            conversations.append(list(conversation))

    if batch_size is not None and len(conversations) != batch_size:
        raise ValueError(
            f"{policy_name} expected {batch_size} semantic-message conversations, got {len(conversations)}."
        )
    for conversation in conversations:
        if not conversation or not all(isinstance(message, Mapping) for message in conversation):
            raise TypeError(f"{policy_name} semantic messages must be non-empty mappings.")
    return conversations


def require_single_semantic_conversation(
    messages: Any,
    *,
    policy_name: str,
) -> list[Mapping[str, Any]]:
    """Return the only semantic conversation expected by the interactive runtime."""
    conversations = normalize_semantic_messages(messages, policy_name=policy_name)
    if len(conversations) != 1:
        raise ValueError(
            f"The interactive runtime expected one {policy_name} prompt, got {len(conversations)}."
        )
    return conversations[0]


def join_semantic_message_text(
    messages: Sequence[Mapping[str, Any]],
    *,
    role: str | None = None,
    separator: str = "\n",
) -> str:
    """Join non-empty text payloads, optionally selecting one message role."""
    texts = [
        semantic_message_content_text(message.get("content"))
        for message in messages
        if role is None or message.get("role") == role
    ]
    return separator.join(text for text in texts if text)


def last_semantic_message_text(
    messages: Sequence[Mapping[str, Any]],
    *,
    role: str | None = None,
) -> str:
    """Return the last non-empty semantic-message text for the selected role."""
    for message in reversed(messages):
        if role is not None and message.get("role") != role:
            continue
        text = semantic_message_content_text(message.get("content"))
        if text:
            return text
    role_suffix = "" if role is None else f" with role {role!r}"
    raise ValueError(f"Semantic messages contain no text{role_suffix}.")


def require_single_text_output(outputs: Sequence[str], *, policy_name: str) -> str:
    """Validate the single-output interactive contract and strip decoder whitespace."""
    if len(outputs) != 1:
        raise ValueError(
            f"The interactive runtime expected one {policy_name} text output, got {len(outputs)}."
        )
    output = outputs[0]
    if not isinstance(output, str):
        raise TypeError(
            f"The interactive runtime expected {policy_name} to return a text output, "
            f"got {type(output).__name__}."
        )
    return output.strip()
