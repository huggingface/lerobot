#!/usr/bin/env python

"""Tests for dependency-light semantic-message helpers."""

import pytest

from lerobot.policies.language import (
    join_semantic_message_text,
    last_semantic_message_text,
    normalize_semantic_messages,
    require_single_semantic_conversation,
    require_single_text_output,
    semantic_message_content_text,
)


def test_semantic_message_helpers_support_multimodal_conversations():
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "feature": "observation.images.front"},
                {"type": "text", "text": "What is visible?"},
            ],
        },
        {"role": "assistant", "content": "A blue cube."},
    ]

    assert normalize_semantic_messages(conversation, policy_name="Test") == [conversation]
    assert require_single_semantic_conversation([conversation], policy_name="Test") == conversation
    assert semantic_message_content_text(conversation[0]["content"]) == "What is visible?"
    assert join_semantic_message_text(conversation) == "What is visible?\nA blue cube."
    assert last_semantic_message_text(conversation, role="user") == "What is visible?"
    assert require_single_text_output([" answer "], policy_name="Test") == "answer"


def test_semantic_message_helpers_reject_invalid_cardinality():
    conversation = [{"role": "user", "content": "question"}]

    with pytest.raises(ValueError, match="expected one Test prompt"):
        require_single_semantic_conversation([conversation, conversation], policy_name="Test")
    with pytest.raises(ValueError, match="expected one Test text output"):
        require_single_text_output(["one", "two"], policy_name="Test")
