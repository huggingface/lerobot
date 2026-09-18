# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.rollout.agent.provider import OpenAIReasoner
from lerobot.rollout.agent.tools import TOOLS


def test_supervisor_sends_images_and_parses_one_tool_without_storing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    response = MagicMock()
    response.json.return_value = {
        "output": [{"type": "function_call", "name": "steer", "arguments": '{"instruction":"grasp tape"}'}]
    }
    post = MagicMock(return_value=response)
    monkeypatch.setattr("lerobot.rollout.agent.provider.requests.post", post)
    reasoner = OpenAIReasoner()
    snapshot = {"revision": 7, "raw": {"base": np.zeros((8, 8, 3), dtype=np.uint8)}}
    assert reasoner.decide(snapshot, TOOLS, {}) == [
        {"name": "steer", "arguments": {"instruction": "grasp tape"}}
    ]
    body = post.call_args.kwargs["json"]
    assert body["store"] is False
    assert body["parallel_tool_calls"] is False
    assert body["reasoning"] == {"effort": "low"}
    content = body["input"][0]["content"]
    assert json.loads(content[0]["text"])["observation"]["revision"] == 7
    assert content[-1]["image_url"].startswith("data:image/jpeg;base64,")
    response.json.return_value["output"] *= 2
    with pytest.raises(ValueError, match="at most one"):
        reasoner.decide(snapshot, TOOLS, {})
