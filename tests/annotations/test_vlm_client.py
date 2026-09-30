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
"""Unit tests for ``vlm_client`` helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.annotations.steerable_pipeline.vlm_client import _bind_serve_port  # noqa: E402


def test_bind_serve_port_substitutes_placeholder() -> None:
    # The {port} placeholder is replaced everywhere it appears, regardless of
    # parallel vs single server — the bug was the single-server path passing
    # it through unsubstituted.
    cmd = "vllm serve M --max-model-len 32768 --port {port}"
    assert _bind_serve_port(cmd, 8000) == "vllm serve M --max-model-len 32768 --port 8000"


def test_bind_serve_port_appends_when_missing() -> None:
    assert _bind_serve_port("vllm serve M", 8001) == "vllm serve M --port 8001"


def test_bind_serve_port_leaves_explicit_port_untouched() -> None:
    cmd = "vllm serve M --port 9000"
    assert _bind_serve_port(cmd, 8000) == cmd


def test_responses_client_uses_env_key_and_vision_without_chat_parameters(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from PIL import Image

    from lerobot.annotations.steerable_pipeline.config import VlmConfig
    from lerobot.annotations.steerable_pipeline.vlm_client import make_vlm_client

    sdk = MagicMock()
    sdk.return_value.responses.create.return_value = SimpleNamespace(
        status="completed", output_text='{"answer":"ok"}'
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=sdk))
    monkeypatch.setenv("TEST_VLM_KEY", "synthetic-key")
    cfg = VlmConfig(
        api_mode="responses",
        api_key_env="TEST_VLM_KEY",
        api_base="https://api.openai.com/v1",
        auto_serve=False,
        model_id="gpt-6.1-sol",
        reasoning_effort="low",
        request_timeout_s=20,
        request_max_retries=0,
    )
    client = make_vlm_client(cfg)
    result = client.generate_json(
        [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "JSON please"},
                        {"type": "image", "image": Image.new("RGB", (4, 4))},
                    ],
                }
            ]
        ]
    )
    assert result == [{"answer": "ok"}]
    sdk.assert_called_once_with(base_url=cfg.api_base, api_key="synthetic-key", timeout=20, max_retries=0)
    request = sdk.return_value.responses.create.call_args.kwargs
    assert request["input"][0]["content"][1]["type"] == "input_image"
    assert request["input"][0]["content"][1]["image_url"].startswith("data:image/")
    assert request["model"] == "gpt-6.1-sol"
    assert request["reasoning"] == {"effort": "low"}
    assert "temperature" not in request and "max_tokens" not in request and "extra_body" not in request
    assert not request["store"]
    assert cfg.api_key == "EMPTY"  # secret never copied into the logged dataclass


def test_missing_env_key_fails_before_client_creation(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from lerobot.annotations.steerable_pipeline.config import VlmConfig
    from lerobot.annotations.steerable_pipeline.vlm_client import make_vlm_client

    sdk = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=sdk))
    monkeypatch.delenv("TEST_VLM_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_VLM_KEY"):
        make_vlm_client(VlmConfig(api_key_env="TEST_VLM_KEY", auto_serve=False))
    sdk.assert_not_called()
