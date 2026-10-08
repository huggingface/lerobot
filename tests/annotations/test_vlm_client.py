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

import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.annotations.steerable_pipeline.config import VlmConfig  # noqa: E402
from lerobot.annotations.steerable_pipeline.vlm_client import _bind_serve_port, make_vlm_client  # noqa: E402


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


def test_api_key_is_read_from_environment(monkeypatch) -> None:
    constructor = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))
    config = VlmConfig(
        auto_serve=False, api_key_env="TEST_PLANNER_KEY", request_timeout_s=30, request_max_retries=0
    )
    monkeypatch.delenv("TEST_PLANNER_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_PLANNER_KEY is not set"):
        make_vlm_client(config)
    constructor.assert_not_called()
    monkeypatch.setenv("TEST_PLANNER_KEY", "test-secret")
    make_vlm_client(config)
    constructor.assert_called_once_with(
        base_url=config.api_base, api_key="test-secret", timeout=30, max_retries=0
    )
    assert "test-secret" not in str(asdict(config))
