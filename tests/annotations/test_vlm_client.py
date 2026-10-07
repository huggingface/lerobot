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
