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

"""PI0Fast policy tests that need no model download."""

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from tests.processor.test_tokenizer_processor import (  # noqa: E402
    _assert_tiny_action_processor,
    _save_action_processor,
    _save_tiny_tokenizer,
)


def test_pi0_fast_policy_loads_action_tokenizer_saved_by_transformers_4(tmp_path, monkeypatch):
    import transformers.dynamic_module_utils

    from lerobot.policies.pi0_fast import modeling_pi0_fast
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig

    monkeypatch.setattr(transformers.dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    # The test only covers tokenizer loading, so it skips building the PaliGemma model.
    monkeypatch.setattr(modeling_pi0_fast, "PI0FastPytorch", lambda *_args, **_kwargs: torch.nn.Identity())
    _save_action_processor(tmp_path / "legacy_fast")
    _save_tiny_tokenizer(tmp_path / "text")
    config = PI0FastConfig(
        input_features={},
        output_features={},
        action_tokenizer_name=str(tmp_path / "legacy_fast"),
        text_tokenizer_name=str(tmp_path / "text"),
        device="cpu",
    )

    policy = modeling_pi0_fast.PI0FastPolicy(config)

    _assert_tiny_action_processor(policy.action_tokenizer)
