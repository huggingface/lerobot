#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import os

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi0 import PI0Config  # noqa: E402
from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch  # noqa: E402
from tests.policies.pi0_pi05.utils.torch_compile import (  # noqa: E402
    assert_cache_stability,
    assert_compiled_output_matches_eager,
    assert_explain_has_no_graph_breaks,
    benchmark_runtime,
    make_compile_config,
    reset_compile_state,
)
from tests.utils import require_cuda  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="torch.compile benchmark is too slow for CI; run manually on GPU nodes",
)


def _make_model(*, compile_model):
    return PI0Pytorch(make_compile_config(PI0Config, compile_model=compile_model)).cuda().eval()


def _make_dummy_inputs(config):
    device = torch.device("cuda")
    common = {
        "images": [torch.randn(1, 3, *config.image_resolution, device=device)],
        "img_masks": [torch.ones(1, dtype=torch.bool, device=device)],
        "lang_tokens": torch.randint(0, 1024, (1, 5), dtype=torch.long, device=device),
        "lang_masks": torch.ones(1, 5, dtype=torch.bool, device=device),
        "state": torch.randn(1, config.max_state_dim, device=device),
    }
    forward_kwargs = {
        **common,
        "actions": torch.randn(1, config.chunk_size, config.max_action_dim, device=device),
        "noise": torch.randn(1, config.chunk_size, config.max_action_dim, device=device),
        "time": torch.rand(1, device=device),
    }
    sample_kwargs = {
        **common,
        "noise": torch.randn(1, config.chunk_size, config.max_action_dim, device=device),
        "num_steps": config.num_inference_steps,
    }
    return forward_kwargs, sample_kwargs


@require_cuda
def test_pi0_torch_compile_forward_and_sample_actions():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available")
    if not torch._dynamo.is_dynamo_supported():
        pytest.skip("torch._dynamo is not supported on this platform")

    torch.manual_seed(0)
    eager_model = _make_model(compile_model=False)
    torch.manual_seed(0)
    compiled_model = _make_model(compile_model=True)
    forward_kwargs, sample_kwargs = _make_dummy_inputs(compiled_model.config)

    try:
        assert_compiled_output_matches_eager(eager_model, compiled_model, forward_kwargs, sample_kwargs)

        assert_explain_has_no_graph_breaks(eager_model.forward, forward_kwargs, "pi0.forward")
        assert_explain_has_no_graph_breaks(eager_model.sample_actions, sample_kwargs, "pi0.sample_actions")

        assert_cache_stability(compiled_model.forward, forward_kwargs, "pi0.forward")
        assert_cache_stability(compiled_model.sample_actions, sample_kwargs, "pi0.sample_actions")

        benchmark_runtime(eager_model.forward, compiled_model.forward, forward_kwargs, "pi0.forward")
        benchmark_runtime(
            eager_model.sample_actions, compiled_model.sample_actions, sample_kwargs, "pi0.sample_actions"
        )
    finally:
        reset_compile_state()
        del eager_model
        del compiled_model
        torch.cuda.empty_cache()
