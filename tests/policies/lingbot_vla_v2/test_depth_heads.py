# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Parity tests for the ported alignment heads (``model_core/depth_heads.py``).

The heads are a verbatim port of upstream
``lingbotvla/models/vla/vision_models/align_heads`` — module and parameter
names are load-bearing because the released 6B "Native Depth" checkpoint stores
them under ``model.<head>.projector.*`` (exact-name strict loading). These
tests pin:

1. the state-dict key set matches the checkpoint contract exactly
   (proj_in1/proj_in2 — the TaskTokenResampler signature — layers.<L>.0
   PerceiverAttention internals, layers.<L>.1 FeedForward Sequential indices,
   proj_out, norm_out);
2. forward parity against the upstream implementation, bit-for-bit under the
   same seed and dtype (skipped when the upstream checkout is unavailable);
3. the geometry of the official recipe (2560 -> 1024, 256 queries).

Pure CPU, no teachers, no weights download.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lerobot.policies.lingbot_vla_v2.model_core.depth_heads import TaskTokenDepthHead  # noqa: E402

# Official RoboTwin geometry (robotwin.yaml): llm dim 2560, teacher dim 1024,
# 256 alignment queries, 1 perceiver layer, 4 heads x 32 dim_head, ff_mult 1.
HEAD_CFG = {
    "num_layers": 1,
    "num_heads": 4,
    "dim_head": 32,
    "ff_mult": 1,
    "num_backbone_tokens": 256,
    "dim_out": 1024,
}
LLM_HIDDEN = 2560

# The exact key set the released checkpoint carries per head (verified against
# /home/nvidia/platform/30-models/weights/lingbot-vla-v2-6b).
EXPECTED_STATE_KEYS = {
    "projector.proj_in1.weight",
    "projector.proj_in1.bias",
    "projector.proj_in2.weight",
    "projector.proj_in2.bias",
    "projector.layers.0.0.norm1.weight",
    "projector.layers.0.0.norm1.bias",
    "projector.layers.0.0.norm2.weight",
    "projector.layers.0.0.norm2.bias",
    "projector.layers.0.0.to_q.weight",
    "projector.layers.0.0.to_kv.weight",
    "projector.layers.0.0.to_out.weight",
    "projector.layers.0.1.0.weight",
    "projector.layers.0.1.0.bias",
    "projector.layers.0.1.1.weight",
    "projector.layers.0.1.3.weight",
    "projector.proj_out.weight",
    "projector.proj_out.bias",
    "projector.norm_out.weight",
    "projector.norm_out.bias",
}


def _import_upstream_head():
    """Load the upstream head modules by file path.

    Importing ``lingbotvla.models...`` normally runs the package ``__init__``
    chain, which imports transformers APIs that may not exist in the test env.
    The head modules are torch-only, so loading them directly from their files
    sidesteps the package init entirely.
    """
    import importlib.util
    import sys
    import types

    root = Path.home() / "lingbot-vla-v2-upstream"
    heads_dir = root / "lingbotvla" / "models" / "vla" / "vision_models" / "align_heads"
    if not (heads_dir / "depth_head.py").is_file():
        return None, None

    # Supply only the synthetic package chain needed to resolve depth_head.py's
    # relative ``from .resampler`` import; none of these package __init__ files
    # are executed.
    package = "_upstream_align_heads"
    package_module = types.ModuleType(package)
    package_module.__path__ = [str(heads_dir)]
    sys.modules[package] = package_module
    try:
        modules = {}
        for name, filename in (("resampler", "resampler.py"), ("depth_head", "depth_head.py")):
            qualified_name = f"{package}.{name}"
            spec = importlib.util.spec_from_file_location(qualified_name, heads_dir / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified_name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    except Exception:
        return None, None
    return modules["depth_head"].TaskTokenDepthHead, modules["resampler"].TaskTokenResampler


UPSTREAM = pytest.mark.skipif(_import_upstream_head()[0] is None, reason="upstream repo not available")


def test_state_dict_keys_match_checkpoint_contract():
    head = TaskTokenDepthHead(dict(HEAD_CFG), llm_hidden_size=LLM_HIDDEN)
    assert set(head.state_dict().keys()) == EXPECTED_STATE_KEYS
    # Geometry of the official recipe.
    assert head.projector.proj_in1.weight.shape == (LLM_HIDDEN, LLM_HIDDEN)
    assert head.projector.proj_out.weight.shape == (HEAD_CFG["dim_out"], LLM_HIDDEN)
    assert head.projector.layers[0][0].to_q.weight.shape == (
        HEAD_CFG["num_heads"] * HEAD_CFG["dim_head"],
        LLM_HIDDEN,
    )


def test_forward_shape_and_dtype():
    head = TaskTokenDepthHead(dict(HEAD_CFG), llm_hidden_size=LLM_HIDDEN).to(dtype=torch.bfloat16)
    torch.manual_seed(0)
    llm_feats = torch.randn(2, 73, LLM_HIDDEN, dtype=torch.bfloat16)  # 64 image tokens + 8 task tokens + 1
    queries = torch.randn(2, HEAD_CFG["num_backbone_tokens"], LLM_HIDDEN, dtype=torch.bfloat16)
    out = head(llm_feats, queries)
    assert out.shape == (2, HEAD_CFG["num_backbone_tokens"], HEAD_CFG["dim_out"])
    assert out.dtype == torch.bfloat16


@UPSTREAM
def test_forward_parity_with_upstream():
    upstream_head_cls, _ = _import_upstream_head()
    torch.manual_seed(42)
    ours = TaskTokenDepthHead(dict(HEAD_CFG), llm_hidden_size=LLM_HIDDEN).to(dtype=torch.float64)
    theirs = upstream_head_cls(dict(HEAD_CFG), llm_hidden_size=LLM_HIDDEN).to(dtype=torch.float64)
    # Same init seed => identical weights.
    ours.load_state_dict(theirs.state_dict())

    torch.manual_seed(7)
    llm_feats = torch.randn(3, 72, LLM_HIDDEN, dtype=torch.float64)
    queries = torch.randn(3, HEAD_CFG["num_backbone_tokens"], LLM_HIDDEN, dtype=torch.float64)
    out_ours = ours(llm_feats, queries)
    out_theirs = theirs(llm_feats, queries)
    torch.testing.assert_close(out_ours, out_theirs, rtol=0, atol=0)


@UPSTREAM
def test_state_dict_interchangeable_with_upstream():
    upstream_head_cls, _ = _import_upstream_head()
    torch.manual_seed(123)
    theirs = upstream_head_cls(dict(HEAD_CFG), llm_hidden_size=LLM_HIDDEN)
    ours = TaskTokenDepthHead(dict(HEAD_CFG), llm_hidden_size=LLM_HIDDEN)
    # strict=True: any name/shape mismatch fails here — this is the converter contract.
    ours.load_state_dict(theirs.state_dict(), strict=True)
