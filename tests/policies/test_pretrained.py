#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Regression tests for save/load helpers in :mod:`lerobot.policies.pretrained`.

Covers issue #3384 — ``RuntimeError: ... None is covering the entire storage`` that fired from
``safetensors.torch._remove_duplicate_names`` when a sub-module exposed a parameter that was a
non-complete view of a larger underlying storage.
"""

from __future__ import annotations

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from torch import nn

from lerobot.policies.pretrained import (
    _materialized_state_dict,
    _safe_load_state_into_model,
)


class _ModuleWithViewParam(nn.Module):
    """Module whose ``state_dict`` exposes a parameter that is a non-complete view.

    This mirrors what some ``transformers`` sub-modules can produce in the wild (issue #3384):
    ``q_proj.weight`` ends up being a slice of a larger storage allocated by the parent module,
    so it is registered as a parameter but does not own the entire underlying storage.
    """

    def __init__(self, in_features: int = 4, out_features: int = 8) -> None:
        super().__init__()
        # Allocate a single buffer big enough to hold three projections (Q, K, V).
        backing = torch.randn(3 * out_features, in_features)
        # Standard Linear modules — we then re-point each weight at a slice of the shared
        # backing storage so each parameter becomes a non-complete view. This reproduces the
        # state-dict shape that triggered ``RuntimeError: ... None is covering the entire
        # storage`` from ``safetensors._remove_duplicate_names`` before this fix.
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.v_proj = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            self.q_proj.weight.data = backing[:out_features]
            self.k_proj.weight.data = backing[out_features : 2 * out_features]
            self.v_proj.weight.data = backing[2 * out_features :]
        # Keep the backing storage alive on the module so the slices remain valid.
        self._backing = backing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


def _is_view_param(t: torch.Tensor) -> bool:
    return t.data_ptr() != t.untyped_storage().data_ptr() or (
        t.nelement() * t.element_size() != t.untyped_storage().nbytes()
    )


def test_module_with_view_param_actually_has_a_view():
    """Sanity check — without the fixture having an actual non-complete view there is no bug
    to regress. If pytorch ever changes ``nn.Linear`` so that ``.data = slice`` no longer
    creates a view, this test will tell us so we can update the fixture."""
    module = _ModuleWithViewParam()
    assert _is_view_param(module.q_proj.weight), (
        "Expected q_proj.weight to be a non-complete storage view; the regression test "
        "fixture is no longer reproducing issue #3384."
    )


def test_materialized_state_dict_breaks_views(tmp_path):
    """``_materialized_state_dict`` must produce tensors that own their own complete storage,
    so that ``safetensors.torch.save_file`` can persist them without raising the
    ``_remove_duplicate_names`` error."""
    module = _ModuleWithViewParam()
    materialized = _materialized_state_dict(module)

    for name, tensor in materialized.items():
        assert not _is_view_param(tensor), (
            f"{name} is still a non-complete view after materialization; saving would still fail."
        )

    # And we can actually save it. Before the fix this raised ``RuntimeError: ... None is
    # covering the entire storage``.
    from safetensors.torch import save_file

    save_path = tmp_path / SAFETENSORS_SINGLE_FILE
    save_file(materialized, str(save_path))
    assert save_path.exists()


def test_safe_load_round_trip_with_view_param(tmp_path):
    """End-to-end regression for issue #3384: save a module that has a view parameter, then
    load the resulting safetensors file back into a freshly-constructed module that *also*
    has a view parameter. The high-level ``safetensors.torch.load_model`` raised here; the
    new ``_safe_load_state_into_model`` helper must succeed and restore the values."""
    torch.manual_seed(0)
    src = _ModuleWithViewParam()

    # Save through the same path the policy uses.
    from safetensors.torch import save_file

    save_path = tmp_path / SAFETENSORS_SINGLE_FILE
    save_file(_materialized_state_dict(src), str(save_path))

    torch.manual_seed(1)  # different init to make sure load actually restores
    dst = _ModuleWithViewParam()
    assert not torch.allclose(src.q_proj.weight, dst.q_proj.weight), (
        "Source and destination modules accidentally share initial values; the round-trip "
        "test would not detect a no-op load."
    )

    missing, unexpected = _safe_load_state_into_model(
        dst, str(save_path), device="cpu", strict=True
    )
    assert missing == [], f"Unexpected missing keys after round-trip: {missing}"
    assert unexpected == [], f"Unexpected extra keys after round-trip: {unexpected}"

    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
        assert torch.allclose(src.state_dict()[name], dst.state_dict()[name]), (
            f"{name} was not restored correctly after the safetensors round-trip."
        )


def test_safe_load_handles_tied_weights(tmp_path):
    """When the destination model has tied weights (a parameter that points at another
    parameter's storage), keys that are absent from the file *because they are tied* should
    not show up in ``missing_keys`` — otherwise ``strict=True`` callers would always fail."""

    class TiedModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8, bias=False)
            # Tie the head weight to the embedding (classic LM tie).
            self.head.weight = self.embed.weight

    from safetensors.torch import save_file

    src = TiedModule()
    # The on-disk file should contain only one of the tied keys (whichever pytorch picks
    # first in state_dict iteration). We materialize and drop the tied alias to emulate a
    # "saved by transformers" file shape.
    sd = _materialized_state_dict(src)
    # Keep "embed.weight" only, drop the tied "head.weight" alias.
    sd.pop("head.weight", None)
    save_path = tmp_path / SAFETENSORS_SINGLE_FILE
    save_file(sd, str(save_path))

    dst = TiedModule()
    missing, unexpected = _safe_load_state_into_model(
        dst, str(save_path), device="cpu", strict=True
    )
    assert missing == [], (
        f"head.weight was reported as missing even though it is tied to embed.weight: {missing}"
    )
    assert unexpected == []
