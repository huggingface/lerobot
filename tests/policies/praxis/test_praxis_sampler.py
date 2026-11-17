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

"""The window-joint sampler on the tiny random model: the passes it skips keep the bits.

The sampler runs the real tiny backbone and action expert on the CPU; spies record the trunk
calls (``decode``) and the expert calls. A traced call runs every pass, so it is the reference
of the untraced call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from lerobot.policies.praxis.praxis_core import model as model_mod
from lerobot.policies.praxis.praxis_core.layers import RotaryEmbedding, to_device_async
from lerobot.policies.praxis.praxis_core.model import PraxisModel, SamplerSettings

from .tiny_model import LATENT_CHANNELS, TEXT_DIM, make_tiny_policy

pytest.importorskip("transformers", reason="praxis requires the praxis extra (transformers)")

LAT = 6  # latents per chunk
GRID = (1, 4)  # the latent grid of the tiny canvas (two 32 x 64 tiles)
P_DIM = 10  # the single-arm proprio
#: The real expert call, wrapped by the spies (captured once, so spies never wrap spies).
_RUN_EXPERT = model_mod.run_expert_on_taps_k

# (depth, steps, t_shift, cfg_scale, cfg_video_scale)
CASES = [
    (0, 2, 12.5, 1.0, 4.0),  # the recommended settings: video guidance only
    (1, 2, 12.5, 1.0, 4.0),
    (0, 4, 12.5, 1.0, 4.0),  # the so101 serve preset
    (1, 3, 8.0, 3.0, 1.0),  # action guidance only
    (2, 2, 1.0, 3.0, 2.0),  # both
    (1, 2, 1.0, 1.0, 1.0),  # no guidance
]


@dataclass
class Calls:
    trunk: list[dict] = field(default_factory=list)
    expert: list[dict] = field(default_factory=list)


def _model(calls: Calls, monkeypatch: pytest.MonkeyPatch, seed: int = 0) -> PraxisModel:
    """The tiny model with spies on its trunk and expert calls (the real modules run)."""
    model = make_tiny_policy(seed=seed).model  # the tiny policy's model, with its CPU attention settings
    trunk = model.backbone.forward_chunk_causal

    def trunk_spy(*args, decode: bool = True, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.trunk.append({"decode": decode, "noised": args[3].clone()})
        return trunk(*args, decode=decode, **kwargs)

    def expert_spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.expert.append({"noised_actions": kwargs["noised_actions"].clone()})
        return _RUN_EXPERT(*args, **kwargs)

    monkeypatch.setattr(model.backbone, "forward_chunk_causal", trunk_spy)
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", expert_spy)
    return model


def _inputs(depth: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(1000 + seed)
    return {
        "text": torch.randn(1, 3, TEXT_DIM, generator=g).to(torch.bfloat16),
        "text_uncond": torch.randn(1, 1, TEXT_DIM, generator=g).to(torch.bfloat16),
        "anchor": torch.randn(1, LATENT_CHANNELS, 1, *GRID, generator=g).to(torch.bfloat16),
        "cleans": [
            torch.randn(1, LATENT_CHANNELS, LAT, *GRID, generator=g).to(torch.bfloat16) for _ in range(depth)
        ],
        "proprio_all": torch.randn(1, depth + 1, P_DIM, generator=g),
    }


def _settings(steps: int, shift: float, cfg: float, cfg_v: float) -> SamplerSettings:
    return SamplerSettings(num_steps=steps, t_shift=shift, cfg_scale=cfg, cfg_video_scale=cfg_v)


@pytest.mark.parametrize(("depth", "steps", "shift", "cfg", "cfg_v"), CASES)
def test_unguided_pass_runs_the_expert_only_for_action_guidance(
    monkeypatch: pytest.MonkeyPatch, depth: int, steps: int, shift: float, cfg: float, cfg_v: float
) -> None:
    """With ``cfg_scale`` 1.0 the unguided pass runs no expert, and the bits stay those of a traced call.

    A trace keeps the unguided actions (and the unguided video), so a traced call runs every pass.
    """
    settings = _settings(steps, shift, cfg, cfg_v)
    guided = cfg != 1.0 or cfg_v != 1.0
    out, calls_of = {}, {}
    for traced in (False, True):
        calls = Calls()
        model = _model(calls, monkeypatch)
        out[traced] = model.sample_window_joint(
            seed=5, settings=settings, trace=(lambda n, t: None) if traced else None, **_inputs(depth)
        )
        calls_of[traced] = calls
    assert len(calls_of[False].trunk) == steps * (2 if guided else 1)
    assert len(calls_of[False].expert) == steps * (2 if cfg != 1.0 else 1)
    assert len(calls_of[True].expert) == steps * (2 if guided else 1)
    assert torch.equal(out[False][0], out[True][0])
    assert torch.equal(out[False][1], out[True][1])


def test_to_device_async_gives_the_values_of_to() -> None:
    t = torch.arange(37, dtype=torch.long)
    assert torch.equal(to_device_async(t, torch.device("cpu")), t.to("cpu"))
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        got = to_device_async(t, dev)
        assert got.device.type == "cuda" and torch.equal(got.cpu(), t)
        back = to_device_async(got, dev)  # a device tensor is moved as `.to` moves it
        assert back.data_ptr() == got.data_ptr()


def test_rotary_table_keeps_one_frequency_copy_per_device() -> None:
    rope = RotaryEmbedding(dim=16, effective_dim=8)
    positions = torch.arange(12, dtype=torch.long)[None] * 1_000_003
    reference = RotaryEmbedding(dim=16, effective_dim=8)
    first = rope(positions)
    second = rope(positions)
    assert torch.equal(first, second) and torch.equal(first, reference(positions))
    assert list(rope._theta_on) == [positions.device]
    assert rope._theta_on[positions.device].dtype == torch.float64
