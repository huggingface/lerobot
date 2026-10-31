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
"""The video VAE attention without NATTEN: NATTEN's neighborhoods in plain PyTorch, also under torch.export
with static shapes."""

import itertools
import logging
import math

import pytest
import torch

from lerobot.policies.flux3.f3 import neighborhood, video_vae
from lerobot.policies.flux3.f3.neighborhood import neighborhood_attention
from lerobot.policies.flux3.f3.runtime import random_init_
from lerobot.policies.flux3.f3.video_vae import Natten3D, VideoVAE, ViTNormInference, ViTNormInferenceParams

# The keys each query of one axis attends to, measured on NATTEN 0.21.6 and 0.21.7:
# (axis size, window, causal) -> keys.
NATTEN_WINDOWS = {
    (6, 4, False): [[0, 1, 2, 3]] * 3 + [[1, 2, 3, 4]] + [[2, 3, 4, 5]] * 2,
}


@pytest.mark.parametrize(("size", "kernel", "causal"), list(NATTEN_WINDOWS))
def test_windows_match_natten(size, kernel, causal):
    # Zero queries attend uniformly, so one-hot values show which keys each query sees.
    q = torch.zeros(1, size, 1, size)
    v = torch.eye(size).reshape(1, size, 1, size)
    out = neighborhood_attention(q, torch.randn(1, size, 1, size), v, [kernel], [causal])[0, :, 0]
    expected = torch.zeros(size, size)
    for query, keys in enumerate(NATTEN_WINDOWS[(size, kernel, causal)]):
        expected[query, keys] = 1 / len(keys)
    torch.testing.assert_close(out, expected)


def test_window_longer_than_its_axis_raises():
    q = torch.zeros(1, 4, 1, 8)
    with pytest.raises(ValueError, match="needs at least 5 positions"):
        neighborhood_attention(q, q, q, [5], [False])


def _reference(q, k, v, kernel_size, is_causal):
    """Softmax attention of each query over the product of its per-axis windows, one query at a time."""
    spatial = q.shape[1:-2]
    out = torch.empty(q.shape, dtype=torch.float64)
    for position in itertools.product(*map(range, spatial)):
        axes = []
        for i, size, kernel, causal in zip(position, spatial, kernel_size, is_causal, strict=True):
            start = min(max(i - (kernel - 1 if causal else kernel // 2), 0), size - kernel)
            axes.append([j for j in range(start, start + kernel) if not causal or j <= i])
        keys = torch.stack([k[(slice(None), *j)] for j in itertools.product(*axes)], -2).double()
        values = torch.stack([v[(slice(None), *j)] for j in itertools.product(*axes)], -2).double()
        query = q[(slice(None), *position)].double().unsqueeze(-1)
        weights = ((keys @ query).squeeze(-1) * q.shape[-1] ** -0.5).softmax(-1)
        out[(slice(None), *position)] = (weights.unsqueeze(-2) @ values).squeeze(-2)
    return out


def test_matches_reference(monkeypatch):
    # Two causal axes, in chunks of three queries, so chunks also end within a row.
    spatial, kernel_size, is_causal = (5, 4, 4), (5, 3, 3), (True, True, False)
    # The target counts the keys a chunk gathers: batch x heads x head_dim x window elements per query.
    monkeypatch.setattr(neighborhood, "_NEIGHBORHOOD_CHUNK_ELEMENTS", 3 * 2 * 2 * 8 * math.prod(kernel_size))
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 2, *spatial, 2, 8).unbind(0)
    out = neighborhood_attention(q, k, v, kernel_size, is_causal)
    torch.testing.assert_close(out, _reference(q, k, v, kernel_size, is_causal).float())


def test_chunks_keep_to_the_target(monkeypatch):
    # A frame larger than the target is split by rows, and a row larger than it into parts, so no chunk
    # gathers more keys than the target.
    gathered = []
    gather = neighborhood._neighbors

    def recorded(x, windows):
        keys = gather(x, windows)
        gathered.append(keys.numel())
        return keys

    monkeypatch.setattr(neighborhood, "_neighbors", recorded)
    target = 4 * 2 * 8 * 27  # four queries' keys (heads x head_dim x window elements each), less than a row
    monkeypatch.setattr(neighborhood, "_NEIGHBORHOOD_CHUNK_ELEMENTS", target)
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 1, 4, 6, 6, 2, 8).unbind(0)  # frames of 36 queries
    out = neighborhood_attention(q, k, v, [3, 3, 3], [True, False, False])
    assert max(gathered) <= target
    torch.testing.assert_close(out, _reference(q, k, v, [3, 3, 3], [True, False, False]).float())


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
    ],
)
def test_computes_in_float32_under_autocast(device):
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 1, 4, 5, 6, 2, 16, device=device).unbind(0)
    expected = neighborhood_attention(q, k, v, [3, 3, 3], [True, False, False])
    with torch.autocast(device, dtype=torch.bfloat16):
        out = neighborhood_attention(q, k, v, [3, 3, 3], [True, False, False])
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not (video_vae._natten_available and torch.cuda.is_available()), reason="needs NATTEN and CUDA"
)
@pytest.mark.parametrize(
    ("spatial", "kernel_size", "is_causal"),
    [((7, 9), (5, 5), (False, False)), ((6, 7, 9), (5, 5, 5), (True, False, False))],
)
def test_matches_natten(spatial, kernel_size, is_causal):
    # The VAE's windows against NATTEN's own kernels, so a NATTEN upgrade that changes them shows here.
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 1, *spatial, 4, 64, device="cuda").unbind(0)
    na = video_vae.na2d if len(spatial) == 2 else video_vae.na3d
    expected = na(q, k, v, kernel_size=kernel_size, is_causal=is_causal)
    out = neighborhood_attention(q, k, v, kernel_size, is_causal)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_computes_bfloat16_inputs_in_float32():
    # The VAE runs in bfloat16, and bfloat16 products would round the logits.
    torch.manual_seed(0)
    q, k, v = (4 * torch.randn(3, 1, 4, 5, 6, 2, 16)).bfloat16().unbind(0)
    out = neighborhood_attention(q, k, v, [3, 3, 3], [True, False, False])
    expected = neighborhood_attention(q.float(), k.float(), v.float(), [3, 3, 3], [True, False, False])
    torch.testing.assert_close(out, expected.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize(("frames", "moved_frame", "changed_frames"), [(1, 0, [0]), (4, 1, [1, 2, 3])])
def test_attention_block_attends_its_window(frames, moved_frame, changed_frames):
    # Moving one input token changes the outputs whose window holds it: a window of 5 rows reaches all 7 rows
    # from row 3, a window of 3 columns reaches columns 2 to 4, and a video is causal in time.
    torch.manual_seed(0)
    block = Natten3D(16, [3, 5, 3], 2, causal=True).eval()
    x = torch.randn(1, frames, 7, 7, 16)
    moved = x.clone()
    moved[0, moved_frame, 3, 3] += 1
    with torch.no_grad():
        changed = (block(moved) - block(x)).abs().amax(-1)[0] > 0
    expected = torch.zeros(frames, 7, 7, dtype=torch.bool)
    expected[changed_frames, :, 2:5] = True
    assert torch.equal(changed, expected)


def test_attention_block_exports():
    torch.manual_seed(0)
    block = Natten3D(32, [3, 3, 3], 2, causal=True, qk_norm=True).eval()
    x = torch.randn(1, 4, 5, 6, 32)
    program = torch.export.export(block, (x,))
    torch.testing.assert_close(program.module()(x), block(x))


def test_attention_block_gives_natten_kernels_no_cpu_tensors(monkeypatch):
    def natten_kernel(*args, **kwargs):
        raise AssertionError("NATTEN's kernel got a CPU tensor")

    monkeypatch.setattr(video_vae, "_natten_available", True)
    monkeypatch.setattr(video_vae, "na2d", natten_kernel, raising=False)
    monkeypatch.setattr(video_vae, "na3d", natten_kernel, raising=False)
    torch.manual_seed(0)
    block = Natten3D(16, [3, 3, 3], 2, causal=True).eval()
    for frames in (1, 3):
        with torch.no_grad():
            assert block(torch.randn(1, frames, 4, 4, 16)).shape == (1, frames, 4, 4, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_attention_block_gives_natten_kernels_cuda_tensors(monkeypatch):
    calls = []

    def recorder(name):
        def natten_kernel(q, k, v, kernel_size, is_causal, backend):
            assert q.is_cuda and k.is_cuda and v.is_cuda
            calls.append((name, tuple(q.shape), list(kernel_size), list(is_causal)))
            return v

        return natten_kernel

    monkeypatch.setattr(video_vae, "_natten_available", True)
    monkeypatch.setattr(video_vae, "na2d", recorder("na2d"), raising=False)
    monkeypatch.setattr(video_vae, "na3d", recorder("na3d"), raising=False)
    torch.manual_seed(0)
    block = Natten3D(16, [3, 5, 3], 2, causal=True).eval().cuda()
    for frames in (1, 4):
        with torch.no_grad():
            assert block(torch.randn(1, frames, 6, 6, 16, device="cuda")).shape == (1, frames, 6, 6, 16)
    assert calls == [
        ("na2d", (1, 6, 6, 2, 8), [5, 3], [False, False]),
        ("na3d", (1, 4, 6, 6, 2, 8), [3, 5, 3], [True, False, False]),
    ]


def _small_vae() -> ViTNormInference:
    """The real VAE with fewer, smaller layers and random bfloat16 weights, like ``load_video_vae(None)``."""
    params = ViTNormInferenceParams(
        embed_dim=16,
        window_size=[3, 3, 3],
        enc_depths=[1, 1, 1, 1],
        dec_depths=[1, 1, 1, 1],
        num_heads=[1, 1, 2, 2],
        use_compile=False,
    )
    model = ViTNormInference(params)
    random_init_(model)
    return model


@pytest.mark.parametrize("frames", [1, 9])
def test_vae_encodes_with_the_plain_attention(frames, monkeypatch):
    # A small VAE on the CPU, in bfloat16 like the real one: a single frame takes 2D windows at every stage,
    # 9 frames take 3D windows through both temporal merges.
    dims = []

    def recorded(q, k, v, kernel_size, is_causal):
        dims.append(len(kernel_size))
        return neighborhood_attention(q, k, v, kernel_size, is_causal)

    monkeypatch.setattr(video_vae, "neighborhood_attention", recorded)
    model = _small_vae()
    torch.manual_seed(0)
    video = torch.rand(1, 3, frames, 96, 96, dtype=torch.bfloat16) * 2 - 1
    with torch.no_grad():
        latent = model.encode(video, chunked_encode=False)
    assert latent.shape == (1, model.params.z_dim, 1 + (frames - 1) // 4, 3, 3)
    assert latent.dtype == torch.bfloat16 and latent.isfinite().all()
    assert set(dims) == {2 if frames == 1 else 3}


@pytest.mark.parametrize(
    ("device", "warnings"),
    [
        ("cpu", 0),
        pytest.param("cuda", 1, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
    ],
)
def test_vae_warns_once_when_cuda_tensors_get_the_plain_attention(device, warnings, caplog, monkeypatch):
    # A GPU host that runs the policy on the CPU gets no warning: it depends on the tensors, not on the host.
    monkeypatch.setattr(video_vae, "_natten_available", False)
    video_vae._note_attention.cache_clear()
    vae = VideoVAE(_small_vae().to(device))
    video = torch.rand(1, 3, 1, 96, 96, dtype=torch.bfloat16, device=device) * 2 - 1
    with caplog.at_level(logging.WARNING, logger=video_vae.__name__):
        for _ in range(2):
            vae.encode_task(video)
    assert sum("NATTEN is not installed" in record.getMessage() for record in caplog.records) == warnings
    video_vae._note_attention.cache_clear()
