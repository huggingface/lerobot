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
"""The camera canvas of the two single-arm checkpoints (CPU, no weights).

`praxis-1-droid` (the released checkpoint) has 224 x 416 tiles with an
antialiased resize (`video.antialias: true`): a centre crop to the native 720 x 1280 aspect, one
antialiased bilinear resize to 234 x 416 and a centre trim to 224 x 416. The earlier DROID checkpoint has 256 x 416 tiles
with a plain resize. The pinned hashes come from the praxis-1 repository's tests (`tests/test_data_path.py`
and `tests/test_single_arm.py`), computed with the original serve and training code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.praxis.configuration_praxis import PraxisConfig
from lerobot.policies.praxis.praxis_core import single_arm as sa
from lerobot.policies.praxis.praxis_core.checkpoint import CheckpointError, model_variant
from lerobot.policies.praxis.praxis_core.model import SamplerSettings
from lerobot.policies.praxis.processor_praxis import PraxisTileViewsProcessorStep, tile_views

from .tiny_model import tiny_lerobot_config

pytest.importorskip("transformers", reason="praxis requires the praxis extra (transformers)")

DATA = Path(__file__).parent / "checkpoint_configs"
M2_CONFIG = DATA / "praxis-1-droid-m2.json"
K_CONFIG = DATA / "praxis-1-droid-k.json"


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _views(seed: int, h: int = 288, w: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ext = rng.integers(0, 256, (2, h, w, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (2, h, w, 3), dtype=np.uint8)
    return ext, wrist


# SHA-256 of the uint8 canvas of the earlier DROID checkpoint (1, 3, 2, 256, 832) and a few pixels (channel, frame, row,
# column) of the canvas of the original implementation.
K_CANVAS_REF = {
    (720, 1280): (
        "82ab2a59e5177cf4ac791bd44b514b8ba3c4bc1fba0a5ad76dec4cc808bbccca",
        {
            (0, 0, 0, 0): 90,
            (1, 1, 100, 200): 80,
            (2, 0, 255, 415): 209,
            (0, 1, 17, 416): 38,
            (2, 1, 200, 831): 143,
        },
    ),
    (288, 512): (
        "8a6e0391dede7681e5c389e7a1e095cb1676678cfcf6c6e88cc21b932de49d54",
        {
            (0, 0, 0, 0): 198,
            (1, 1, 100, 200): 125,
            (2, 0, 255, 415): 100,
            (0, 1, 17, 416): 123,
            (2, 1, 200, 831): 121,
        },
    ),
}
# SHA-256 of the uint8 `praxis-1-droid` canvas (1, 3, 2, 224, 832) of the training data path and the serve
# code, on the same frames. A 416 x 234 frame, the size the client sends, is trimmed with no resize.
M2_CANVAS_REF = {
    (720, 1280): "e9a050a35e7c8d6fff1940109412f0375d5772b33293327c5cbf3c14dc47db48",
    (234, 416): "8046acf194e1b67ebedc19e5169035f9a3bf772192986c362ac706cadfbd5495",
    (288, 512): "cbec75a15f3995b104e092cc460303d5f4a3bdaa893e15db8aa01912106017ad",
}
# torch's scalar CPU kernels (`get_cpu_capability()` "DEFAULT": torch 2.11 on Neoverse V2, the GB300
# host CPU, which has no SVE128 kernels in that build) round a few pixels of the antialiased 3x
# downscale of a 720 x 1280 frame one level differently from the vectorized kernels (AVX2, AVX-512,
# SVE128), which give the praxis-1 hashes above. praxis-1's own test differs the same way there.
M2_CANVAS_REF_SCALAR_CPU = {
    (720, 1280): "00c9900f044319cb3f1e953eac0babf26d4cd4774380a3032182cef6d53f4770",
}


def _frames(size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = size
    rng = np.random.default_rng(0)
    ext = rng.integers(0, 256, size=(2, h, w, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(2, h, w, 3), dtype=np.uint8)
    return ext, wrist


@pytest.mark.parametrize("size", sorted(K_CANVAS_REF))
def test_k_tile_clip_equals_the_original_canvas(size: tuple[int, int]) -> None:
    clip = sa.tile_clip(*_frames(size))
    assert clip.shape == (1, 3, 2, 256, 832) and clip.dtype == torch.float32
    u8 = clip.to(torch.uint8)
    assert torch.equal(u8.float(), clip)
    sha, pixels = K_CANVAS_REF[size]
    assert _sha(u8.numpy()) == sha
    for (c, t, y, x), v in pixels.items():
        assert float(clip[0, c, t, y, x]) == v


@pytest.mark.parametrize("size", sorted(M2_CANVAS_REF))
def test_m2_tile_clip_equals_the_training_canvas(size: tuple[int, int]) -> None:
    clip = sa.tile_clip(*_frames(size), 224, 416, 0.03, antialias=True)
    assert clip.shape == (1, 3, 2, 224, 832)
    want = M2_CANVAS_REF[size]
    if torch.backends.cpu.get_cpu_capability() == "DEFAULT":
        want = M2_CANVAS_REF_SCALAR_CPU.get(size, want)
    assert _sha(clip.to(torch.uint8).numpy()) == want


def test_tile_clip_antialias_off_is_the_plain_bilinear_resize() -> None:
    """`antialias=False` (the default) is byte-identical to the resize before the flag."""
    ext, wrist = _views(0)
    clip = sa.tile_clip(ext, wrist, 224, 416, 0.03)
    assert torch.equal(clip, sa.tile_clip(ext, wrist, 224, 416, 0.03, antialias=False))
    fr = torch.from_numpy(ext).permute(0, 3, 1, 2)
    want = (
        torch.nn.functional.interpolate(
            sa.aspect_crop(fr, 224, 416, 0.03).float(), size=(224, 416), mode="bilinear", align_corners=False
        )
        .round()
        .clamp(0, 255)
    )
    assert torch.equal(clip[0, :, :, :, :416].permute(1, 0, 2, 3), want)


def test_tile_clip_antialias_on_is_the_fit_rule() -> None:
    """`antialias=True` crops to the native aspect, resizes to 234 x 416 and trims to 224 x 416."""
    ext, wrist = _views(1)
    on = sa.tile_clip(ext, wrist, 224, 416, 0.03, antialias=True)
    off = sa.tile_clip(ext, wrist, 224, 416, 0.03, antialias=False)
    assert not torch.equal(on, off)
    assert sa.fit_resize_size(416) == (234, 416)
    fr = torch.from_numpy(wrist).permute(0, 3, 1, 2)
    fit = (
        torch.nn.functional.interpolate(
            sa.aspect_crop(fr, 720, 1280, 0.03).float(),
            size=(234, 416),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)
    )
    assert torch.equal(on[0, :, :, :, 416:].permute(1, 0, 2, 3), fit[..., 5:229, :])


def test_tile_clip_antialias_on_does_not_resize_a_fit_size_input() -> None:
    """A 416 x 234 view (what the client sends) is only trimmed to 416 x 224."""
    rng = np.random.default_rng(3)
    ext = rng.integers(0, 256, (2, 234, 416, 3), dtype=np.uint8)
    clip = sa.tile_clip(ext, ext, 224, 416, 0.03, antialias=True)
    want = torch.from_numpy(ext[:, 5:229]).permute(0, 3, 1, 2).float()
    assert torch.equal(clip[0, :, :, :, :416].permute(1, 0, 2, 3), want)


@pytest.mark.parametrize("antialias", [False, True])
@pytest.mark.parametrize("size", [(720, 1280), (288, 512), (234, 416)])
def test_tile_views_equals_the_praxis_tile(size: tuple[int, int], antialias: bool) -> None:
    """The processor function gives the bytes of the praxis tile for both canvases, float or uint8 input."""
    rng = np.random.default_rng(2)
    frames = rng.integers(0, 256, (2, 3, *size, 3), dtype=np.uint8)  # (B, T, H, W, 3)
    th = 224 if antialias else 256
    want = torch.stack([sa._tile_view(f, th, 416, 0.03, antialias) for f in frames]).float() / 255
    as_tensor = torch.from_numpy(frames).permute(0, 1, 4, 2, 3)
    assert torch.equal(tile_views(as_tensor.float() / 255, th, 416, 0.03, antialias), want)
    assert torch.equal(tile_views(as_tensor, th, 416, 0.03, antialias), want)


@pytest.mark.parametrize(
    ("path", "tile_size", "antialias"), [(M2_CONFIG, (224, 416), True), (K_CONFIG, (256, 416), False)]
)
def test_checkpoint_configs_validate(path: Path, tile_size: tuple[int, int], antialias: bool) -> None:
    """The config.json of both checkpoints load unchanged, with their canvas."""
    config = json.loads(path.read_text())
    variant = model_variant(config)
    assert variant.tile_size == tile_size and variant.antialias is antialias
    assert sa.canvas_size(config) == (tile_size[0], 2 * tile_size[1])
    policy_config = PraxisConfig(model_config=config)
    assert policy_config.model_config == config


def test_video_antialias_is_validated() -> None:
    k = json.loads(K_CONFIG.read_text())
    with pytest.raises(CheckpointError, match="video.antialias"):
        model_variant({**k, "video": {**k["video"], "antialias": "yes"}})
    # The fit rule trims 234 x 416 to the tile: a 256 x 416 tile cannot use it.
    with pytest.raises(CheckpointError, match="no taller than 234"):
        model_variant({**k, "video": {**k["video"], "antialias": True}})


@pytest.mark.parametrize(
    ("path", "tile_height", "antialias"), [(M2_CONFIG, 224, True), (K_CONFIG, 256, False)]
)
def test_pipelines_follow_the_checkpoint_canvas(path: Path, tile_height: int, antialias: bool) -> None:
    config = tiny_lerobot_config(model_config=json.loads(path.read_text()))
    pre, _ = make_pre_post_processors(config, None)
    step = next(s for s in pre.steps if isinstance(s, PraxisTileViewsProcessorStep))
    assert (step.tile_height, step.tile_width, step.antialias) == (tile_height, 416, antialias)
    assert step.get_config()["antialias"] is antialias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compares the CUDA input path with the CPU one")
def test_tile_views_on_cuda_frames_gives_the_cpu_bytes() -> None:
    """Frames on the GPU get the CPU canvas (the CUDA antialiased resize rounds differently)."""
    rng = np.random.default_rng(4)
    frames = torch.from_numpy(rng.integers(0, 256, (2, 3, 288, 512), dtype=np.uint8)).float() / 255
    for antialias, th in ((True, 224), (False, 256)):
        cpu = tile_views(frames, th, 416, 0.03, antialias)
        gpu = tile_views(frames.cuda(), th, 416, 0.03, antialias)
        assert gpu.device.type == "cuda" and torch.equal(gpu.cpu(), cpu)


def test_sampler_defaults_are_the_recommended_settings() -> None:
    """The defaults of the policy config and of the vendored sampler are the recommended settings of `praxis-1-droid`."""
    s, c = SamplerSettings(), PraxisConfig()
    assert (s.num_steps, s.t_shift, s.cfg_scale, s.cfg_video_scale) == (2, 12.5, 1.0, 4.0)
    assert (c.num_steps, c.t_shift, c.cfg_scale, c.cfg_video_scale, c.max_context_chunks) == (
        2,
        12.5,
        1.0,
        4.0,
        1,
    )
