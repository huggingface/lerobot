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

"""The batched image transforms reproduce the per-sample ones and draw independent parameters per sample."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as F  # noqa: N812

from lerobot.transforms import (
    BatchedColorJitter,
    BatchedImageTransforms,
    BatchedRandomRotation,
    BatchedRandomSubsetApply,
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
    make_batched_transform_from_config,
    transforms as tf,
)
from lerobot.utils.random_utils import seeded_context
from tests.utils import DEVICE

BATCH, FRAMES, HEIGHT, WIDTH = 3, 2, 32, 40
CPU = torch.device("cpu")


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.fixture
def frames():
    """A (B, T, C, H, W) float batch; every sample carries T frames that share its parameters."""
    return torch.rand(BATCH, FRAMES, 3, HEIGHT, WIDTH, generator=_generator(0))


def _batched(op, frames, *params):
    """Run a batched op on sample 0 alone so it can be compared with a per-sample kernel."""
    return op(frames[:1], *(torch.tensor([p]) for p in params))[0]


@pytest.mark.parametrize(
    ("batched", "reference", "value"),
    [
        (tf.adjust_brightness_batched, F.adjust_brightness, 1.3),
        (tf.adjust_contrast_batched, F.adjust_contrast, 0.7),
        (tf.adjust_saturation_batched, F.adjust_saturation, 1.4),
        (tf.adjust_hue_batched, F.adjust_hue, 0.04),
        (tf.adjust_sharpness_batched, F.adjust_sharpness, 1.6),
    ],
)
def test_photometric_ops_match_torchvision(frames, batched, reference, value):
    torch.testing.assert_close(
        _batched(batched, frames, value), reference(frames[0], value), atol=2e-6, rtol=0
    )


def test_rotation_rejects_expand():
    with pytest.raises(ValueError, match="expand"):
        BatchedRandomRotation(degrees=5, expand=True)


# --- Parity with the per-sample transforms, over the whole registry ---------------------------
#
# For every type in `_BATCHED_TRANSFORMS`, draw parameters for a batch of one from the batched
# transform, hand the same parameters to the per-sample implementation (torchvision's functional
# kernels or the LeRobot transform's `transform`), and compare. A transform added to the registry
# without an entry here fails `test_parity_cases_cover_the_registry`.

Expected = Callable[[Tensor, dict[str, Any], torch.Generator], Tensor]


def _scalar(value: Tensor) -> float:
    return value[0].item()


def _identity(frame: Tensor, params: dict[str, Any], generator: torch.Generator) -> Tensor:
    return frame


def _color_jitter(frame: Tensor, params: dict[str, Any], generator: torch.Generator) -> Tensor:
    ops = {
        "brightness": F.adjust_brightness,
        "contrast": F.adjust_contrast,
        "saturation": F.adjust_saturation,
        "hue": F.adjust_hue,
    }
    names = list(ops)
    for index in params["order"][0].tolist():
        frame = ops[names[index]](frame, _scalar(params[names[index]]))
    return frame


def _affine(frame: Tensor, params: dict[str, Any], generator: torch.Generator) -> Tensor:
    return F.affine(
        frame,
        angle=_scalar(params["angle"]),
        translate=[int(v) for v in params["translate"][0].tolist()],
        scale=_scalar(params["scale"]),
        shear=params["shear"][0].tolist(),
        interpolation=InterpolationMode.BILINEAR,
        fill=0.25,
    )


def _rotation(frame: Tensor, params: dict[str, Any], generator: torch.Generator) -> Tensor:
    return F.rotate(frame, angle=_scalar(params["angle"]), interpolation=InterpolationMode.BILINEAR, fill=0)


def _custom(reference_cls: type[v2.Transform], lift: Callable[[dict[str, Any]], dict[str, Any]]) -> Expected:
    """Compare with a LeRobot per-sample transform, whose params are python scalars and lists."""

    def expected(frame: Tensor, params: dict[str, Any], generator: torch.Generator) -> Tensor:
        return reference_cls(**PARITY_KWARGS[reference_cls.__name__]).transform(frame, lift(params))

    return expected


def _gaussian_noise(frame: Tensor, params: dict[str, Any], generator: torch.Generator) -> Tensor:
    # The batched transform draws its noise first, so `GaussianNoise` seeded like the generator was draws
    # the very same values.
    reference = tf.GaussianNoise(**PARITY_KWARGS["GaussianNoise"])
    return reference.transform(frame, {"std": _scalar(params["std"]), "seed": generator.initial_seed()})


PARITY_KWARGS: dict[str, dict[str, Any]] = {
    "Identity": {},
    "ColorJitter": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.4, "hue": 0.05},
    "RandomAffine": {
        "degrees": 10,
        "translate": (0.1, 0.1),
        "scale": (0.9, 1.1),
        "shear": (-3, 3, -2, 2),
        "interpolation": InterpolationMode.BILINEAR,
        "fill": 0.25,
    },
    "RandomRotation": {"degrees": 15, "interpolation": InterpolationMode.BILINEAR},
    "SharpnessJitter": {"sharpness": (0.5, 1.5)},
    "GaussianNoise": {"std": (5.0, 25.0)},
    "MotionBlur": {"kernel_size": (3, 11)},
    "GaussianPatchBrightness": {"num_patches": (1, 3)},
    "RandomShadow": {},
    "CoarseDropout": {"max_holes": 3},
    "GammaCorrection": {},
    "PlanckianJitter": {},
}

PARITY_EXPECTED: dict[str, Expected] = {
    "Identity": _identity,
    "ColorJitter": _color_jitter,
    "RandomAffine": _affine,
    "RandomRotation": _rotation,
    "SharpnessJitter": _custom(
        tf.SharpnessJitter, lambda p: {"sharpness_factor": _scalar(p["sharpness_factor"])}
    ),
    "GaussianNoise": _gaussian_noise,
    "MotionBlur": _custom(
        tf.MotionBlur, lambda p: {"kernel_size": int(_scalar(p["kernel_size"])), "angle": _scalar(p["angle"])}
    ),
    "GaussianPatchBrightness": _custom(
        tf.GaussianPatchBrightness,
        lambda p: {
            "centers": p["centers"][0][p["valid"][0]].tolist(),
            "sigmas": p["sigmas"][0][p["valid"][0]].tolist(),
            "factors": p["factors"][0][p["valid"][0]].tolist(),
        },
    ),
    "RandomShadow": _custom(
        tf.RandomShadow,
        lambda p: {name: _scalar(p[name]) for name in ("opacity", "start", "width", "direction")},
    ),
    "CoarseDropout": _custom(
        tf.CoarseDropout,
        lambda p: {
            "sizes": p["sizes"][0][p["valid"][0]].tolist(),
            "positions": p["positions"][0][p["valid"][0]].tolist(),
        },
    ),
    "GammaCorrection": _custom(tf.GammaCorrection, lambda p: {"gamma": _scalar(p["gamma"])}),
    "PlanckianJitter": _custom(tf.PlanckianJitter, lambda p: {"temperature": int(_scalar(p["temperature"]))}),
}


def test_parity_cases_cover_the_registry():
    assert set(PARITY_EXPECTED) == set(tf._BATCHED_TRANSFORMS) == set(PARITY_KWARGS)  # noqa: SLF001


def test_every_batched_type_has_a_per_sample_counterpart():
    for name in tf._BATCHED_TRANSFORMS:  # noqa: SLF001
        assert name in tf._CUSTOM_TRANSFORMS or hasattr(v2, name), name  # noqa: SLF001


@pytest.mark.parametrize("type_name", sorted(PARITY_EXPECTED))
def test_batch_of_one_matches_the_per_sample_transform(frames, type_name):
    """Over many parameter draws, the batched transform on one sample equals the per-sample transform."""
    batched = make_batched_transform_from_config(
        ImageTransformConfig(type=type_name, kwargs=PARITY_KWARGS[type_name])
    )
    for seed in range(25):
        generator = _generator(seed)
        params = batched.make_params(frames[:1].shape, CPU, generator)
        out = batched.transform(frames[:1].clone(), params)[0]
        expected = PARITY_EXPECTED[type_name](frames[0], params, _generator(seed))
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0, msg=f"{type_name} seed={seed}")


# --- Randomness lives in `make_params` ---------------------------------------------------------


def test_color_jitter_applies_components_in_sampled_order(frames):
    jitter = BatchedColorJitter(
        brightness=(1.2, 1.2), contrast=(0.8, 0.8), saturation=(1.3, 1.3), hue=(0.03, 0.03)
    )
    params = jitter.make_params(frames[:1].shape, CPU)
    params["order"] = torch.tensor([[3, 0, 2, 1]])  # hue, brightness, saturation, contrast
    expected = F.adjust_contrast(
        F.adjust_saturation(F.adjust_brightness(F.adjust_hue(frames[0], 0.03), 1.2), 1.3), 0.8
    )
    torch.testing.assert_close(jitter.transform(frames[:1], params)[0], expected, atol=2e-6, rtol=0)


def test_color_jitter_drops_identity_ranges():
    assert BatchedColorJitter(brightness=(1.0, 1.0), hue=0.0).components == []
    assert [name for name, _ in BatchedColorJitter(brightness=0.2, hue=0.1).components] == [
        "brightness",
        "hue",
    ]


def test_gaussian_noise_std_is_per_sample():
    transform = tf.BatchedGaussianNoise((0.0, 30.0))
    flat = torch.full((2, 1, 3, 128, 128), 0.5)
    std = torch.tensor([5.0, 25.0])
    noise = torch.randn(flat.shape, generator=_generator(0)) * tf._per_sample(std / 255.0)  # noqa: SLF001
    out = transform.transform(flat, {"std": std, "noise": noise})
    measured = (out - 0.5).flatten(1).std(dim=1) * 255
    torch.testing.assert_close(measured, std, atol=1.0, rtol=0)


def test_parameters_are_independent_per_sample_and_shared_across_frames():
    """Two samples get different brightness factors; both frames of a sample get the same one."""
    frames = torch.full((2, FRAMES, 3, HEIGHT, WIDTH), 0.5)
    out = BatchedColorJitter(brightness=(0.5, 1.5))(frames, generator=_generator(1))
    factors = out[:, :, 0, 0, 0] / 0.5  # (B, T)
    assert torch.allclose(factors[:, 0], factors[:, 1])
    assert not torch.allclose(factors[0], factors[1])


def test_factor_ranges_are_respected():
    frames = torch.full((256, 1, 3, 8, 8), 0.5)
    out = BatchedColorJitter(brightness=(0.75, 1.25))(frames, generator=_generator(2))
    brightness = out[:, 0, 0, 0, 0] / 0.5
    assert brightness.min() >= 0.75 - 1e-6 and brightness.max() <= 1.25 + 1e-6
    assert brightness.min() < 0.8 and brightness.max() > 1.2  # actually spread over the range


class _AddOne(tf.BatchedTransform):
    def transform(self, frames, params):
        return frames + 1.0


def test_random_subset_apply_selects_n_subset_per_sample_without_replacement():
    """Each sample gets exactly n_subset distinct transforms, each with probability n_subset / n."""
    n_transforms, n_subset, batch = 7, 4, 4000
    subset = BatchedRandomSubsetApply([_AddOne() for _ in range(n_transforms)], n_subset=n_subset)
    shape = torch.Size((batch, 1, 1, 1, 1))
    params = subset.make_params(shape, CPU, _generator(3))
    # With every transform adding 1 and masked per sample, the sum over transforms is n_subset for everyone.
    assert torch.all(subset.transform(torch.zeros(shape), params) == n_subset)
    selected = params["selected"]
    assert selected.shape == (batch, n_subset)
    assert all(len(set(row)) == n_subset for row in selected.tolist())
    frequency = torch.stack([(selected == i).any(dim=1).float().mean() for i in range(n_transforms)])
    torch.testing.assert_close(
        frequency, torch.full((n_transforms,), n_subset / n_transforms), atol=0.03, rtol=0
    )


def test_random_subset_apply_respects_weights():
    frames = torch.zeros(2000, 1, 1, 1, 1)
    identity = make_batched_transform_from_config(ImageTransformConfig(type="Identity"))
    out = BatchedRandomSubsetApply([identity, _AddOne()], p=[0.0, 1.0], n_subset=1)(frames, _generator(4))
    assert torch.all(out == 1.0)


def test_random_order_applies_every_selected_transform(frames):
    cfg = ImageTransformsConfig(enable=True, random_order=True, max_num_transforms=2)
    out = BatchedImageTransforms(cfg)(frames, generator=_generator(0))
    assert out.shape == frames.shape
    assert not torch.equal(out, frames)


def test_slice_params_recurses_into_nested_structures():
    params = {
        "selected": torch.arange(6).view(3, 2),
        "transforms": [{"a": torch.arange(3)}, {"b": torch.arange(3, 6), "n": 5}],
    }
    sliced = tf.slice_params(params, 1, 3)
    assert torch.equal(sliced["selected"], torch.tensor([[2, 3], [4, 5]]))
    assert torch.equal(sliced["transforms"][0]["a"], torch.tensor([1, 2]))
    assert torch.equal(sliced["transforms"][1]["b"], torch.tensor([4, 5]))
    assert sliced["transforms"][1]["n"] == 5


# --- `BatchedImageTransforms` -------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(BATCH, 3, HEIGHT, WIDTH), (BATCH, FRAMES, 3, HEIGHT, WIDTH), (BATCH, 1, HEIGHT, WIDTH)]
)
def test_image_transforms_keep_shape_and_dtype(shape):
    cfg = ImageTransformsConfig(enable=True, max_num_transforms=4)
    transform = BatchedImageTransforms(cfg)
    images = torch.rand(shape)
    out = transform(images)
    assert out.shape == images.shape and out.dtype == images.dtype
    assert out.min() >= 0.0 and out.max() <= 1.0
    uint8 = (images * 255).round().to(torch.uint8)
    out8 = transform(uint8)
    assert out8.shape == uint8.shape and out8.dtype == torch.uint8


def test_image_transforms_low_precision_input_is_computed_in_float32():
    cfg = ImageTransformsConfig(enable=True)
    out = BatchedImageTransforms(cfg)(torch.rand(2, 3, HEIGHT, WIDTH).to(torch.bfloat16))
    assert out.dtype == torch.bfloat16


def test_image_transforms_disabled_is_identity(frames):
    assert torch.equal(BatchedImageTransforms(ImageTransformsConfig(enable=False))(frames), frames)
    cfg = ImageTransformsConfig(enable=True, tfs={})
    assert torch.equal(BatchedImageTransforms(cfg)(frames), frames)


def test_generator_makes_the_augmentation_reproducible(frames):
    cfg = ImageTransformsConfig(enable=True, max_num_transforms=4)
    transform = BatchedImageTransforms(cfg)
    first = transform(frames, generator=_generator(5))
    assert torch.equal(first, transform(frames, generator=_generator(5)))
    assert not torch.equal(first, transform(frames, generator=_generator(6)))


def test_generator_leaves_the_default_rng_untouched(frames):
    """The policy's own sampling must not move because augmentation ran in front of it."""
    with seeded_context(7):
        control = torch.rand(4)
    with seeded_context(7):
        BatchedImageTransforms(ImageTransformsConfig(enable=True))(frames, generator=_generator(0))
        after = torch.rand(4)
    assert torch.equal(control, after)


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_chunking_does_not_change_the_augmentation(frames, chunk_size):
    """Parameters are drawn for the whole batch and sliced, so a chunk size is only a memory knob."""
    cfg = ImageTransformsConfig(enable=True, max_num_transforms=4)
    whole = BatchedImageTransforms(cfg)(frames, generator=_generator(5))
    chunked = BatchedImageTransforms(cfg, chunk_size=chunk_size)(frames, generator=_generator(5))
    assert chunk_size < BATCH  # the chunked run really splits the batch
    assert torch.equal(whole, chunked)


def test_chunk_size_is_validated():
    with pytest.raises(ValueError, match="chunk_size"):
        BatchedImageTransforms(ImageTransformsConfig(enable=True), chunk_size=0)


@pytest.mark.parametrize("bad", [torch.rand(3, HEIGHT, WIDTH), torch.rand(BATCH, 4, HEIGHT, WIDTH)])
def test_image_transforms_reject_unexpected_shapes(bad):
    with pytest.raises(ValueError):
        BatchedImageTransforms(ImageTransformsConfig(enable=True))(bad)


def test_default_config_transforms_have_batched_counterparts():
    for tf_cfg in ImageTransformsConfig().tfs.values():
        assert isinstance(make_batched_transform_from_config(tf_cfg), tf.BatchedTransform)


@pytest.mark.parametrize("type_name", ["JPEGCompression", "RandomResizedCrop", "GaussianBlur"])
def test_unsupported_transform_types_raise(type_name):
    with pytest.raises(ValueError, match="no batched implementation"):
        make_batched_transform_from_config(ImageTransformConfig(type=type_name, kwargs={}))


def test_backend_is_validated():
    with pytest.raises(ValueError, match="backend"):
        ImageTransformsConfig(backend="tpu")
    with pytest.raises(ValueError, match="gpu_chunk_size"):
        ImageTransformsConfig(gpu_chunk_size=0)
    cfg = ImageTransformsConfig()
    assert cfg.backend == "dataloader" and cfg.gpu_compile is True and cfg.gpu_chunk_size == 32


def test_per_sample_and_batched_paths_agree_on_the_same_parameters():
    """The dataloader backend on one (T, C, H, W) camera tensor equals the GPU backend on a batch of one."""
    cfg = ImageTransformsConfig(
        enable=True,
        max_num_transforms=2,
        tfs={
            "brightness": ImageTransformConfig(type="ColorJitter", kwargs={"brightness": (0.8, 0.8)}),
            "sharpness": ImageTransformConfig(type="SharpnessJitter", kwargs={"sharpness": (1.5, 1.5)}),
        },
    )
    frames = torch.rand(FRAMES, 3, HEIGHT, WIDTH, generator=_generator(6))
    per_sample = ImageTransforms(cfg)(frames)
    batched = BatchedImageTransforms(cfg)(frames.unsqueeze(0))[0]
    torch.testing.assert_close(per_sample, batched, atol=2e-6, rtol=0)


@pytest.mark.skipif(DEVICE == "cpu", reason="checks the transforms on the accelerator")
def test_image_transforms_run_on_device(frames):
    cfg = ImageTransformsConfig(enable=True, max_num_transforms=4)
    out = BatchedImageTransforms(cfg)(frames.to(DEVICE))
    assert out.device.type == torch.device(DEVICE).type and out.shape == frames.shape


@pytest.mark.skipif(DEVICE == "cpu", reason="compiles the transforms on the accelerator")
def test_compile_does_not_change_the_augmentation(frames):
    cfg = ImageTransformsConfig(enable=True, max_num_transforms=4)
    device = torch.device(DEVICE)
    eager = BatchedImageTransforms(cfg)(frames.to(device), torch.Generator(device).manual_seed(5))
    compiled = BatchedImageTransforms(cfg, compile_model=True)(
        frames.to(device), torch.Generator(device).manual_seed(5)
    )
    torch.testing.assert_close(eager, compiled, atol=1e-5, rtol=0)
