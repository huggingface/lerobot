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

import pytest
import torch
from packaging import version
from safetensors.torch import load_file
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F  # noqa: N812

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_imgtransform_viz import (
    save_all_transforms,
    save_each_transform,
)
from lerobot.transforms import (
    BatchedCoarseDropout,
    BatchedGammaCorrection,
    BatchedGaussianNoise,
    BatchedGaussianPatchBrightness,
    BatchedIdentity,
    BatchedMotionBlur,
    BatchedPlanckianJitter,
    BatchedRandomAffine,
    BatchedRandomShadow,
    BatchedRandomSubsetApply,
    BatchedSharpnessJitter,
    BatchedTransform,
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
    JPEGCompression,
    PerSampleTransform,
    make_batched_transform_from_config,
)
from lerobot.utils.random_utils import seeded_context
from tests.artifacts.image_transforms.save_image_transforms_to_safetensors import ARTIFACT_DIR
from tests.utils import require_x86_64_kernel

CPU = torch.device("cpu")


def _single(tf_cfg: ImageTransformConfig) -> ImageTransforms:
    """`ImageTransforms` that always applies exactly the one configured transform."""
    return ImageTransforms(ImageTransformsConfig(enable=True, max_num_transforms=1, tfs={"only": tf_cfg}))


def _one(transform: BatchedTransform, img: torch.Tensor, generator: torch.Generator | None = None):
    """Run a batched transform on a single `(C, H, W)` image."""
    return transform(img[None, None], generator=generator)[0, 0]


@pytest.fixture
def color_jitters():
    return [
        make_batched_transform_from_config(
            ImageTransformConfig(type="ColorJitter", kwargs={"brightness": 0.5})
        ),
        make_batched_transform_from_config(
            ImageTransformConfig(type="ColorJitter", kwargs={"contrast": 0.5})
        ),
        make_batched_transform_from_config(
            ImageTransformConfig(type="ColorJitter", kwargs={"saturation": 0.5})
        ),
    ]


@pytest.fixture
def single_transforms():
    return load_file(ARTIFACT_DIR / "single_transforms.safetensors")


@pytest.fixture
def img_tensor(single_transforms):
    return single_transforms["original_frame"]


@pytest.fixture
def default_transforms():
    return load_file(ARTIFACT_DIR / "default_transforms.safetensors")


def test_get_image_transforms_no_transform_enable_false(img_tensor_factory):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig()  # default is enable=False
    tf_actual = ImageTransforms(tf_cfg)
    torch.testing.assert_close(tf_actual(img_tensor), img_tensor)


def test_get_image_transforms_no_transform_max_num_transforms_0(img_tensor_factory):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(enable=True, max_num_transforms=0)
    tf_actual = ImageTransforms(tf_cfg)
    torch.testing.assert_close(tf_actual(img_tensor), img_tensor)


@pytest.mark.parametrize("min_max", [(0.5, 0.5), (2.0, 2.0)])
def test_get_image_transforms_brightness(img_tensor_factory, min_max):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(
        enable=True,
        tfs={"brightness": ImageTransformConfig(type="ColorJitter", kwargs={"brightness": min_max})},
    )
    tf_actual = ImageTransforms(tf_cfg)
    tf_expected = v2.ColorJitter(brightness=min_max)
    torch.testing.assert_close(tf_actual(img_tensor), tf_expected(img_tensor))


@pytest.mark.parametrize("min_max", [(0.5, 0.5), (2.0, 2.0)])
def test_get_image_transforms_contrast(img_tensor_factory, min_max):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(
        enable=True, tfs={"contrast": ImageTransformConfig(type="ColorJitter", kwargs={"contrast": min_max})}
    )
    tf_actual = ImageTransforms(tf_cfg)
    tf_expected = v2.ColorJitter(contrast=min_max)
    torch.testing.assert_close(tf_actual(img_tensor), tf_expected(img_tensor))


@pytest.mark.parametrize("min_max", [(0.5, 0.5), (2.0, 2.0)])
def test_get_image_transforms_saturation(img_tensor_factory, min_max):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(
        enable=True,
        tfs={"saturation": ImageTransformConfig(type="ColorJitter", kwargs={"saturation": min_max})},
    )
    tf_actual = ImageTransforms(tf_cfg)
    tf_expected = v2.ColorJitter(saturation=min_max)
    torch.testing.assert_close(tf_actual(img_tensor), tf_expected(img_tensor))


@pytest.mark.parametrize("min_max", [(-0.25, -0.25), (0.25, 0.25)])
def test_get_image_transforms_hue(img_tensor_factory, min_max):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(
        enable=True, tfs={"hue": ImageTransformConfig(type="ColorJitter", kwargs={"hue": min_max})}
    )
    tf_actual = ImageTransforms(tf_cfg)
    tf_expected = v2.ColorJitter(hue=min_max)
    torch.testing.assert_close(tf_actual(img_tensor), tf_expected(img_tensor))


@pytest.mark.parametrize("min_max", [(0.5, 0.5), (2.0, 2.0)])
def test_get_image_transforms_sharpness(img_tensor_factory, min_max):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(
        enable=True,
        tfs={"sharpness": ImageTransformConfig(type="SharpnessJitter", kwargs={"sharpness": min_max})},
    )
    tf_actual = ImageTransforms(tf_cfg)
    torch.testing.assert_close(tf_actual(img_tensor), F.adjust_sharpness(img_tensor, min_max[0]))


@pytest.mark.parametrize("degrees, translate", [((-5.0, 5.0), (0.05, 0.05)), ((10.0, 10.0), (0.1, 0.1))])
def test_get_image_transforms_affine(img_tensor_factory, degrees, translate):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(
        enable=True,
        tfs={
            "affine": ImageTransformConfig(
                type="RandomAffine", kwargs={"degrees": degrees, "translate": translate}
            )
        },
    )
    tf = ImageTransforms(tf_cfg)
    output = tf(img_tensor)
    # Verify output shape is preserved
    assert output.shape == img_tensor.shape
    # Verify transform is type RandomAffine
    assert isinstance(tf.transforms["affine"], BatchedRandomAffine)


def test_get_image_transforms_max_num_transforms(img_tensor_factory):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(
        enable=True,
        max_num_transforms=5,
        tfs={
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.5, 0.5)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.5, 0.5)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 0.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (0.5, 0.5)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 0.5)},
            ),
        },
    )
    tf_actual = ImageTransforms(tf_cfg)
    tf_expected = v2.Compose(
        [
            v2.ColorJitter(brightness=(0.5, 0.5)),
            v2.ColorJitter(contrast=(0.5, 0.5)),
            v2.ColorJitter(saturation=(0.5, 0.5)),
            v2.ColorJitter(hue=(0.5, 0.5)),
            v2.RandomAdjustSharpness(sharpness_factor=0.5, p=1.0),
        ]
    )
    torch.testing.assert_close(tf_actual(img_tensor), tf_expected(img_tensor))


def test_get_image_transforms_random_order(img_tensor_factory):
    """With `random_order`, the same seed gives the same result and different draws give different orders."""
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(
        enable=True,
        random_order=True,
        max_num_transforms=2,
        tfs={
            "brightness": ImageTransformConfig(type="ColorJitter", kwargs={"brightness": (0.5, 0.5)}),
            "contrast": ImageTransformConfig(type="ColorJitter", kwargs={"contrast": (0.5, 0.5)}),
            "saturation": ImageTransformConfig(type="ColorJitter", kwargs={"saturation": (0.5, 0.5)}),
            "hue": ImageTransformConfig(type="ColorJitter", kwargs={"hue": (0.5, 0.5)}),
            "sharpness": ImageTransformConfig(type="SharpnessJitter", kwargs={"sharpness": (0.5, 0.5)}),
        },
    )
    tf = ImageTransforms(tf_cfg)
    first = tf(img_tensor, generator=torch.Generator().manual_seed(1338))
    torch.testing.assert_close(first, tf(img_tensor, generator=torch.Generator().manual_seed(1338)))

    out_imgs = [tf(img_tensor, generator=torch.Generator().manual_seed(seed)) for seed in range(10)]
    for i in range(1, len(out_imgs)):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out_imgs[0], out_imgs[i])


@pytest.mark.parametrize(
    "tf_type, tf_name, min_max_values",
    [
        ("ColorJitter", "brightness", [(0.5, 0.5), (2.0, 2.0)]),
        ("ColorJitter", "contrast", [(0.5, 0.5), (2.0, 2.0)]),
        ("ColorJitter", "saturation", [(0.5, 0.5), (2.0, 2.0)]),
        ("ColorJitter", "hue", [(-0.25, -0.25), (0.25, 0.25)]),
        ("SharpnessJitter", "sharpness", [(0.5, 0.5), (2.0, 2.0)]),
    ],
)
def test_backward_compatibility_single_transforms(
    img_tensor, tf_type, tf_name, min_max_values, single_transforms
):
    for min_max in min_max_values:
        tf_cfg = ImageTransformConfig(type=tf_type, kwargs={tf_name: min_max})
        actual = _single(tf_cfg)(img_tensor)
        key = f"{tf_name}_{min_max[0]}_{min_max[1]}"
        expected = single_transforms[key]
        torch.testing.assert_close(actual, expected)


@require_x86_64_kernel
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.7.0"),
    reason="Test artifacts were generated with PyTorch >= 2.7.0 which has different multinomial behavior",
)
def test_backward_compatibility_default_config(img_tensor, default_transforms):
    # NOTE: PyTorch versions have different randomness, it might break this test.
    # See this PR: https://github.com/huggingface/lerobot/pull/1127.

    # Use config without affine to match original test artifacts
    cfg = ImageTransformsConfig(
        enable=True,
        tfs={
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
        },
    )
    default_tf = ImageTransforms(cfg)

    with seeded_context(1337):
        actual = default_tf(img_tensor)

    expected = default_transforms["default"]

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("p", [[0, 1], [1, 0]])
def test_random_subset_apply_single_choice(img_tensor_factory, p):
    img_tensor = img_tensor_factory()
    flips = [PerSampleTransform(v2.RandomHorizontalFlip(p=1)), PerSampleTransform(v2.RandomVerticalFlip(p=1))]
    random_choice = BatchedRandomSubsetApply(flips, p=p, n_subset=1, random_order=False)
    actual = _one(random_choice, img_tensor)

    p_horz, _ = p
    if p_horz:
        torch.testing.assert_close(actual, F.horizontal_flip(img_tensor))
    else:
        torch.testing.assert_close(actual, F.vertical_flip(img_tensor))


def test_random_subset_apply_random_order(img_tensor_factory):
    img_tensor = img_tensor_factory()
    flips = [PerSampleTransform(v2.RandomHorizontalFlip(p=1)), PerSampleTransform(v2.RandomVerticalFlip(p=1))]
    random_order = BatchedRandomSubsetApply(flips, p=[0.5, 0.5], n_subset=2, random_order=True)
    # We can't really check whether the transforms are actually applied in random order. However,
    # horizontal and vertical flip are commutative. Meaning, even under the assumption that the transform
    # applies them in random order, we can use a fixed order to compute the expected value.
    actual = _one(random_order, img_tensor)
    expected = F.vertical_flip(F.horizontal_flip(img_tensor))
    torch.testing.assert_close(actual, expected)


def test_random_subset_apply_valid_transforms(img_tensor_factory, color_jitters):
    img_tensor = img_tensor_factory()
    transform = BatchedRandomSubsetApply(color_jitters)
    output = _one(transform, img_tensor)
    assert output.shape == img_tensor.shape


def test_random_subset_apply_probability_length_mismatch(color_jitters):
    with pytest.raises(ValueError):
        BatchedRandomSubsetApply(color_jitters, p=[0.5, 0.5])


@pytest.mark.parametrize("n_subset", [0, 5])
def test_random_subset_apply_invalid_n_subset(color_jitters, n_subset):
    with pytest.raises(ValueError):
        BatchedRandomSubsetApply(color_jitters, n_subset=n_subset)


def test_sharpness_jitter_valid_range_tuple(img_tensor_factory):
    img_tensor = img_tensor_factory()
    tf = BatchedSharpnessJitter((0.1, 2.0))
    output = _one(tf, img_tensor)
    assert output.shape == img_tensor.shape


def test_sharpness_jitter_valid_range_float(img_tensor_factory):
    img_tensor = img_tensor_factory()
    tf = BatchedSharpnessJitter(0.5)
    assert tf.sharpness == (0.5, 1.5)
    output = _one(tf, img_tensor)
    assert output.shape == img_tensor.shape


def test_sharpness_jitter_invalid_range_min_negative():
    with pytest.raises(ValueError):
        BatchedSharpnessJitter((-0.1, 2.0))


def test_sharpness_jitter_invalid_range_max_smaller():
    with pytest.raises(ValueError):
        BatchedSharpnessJitter((2.0, 0.1))


def test_make_batched_transform_from_config_with_v2_resize(img_tensor_factory):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformConfig(type="Resize", kwargs={"size": (32, 32)})
    tf = make_batched_transform_from_config(tf_cfg)
    assert isinstance(tf, PerSampleTransform) and isinstance(tf.per_sample, v2.Resize)
    output = _single(tf_cfg)(img_tensor)
    assert output.shape[-2:] == (32, 32)


def test_make_batched_transform_from_config_with_v2_identity(img_tensor_factory):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformConfig(type="Identity", kwargs={})
    tf = make_batched_transform_from_config(tf_cfg)
    assert isinstance(tf, BatchedIdentity)
    output = _single(tf_cfg)(img_tensor)
    assert torch.equal(output, img_tensor)


def test_make_batched_transform_from_config_invalid_type():
    tf_cfg = ImageTransformConfig(type="NotARealTransform", kwargs={})
    with pytest.raises(ValueError, match="not valid"):
        make_batched_transform_from_config(tf_cfg)


@pytest.mark.parametrize("shape", [(3, 24, 32), (2, 3, 24, 32)])
def test_image_transforms_accept_uint8_and_frame_stacks(shape):
    """The dataloader backend sees uint8 frames and `(T, C, H, W)` stacks; both are transformed in place."""
    img = torch.randint(0, 256, shape, dtype=torch.uint8)
    tf = ImageTransforms(ImageTransformsConfig(enable=True, max_num_transforms=2))
    out = tf(img)
    assert out.shape == img.shape and out.dtype == torch.uint8
    assert not torch.equal(out, img)


def test_image_transforms_reject_unexpected_ranks():
    with pytest.raises(ValueError, match="Expected"):
        ImageTransforms(ImageTransformsConfig(enable=True))(torch.rand(1, 2, 3, 8, 8))


def test_save_all_transforms(img_tensor_factory, tmp_path):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(enable=True)
    n_examples = 3

    save_all_transforms(tf_cfg, img_tensor, tmp_path, n_examples)

    # Check if the combined transforms directory exists and contains the right files
    combined_transforms_dir = tmp_path / "all"
    assert combined_transforms_dir.exists(), "Combined transforms directory was not created."
    assert any(combined_transforms_dir.iterdir()), (
        "No transformed images found in combined transforms directory."
    )
    for i in range(1, n_examples + 1):
        assert (combined_transforms_dir / f"{i}.png").exists(), (
            f"Combined transform image {i}.png was not found."
        )


def test_save_each_transform(img_tensor_factory, tmp_path):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(enable=True)
    n_examples = 3

    save_each_transform(tf_cfg, img_tensor, tmp_path, n_examples)

    # Check if the transformed images exist for each transform type
    transforms = ["brightness", "contrast", "saturation", "hue", "sharpness", "affine"]
    for transform in transforms:
        transform_dir = tmp_path / transform
        assert transform_dir.exists(), f"{transform} directory was not created."
        assert any(transform_dir.iterdir()), f"No transformed images found in {transform} directory."

        # Check for specific files within each transform directory
        expected_files = [f"{i}.png" for i in range(1, n_examples + 1)] + ["min.png", "max.png", "mean.png"]
        for file_name in expected_files:
            assert (transform_dir / file_name).exists(), (
                f"{file_name} was not found in {transform} directory."
            )


# --- Tests for robotics-relevant augmentations ---

ROBOTICS_TRANSFORMS = [
    ("GaussianNoise", BatchedGaussianNoise, {"std": (5.0, 25.0)}),
    ("MotionBlur", BatchedMotionBlur, {"kernel_size": (3, 11)}),
    ("JPEGCompression", PerSampleTransform, {"quality": (15, 75)}),
    ("GaussianPatchBrightness", BatchedGaussianPatchBrightness, {}),
    ("RandomShadow", BatchedRandomShadow, {"opacity": (0.3, 0.6)}),
    ("CoarseDropout", BatchedCoarseDropout, {"max_holes": 8}),
    ("GammaCorrection", BatchedGammaCorrection, {"gamma": (0.5, 2.0)}),
    ("PlanckianJitter", BatchedPlanckianJitter, {"temperature": (3_000, 15_000)}),
]


@pytest.mark.parametrize("name,cls,kwargs", ROBOTICS_TRANSFORMS, ids=[t[0] for t in ROBOTICS_TRANSFORMS])
def test_robotics_transform_shape_preserved(name, cls, kwargs, img_tensor_factory):
    img = img_tensor_factory()
    out = _single(ImageTransformConfig(type=name, kwargs=kwargs))(img)
    assert out.shape == img.shape, f"{name} changed shape: {img.shape} -> {out.shape}"


@pytest.mark.parametrize("name,cls,kwargs", ROBOTICS_TRANSFORMS, ids=[t[0] for t in ROBOTICS_TRANSFORMS])
def test_robotics_transform_output_range(name, cls, kwargs, img_tensor_factory):
    img = img_tensor_factory()
    out = _single(ImageTransformConfig(type=name, kwargs=kwargs))(img)
    assert out.min() >= -0.01, f"{name} min below range: {out.min():.4f}"
    assert out.max() <= 1.01, f"{name} max above range: {out.max():.4f}"


@pytest.mark.parametrize("name,cls,kwargs", ROBOTICS_TRANSFORMS, ids=[t[0] for t in ROBOTICS_TRANSFORMS])
def test_robotics_transform_float_output(name, cls, kwargs, img_tensor_factory):
    img = img_tensor_factory()
    out = _single(ImageTransformConfig(type=name, kwargs=kwargs))(img)
    assert out.is_floating_point(), f"{name} output dtype={out.dtype}"


@pytest.mark.parametrize("name,cls,kwargs", ROBOTICS_TRANSFORMS, ids=[t[0] for t in ROBOTICS_TRANSFORMS])
def test_robotics_transform_applies_to_uint8(name, cls, kwargs):
    """uint8 frames, as the DataLoader workers see them, are transformed and handed back as uint8."""
    int_img = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
    out = _single(ImageTransformConfig(type=name, kwargs=kwargs))(int_img)
    assert out.dtype == torch.uint8 and out.shape == int_img.shape
    assert not torch.equal(out, int_img), f"{name} left uint8 input untouched"


@pytest.mark.parametrize("name,cls,kwargs", ROBOTICS_TRANSFORMS, ids=[t[0] for t in ROBOTICS_TRANSFORMS])
def test_robotics_transform_via_config(name, cls, kwargs):
    cfg = ImageTransformConfig(type=name, kwargs=kwargs)
    tf = make_batched_transform_from_config(cfg)
    assert isinstance(tf, cls), f"Config produced {type(tf)}, expected {cls}"


def test_jpeg_compression_goes_through_the_per_sample_adapter():
    tf = make_batched_transform_from_config(ImageTransformConfig(type="JPEGCompression", kwargs={}))
    assert isinstance(tf, PerSampleTransform) and isinstance(tf.per_sample, JPEGCompression)


def test_make_transform_error_message_includes_custom():
    """Error message should list all registered custom transforms."""
    with pytest.raises(ValueError, match="GaussianNoise"):
        make_batched_transform_from_config(ImageTransformConfig(type="NonExistent"))


@pytest.mark.parametrize("name,cls,kwargs", ROBOTICS_TRANSFORMS, ids=[t[0] for t in ROBOTICS_TRANSFORMS])
@pytest.mark.parametrize("shape", [(4, 3, 32, 32), (2, 4, 3, 16, 16)])
def test_robotics_transform_supports_temporal_batches(name, cls, kwargs, shape):
    img = torch.rand(shape)
    cfg = ImageTransformsConfig(enable=True, tfs={name: ImageTransformConfig(type=name, kwargs=kwargs)})
    out = ImageTransforms(cfg).batched(img)
    assert out.shape == img.shape, f"{name} changed shape: {img.shape} -> {out.shape}"
    assert out.min() >= 0
    assert out.max() <= 1


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (BatchedGaussianNoise, {"std": (25.0, 25.0)}),
        (BatchedMotionBlur, {"kernel_size": 5}),
        (
            BatchedGaussianPatchBrightness,
            {"num_patches": 1, "sigma_range": (0.2, 0.2), "factor_range": (0.5, 0.5)},
        ),
        (BatchedRandomShadow, {"opacity": 0.5}),
        (BatchedCoarseDropout, {"max_holes": 1, "fill_value": 0.0}),
        (BatchedGammaCorrection, {"gamma": (2.0, 2.0)}),
        (BatchedPlanckianJitter, {"temperature": 3_000}),
    ],
)
def test_robotics_transform_is_not_silent_noop(cls, kwargs):
    img = torch.rand(3, 32, 32)
    out = _one(cls(**kwargs), img)
    assert not torch.equal(out, img)


def test_jpeg_compression_is_not_silent_noop():
    img = torch.rand(3, 32, 32)
    out = _one(PerSampleTransform(JPEGCompression(quality=10)), img)
    assert not torch.equal(out, img)


@pytest.mark.parametrize(
    "transform",
    [
        BatchedGaussianNoise(std=25),
        BatchedRandomShadow(opacity=0.5),
        BatchedCoarseDropout(max_holes=4),
        PerSampleTransform(JPEGCompression(quality=(5, 95))),
    ],
)
def test_robotics_transform_random_params_are_reused(transform):
    frames = torch.rand(1, 1, 3, 32, 32)
    params = transform.make_params(frames.shape, CPU)
    torch.testing.assert_close(transform.transform(frames, params), transform.transform(frames, params))


def test_motion_blur_kernel_size_stays_in_configured_range():
    transform = BatchedMotionBlur(kernel_size=(4, 10))
    sampled_sizes = set(transform.make_params(torch.Size((200, 1, 3, 8, 8)), CPU)["kernel_size"].tolist())
    assert sampled_sizes <= {5, 7, 9}
    assert sampled_sizes


def test_motion_blur_rejects_a_range_without_an_odd_size():
    with pytest.raises(ValueError, match="odd"):
        BatchedMotionBlur(kernel_size=(4, 4))


def test_gamma_correction_scalar_below_one_defines_symmetric_range():
    transform = BatchedGammaCorrection(gamma=0.5)
    assert transform.gamma == (0.5, 2.0)
    assert _one(transform, torch.rand(3, 8, 8)).shape == (3, 8, 8)


def test_planckian_jitter_uses_correlated_temperature_coefficients():
    frames = torch.full((1, 2, 3, 8, 8), 0.25)
    out = BatchedPlanckianJitter(temperature=3_000)(frames)[0]
    torch.testing.assert_close(out[:, 1], frames[0, :, 1])
    assert torch.all(out[:, 0] > out[:, 1])
    assert torch.all(out[:, 2] < out[:, 1])


def test_random_shadow_supports_small_images():
    img = torch.rand(3, 7, 7)
    assert _one(BatchedRandomShadow(), img).shape == img.shape


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (BatchedGaussianNoise, {"std": (-1.0, 1.0)}),
        (BatchedMotionBlur, {"kernel_size": 4}),
        (JPEGCompression, {"quality": (0, 75)}),
        (BatchedGaussianPatchBrightness, {"sigma_range": (0.0, 0.25)}),
        (BatchedGaussianPatchBrightness, {"num_patches": (3, 1)}),
        (BatchedRandomShadow, {"opacity": (0.3, 1.1)}),
        (BatchedCoarseDropout, {"max_holes": 0}),
        (BatchedCoarseDropout, {"max_height_frac": 1.5}),
        (BatchedGammaCorrection, {"gamma": 0.0}),
        (BatchedGammaCorrection, {"gamma": (2.0, 0.5)}),
        (BatchedPlanckianJitter, {"temperature": (2_000, 6_500)}),
    ],
)
def test_robotics_transform_rejects_invalid_config(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (BatchedGaussianNoise, {"std": (1.0, 2.0, 3.0)}),
        (BatchedMotionBlur, {"kernel_size": "5"}),
        (BatchedGaussianPatchBrightness, {"sigma_range": 0.1}),
        (BatchedRandomShadow, {"opacity": "dark"}),
        (BatchedCoarseDropout, {"max_holes": 2.0}),
        (BatchedGammaCorrection, {"gamma": None}),
        (BatchedPlanckianJitter, {"temperature": 3_000.0}),
    ],
)
def test_robotics_transform_rejects_wrong_argument_types(cls, kwargs):
    with pytest.raises(TypeError):
        cls(**kwargs)
