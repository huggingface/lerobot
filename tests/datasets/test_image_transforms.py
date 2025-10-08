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

from lerobot.datasets.transforms import (
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
    RandomSubsetApply,
    SharpnessJitter,
    make_transform_from_config,
)
from lerobot.scripts.lerobot_imgtransform_viz import (
    save_all_transforms,
    save_each_transform,
)
from lerobot.utils.random_utils import seeded_context
from tests.artifacts.image_transforms.save_image_transforms_to_safetensors import ARTIFACT_DIR
from tests.utils import require_x86_64_kernel


@pytest.fixture
def color_jitters():
    return [
        v2.ColorJitter(brightness=0.5),
        v2.ColorJitter(contrast=0.5),
        v2.ColorJitter(saturation=0.5),
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
    tf_expected = SharpnessJitter(sharpness=min_max)
    torch.testing.assert_close(tf_actual(img_tensor), tf_expected(img_tensor))


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
    assert isinstance(tf.transforms["affine"], v2.RandomAffine)


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
            SharpnessJitter(sharpness=(0.5, 0.5)),
        ]
    )
    torch.testing.assert_close(tf_actual(img_tensor), tf_expected(img_tensor))


@require_x86_64_kernel
def test_get_image_transforms_random_order(img_tensor_factory):
    out_imgs = []
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(
        enable=True,
        random_order=True,
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
    tf = ImageTransforms(tf_cfg)

    with seeded_context(1338):
        for _ in range(10):
            out_imgs.append(tf(img_tensor))

            tmp_img_tensor = img_tensor
            for sub_tf in tf.tf.selected_transforms:
                tmp_img_tensor = sub_tf(tmp_img_tensor)
            torch.testing.assert_close(tmp_img_tensor, out_imgs[-1])

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
        tf = make_transform_from_config(tf_cfg)
        actual = tf(img_tensor)
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
    flips = [v2.RandomHorizontalFlip(p=1), v2.RandomVerticalFlip(p=1)]
    random_choice = RandomSubsetApply(flips, p=p, n_subset=1, random_order=False)
    actual = random_choice(img_tensor)

    p_horz, _ = p
    if p_horz:
        torch.testing.assert_close(actual, F.horizontal_flip(img_tensor))
    else:
        torch.testing.assert_close(actual, F.vertical_flip(img_tensor))


def test_random_subset_apply_random_order(img_tensor_factory):
    img_tensor = img_tensor_factory()
    flips = [v2.RandomHorizontalFlip(p=1), v2.RandomVerticalFlip(p=1)]
    random_order = RandomSubsetApply(flips, p=[0.5, 0.5], n_subset=2, random_order=True)
    # We can't really check whether the transforms are actually applied in random order. However,
    # horizontal and vertical flip are commutative. Meaning, even under the assumption that the transform
    # applies them in random order, we can use a fixed order to compute the expected value.
    actual = random_order(img_tensor)
    expected = v2.Compose(flips)(img_tensor)
    torch.testing.assert_close(actual, expected)


def test_random_subset_apply_valid_transforms(img_tensor_factory, color_jitters):
    img_tensor = img_tensor_factory()
    transform = RandomSubsetApply(color_jitters)
    output = transform(img_tensor)
    assert output.shape == img_tensor.shape


def test_random_subset_apply_probability_length_mismatch(color_jitters):
    with pytest.raises(ValueError):
        RandomSubsetApply(color_jitters, p=[0.5, 0.5])


@pytest.mark.parametrize("n_subset", [0, 5])
def test_random_subset_apply_invalid_n_subset(color_jitters, n_subset):
    with pytest.raises(ValueError):
        RandomSubsetApply(color_jitters, n_subset=n_subset)


def test_sharpness_jitter_valid_range_tuple(img_tensor_factory):
    img_tensor = img_tensor_factory()
    tf = SharpnessJitter((0.1, 2.0))
    output = tf(img_tensor)
    assert output.shape == img_tensor.shape


def test_sharpness_jitter_valid_range_float(img_tensor_factory):
    img_tensor = img_tensor_factory()
    tf = SharpnessJitter(0.5)
    output = tf(img_tensor)
    assert output.shape == img_tensor.shape


def test_sharpness_jitter_invalid_range_min_negative():
    with pytest.raises(ValueError):
        SharpnessJitter((-0.1, 2.0))


def test_sharpness_jitter_invalid_range_max_smaller():
    with pytest.raises(ValueError):
        SharpnessJitter((2.0, 0.1))


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
