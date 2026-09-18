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

"""The deprecated per-sample names warn, and forward to their batched replacement on a batch of one."""

import pytest
import torch
from torchvision.transforms import v2

from lerobot.transforms import (
    BatchedCoarseDropout,
    BatchedGammaCorrection,
    BatchedGaussianNoise,
    BatchedGaussianPatchBrightness,
    BatchedIdentity,
    BatchedMotionBlur,
    BatchedPlanckianJitter,
    BatchedRandomShadow,
    BatchedSharpnessJitter,
    CoarseDropout,
    GammaCorrection,
    GaussianNoise,
    GaussianPatchBrightness,
    ImageTransformConfig,
    MotionBlur,
    PlanckianJitter,
    RandomShadow,
    RandomSubsetApply,
    SharpnessJitter,
    make_transform_from_config,
)

# (deprecated class, batched replacement, constructor arguments)
DEPRECATED = [
    (SharpnessJitter, BatchedSharpnessJitter, (0.5,)),
    (GaussianNoise, BatchedGaussianNoise, ()),
    (MotionBlur, BatchedMotionBlur, ()),
    (GaussianPatchBrightness, BatchedGaussianPatchBrightness, ()),
    (RandomShadow, BatchedRandomShadow, ()),
    (CoarseDropout, BatchedCoarseDropout, ()),
    (GammaCorrection, BatchedGammaCorrection, ()),
    (PlanckianJitter, BatchedPlanckianJitter, ()),
]
IDS = [cls.__name__ for cls, _, _ in DEPRECATED]


@pytest.mark.parametrize(("deprecated_cls", "batched_cls", "args"), DEPRECATED, ids=IDS)
def test_deprecated_name_warns_and_names_its_replacement(deprecated_cls, batched_cls, args):
    with pytest.warns(FutureWarning, match=batched_cls.__name__):
        deprecated_cls(*args)


@pytest.mark.parametrize(("deprecated_cls", "batched_cls", "args"), DEPRECATED, ids=IDS)
def test_deprecated_name_matches_its_batched_replacement(deprecated_cls, batched_cls, args):
    image = torch.rand(3, 32, 32)
    with pytest.warns(FutureWarning):
        deprecated = deprecated_cls(*args)

    actual = deprecated(image, generator=torch.Generator().manual_seed(0))
    expected = batched_cls(*args)(image[None, None], generator=torch.Generator().manual_seed(0))

    assert actual.shape == image.shape
    torch.testing.assert_close(actual, expected[0, 0])


@pytest.mark.parametrize(("deprecated_cls", "batched_cls", "args"), DEPRECATED, ids=IDS)
def test_deprecated_name_keeps_the_dtype_and_the_frame_stack(deprecated_cls, batched_cls, args):
    with pytest.warns(FutureWarning):
        deprecated = deprecated_cls(*args)

    single = deprecated(torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8))
    stacked = deprecated(torch.rand(2, 3, 32, 32))

    assert single.dtype == torch.uint8
    assert single.shape == (3, 32, 32)
    assert stacked.shape == (2, 3, 32, 32)


@pytest.mark.parametrize("images", [torch.rand(32, 32), torch.rand(2, 2, 3, 32, 32)])
def test_deprecated_name_refuses_a_rank_it_never_took(images):
    with pytest.warns(FutureWarning):
        deprecated = GammaCorrection()

    with pytest.raises(ValueError, match="C, H, W"):
        deprecated(images)


def test_random_subset_apply_still_takes_plain_torchvision_transforms():
    with pytest.warns(FutureWarning, match="BatchedRandomSubsetApply"):
        subset = RandomSubsetApply([v2.Identity(), BatchedIdentity()], n_subset=2)

    image = torch.rand(3, 32, 32)

    torch.testing.assert_close(subset(image, generator=torch.Generator().manual_seed(0)), image)


def test_make_transform_from_config_warns_and_builds_the_batched_transform():
    cfg = ImageTransformConfig(type="GammaCorrection", kwargs={"gamma": (0.5, 2.0)})

    with pytest.warns(FutureWarning, match="make_batched_transform_from_config"):
        transform = make_transform_from_config(cfg)

    image = torch.rand(3, 32, 32)
    actual = transform(image, generator=torch.Generator().manual_seed(0))
    expected = BatchedGammaCorrection(gamma=(0.5, 2.0))(
        image[None, None], generator=torch.Generator().manual_seed(0)
    )

    torch.testing.assert_close(actual, expected[0, 0])
