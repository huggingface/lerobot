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

"""Deprecated per-sample transform names, kept as thin front ends over the batched transforms.

The per-sample implementations are gone: every transform is now batched, and `ImageTransforms` runs the
batched pipeline on a batch of one. The names below were public API in v0.6.1, so each is kept for one
release as a front end that adds and removes the batch dimension around its batched replacement. They
hold no augmentation logic; deleting this module removes them.

Configurations are unaffected either way: `ImageTransformConfig.type` never named these classes.
"""

import warnings
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .transforms import (
    BatchedCoarseDropout,
    BatchedGammaCorrection,
    BatchedGaussianNoise,
    BatchedGaussianPatchBrightness,
    BatchedMotionBlur,
    BatchedPlanckianJitter,
    BatchedRandomShadow,
    BatchedRandomSubsetApply,
    BatchedSharpnessJitter,
    BatchedTransform,
    ImageTransformConfig,
    PerSampleTransform,
    make_batched_transform_from_config,
)


def _warn(name: str, replacement: str) -> None:
    """Warn that `name` is deprecated in favour of `replacement`."""
    warnings.warn(
        f"`lerobot.transforms.{name}` is deprecated and will be removed in a future release. Use "
        f"`{replacement}`, which transforms a batch of `(B, N, C, H, W)` frames; `ImageTransforms` "
        "applies the configured transforms to one sample. Configuration files are unaffected.",
        FutureWarning,
        stacklevel=3,
    )


class _OneSample(nn.Module):
    """Apply a batched transform to a single image or frame stack.

    The batched transforms consume float `(B, N, C, H, W)` frames in `[0, 1]`, while the deprecated names
    took one `(C, H, W)` or `(T, C, H, W)` image of either dtype. This adds and removes the batch
    dimension and converts the dtype exactly as `BatchedImageTransforms.forward` does for a batch, so a
    deprecated name keeps the calling convention it had.

    Args:
        batched (`BatchedTransform`):
            The transform to apply to the single sample.
    """

    def __init__(self, batched: BatchedTransform) -> None:
        super().__init__()
        self.batched = batched

    def forward(self, images: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Transform one image or one stack of frames.

        Args:
            images (`torch.Tensor`):
                `(C, H, W)` or `(T, C, H, W)` with `C` in `{1, 3}`; the `T` frames share one parameter
                draw. `uint8` in `[0, 255]` or floating point in `[0, 1]`.
            generator (`torch.Generator`, *optional*):
                Generator on the images' device to draw the parameters from, the device's default
                generator if `None`.

        Returns:
            `torch.Tensor`: The transformed images, same dtype as the input and, unless the transform
            changes the frame size, the same shape.
        """
        if images.ndim not in (3, 4):
            raise ValueError(f"Expected (C, H, W) or (T, C, H, W) images, got shape {tuple(images.shape)}.")
        if images.shape[-3] not in (1, 3):
            raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(images.shape)}.")
        frames = images.unsqueeze(0) if images.ndim == 3 else images
        if frames.dtype == torch.uint8:
            work = frames.to(torch.float32) / 255.0
        elif frames.is_floating_point():
            work = frames.to(torch.float32)
        else:
            raise TypeError(f"Expected uint8 or floating point images, got {images.dtype}.")
        out = self.batched(work.unsqueeze(0), generator=generator).squeeze(0)
        out = (out * 255.0).round_().to(torch.uint8) if frames.dtype == torch.uint8 else out.to(frames.dtype)
        return out.squeeze(0) if images.ndim == 3 else out


class _DeprecatedPerSample(_OneSample):
    """Base for the deprecated names: warn, then build the batched transform to forward to."""

    _batched_cls: type[BatchedTransform]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn(type(self).__name__, self._batched_cls.__name__)
        super().__init__(self._batched_cls(*args, **kwargs))


class SharpnessJitter(_DeprecatedPerSample):
    """Deprecated, use `BatchedSharpnessJitter`."""

    _batched_cls = BatchedSharpnessJitter


class GaussianNoise(_DeprecatedPerSample):
    """Deprecated, use `BatchedGaussianNoise`."""

    _batched_cls = BatchedGaussianNoise


class MotionBlur(_DeprecatedPerSample):
    """Deprecated, use `BatchedMotionBlur`."""

    _batched_cls = BatchedMotionBlur


class GaussianPatchBrightness(_DeprecatedPerSample):
    """Deprecated, use `BatchedGaussianPatchBrightness`."""

    _batched_cls = BatchedGaussianPatchBrightness


class RandomShadow(_DeprecatedPerSample):
    """Deprecated, use `BatchedRandomShadow`."""

    _batched_cls = BatchedRandomShadow


class CoarseDropout(_DeprecatedPerSample):
    """Deprecated, use `BatchedCoarseDropout`."""

    _batched_cls = BatchedCoarseDropout


class GammaCorrection(_DeprecatedPerSample):
    """Deprecated, use `BatchedGammaCorrection`."""

    _batched_cls = BatchedGammaCorrection


class PlanckianJitter(_DeprecatedPerSample):
    """Deprecated, use `BatchedPlanckianJitter`."""

    _batched_cls = BatchedPlanckianJitter


class RandomSubsetApply(_DeprecatedPerSample):
    """Deprecated, use `BatchedRandomSubsetApply`.

    `BatchedRandomSubsetApply` takes batched transforms; a transform that is not one is wrapped in
    `PerSampleTransform`, so the sequences of torchvision transforms this used to accept still work.

    Args:
        transforms (`Sequence[Callable]`):
            The transforms to draw from.
        p (`list[float]`, *optional*):
            Sampling weights, uniform if `None`.
        n_subset (`int`, *optional*):
            How many to apply, all of them if `None`.
        random_order (`bool`, *optional*, defaults to `False`):
            Apply the drawn subset in a random order rather than the given one.
    """

    _batched_cls = BatchedRandomSubsetApply

    def __init__(
        self,
        transforms: Sequence[Any],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of transforms")
        super().__init__(
            [tf if isinstance(tf, BatchedTransform) else PerSampleTransform(tf) for tf in transforms],
            p=p,
            n_subset=n_subset,
            random_order=random_order,
        )


def make_transform_from_config(cfg: ImageTransformConfig) -> _OneSample:
    """Deprecated, use `make_batched_transform_from_config`.

    Args:
        cfg (`ImageTransformConfig`):
            The transform's type and keyword arguments.

    Returns:
        `_OneSample`: The batched transform for `cfg`, callable on one image or frame stack.
    """
    _warn("make_transform_from_config", "make_batched_transform_from_config")
    return _OneSample(make_batched_transform_from_config(cfg))
