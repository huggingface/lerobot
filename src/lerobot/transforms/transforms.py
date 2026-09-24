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
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torchvision.io import decode_image, encode_jpeg
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class JPEGCompression(Transform):
    """Simulate JPEG compression artifacts (block artifacts, color banding).

    Models quality degradation from video compression in network-streamed camera feeds.

    Args:
        quality: Range (min, max) for JPEG quality factor (lower = more artifacts).
    """

    def __init__(self, quality: int | Sequence[int] = (15, 75)) -> None:
        super().__init__()
        if isinstance(quality, int):
            self.quality = (quality, quality)
        elif isinstance(quality, Sequence) and len(quality) == 2:
            self.quality = (int(quality[0]), int(quality[1]))
        else:
            raise TypeError("quality must be an int or a sequence with length 2.")
        if not 1 <= self.quality[0] <= self.quality[1] <= 100:
            raise ValueError(f"quality must satisfy 1 <= min <= max <= 100, but got {self.quality}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"quality": int(torch.randint(self.quality[0], self.quality[1] + 1, (1,)).item())}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3:
            raise ValueError(f"JPEGCompression expects [..., C, H, W] input, but got shape {inpt.shape}.")

        channels, height, width = inpt.shape[-3:]
        if channels not in (1, 3):
            raise ValueError(f"JPEGCompression expects 1 or 3 channels, but got {channels}.")

        flat_input = inpt.reshape(-1, channels, height, width)
        flat_uint8 = (flat_input.clamp(0.0, 1.0) * 255).round().to(torch.uint8).cpu()
        decoded_frames = [
            decode_image(encode_jpeg(frame, quality=params["quality"])) for frame in flat_uint8.unbind()
        ]
        output = torch.stack(decoded_frames).to(device=inpt.device, dtype=inpt.dtype) / 255.0
        return output.reshape(inpt.shape)


# From the paper authors' MIT-licensed reference implementation:
# https://github.com/TheZino/PlanckianJitter
_PLANCKIAN_BLACKBODY_COEFFICIENTS = (
    (0.6743, 0.4029, 0.0013),
    (0.6281, 0.4241, 0.1665),
    (0.5919, 0.4372, 0.2513),
    (0.5623, 0.4457, 0.3154),
    (0.5376, 0.4515, 0.3672),
    (0.5163, 0.4555, 0.4103),
    (0.4979, 0.4584, 0.4468),
    (0.4816, 0.4604, 0.4782),
    (0.4672, 0.4619, 0.5053),
    (0.4542, 0.4630, 0.5289),
    (0.4426, 0.4638, 0.5497),
    (0.4320, 0.4644, 0.5681),
    (0.4223, 0.4648, 0.5844),
    (0.4135, 0.4651, 0.5990),
    (0.4054, 0.4653, 0.6121),
    (0.3980, 0.4654, 0.6239),
    (0.3911, 0.4655, 0.6346),
    (0.3847, 0.4656, 0.6444),
    (0.3787, 0.4656, 0.6532),
    (0.3732, 0.4656, 0.6613),
    (0.3680, 0.4655, 0.6688),
    (0.3632, 0.4655, 0.6756),
    (0.3586, 0.4655, 0.6820),
    (0.3544, 0.4654, 0.6878),
    (0.3503, 0.4653, 0.6933),
)
_PLANCKIAN_MIN_TEMPERATURE = 3_000
_PLANCKIAN_MAX_TEMPERATURE = 15_000
_PLANCKIAN_TEMPERATURE_STEP = 500


# --- Batched, per-sample transforms --------------------------------------------------------
#
# Every transform below applies to a whole batch at once, with independent random parameters
# for every sample. The GPU backend hands it a training batch on the policy device; the
# dataloader backend hands it one sample at a time, as a batch of one, through `ImageTransforms`
# in the DataLoader workers. `ImageTransformsConfig.backend` selects between the two, and a
# sample augmented on one equals the same sample augmented on the other.
#
# Every batched transform consumes float frames in [0, 1] of shape (B, N, C, H, W): B samples,
# each with N frames that share that sample's parameters (the observation history of one
# camera), C in {1, 3}. Where a torchvision kernel exists, the math below mirrors it so that
# a batch of one reproduces the torchvision transform up to floating-point rounding; the
# uint8 rounding torchvision applies between operations on uint8 input is not reproduced.
# Transform types with no batched implementation (any `torchvision.transforms.v2` class by
# name, and `JPEGCompression`) run through `PerSampleTransform`, which loops over the batch.
#
# Randomness is confined to `make_params`, which draws from an explicit `torch.Generator`, and
# `transform` is a pure function of the frames and those parameters. That is what lets the
# augmentation be reproduced from a seed, leave the default generators (and so the policy's own
# noise) untouched, and run compiled or one chunk at a time without changing what is applied.

ImageTransformsBackend = Literal["dataloader", "gpu"]


def _per_sample(values: Tensor) -> Tensor:
    """Reshape a `(B,)` parameter tensor so it broadcasts over `(B, N, C, H, W)` frames."""
    return values.view(-1, 1, 1, 1, 1)


def _uniform(
    batch_size: int, low: float, high: float, device: torch.device, generator: torch.Generator | None = None
) -> Tensor:
    return torch.empty(batch_size, device=device).uniform_(low, high, generator=generator)


def _check_range(
    value: float | Sequence[float], name: str, center: float, bound: tuple[float, float]
) -> tuple[float, float]:
    """Parse a jitter range the way `torchvision.transforms.v2.ColorJitter` does."""
    if isinstance(value, int | float):
        if value < 0:
            raise ValueError(f"If {name} is a single number, it must be non negative.")
        low, high = center - value, center + value
        if bound[0] == 0:
            low = max(low, 0.0)
    elif isinstance(value, Sequence) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
    else:
        raise TypeError(f"{name}={value} should be a single number or a sequence with length 2.")
    if not bound[0] <= low <= high <= bound[1]:
        raise ValueError(f"{name} values should be between {bound} and increasing, but got {(low, high)}.")
    return low, high


def _check_angle(value: float | Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, int | float):
        if value < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        return -float(value), float(value)
    if isinstance(value, Sequence) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise TypeError(f"{name}={value} should be a single number or a sequence with length 2.")


def _check_interpolation(interpolation: InterpolationMode | int | str) -> str:
    if isinstance(interpolation, int):
        interpolation = F._utils._interpolation_modes_from_int(interpolation)  # noqa: SLF001
    mode = InterpolationMode(interpolation).value
    if mode not in ("nearest", "bilinear"):
        raise ValueError(
            f"Batched affine transforms support nearest and bilinear interpolation, got {mode!r}."
        )
    return mode


def _check_fill(fill: Any) -> list[float] | None:
    if fill is None:
        return None
    if isinstance(fill, int | float):
        return [float(fill)]
    if isinstance(fill, Sequence) and all(isinstance(v, int | float) for v in fill):
        return [float(v) for v in fill]
    raise TypeError(f"fill must be a number, a sequence of numbers or None, got {fill!r}.")


def _check_pair(
    value: Any, name: str, cast: type, scalar: Callable[[Any], tuple[Any, Any]] | None = None
) -> tuple[Any, Any]:
    """Parse an argument given as a single number or a `(min, max)` pair into a pair of `cast`.

    A single number becomes `(value, value)` unless `scalar` maps it to a different pair.
    """
    number = int if cast is int else int | float
    if isinstance(value, number) and not isinstance(value, bool):
        low, high = (value, value) if scalar is None else scalar(value)
    elif isinstance(value, Sequence) and len(value) == 2:
        low, high = value
    else:
        raise TypeError(
            f"{name} must be a{'n int' if cast is int else ' number'} or a sequence with length 2."
        )
    return cast(low), cast(high)


def _rgb_to_grayscale(frames: Tensor) -> Tensor:
    r, g, b = frames.unbind(dim=-3)
    return (r * 0.2989 + g * 0.587 + b * 0.114).unsqueeze(-3)


def _blend(frames: Tensor, other: Tensor, ratio: Tensor) -> Tensor:
    ratio = _per_sample(ratio)
    return (frames * ratio + other * (1.0 - ratio)).clamp_(0.0, 1.0)


def _rgb_to_hsv(frames: Tensor) -> Tensor:
    # Same algorithm as torchvision (after Pillow): S and H are zeroed where max == min.
    r, g, _ = frames.unbind(dim=-3)
    minc, maxc = torch.aminmax(frames, dim=-3)
    eqc = maxc == minc
    channels_range = maxc - minc
    ones = torch.ones_like(maxc)
    s = channels_range / torch.where(eqc, ones, maxc)
    divisor = torch.where(eqc, ones, channels_range).unsqueeze(-3)
    rc, gc, bc = ((maxc.unsqueeze(-3) - frames) / divisor).unbind(dim=-3)
    maxc_neq_r = maxc != r
    maxc_eq_g = maxc == g
    hg = (rc + 2.0 - bc) * (maxc_eq_g & maxc_neq_r)
    hr = (bc - gc) * ~maxc_neq_r
    hb = (gc + 4.0 - rc) * (maxc_neq_r & ~maxc_eq_g)
    h = ((hr + hg + hb) / 6.0 + 1.0).fmod_(1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv_to_rgb(frames: Tensor) -> Tensor:
    h, s, v = frames.unbind(dim=-3)
    h6 = h * 6.0
    i = torch.floor(h6)
    f = h6 - i
    i = i.to(torch.int32).remainder_(6)
    sxf = s * f
    one_minus_s = 1.0 - s
    q = ((1.0 - sxf) * v).clamp_(0.0, 1.0)
    t = ((sxf + one_minus_s) * v).clamp_(0.0, 1.0)
    p = (one_minus_s * v).clamp_(0.0, 1.0)
    vpqt = torch.stack((v, p, q, t), dim=-3)
    select = torch.tensor(
        [[0, 2, 1, 1, 3, 0], [3, 0, 0, 2, 1, 1], [1, 1, 3, 0, 0, 2]], dtype=torch.long, device=frames.device
    )
    select = select[:, i].moveaxis(0, -3)
    return vpqt.gather(-3, select)


def adjust_brightness_batched(frames: Tensor, factor: Tensor) -> Tensor:
    """Scale each sample's frames by its own brightness factor.

    Args:
        frames (`torch.Tensor`):
            Float frames in `[0, 1]` of shape `(B, N, C, H, W)`.
        factor (`torch.Tensor`):
            Non-negative factors of shape `(B,)`.

    Returns:
        `torch.Tensor`: The adjusted frames, same shape as the input.
    """
    return (frames * _per_sample(factor)).clamp_(0.0, 1.0)


def adjust_contrast_batched(frames: Tensor, factor: Tensor) -> Tensor:
    """Blend each frame with its own grayscale mean, one factor per sample.

    Args:
        frames (`torch.Tensor`):
            Float frames in `[0, 1]` of shape `(B, N, C, H, W)`.
        factor (`torch.Tensor`):
            Non-negative factors of shape `(B,)`.

    Returns:
        `torch.Tensor`: The adjusted frames, same shape as the input.
    """
    grayscale = _rgb_to_grayscale(frames) if frames.shape[-3] == 3 else frames
    mean = grayscale.mean(dim=(-3, -2, -1), keepdim=True)
    return _blend(frames, mean, factor)


def adjust_saturation_batched(frames: Tensor, factor: Tensor) -> Tensor:
    """Blend each frame with its grayscale version, one factor per sample.

    Args:
        frames (`torch.Tensor`):
            Float frames in `[0, 1]` of shape `(B, N, C, H, W)`.
        factor (`torch.Tensor`):
            Non-negative factors of shape `(B,)`.

    Returns:
        `torch.Tensor`: The adjusted frames, same shape as the input. Single-channel frames are returned as is.
    """
    if frames.shape[-3] == 1:
        return frames
    return _blend(frames, _rgb_to_grayscale(frames), factor)


def adjust_hue_batched(frames: Tensor, factor: Tensor) -> Tensor:
    """Shift the hue of each sample's frames by its own factor.

    Args:
        frames (`torch.Tensor`):
            Float frames in `[0, 1]` of shape `(B, N, C, H, W)`.
        factor (`torch.Tensor`):
            Hue shifts in `[-0.5, 0.5]` of shape `(B,)`.

    Returns:
        `torch.Tensor`: The adjusted frames, same shape as the input. Single-channel frames are returned as is.
    """
    if frames.shape[-3] == 1:
        return frames
    h, s, v = _rgb_to_hsv(frames).unbind(dim=-3)
    h = (h + factor.view(-1, 1, 1, 1)).remainder_(1.0)
    return _hsv_to_rgb(torch.stack((h, s, v), dim=-3))


def adjust_sharpness_batched(frames: Tensor, factor: Tensor) -> Tensor:
    """Blend each frame with a 3x3 blurred copy, one factor per sample, like torchvision's `adjust_sharpness`.

    Args:
        frames (`torch.Tensor`):
            Float frames in `[0, 1]` of shape `(B, N, C, H, W)`.
        factor (`torch.Tensor`):
            Non-negative factors of shape `(B,)`; 0 blurs, 1 is the identity, 2 sharpens.

    Returns:
        `torch.Tensor`: The adjusted frames, same shape as the input. The one-pixel border is left untouched.
    """
    batch_size, num_frames, channels, height, width = frames.shape
    if height <= 2 or width <= 2:
        return frames
    edge, center = 1.0 / 13.0, 5.0 / 13.0
    kernel = torch.tensor(
        [[edge, edge, edge], [edge, center, edge], [edge, edge, edge]],
        dtype=frames.dtype,
        device=frames.device,
    ).expand(channels, 1, 3, 3)
    blurred = nn.functional.conv2d(frames.reshape(-1, channels, height, width), kernel, groups=channels)
    blurred = blurred.reshape(batch_size, num_frames, channels, height - 2, width - 2)
    output = frames.clone()
    output[..., 1:-1, 1:-1] = _blend(frames[..., 1:-1, 1:-1], blurred, factor)
    return output


def inverse_affine_matrix_batched(
    angle: Tensor, translate: Tensor, scale: Tensor, shear: Tensor, center: Tensor
) -> Tensor:
    """Per-sample inverse affine matrices, matching torchvision's `_get_inverse_affine_matrix`.

    Args:
        angle (`torch.Tensor`):
            Rotation angles in degrees, shape `(B,)`.
        translate (`torch.Tensor`):
            Translations in pixels, shape `(B, 2)` as `(tx, ty)`.
        scale (`torch.Tensor`):
            Isotropic scale factors, shape `(B,)`.
        shear (`torch.Tensor`):
            Shear angles in degrees, shape `(B, 2)` as `(sx, sy)`.
        center (`torch.Tensor`):
            Rotation centers in pixels relative to the image center, shape `(B, 2)`.

    Returns:
        `torch.Tensor`: Inverse affine matrices of shape `(B, 2, 3)` for `affine_grid_batched`.
    """
    rot = torch.deg2rad(angle)
    sx = torch.deg2rad(shear[:, 0])
    sy = torch.deg2rad(shear[:, 1])
    cx, cy = center.unbind(dim=1)
    tx, ty = translate.unbind(dim=1)
    cos_sy = torch.cos(sy)
    tan_sx = torch.tan(sx)
    a = torch.cos(rot - sy) / cos_sy
    b = -(a * tan_sx + torch.sin(rot))
    c = torch.sin(rot - sy) / cos_sy
    d = torch.cos(rot) - c * tan_sx
    m0, m1, m3, m4 = d / scale, -b / scale, -c / scale, a / scale
    m2 = cx - m0 * (cx + tx) - m1 * (cy + ty)
    m5 = cy - m3 * (cx + tx) - m4 * (cy + ty)
    return torch.stack((m0, m1, m2, m3, m4, m5), dim=1).view(-1, 2, 3)


def affine_grid_batched(theta: Tensor, height: int, width: int) -> Tensor:
    """Sampling grid for `torch.nn.functional.grid_sample`, matching torchvision's pixel-centered convention.

    Args:
        theta (`torch.Tensor`):
            Inverse affine matrices of shape `(B, 2, 3)`.
        height (`int`):
            Image height in pixels.
        width (`int`):
            Image width in pixels.

    Returns:
        `torch.Tensor`: A grid of shape `(B, H, W, 2)` in normalized coordinates.
    """
    dtype, device = theta.dtype, theta.device
    base_grid = torch.empty(1, height, width, 3, dtype=dtype, device=device)
    base_grid[..., 0].copy_(
        torch.linspace((1.0 - width) * 0.5, (width - 1.0) * 0.5, steps=width, device=device)
    )
    base_grid[..., 1].copy_(
        torch.linspace((1.0 - height) * 0.5, (height - 1.0) * 0.5, steps=height, device=device).unsqueeze_(-1)
    )
    base_grid[..., 2].fill_(1)
    rescaled_theta = theta.transpose(1, 2) / torch.tensor(
        [0.5 * width, 0.5 * height], dtype=dtype, device=device
    )
    grid = base_grid.view(1, height * width, 3).expand(theta.shape[0], -1, -1).bmm(rescaled_theta)
    return grid.view(-1, height, width, 2)


def warp_batched(frames: Tensor, theta: Tensor, interpolation: str, fill: list[float] | None) -> Tensor:
    """Resample each sample's frames through its own inverse affine matrix.

    Args:
        frames (`torch.Tensor`):
            Float frames of shape `(B, N, C, H, W)`.
        theta (`torch.Tensor`):
            Inverse affine matrices of shape `(B, 2, 3)`.
        interpolation (`str`):
            `"nearest"` or `"bilinear"`.
        fill (`list[float] | None`):
            Fill value(s) for pixels sampled outside the image, one value or one per channel. `None` fills
            with zeros.

    Returns:
        `torch.Tensor`: The warped frames, same shape as the input.
    """
    batch_size, num_frames, channels, height, width = frames.shape
    grid = affine_grid_batched(theta.to(frames.dtype), height, width)
    flat = frames.reshape(batch_size, num_frames * channels, height, width)
    if fill is not None:
        mask = torch.ones((batch_size, 1, height, width), dtype=flat.dtype, device=flat.device)
        flat = torch.cat((flat, mask), dim=1)
    flat = nn.functional.grid_sample(
        flat, grid, mode=interpolation, padding_mode="zeros", align_corners=False
    )
    if fill is not None:
        flat, mask = torch.tensor_split(flat, indices=(-1,), dim=1)
        mask = mask.expand_as(flat)
        if len(fill) not in (1, channels):
            raise ValueError(f"fill must have 1 or {channels} values, got {len(fill)}.")
        fill_values = torch.tensor(
            fill * (num_frames * channels // len(fill)), dtype=flat.dtype, device=flat.device
        )
        fill_values = fill_values.view(1, -1, 1, 1)
        if interpolation == "nearest":
            flat = torch.where(mask < 0.5, fill_values.expand_as(flat), flat)
        else:
            flat = (flat - fill_values) * mask + fill_values
    return flat.reshape(batch_size, num_frames, channels, height, width)


def slice_params(params: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """Select samples `start:end` of every per-sample parameter, recursing into nested dicts and lists.

    Args:
        params (`dict[str, Any]`):
            The output of a `BatchedTransform.make_params` call for a whole batch.
        start (`int`):
            First sample of the slice.
        end (`int`):
            One past the last sample of the slice.

    Returns:
        `dict[str, Any]`: The same structure, every tensor sliced along its leading dimension.
    """

    def _slice(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value[start:end]
        if isinstance(value, dict):
            return {key: _slice(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_slice(item) for item in value]
        return value

    return _slice(params)


class BatchedTransform(nn.Module):
    """Base class for a random transform applied to a batch of frames with independent parameters per sample.

    Subclasses implement `make_params`, which draws one parameter set per batch item from an explicit
    generator, and `transform`, which consumes them and holds all of the math. Every random draw lives in
    `make_params`, so `transform` can be compiled, or run one chunk of the batch at a time, without changing
    the augmentation. Inputs are float frames in `[0, 1]` of shape `(B, N, C, H, W)`, where the `N` frames of
    a sample share its parameters.
    """

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        """Draw one set of transform parameters per batch item.

        Args:
            shape (`torch.Size`):
                Shape of the frames about to be transformed, `(B, N, C, H, W)`.
            device (`torch.device`):
                Device to draw the parameters on.
            generator (`torch.Generator`, *optional*):
                Generator on `device` to draw from; the default generator of `device` if `None`.

        Returns:
            `dict[str, Any]`: Parameter tensors whose leading dimension is `B`.
        """
        return {}

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        """Transform the frames with the given per-sample parameters.

        Args:
            frames (`torch.Tensor`):
                Float frames in `[0, 1]` of shape `(B, N, C, H, W)`.
            params (`dict[str, Any]`):
                The output of `make_params` for these frames.

        Returns:
            `torch.Tensor`: The transformed frames, same shape as the input.
        """
        raise NotImplementedError

    def forward(self, frames: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Draw parameters for the batch and transform it."""
        return self.transform(frames, self.make_params(frames.shape, frames.device, generator))


class BatchedIdentity(BatchedTransform):
    """Return the frames unchanged."""

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        return frames


class BatchedColorJitter(BatchedTransform):
    """Per-sample brightness, contrast, saturation and hue jitter, like `torchvision.transforms.v2.ColorJitter`.

    The configured components are applied in a random order that is drawn independently for every sample, as
    torchvision does per call.

    Args:
        brightness (`float | Sequence[float]`, *optional*):
            Factor range, or a non-negative number `b` for `[max(0, 1 - b), 1 + b]`.
        contrast (`float | Sequence[float]`, *optional*):
            Factor range, or a non-negative number `c` for `[max(0, 1 - c), 1 + c]`.
        saturation (`float | Sequence[float]`, *optional*):
            Factor range, or a non-negative number `s` for `[max(0, 1 - s), 1 + s]`.
        hue (`float | Sequence[float]`, *optional*):
            Shift range within `[-0.5, 0.5]`, or a non-negative number `h` for `[-h, h]`.
    """

    _COMPONENTS: tuple[tuple[str, Callable[[Tensor, Tensor], Tensor]], ...] = (
        ("brightness", adjust_brightness_batched),
        ("contrast", adjust_contrast_batched),
        ("saturation", adjust_saturation_batched),
        ("hue", adjust_hue_batched),
    )

    def __init__(
        self,
        brightness: float | Sequence[float] | None = None,
        contrast: float | Sequence[float] | None = None,
        saturation: float | Sequence[float] | None = None,
        hue: float | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        inf = float("inf")
        self.ranges: dict[str, tuple[float, float]] = {}
        for name, value, center, bound in (
            ("brightness", brightness, 1.0, (0.0, inf)),
            ("contrast", contrast, 1.0, (0.0, inf)),
            ("saturation", saturation, 1.0, (0.0, inf)),
            ("hue", hue, 0.0, (-0.5, 0.5)),
        ):
            if value is None:
                continue
            low, high = _check_range(value, name, center, bound)
            if low == high == center:
                continue
            self.ranges[name] = (low, high)
        self.components = [(name, fn) for name, fn in self._COMPONENTS if name in self.ranges]

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        batch_size = shape[0]
        params: dict[str, Any] = {
            name: _uniform(batch_size, *self.ranges[name], device, generator) for name, _ in self.components
        }
        order = torch.rand(batch_size, len(self.components), device=device, generator=generator)
        params["order"] = order.argsort(dim=1)
        return params

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        if len(self.components) == 1:
            name, fn = self.components[0]
            return fn(frames, params[name])
        order = params["order"]
        for position in range(len(self.components)):
            for index, (name, fn) in enumerate(self.components):
                frames = torch.where(
                    _per_sample(order[:, position] == index), fn(frames, params[name]), frames
                )
        return frames


class BatchedSharpnessJitter(BatchedTransform):
    """Per-sample sharpness jitter, like `torchvision.transforms.v2.RandomAdjustSharpness` with a random factor.

    A factor of 0 blurs, 1 is the identity and 2 sharpens; every call draws a fresh factor.

    Args:
        sharpness (`float | Sequence[float]`):
            Factor range, or a non-negative number `s` for `[max(0, 1 - s), 1 + s]`.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = _check_range(sharpness, "sharpness", 1.0, (0.0, float("inf")))

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        return {"sharpness_factor": _uniform(shape[0], *self.sharpness, device, generator)}

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        return adjust_sharpness_batched(frames, params["sharpness_factor"])


class BatchedRandomAffine(BatchedTransform):
    """Per-sample rotation, translation, scale and shear, like `torchvision.transforms.v2.RandomAffine`.

    Args:
        degrees (`float | Sequence[float]`):
            Rotation range in degrees, or a non-negative number `d` for `[-d, d]`.
        translate (`Sequence[float]`, *optional*):
            Maximum absolute translation as a fraction of width and height.
        scale (`Sequence[float]`, *optional*):
            Scale factor range.
        shear (`float | Sequence[float]`, *optional*):
            Shear range in degrees: a number `s` for `[-s, s]` on x, two values for x, four for x then y.
        interpolation (`InterpolationMode | int | str`, *optional*, defaults to `InterpolationMode.NEAREST`):
            `NEAREST` or `BILINEAR`.
        fill (`float | Sequence[float] | None`, *optional*, defaults to `0`):
            Fill value for pixels outside the image, one value or one per channel.
        center (`Sequence[float]`, *optional*):
            Rotation center in pixels from the top-left corner. Defaults to the image center.
    """

    def __init__(
        self,
        degrees: float | Sequence[float],
        translate: Sequence[float] | None = None,
        scale: Sequence[float] | None = None,
        shear: float | Sequence[float] | None = None,
        interpolation: InterpolationMode | int | str = InterpolationMode.NEAREST,
        fill: float | Sequence[float] | None = 0,
        center: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.degrees = _check_angle(degrees, "degrees")
        if translate is not None:
            if not (isinstance(translate, Sequence) and len(translate) == 2):
                raise TypeError("translate should be a sequence of length 2.")
            if not all(0.0 <= t <= 1.0 for t in translate):
                raise ValueError("translation values should be between 0 and 1")
        self.translate = (float(translate[0]), float(translate[1])) if translate is not None else None
        if scale is not None:
            if not (isinstance(scale, Sequence) and len(scale) == 2):
                raise TypeError("scale should be a sequence of length 2.")
            if not all(s > 0 for s in scale):
                raise ValueError("scale values should be positive")
        self.scale = (float(scale[0]), float(scale[1])) if scale is not None else None
        if shear is None:
            self.shear: tuple[float, ...] | None = None
        elif isinstance(shear, int | float):
            self.shear = _check_angle(shear, "shear")
        elif isinstance(shear, Sequence) and len(shear) in (2, 4):
            self.shear = tuple(float(v) for v in shear)
        else:
            raise TypeError("shear should be a number or a sequence of length 2 or 4.")
        self.interpolation = _check_interpolation(interpolation)
        self.fill = _check_fill(fill)
        if center is not None and not (isinstance(center, Sequence) and len(center) == 2):
            raise TypeError("center should be a sequence of length 2.")
        self.center = tuple(float(c) for c in center) if center is not None else None

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        batch_size, height, width = shape[0], shape[-2], shape[-1]
        angle = _uniform(batch_size, *self.degrees, device, generator)
        if self.translate is not None:
            max_dx, max_dy = self.translate[0] * width, self.translate[1] * height
            translate = torch.stack(
                (
                    _uniform(batch_size, -max_dx, max_dx, device, generator),
                    _uniform(batch_size, -max_dy, max_dy, device, generator),
                ),
                dim=1,
            ).round_()
        else:
            translate = torch.zeros(batch_size, 2, device=device)
        scale = (
            _uniform(batch_size, *self.scale, device, generator)
            if self.scale is not None
            else torch.ones(batch_size, device=device)
        )
        shear = torch.zeros(batch_size, 2, device=device)
        if self.shear is not None:
            shear[:, 0] = _uniform(batch_size, self.shear[0], self.shear[1], device, generator)
            if len(self.shear) == 4:
                shear[:, 1] = _uniform(batch_size, self.shear[2], self.shear[3], device, generator)
        return {"angle": angle, "translate": translate, "scale": scale, "shear": shear}

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        batch_size, _, _, height, width = frames.shape
        center = torch.zeros(batch_size, 2, device=frames.device)
        if self.center is not None:
            center[:, 0] = self.center[0] - width * 0.5
            center[:, 1] = self.center[1] - height * 0.5
        theta = inverse_affine_matrix_batched(
            params["angle"], params["translate"], params["scale"], params["shear"], center
        )
        return warp_batched(frames, theta, self.interpolation, self.fill)


class BatchedRandomRotation(BatchedTransform):
    """Per-sample rotation, like `torchvision.transforms.v2.RandomRotation` with `expand=False`.

    Args:
        degrees (`float | Sequence[float]`):
            Rotation range in degrees, or a non-negative number `d` for `[-d, d]`.
        interpolation (`InterpolationMode | int | str`, *optional*, defaults to `InterpolationMode.NEAREST`):
            `NEAREST` or `BILINEAR`.
        expand (`bool`, *optional*, defaults to `False`):
            Must be `False`; a batch shares one frame size.
        center (`Sequence[float]`, *optional*):
            Rotation center in pixels from the top-left corner. Defaults to the image center.
        fill (`float | Sequence[float] | None`, *optional*, defaults to `0`):
            Fill value for pixels outside the image, one value or one per channel.
    """

    def __init__(
        self,
        degrees: float | Sequence[float],
        interpolation: InterpolationMode | int | str = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Sequence[float] | None = None,
        fill: float | Sequence[float] | None = 0,
    ) -> None:
        super().__init__()
        if expand:
            raise ValueError(
                "BatchedRandomRotation does not support expand=True: all frames of a batch share one size."
            )
        self.affine = BatchedRandomAffine(degrees, interpolation=interpolation, fill=fill, center=center)

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        return {"angle": _uniform(shape[0], *self.affine.degrees, device, generator)}

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        batch_size = frames.shape[0]
        # torchvision's rotate and affine kernels disagree on the sign of the angle; rotate negates it.
        affine_params = {
            "angle": -params["angle"],
            "translate": torch.zeros(batch_size, 2, device=frames.device),
            "scale": torch.ones(batch_size, device=frames.device),
            "shear": torch.zeros(batch_size, 2, device=frames.device),
        }
        return self.affine.transform(frames, affine_params)


class BatchedGaussianNoise(BatchedTransform):
    """Per-sample Gaussian noise, to simulate camera sensor noise.

    Models readout noise from ADC quantization, which increases in low-light conditions and is common in
    real-robot setups where wrist cameras operate in suboptimal lighting.

    Args:
        std (`float | Sequence[float]`, *optional*, defaults to `(5.0, 25.0)`):
            Range for the noise standard deviation in pixel-value scale (0-255), or a number `s` for `(0, s)`.
    """

    def __init__(self, std: float | Sequence[float] = (5.0, 25.0)) -> None:
        super().__init__()
        self.std = _check_pair(std, "std", float, scalar=lambda s: (0.0, s))
        if not 0.0 <= self.std[0] <= self.std[1]:
            raise ValueError(f"std must satisfy 0 <= min <= max, but got {self.std}.")

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        # The noise is a parameter like any other, so it is drawn here rather than inside `transform`.
        noise = torch.randn(shape, device=device, generator=generator)
        std = _uniform(shape[0], *self.std, device, generator)
        return {"std": std, "noise": noise * _per_sample(std / 255.0)}

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        if params["noise"].shape != frames.shape:
            raise ValueError(
                f"GaussianNoise drew noise for frames of shape {tuple(params['noise'].shape)} but received "
                f"{tuple(frames.shape)}. Place it before any transform that changes the frame size."
            )
        return (frames + params["noise"]).clamp_(0.0, 1.0)


class BatchedMotionBlur(BatchedTransform):
    """Per-sample directional motion blur, to simulate fast robot or object movement.

    A 1D averaging kernel along a random direction, applied as a depthwise convolution.

    Args:
        kernel_size (`int | Sequence[int]`, *optional*, defaults to `(3, 11)`):
            An odd kernel size or a range containing at least one odd kernel size.
    """

    def __init__(self, kernel_size: int | Sequence[int] = (3, 11)) -> None:
        super().__init__()
        self.kernel_size = _check_pair(kernel_size, "kernel_size", int)
        if not 1 <= self.kernel_size[0] <= self.kernel_size[1]:
            raise ValueError(f"kernel_size must satisfy 1 <= min <= max, but got {self.kernel_size}.")
        self.first_odd_kernel_size = self.kernel_size[0] + (self.kernel_size[0] + 1) % 2
        if self.first_odd_kernel_size > self.kernel_size[1]:
            raise ValueError(f"kernel_size range must contain an odd value, but got {self.kernel_size}.")
        self.num_odd_sizes = (self.kernel_size[1] - self.first_odd_kernel_size) // 2 + 1
        self.max_kernel_size = self.first_odd_kernel_size + 2 * (self.num_odd_sizes - 1)

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        batch_size = shape[0]
        size_index = torch.randint(0, self.num_odd_sizes, (batch_size,), device=device, generator=generator)
        return {
            "kernel_size": self.first_odd_kernel_size + 2 * size_index,
            "angle": _uniform(batch_size, 0.0, 360.0, device, generator),
        }

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        batch_size, num_frames, channels, height, width = frames.shape
        max_size = self.max_kernel_size
        max_radius = max_size // 2
        kernel_size = params["kernel_size"]
        radius = (kernel_size // 2).to(frames.dtype)
        angle = torch.deg2rad(params["angle"])
        # Tap positions of each sample's 1D kernel, laid out in a (B, max_size) grid; invalid taps masked.
        tap = torch.arange(max_size, device=frames.device)
        valid = tap.unsqueeze(0) < kernel_size.unsqueeze(1)
        spacing = torch.where(
            kernel_size > 1, 2.0 * radius / (kernel_size - 1).clamp(min=1), torch.zeros_like(radius)
        )
        positions = -radius.unsqueeze(1) + tap.unsqueeze(0).to(frames.dtype) * spacing.unsqueeze(1)
        x_coords = (positions * torch.cos(angle).unsqueeze(1)).round().long() + max_radius
        y_coords = (positions * torch.sin(angle).unsqueeze(1)).round().long() + max_radius
        sample_index = torch.arange(batch_size, device=frames.device).unsqueeze(1).expand_as(x_coords)
        kernels = torch.zeros(batch_size, max_size, max_size, dtype=frames.dtype, device=frames.device)
        kernels.index_put_(
            (sample_index[valid], y_coords[valid], x_coords[valid]),
            torch.ones((), dtype=frames.dtype, device=frames.device),
        )
        kernels = kernels / kernels.sum(dim=(1, 2), keepdim=True)
        groups = batch_size * num_frames * channels
        weight = kernels.repeat_interleave(num_frames * channels, dim=0).unsqueeze(1)
        padded = nn.functional.pad(
            frames.reshape(1, groups, height, width), (max_radius,) * 4, mode="replicate"
        )
        output = nn.functional.conv2d(padded, weight, groups=groups)
        return output.reshape(batch_size, num_frames, channels, height, width).clamp_(0.0, 1.0)


class BatchedGaussianPatchBrightness(BatchedTransform):
    """Per-sample spatially varying brightness with Gaussian patches.

    Simulates uneven overhead lighting, spotlights and shadow patches, as found in robot workspaces with
    several light sources.

    Args:
        num_patches (`int | Sequence[int]`, *optional*, defaults to `(1, 4)`):
            Range for the number of patches.
        sigma_range (`Sequence[float]`, *optional*, defaults to `(0.05, 0.25)`):
            Range for the Gaussian sigma as a fraction of the image size.
        factor_range (`Sequence[float]`, *optional*, defaults to `(0.4, 1.6)`):
            Range for the brightness factor at a patch center.
    """

    def __init__(
        self,
        num_patches: int | Sequence[int] = (1, 4),
        sigma_range: Sequence[float] = (0.05, 0.25),
        factor_range: Sequence[float] = (0.4, 1.6),
    ) -> None:
        super().__init__()
        self.num_patches = _check_pair(num_patches, "num_patches", int)
        if not 1 <= self.num_patches[0] <= self.num_patches[1]:
            raise ValueError(f"num_patches must satisfy 1 <= min <= max, but got {self.num_patches}.")
        if not isinstance(sigma_range, Sequence) or len(sigma_range) != 2:
            raise TypeError("sigma_range must be a sequence with length 2.")
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))
        if not 0.0 < self.sigma_range[0] <= self.sigma_range[1]:
            raise ValueError(f"sigma_range must satisfy 0 < min <= max, but got {self.sigma_range}.")
        if not isinstance(factor_range, Sequence) or len(factor_range) != 2:
            raise TypeError("factor_range must be a sequence with length 2.")
        self.factor_range = (float(factor_range[0]), float(factor_range[1]))
        if not 0.0 <= self.factor_range[0] <= self.factor_range[1]:
            raise ValueError(f"factor_range must satisfy 0 <= min <= max, but got {self.factor_range}.")

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        batch_size, max_patches = shape[0], self.num_patches[1]
        count = torch.randint(
            self.num_patches[0], max_patches + 1, (batch_size,), device=device, generator=generator
        )
        valid = torch.arange(max_patches, device=device).unsqueeze(0) < count.unsqueeze(1)
        factors = torch.empty(batch_size, max_patches, device=device).uniform_(
            *self.factor_range, generator=generator
        )
        return {
            "centers": torch.rand(batch_size, max_patches, 2, device=device, generator=generator),
            "sigmas": torch.empty(batch_size, max_patches, device=device).uniform_(
                *self.sigma_range, generator=generator
            ),
            # An unused patch has factor 1 and therefore no effect.
            "factors": torch.where(valid, factors, torch.ones_like(factors)),
            "valid": valid,
        }

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        height, width = frames.shape[-2:]
        grid_y = torch.linspace(0, 1, height, device=frames.device, dtype=frames.dtype).view(1, 1, height, 1)
        grid_x = torch.linspace(0, 1, width, device=frames.device, dtype=frames.dtype).view(1, 1, 1, width)
        centers = params["centers"].to(frames.dtype)
        cy = centers[..., 0].unsqueeze(-1).unsqueeze(-1)
        cx = centers[..., 1].unsqueeze(-1).unsqueeze(-1)
        sigma = params["sigmas"].to(frames.dtype).unsqueeze(-1).unsqueeze(-1)
        factor = params["factors"].to(frames.dtype).unsqueeze(-1).unsqueeze(-1)
        gauss = torch.exp(-((grid_y - cy) ** 2 + (grid_x - cx) ** 2) / (2 * sigma**2))
        mask = (1.0 + (factor - 1.0) * gauss).prod(dim=1)
        return (frames * mask.unsqueeze(1).unsqueeze(1)).clamp_(0.0, 1.0)


class BatchedRandomShadow(BatchedTransform):
    """Per-sample vertical shadow or highlight band with smooth edges.

    Simulates cast shadows from objects or people near the workspace. Symmetric: the band darkens or
    brightens with equal probability, so BatchNorm statistics do not shift.

    Args:
        opacity (`float | Sequence[float]`, *optional*, defaults to `(0.3, 0.6)`):
            Range for the band opacity, within `[0, 1]`.
    """

    def __init__(self, opacity: float | Sequence[float] = (0.3, 0.6)) -> None:
        super().__init__()
        self.opacity = _check_pair(opacity, "opacity", float)
        if not 0.0 <= self.opacity[0] <= self.opacity[1] <= 1.0:
            raise ValueError(f"opacity must satisfy 0 <= min <= max <= 1, but got {self.opacity}.")

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        batch_size = shape[0]
        coin = torch.rand(batch_size, device=device, generator=generator)
        return {
            "opacity": _uniform(batch_size, *self.opacity, device, generator),
            "start": torch.rand(batch_size, device=device, generator=generator),
            "width": _uniform(batch_size, 1 / 3, 2 / 3, device, generator),
            "direction": torch.where(coin < 0.5, -1.0, 1.0),
        }

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        batch_size, _, _, height, width = frames.shape
        band_width = (params["width"] * width).round().clamp(1, width)
        x_start = (params["start"] * (width - band_width)).round()
        columns = torch.arange(width, device=frames.device, dtype=frames.dtype).unsqueeze(0)
        in_band = (columns >= x_start.unsqueeze(1)) & (columns < (x_start + band_width).unsqueeze(1))
        band_value = (1.0 + params["direction"] * params["opacity"]).unsqueeze(1)
        mask = torch.where(in_band, band_value, torch.ones_like(band_value)).to(frames.dtype)
        mask = mask.view(batch_size, 1, 1, width).expand(batch_size, 1, height, width)
        smoothing_size = min(8, height, width)
        if smoothing_size > 1:
            small = nn.functional.avg_pool2d(mask, smoothing_size, stride=smoothing_size)
            mask = nn.functional.interpolate(
                small, size=(height, width), mode="bilinear", align_corners=False
            )
        return (frames * mask.unsqueeze(1)).clamp_(0.0, 1.0)


class BatchedCoarseDropout(BatchedTransform):
    """Per-sample rectangular dropout, to simulate partial occlusion.

    Models objects, hands or cables passing through the camera's field of view.

    Args:
        max_holes (`int`, *optional*, defaults to `8`):
            Maximum number of dropped patches.
        max_height_frac (`float`, *optional*, defaults to `0.07`):
            Maximum patch height as a fraction of the image height.
        max_width_frac (`float`, *optional*, defaults to `0.07`):
            Maximum patch width as a fraction of the image width.
        fill_value (`float`, *optional*, defaults to `0.0`):
            Value written into dropped patches.
    """

    def __init__(
        self,
        max_holes: int = 8,
        max_height_frac: float = 0.07,
        max_width_frac: float = 0.07,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(max_holes, int):
            raise TypeError("max_holes must be an int.")
        if max_holes < 1:
            raise ValueError(f"max_holes must be at least 1, but got {max_holes}.")
        if not 0.0 < max_height_frac <= 1.0:
            raise ValueError(f"max_height_frac must be in (0, 1], but got {max_height_frac}.")
        if not 0.0 < max_width_frac <= 1.0:
            raise ValueError(f"max_width_frac must be in (0, 1], but got {max_width_frac}.")
        if not 0.0 <= fill_value <= 1.0:
            raise ValueError(f"fill_value must be in [0, 1], but got {fill_value}.")
        self.max_holes = max_holes
        self.max_height_frac = max_height_frac
        self.max_width_frac = max_width_frac
        self.fill_value = fill_value

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        batch_size = shape[0]
        count = torch.randint(1, self.max_holes + 1, (batch_size,), device=device, generator=generator)
        sizes = torch.rand(batch_size, self.max_holes, 2, device=device, generator=generator)
        sizes[..., 0] *= self.max_height_frac
        sizes[..., 1] *= self.max_width_frac
        return {
            "sizes": sizes,
            "positions": torch.rand(batch_size, self.max_holes, 2, device=device, generator=generator),
            "valid": torch.arange(self.max_holes, device=device).unsqueeze(0) < count.unsqueeze(1),
        }

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        height, width = frames.shape[-2:]
        sizes, positions = params["sizes"], params["positions"]
        hole_h = (sizes[..., 0] * height).round().clamp(1, height)
        hole_w = (sizes[..., 1] * width).round().clamp(1, width)
        y = (positions[..., 0] * (height - hole_h)).round()
        x = (positions[..., 1] * (width - hole_w)).round()
        rows = torch.arange(height, device=frames.device, dtype=hole_h.dtype).view(1, 1, height)
        cols = torch.arange(width, device=frames.device, dtype=hole_w.dtype).view(1, 1, width)
        in_rows = (rows >= y.unsqueeze(-1)) & (rows < (y + hole_h).unsqueeze(-1))
        in_cols = (cols >= x.unsqueeze(-1)) & (cols < (x + hole_w).unsqueeze(-1))
        holes = in_rows.unsqueeze(-1) & in_cols.unsqueeze(-2) & params["valid"].unsqueeze(-1).unsqueeze(-1)
        mask = holes.any(dim=1).unsqueeze(1).unsqueeze(1)
        return torch.where(mask, torch.full_like(frames, self.fill_value), frames)


class BatchedGammaCorrection(BatchedTransform):
    """Per-sample gamma correction, to simulate exposure variation.

    Models different auto-exposure settings and sensor response curves. Gamma is sampled log-uniformly,
    so brightening and darkening are equally likely and BatchNorm statistics do not shift.

    Args:
        gamma (`float | Sequence[float]`, *optional*, defaults to `(0.5, 2.0)`):
            Range for the gamma value (below 1 brightens, above 1 darkens), or a positive number `g` for
            the symmetric range `(min(g, 1/g), max(g, 1/g))`.
    """

    def __init__(self, gamma: float | Sequence[float] = (0.5, 2.0)) -> None:
        super().__init__()
        if isinstance(gamma, int | float) and gamma <= 0:
            raise ValueError(f"gamma must be positive, but got {gamma}.")
        self.gamma = _check_pair(gamma, "gamma", float, scalar=lambda g: (min(g, 1.0 / g), max(g, 1.0 / g)))
        if not 0.0 < self.gamma[0] <= self.gamma[1]:
            raise ValueError(f"gamma must satisfy 0 < min <= max, but got {self.gamma}.")

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        log_gamma = _uniform(shape[0], math.log(self.gamma[0]), math.log(self.gamma[1]), device, generator)
        return {"gamma": log_gamma.exp()}

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        return frames.pow(_per_sample(params["gamma"])).clamp_(0.0, 1.0)


class BatchedPlanckianJitter(BatchedTransform):
    """Per-sample color temperature shift along the Planckian locus.

    Samples one black-body temperature per sample and applies the correlated red and blue channel scaling
    while preserving the green channel; coefficients between the tabulated 500 K steps are interpolated.

    Reference: Zini et al., "Planckian Jitter", CVPR 2022 Workshop.

    Args:
        temperature (`int | Sequence[int]`, *optional*, defaults to `(3000, 15000)`):
            A fixed color temperature or range in Kelvin, between 3000 K and 15000 K.
    """

    def __init__(self, temperature: int | Sequence[int] = (3_000, 15_000)) -> None:
        super().__init__()
        self.temperature = _check_pair(temperature, "temperature", int)
        if not (
            _PLANCKIAN_MIN_TEMPERATURE
            <= self.temperature[0]
            <= self.temperature[1]
            <= _PLANCKIAN_MAX_TEMPERATURE
        ):
            raise ValueError(
                "temperature must satisfy "
                f"{_PLANCKIAN_MIN_TEMPERATURE} <= min <= max <= {_PLANCKIAN_MAX_TEMPERATURE}, "
                f"but got {self.temperature}."
            )
        self.register_buffer("table", torch.tensor(_PLANCKIAN_BLACKBODY_COEFFICIENTS), persistent=False)

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        return {
            "temperature": torch.randint(
                self.temperature[0], self.temperature[1] + 1, (shape[0],), device=device, generator=generator
            )
        }

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        if frames.shape[-3] != 3:
            raise ValueError(
                f"PlanckianJitter expects 3-channel frames, but got shape {tuple(frames.shape)}."
            )
        table = self.table.to(device=frames.device, dtype=frames.dtype)
        position = (
            params["temperature"].to(frames.dtype) - _PLANCKIAN_MIN_TEMPERATURE
        ) / _PLANCKIAN_TEMPERATURE_STEP
        left = position.floor().long()
        right = (left + 1).clamp(max=len(_PLANCKIAN_BLACKBODY_COEFFICIENTS) - 1)
        weight = (position - left.to(frames.dtype)).unsqueeze(1)
        coefficients = torch.lerp(table[left], table[right], weight)
        scale = torch.stack(
            (
                coefficients[:, 0] / coefficients[:, 1],
                torch.ones_like(coefficients[:, 1]),
                coefficients[:, 2] / coefficients[:, 1],
            ),
            dim=1,
        )
        return (frames * scale.view(-1, 1, 3, 1, 1)).clamp_(0.0, 1.0)


class BatchedRandomSubsetApply(BatchedTransform):
    """Apply a random subset of transforms to every sample of a batch.

    Each sample draws its own subset (multinomial, without replacement) and, with `random_order`, its own
    order. On a batch of more than one sample, every transform runs on the whole batch and is masked into
    the samples that selected it, which keeps the work free of host-device synchronization at the cost of
    computing unselected samples too. On a batch of one (the dataloader backend, one sample per call in a
    DataLoader worker) only the selected transforms run, in sequence; that path also admits transforms
    that change the frame size, which the masked path cannot hold.

    Args:
        transforms (`Sequence[BatchedTransform]`):
            The candidate transforms.
        p (`list[float]`, *optional*):
            Sampling weights, normalized to sum to one. Uniform if `None`.
        n_subset (`int`, *optional*):
            Number of transforms applied per sample, in `[1, len(transforms)]`. All of them if `None`.
        random_order (`bool`, *optional*, defaults to `False`):
            Apply the selected transforms in a random order per sample instead of the configured order.
            On the masked path every transform then runs once per position, so the work and the transient
            memory grow by a factor of `n_subset`.
    """

    def __init__(
        self,
        transforms: Sequence[BatchedTransform],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of transforms")
        if p is None:
            p = [1.0] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )
        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")
        total = sum(p)
        self.transforms = nn.ModuleList(transforms)
        self.register_buffer("p", torch.tensor([prob / total for prob in p]), persistent=False)
        self.n_subset = n_subset
        self.random_order = random_order

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        weights = self.p.to(device).expand(shape[0], -1)
        return {
            "selected": torch.multinomial(weights, self.n_subset, replacement=False, generator=generator),
            "transforms": [transform.make_params(shape, device, generator) for transform in self.transforms],
        }

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        selected = params["selected"]
        children = list(zip(self.transforms, params["transforms"], strict=True))
        if frames.shape[0] == 1 and not torch.compiler.is_compiling():
            return self._transform_one(frames, selected[0].tolist(), children)
        if not self.random_order:
            for index, (transform, child_params) in enumerate(children):
                chosen = (selected == index).any(dim=1)
                frames = self._masked(frames, chosen, transform, child_params)
            return frames
        for position in range(self.n_subset):
            for index, (transform, child_params) in enumerate(children):
                chosen = selected[:, position] == index
                frames = self._masked(frames, chosen, transform, child_params)
        return frames

    def _transform_one(
        self, frames: Tensor, indices: list[int], children: list[tuple[BatchedTransform, dict[str, Any]]]
    ) -> Tensor:
        """Run only the selected transforms on a batch of one, in configured or drawn order."""
        for index in indices if self.random_order else sorted(indices):
            transform, child_params = children[index]
            frames = transform.transform(frames, child_params)
        return frames

    @staticmethod
    def _masked(
        frames: Tensor, chosen: Tensor, transform: BatchedTransform, child_params: dict[str, Any]
    ) -> Tensor:
        """Run `transform` on the whole batch and keep its output for the `chosen` samples only."""
        out = transform.transform(frames, child_params)
        if out.shape != frames.shape:
            raise ValueError(
                f"{type(transform).__name__} changed the frame shape from {tuple(frames.shape)} to "
                f"{tuple(out.shape)}. Transforms that change the frame size run one sample at a time only, "
                "as on the dataloader backend."
            )
        return torch.where(_per_sample(chosen), out, frames)


@contextmanager
def _seeded_default_generators(seed: int, device: torch.device) -> Iterator[None]:
    """Seed the CPU default generator, and the CUDA one of `device`, for the block; restore them after."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        with torch.random.fork_rng(devices=[index]):
            torch.default_generator.manual_seed(seed)
            torch.cuda.default_generators[index].manual_seed(seed)
            yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            yield


class PerSampleTransform(BatchedTransform):
    """Run a per-sample transform on each sample of a batch in turn, seeded from the batch generator.

    The adapter for transform types with no batched implementation: any `torchvision.transforms.v2` class
    by name, and `JPEGCompression`. `make_params` draws one seed per sample from the generator, and
    `transform` seeds the default generators with it (under `torch.random.fork_rng`, so they are restored
    afterwards) before calling the wrapped transform on that sample's `(N, C, H, W)` frames. The
    augmentation is therefore reproducible from the batch generator, and the frames of one sample share
    the parameters the wrapped transform draws, as on the dataloader backend.

    The loop is not vectorized: it costs one call per sample, and on an accelerator `JPEGCompression`
    round-trips through the CPU. The wrapped transform may change the frame size; see
    `BatchedRandomSubsetApply` for where that is supported.

    Args:
        transform (`Callable[[Tensor], Tensor]`):
            The per-sample transform, taking one sample's float `(N, C, H, W)` frames in `[0, 1]`.
    """

    def __init__(self, transform: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.per_sample = transform

    def make_params(
        self, shape: torch.Size, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, Any]:
        high = torch.iinfo(torch.int64).max
        return {"seed": torch.randint(0, high, (shape[0],), device=device, generator=generator)}

    @torch.compiler.disable
    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        outputs = []
        for sample, seed in zip(frames.unbind(0), params["seed"].tolist(), strict=True):
            with _seeded_default_generators(seed, frames.device):
                outputs.append(self.per_sample(sample))
        return torch.stack(outputs)


# LeRobot transforms with no batched implementation; `PerSampleTransform` wraps them.
_PER_SAMPLE_TRANSFORMS: dict[str, type[Transform]] = {
    "JPEGCompression": JPEGCompression,
}

# Batched counterparts, keyed by the same type names as `ImageTransformConfig.type`.
_BATCHED_TRANSFORMS: dict[str, type[BatchedTransform]] = {
    "Identity": BatchedIdentity,
    "ColorJitter": BatchedColorJitter,
    "RandomAffine": BatchedRandomAffine,
    "RandomRotation": BatchedRandomRotation,
    "SharpnessJitter": BatchedSharpnessJitter,
    "GaussianNoise": BatchedGaussianNoise,
    "MotionBlur": BatchedMotionBlur,
    "GaussianPatchBrightness": BatchedGaussianPatchBrightness,
    "RandomShadow": BatchedRandomShadow,
    "CoarseDropout": BatchedCoarseDropout,
    "GammaCorrection": BatchedGammaCorrection,
    "PlanckianJitter": BatchedPlanckianJitter,
}


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    `BatchedRandomSubsetApply` samples them per frame.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
    # Where the transforms run. "dataloader" applies them per sample in the DataLoader workers (CPU).
    # "gpu" leaves the workers to decode only and applies the batched, per-sample equivalents on the
    # policy device inside the training preprocessor; see `BatchedImageTransforms` for what is supported.
    # The knobs below sit next to `enable` and `tfs` because they describe how the dataset's transforms
    # run, and every consumer of a dataset config that enables them can then run them unchanged.
    backend: ImageTransformsBackend = "dataloader"
    # "gpu" backend only. torch.compile the batched transforms: about three times faster per batch for a
    # one-off compile of 15-30 s per frame shape (a second, dynamic compile covers the epoch's partial last
    # batch and the chunk remainders). Set to False if compilation is unavailable on the machine. This does
    # not change the augmentation: the random parameters are drawn outside the compiled function.
    gpu_compile: bool = True
    # "gpu" backend only. Augment this many samples at a time. Every configured transform runs on the whole
    # chunk and is masked into the samples that drew it, so the peak memory is two float copies of the batch
    # (the input conversion and the output) plus about four copies of the chunk compiled, ten eager. At
    # 256x256 with two frames per sample that is 0.5 GB for a batch of 160 at the default against 1.1 GB
    # unbounded; at 480x640, 2.4 GB against 6.3 GB. Chunking does not change the augmentation either, since
    # the parameters are drawn for the whole batch and sliced per chunk. None augments the whole batch at once.
    gpu_chunk_size: int | None = 32
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
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
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
        }
    )

    def __post_init__(self) -> None:
        if self.backend not in ("dataloader", "gpu"):
            raise ValueError(f"image_transforms.backend must be 'dataloader' or 'gpu', got {self.backend!r}.")
        if self.gpu_chunk_size is not None and self.gpu_chunk_size < 1:
            raise ValueError(
                f"image_transforms.gpu_chunk_size must be a positive integer or None, got {self.gpu_chunk_size}."
            )


def make_batched_transform_from_config(cfg: ImageTransformConfig) -> BatchedTransform:
    """Build the batched transform for a configured type.

    Types with a batched implementation (`_BATCHED_TRANSFORMS`) get it. `JPEGCompression` and any other
    `torchvision.transforms.v2` class are wrapped in `PerSampleTransform`, which applies them one sample at a
    time.

    Args:
        cfg (`ImageTransformConfig`):
            The transform's type and keyword arguments.

    Returns:
        `BatchedTransform`: The batched transform.

    Raises:
        ValueError: If the type is neither a LeRobot transform nor a `torchvision.transforms.v2` class.
    """
    if cfg.type in _BATCHED_TRANSFORMS:
        return _BATCHED_TRANSFORMS[cfg.type](**cfg.kwargs)
    if cfg.type in _PER_SAMPLE_TRANSFORMS:
        return PerSampleTransform(_PER_SAMPLE_TRANSFORMS[cfg.type](**cfg.kwargs))
    transform_cls = getattr(v2, cfg.type, None)
    if isinstance(transform_cls, type) and issubclass(transform_cls, Transform):
        return PerSampleTransform(transform_cls(**cfg.kwargs))
    valid = ", ".join(sorted({*_BATCHED_TRANSFORMS, *_PER_SAMPLE_TRANSFORMS}))
    raise ValueError(
        f"Transform '{cfg.type}' is not valid. It must be a class in torchvision.transforms.v2 or one of: "
        f"{valid}."
    )


class BatchedImageTransforms(nn.Module):
    """Apply the configured image transforms to a batch, with an independent draw for every sample.

    Draws a transform subset and parameters per sample, so one call on the GPU backend replaces the
    per-sample calls the DataLoader workers make; on the dataloader backend `ImageTransforms` runs the same
    module on a batch of one. The frames of one sample (its observation history, a camera's `(T, C, H, W)`
    tensor) share the sample's parameters.

    All random draws come from the `generator` passed to `forward` (or the device's default generator when
    none is), and happen before the frames are touched. The math itself is a pure function of frames and
    parameters, which is the part `compile_model` compiles and `chunk_size` runs one chunk at a time; neither
    changes the augmentation a given generator state produces.

    Args:
        cfg (`ImageTransformsConfig`):
            The same configuration `ImageTransforms` takes. `backend`, `gpu_compile` and `gpu_chunk_size` are
            not consulted; the caller passes the two knobs below.
        chunk_size (`int`, *optional*):
            Transform at most this many samples at a time to bound the memory of the intermediate frames.
            The whole batch at once if `None`.
        compile_model (`bool`, *optional*, defaults to `False`):
            Wrap the transform math in `torch.compile`. Compilation happens on the first call.

    Example:
        ```python
        >>> import torch
        >>> from lerobot.transforms import BatchedImageTransforms, ImageTransformsConfig
        >>> tf = BatchedImageTransforms(ImageTransformsConfig(enable=True))
        >>> frames = torch.rand(4, 2, 3, 96, 96)  # (B, T, C, H, W), float in [0, 1]
        >>> tf(frames, generator=torch.Generator().manual_seed(0)).shape
        torch.Size([4, 2, 3, 96, 96])
        ```
    """

    def __init__(
        self, cfg: ImageTransformsConfig, chunk_size: int | None = None, compile_model: bool = False
    ) -> None:
        super().__init__()
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive integer or None, got {chunk_size}.")
        self.chunk_size = chunk_size
        weights: list[float] = []
        transforms: dict[str, BatchedTransform] = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue
            transforms[tf_name] = make_batched_transform_from_config(tf_cfg)
            weights.append(tf_cfg.weight)
        self.transforms = nn.ModuleDict(transforms)
        self.weights = weights
        n_subset = min(len(transforms), cfg.max_num_transforms)
        self.tf: BatchedTransform
        if n_subset == 0 or not cfg.enable:
            self.tf = BatchedIdentity()
        else:
            self.tf = BatchedRandomSubsetApply(
                transforms=list(transforms.values()),
                p=weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )
        self._transform = torch.compile(self.transform) if compile_model else self.transform

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        """Transform float `(B, N, C, H, W)` frames with parameters drawn by `self.tf.make_params`."""
        return self.tf.transform(frames, params)

    def forward(self, images: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Transform a batch of images or frame stacks.

        Args:
            images (`torch.Tensor`):
                `(B, C, H, W)` or `(B, T, C, H, W)` with `C` in `{1, 3}`. `uint8` in `[0, 255]` or floating
                point in `[0, 1]`.
            generator (`torch.Generator`, *optional*):
                Generator on the images' device to draw the parameters from. The device's default generator
                if `None`.

        Returns:
            `torch.Tensor`: The transformed batch, same shape and dtype as the input. Floating point inputs
            are computed in `float32`; `uint8` inputs are rounded back after the transform.
        """
        if images.ndim not in (4, 5):
            raise ValueError(
                f"Expected (B, C, H, W) or (B, T, C, H, W) images, got shape {tuple(images.shape)}."
            )
        if images.shape[-3] not in (1, 3):
            raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(images.shape)}.")
        frames = images.unsqueeze(1) if images.ndim == 4 else images
        if frames.dtype == torch.uint8:
            work = frames.to(torch.float32) / 255.0
        elif frames.is_floating_point():
            work = frames.to(torch.float32)
        else:
            raise TypeError(f"Expected uint8 or floating point images, got {images.dtype}.")
        params = self.tf.make_params(work.shape, work.device, generator)
        batch_size = work.shape[0]
        if self.chunk_size is None or batch_size <= self.chunk_size:
            out = self._transform(work, params)
        else:
            out = torch.cat(
                [
                    self._transform(
                        work[start : start + self.chunk_size],
                        slice_params(params, start, start + self.chunk_size),
                    )
                    for start in range(0, batch_size, self.chunk_size)
                ],
                dim=0,
            )
        out = (out * 255.0).round_().to(torch.uint8) if frames.dtype == torch.uint8 else out.to(frames.dtype)
        return out.squeeze(1) if images.ndim == 4 else out


class ImageTransforms(nn.Module):
    """The configured image transforms for one sample at a time, as the DataLoader workers apply them.

    A thin wrapper over `BatchedImageTransforms`: the sample is given a batch dimension, transformed, and
    handed back without it, so both backends run the same code and a sample augmented here equals the same
    sample augmented in a batch on the GPU backend. `LeRobotDataset(image_transforms=ImageTransforms(cfg))`
    is the usual way to use it.

    Args:
        cfg (`ImageTransformsConfig`):
            Which transforms to sample and how. `backend`, `gpu_compile` and `gpu_chunk_size` are not
            consulted: this class always runs eagerly, one sample at a time, where it is called.

    Example:
        ```python
        >>> import torch
        >>> from lerobot.transforms import ImageTransforms, ImageTransformsConfig
        >>> tf = ImageTransforms(ImageTransformsConfig(enable=True))
        >>> tf(torch.rand(3, 96, 96)).shape  # (C, H, W), float in [0, 1]
        torch.Size([3, 96, 96])
        ```
    """

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self.batched = BatchedImageTransforms(cfg)

    @property
    def transforms(self) -> nn.ModuleDict:
        """The configured transforms by name, those with a positive weight."""
        return self.batched.transforms

    @property
    def weights(self) -> list[float]:
        """The sampling weights of `transforms`, in the same order."""
        return self.batched.weights

    @property
    def tf(self) -> BatchedTransform:
        """The transform applied to every sample: a `BatchedRandomSubsetApply`, or `BatchedIdentity` when disabled."""
        return self.batched.tf

    def forward(self, images: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Transform one image or one stack of frames.

        Args:
            images (`torch.Tensor`):
                `(C, H, W)` or `(T, C, H, W)` with `C` in `{1, 3}`; the `T` frames share one parameter draw.
                `uint8` in `[0, 255]` or floating point in `[0, 1]`.
            generator (`torch.Generator`, *optional*):
                Generator on the images' device to draw the parameters from. The device's default generator
                if `None`, which in a DataLoader worker is the worker's own.

        Returns:
            `torch.Tensor`: The transformed images, same dtype as the input and, unless a configured transform
            changes the frame size, the same shape.
        """
        if images.ndim not in (3, 4):
            raise ValueError(f"Expected (C, H, W) or (T, C, H, W) images, got shape {tuple(images.shape)}.")
        return self.batched(images.unsqueeze(0), generator=generator).squeeze(0)
