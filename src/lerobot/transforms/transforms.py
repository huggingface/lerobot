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
import collections
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.io import decode_image, encode_jpeg
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable[..., Any]],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
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

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms: list[Callable[..., Any]] = []

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness: float | Sequence[float]) -> tuple[float, float]:
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


class GaussianNoise(Transform):
    """Add Gaussian noise to simulate camera sensor noise.

    Models readout noise from ADC quantization, which increases in low-light conditions.
    Common in real-robot setups where wrist cameras operate in suboptimal lighting.

    Args:
        std: Range (min, max) for noise standard deviation in pixel-value scale (0-255).
    """

    def __init__(self, std: float | Sequence[float] = (5.0, 25.0)) -> None:
        super().__init__()
        if isinstance(std, (int, float)):
            self.std = (0.0, float(std))
        elif isinstance(std, Sequence) and len(std) == 2:
            self.std = (float(std[0]), float(std[1]))
        else:
            raise TypeError("std must be a number or a sequence with length 2.")
        if not 0.0 <= self.std[0] <= self.std[1]:
            raise ValueError(f"std must satisfy 0 <= min <= max, but got {self.std}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "std": torch.empty(1).uniform_(self.std[0], self.std[1]).item(),
            "seed": torch.randint(0, torch.iinfo(torch.int64).max, ()).item(),
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, torch.Tensor) and inpt.is_floating_point():
            generator = torch.Generator(device=inpt.device).manual_seed(params["seed"])
            noise = torch.randn(inpt.shape, device=inpt.device, dtype=inpt.dtype, generator=generator)
            return (inpt + noise * (params["std"] / 255.0)).clamp(0.0, 1.0)
        return inpt


class MotionBlur(Transform):
    """Apply directional motion blur to simulate fast robot or object movement.

    Generates a 1D averaging kernel along a random direction, applied via depthwise convolution.

    Args:
        kernel_size: An odd kernel size or a range containing at least one odd kernel size.
    """

    def __init__(self, kernel_size: int | Sequence[int] = (3, 11)) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, Sequence) and len(kernel_size) == 2:
            self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        else:
            raise TypeError("kernel_size must be an int or a sequence with length 2.")
        if not 1 <= self.kernel_size[0] <= self.kernel_size[1]:
            raise ValueError(f"kernel_size must satisfy 1 <= min <= max, but got {self.kernel_size}.")
        self._first_odd_kernel_size = self.kernel_size[0] + (self.kernel_size[0] + 1) % 2
        if self._first_odd_kernel_size > self.kernel_size[1]:
            raise ValueError(f"kernel_size range must contain an odd value, but got {self.kernel_size}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        num_odd_sizes = (self.kernel_size[1] - self._first_odd_kernel_size) // 2 + 1
        size_index = int(torch.randint(0, num_odd_sizes, ()).item())
        ks = self._first_odd_kernel_size + 2 * size_index
        angle = torch.empty(1).uniform_(0, 360).item()
        return {"kernel_size": ks, "angle": angle}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3:
            raise ValueError(f"MotionBlur expects [..., C, H, W] input, but got shape {inpt.shape}.")

        kernel_size = params["kernel_size"]
        radius = kernel_size // 2
        angle = math.radians(params["angle"])
        positions = torch.linspace(-radius, radius, kernel_size, device=inpt.device)
        x_coords = (positions * math.cos(angle)).round().to(torch.long) + radius
        y_coords = (positions * math.sin(angle)).round().to(torch.long) + radius
        kernel = torch.zeros((kernel_size, kernel_size), device=inpt.device, dtype=inpt.dtype)
        kernel[y_coords, x_coords] = 1
        kernel /= kernel.sum()

        channels, height, width = inpt.shape[-3:]
        flat_input = inpt.reshape(-1, channels, height, width)
        depthwise_kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        padded = torch.nn.functional.pad(flat_input, (radius,) * 4, mode="replicate")
        output = torch.nn.functional.conv2d(padded, depthwise_kernel, groups=channels)
        return output.reshape(inpt.shape).clamp(0.0, 1.0)


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


class GaussianPatchBrightness(Transform):
    """Apply spatially-varying brightness with Gaussian patches.

    Simulates uneven overhead lighting, spotlights, and shadow patches commonly
    encountered in real robot workspaces with multiple light sources.

    Args:
        num_patches: Range (min, max) for number of brightness patches.
        sigma_range: Range for Gaussian sigma as fraction of image size.
        factor_range: Range for brightness factor (< 1 darkens, > 1 brightens).
    """

    def __init__(
        self,
        num_patches: int | Sequence[int] = (1, 4),
        sigma_range: Sequence[float] = (0.05, 0.25),
        factor_range: Sequence[float] = (0.4, 1.6),
    ) -> None:
        super().__init__()
        if isinstance(num_patches, int):
            self.num_patches = (num_patches, num_patches)
        elif isinstance(num_patches, Sequence) and len(num_patches) == 2:
            self.num_patches = (int(num_patches[0]), int(num_patches[1]))
        else:
            raise TypeError("num_patches must be an int or a sequence with length 2.")
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

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        n = int(torch.randint(self.num_patches[0], self.num_patches[1] + 1, (1,)).item())
        return {
            "centers": torch.rand(n, 2).tolist(),
            "sigmas": torch.empty(n).uniform_(self.sigma_range[0], self.sigma_range[1]).tolist(),
            "factors": torch.empty(n).uniform_(self.factor_range[0], self.factor_range[1]).tolist(),
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        h, w = inpt.shape[-2:]
        mask = torch.ones(h, w, device=inpt.device, dtype=inpt.dtype)
        grid_y = torch.linspace(0, 1, h, device=inpt.device, dtype=inpt.dtype)
        grid_x = torch.linspace(0, 1, w, device=inpt.device, dtype=inpt.dtype)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        for (cy, cx), sigma, factor in zip(
            params["centers"], params["sigmas"], params["factors"], strict=True
        ):
            gauss = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
            mask = mask * (1.0 + (factor - 1.0) * gauss)
        broadcast_shape = (1,) * (inpt.ndim - 2) + (h, w)
        return (inpt * mask.reshape(broadcast_shape)).clamp(0.0, 1.0)


class RandomShadow(Transform):
    """Add random vertical band shadow with smooth edges.

    Simulates cast shadows from objects or people near the robot workspace.
    Symmetric: randomly brightens or darkens to prevent BatchNorm stats shift.

    Args:
        opacity: Range (min, max) for shadow/highlight opacity.
    """

    def __init__(self, opacity: float | Sequence[float] = (0.3, 0.6)) -> None:
        super().__init__()
        if isinstance(opacity, (int, float)):
            self.opacity = (float(opacity), float(opacity))
        elif isinstance(opacity, Sequence) and len(opacity) == 2:
            self.opacity = (float(opacity[0]), float(opacity[1]))
        else:
            raise TypeError("opacity must be a number or a sequence with length 2.")
        if not 0.0 <= self.opacity[0] <= self.opacity[1] <= 1.0:
            raise ValueError(f"opacity must satisfy 0 <= min <= max <= 1, but got {self.opacity}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "opacity": torch.empty(1).uniform_(self.opacity[0], self.opacity[1]).item(),
            "start": torch.rand(1).item(),
            "width": torch.empty(1).uniform_(1 / 3, 2 / 3).item(),
            "direction": -1.0 if torch.rand(1).item() < 0.5 else 1.0,
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3:
            raise ValueError(f"RandomShadow expects [..., C, H, W] input, but got shape {inpt.shape}.")

        h, w = inpt.shape[-2:]
        band_width = max(1, min(w, round(params["width"] * w)))
        x_start = round(params["start"] * (w - band_width))
        x_end = x_start + band_width
        mask = torch.ones(h, w, device=inpt.device, dtype=inpt.dtype)
        mask[:, x_start:x_end] = 1.0 + params["direction"] * params["opacity"]

        smoothing_size = min(8, h, w)
        if smoothing_size > 1:
            batched_mask = mask[None, None]
            small = torch.nn.functional.avg_pool2d(batched_mask, smoothing_size, stride=smoothing_size)
            mask = torch.nn.functional.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)[
                0, 0
            ]

        broadcast_shape = (1,) * (inpt.ndim - 2) + (h, w)
        return (inpt * mask.reshape(broadcast_shape)).clamp(0.0, 1.0)


class CoarseDropout(Transform):
    """Drop random rectangular patches to simulate partial occlusion.

    Models objects, hands, or cables passing through the camera field of view
    during robot manipulation.

    Args:
        max_holes: Maximum number of rectangular patches to drop.
        max_height_frac: Maximum patch height as fraction of image height.
        max_width_frac: Maximum patch width as fraction of image width.
        fill_value: Value to fill dropped regions with.
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

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        n = int(torch.randint(1, self.max_holes + 1, (1,)).item())
        sizes = torch.rand(n, 2)
        sizes[:, 0] *= self.max_height_frac
        sizes[:, 1] *= self.max_width_frac
        return {"sizes": sizes.tolist(), "positions": torch.rand(n, 2).tolist()}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3:
            raise ValueError(f"CoarseDropout expects [..., C, H, W] input, but got shape {inpt.shape}.")

        h, w = inpt.shape[-2:]
        result = inpt.clone()
        for (height_frac, width_frac), (y_frac, x_frac) in zip(
            params["sizes"], params["positions"], strict=True
        ):
            hole_h = max(1, min(h, round(height_frac * h)))
            hole_w = max(1, min(w, round(width_frac * w)))
            y = round(y_frac * (h - hole_h))
            x = round(x_frac * (w - hole_w))
            result[..., y : y + hole_h, x : x + hole_w] = self.fill_value
        return result


class GammaCorrection(Transform):
    """Apply random gamma correction to simulate exposure variation.

    Models different camera auto-exposure settings and sensor response curves.
    Uses log-symmetric sampling so brightening and darkening are equally likely,
    preventing BatchNorm statistics shift.

    Args:
        gamma: Range (min, max) for gamma value. Values < 1 brighten, > 1 darken.
    """

    def __init__(self, gamma: float | Sequence[float] = (0.5, 2.0)) -> None:
        super().__init__()
        if isinstance(gamma, (int, float)):
            gamma = float(gamma)
            if gamma <= 0:
                raise ValueError(f"gamma must be positive, but got {gamma}.")
            self.gamma = (min(gamma, 1.0 / gamma), max(gamma, 1.0 / gamma))
        elif isinstance(gamma, Sequence) and len(gamma) == 2:
            self.gamma = (float(gamma[0]), float(gamma[1]))
        else:
            raise TypeError("gamma must be a number or a sequence with length 2.")
        if not 0.0 < self.gamma[0] <= self.gamma[1]:
            raise ValueError(f"gamma must satisfy 0 < min <= max, but got {self.gamma}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        log_lo = math.log(self.gamma[0])
        log_hi = math.log(self.gamma[1])
        gamma = math.exp(torch.empty(1).uniform_(log_lo, log_hi).item())
        return {"gamma": gamma}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, torch.Tensor) and inpt.is_floating_point():
            return inpt.pow(params["gamma"]).clamp(0.0, 1.0)
        return inpt


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


class PlanckianJitter(Transform):
    """Simulate color temperature shift along the Planckian locus.

    Samples one black-body temperature and applies the corresponding correlated red
    and blue channel scaling while preserving the green channel. Coefficients between
    the tabulated 500 K intervals are linearly interpolated.

    Reference: Zini et al., "Planckian Jitter", CVPR 2022 Workshop.

    Args:
        temperature: A fixed color temperature or range in Kelvin. Supported values
            are between 3000 K and 15000 K.
    """

    def __init__(self, temperature: int | Sequence[int] = (3_000, 15_000)) -> None:
        super().__init__()
        if isinstance(temperature, int):
            self.temperature = (temperature, temperature)
        elif isinstance(temperature, Sequence) and len(temperature) == 2:
            self.temperature = (int(temperature[0]), int(temperature[1]))
        else:
            raise TypeError("temperature must be an int or a sequence with length 2.")
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

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        temperature = int(torch.randint(self.temperature[0], self.temperature[1] + 1, ()).item())
        return {"temperature": temperature}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3 or inpt.shape[-3] != 3:
            raise ValueError(f"PlanckianJitter expects [..., 3, H, W] input, but got shape {inpt.shape}.")

        table_position = (params["temperature"] - _PLANCKIAN_MIN_TEMPERATURE) / _PLANCKIAN_TEMPERATURE_STEP
        left_index = math.floor(table_position)
        right_index = min(left_index + 1, len(_PLANCKIAN_BLACKBODY_COEFFICIENTS) - 1)
        interpolation_weight = table_position - left_index

        left = torch.tensor(
            _PLANCKIAN_BLACKBODY_COEFFICIENTS[left_index],
            device=inpt.device,
            dtype=inpt.dtype,
        )
        right = torch.tensor(
            _PLANCKIAN_BLACKBODY_COEFFICIENTS[right_index],
            device=inpt.device,
            dtype=inpt.dtype,
        )
        coefficients = torch.lerp(left, right, interpolation_weight)
        scale = torch.stack(
            (
                coefficients[0] / coefficients[1],
                coefficients.new_tensor(1.0),
                coefficients[2] / coefficients[1],
            )
        )
        broadcast_shape = (1,) * (inpt.ndim - 3) + (3, 1, 1)
        return (inpt * scale.reshape(broadcast_shape)).clamp(0.0, 1.0)


_CUSTOM_TRANSFORMS: dict[str, type[Transform]] = {
    "SharpnessJitter": SharpnessJitter,
    "GaussianNoise": GaussianNoise,
    "MotionBlur": MotionBlur,
    "JPEGCompression": JPEGCompression,
    "GaussianPatchBrightness": GaussianPatchBrightness,
    "RandomShadow": RandomShadow,
    "CoarseDropout": CoarseDropout,
    "GammaCorrection": GammaCorrection,
    "PlanckianJitter": PlanckianJitter,
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
    We use a custom RandomSubsetApply container to sample them.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
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


def make_transform_from_config(cfg: ImageTransformConfig) -> Transform:
    if cfg.type in _CUSTOM_TRANSFORMS:
        return _CUSTOM_TRANSFORMS[cfg.type](**cfg.kwargs)

    transform_cls = getattr(v2, cfg.type, None)
    if isinstance(transform_cls, type) and issubclass(transform_cls, Transform):
        return transform_cls(**cfg.kwargs)

    valid_custom = ", ".join(sorted(_CUSTOM_TRANSFORMS.keys()))
    raise ValueError(
        f"Transform '{cfg.type}' is not valid. It must be a class in "
        f"torchvision.transforms.v2 or one of: {valid_custom}."
    )


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights: list[float] = []
        self.transforms: dict[str, Transform] = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
