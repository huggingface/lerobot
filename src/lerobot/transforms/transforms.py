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
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
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

        self.selected_transforms = None

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

    def _check_input(self, sharpness):
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
        else:
            self.std = (float(std[0]), float(std[1]))

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"std": torch.empty(1).uniform_(self.std[0], self.std[1]).item()}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, torch.Tensor) and inpt.is_floating_point():
            return (inpt + torch.randn_like(inpt) * (params["std"] / 255.0)).clamp(0.0, 1.0)
        return inpt


class MotionBlur(Transform):
    """Apply directional motion blur to simulate fast robot or object movement.

    Generates a 1D averaging kernel along a random direction, applied via depthwise convolution.

    Args:
        kernel_size: Range (min, max) for blur kernel size. Will be forced odd.
    """

    def __init__(self, kernel_size: int | Sequence[int] = (3, 11)) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        ks = int(torch.randint(self.kernel_size[0], self.kernel_size[1] + 1, (1,)).item())
        if ks % 2 == 0:
            ks += 1
        angle = torch.empty(1).uniform_(0, 360).item()
        return {"kernel_size": ks, "angle": angle}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        ks = params["kernel_size"]
        rad = params["angle"] * math.pi / 180
        cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
        x = inpt.unsqueeze(0) if inpt.dim() == 3 else inpt
        if cos_a > sin_a:
            out = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(x, (ks // 2, ks // 2, 0, 0), mode="replicate"),
                (1, ks),
                stride=1,
            )
        else:
            out = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(x, (0, 0, ks // 2, ks // 2), mode="replicate"),
                (ks, 1),
                stride=1,
            )
        return (out.squeeze(0) if inpt.dim() == 3 else out).clamp(0.0, 1.0)


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
        else:
            self.quality = (int(quality[0]), int(quality[1]))

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"quality": int(torch.randint(self.quality[0], self.quality[1] + 1, (1,)).item())}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        from torchvision.io import decode_image, encode_jpeg

        img_uint8 = (inpt * 255).byte()
        if img_uint8.dim() == 3:
            try:
                buf = encode_jpeg(img_uint8.cpu(), quality=params["quality"])
                return decode_image(buf).to(device=inpt.device, dtype=inpt.dtype) / 255.0
            except Exception:
                return inpt
        return inpt


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
        else:
            self.num_patches = (int(num_patches[0]), int(num_patches[1]))
        self.sigma_range = sigma_range
        self.factor_range = factor_range

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
        if inpt.dim() == 3:
            return (inpt * mask.unsqueeze(0)).clamp(0.0, 1.0)
        return (inpt * mask.unsqueeze(0).unsqueeze(0)).clamp(0.0, 1.0)


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
        else:
            self.opacity = (float(opacity[0]), float(opacity[1]))

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"opacity": torch.empty(1).uniform_(self.opacity[0], self.opacity[1]).item()}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        h, w = inpt.shape[-2:]
        x_start = int(torch.randint(0, w // 2, (1,)).item())
        x_end = int(torch.randint(w // 3, w, (1,)).item())
        mask = torch.ones(h, w, device=inpt.device, dtype=inpt.dtype)
        if torch.rand(1).item() < 0.5:
            mask[:, x_start:x_end] = 1.0 - params["opacity"]
        else:
            mask[:, x_start:x_end] = 1.0 + params["opacity"]
        mask = mask.unsqueeze(0).unsqueeze(0)
        small = torch.nn.functional.avg_pool2d(mask, 8, stride=8)
        mask = (
            torch.nn.functional.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
            .squeeze(0)
            .squeeze(0)
        )
        if inpt.dim() == 3:
            return (inpt * mask.unsqueeze(0)).clamp(0.0, 1.0)
        return (inpt * mask.unsqueeze(0).unsqueeze(0)).clamp(0.0, 1.0)


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
        self.max_holes = max_holes
        self.max_height_frac = max_height_frac
        self.max_width_frac = max_width_frac
        self.fill_value = fill_value

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        n = int(torch.randint(1, self.max_holes + 1, (1,)).item())
        return {"n_holes": n}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        h, w = inpt.shape[-2:]
        result = inpt.clone()
        for _ in range(params["n_holes"]):
            hole_h = int(torch.randint(1, max(2, int(h * self.max_height_frac)), (1,)).item())
            hole_w = int(torch.randint(1, max(2, int(w * self.max_width_frac)), (1,)).item())
            y = int(torch.randint(0, h - hole_h + 1, (1,)).item())
            x = int(torch.randint(0, w - hole_w + 1, (1,)).item())
            if result.dim() == 3:
                result[:, y : y + hole_h, x : x + hole_w] = self.fill_value
            else:
                result[:, :, y : y + hole_h, x : x + hole_w] = self.fill_value
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
            self.gamma = (1.0 / float(gamma), float(gamma))
        else:
            self.gamma = (float(gamma[0]), float(gamma[1]))

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        log_lo = math.log(self.gamma[0])
        log_hi = math.log(self.gamma[1])
        gamma = math.exp(torch.empty(1).uniform_(log_lo, log_hi).item())
        return {"gamma": gamma}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, torch.Tensor) and inpt.is_floating_point():
            return inpt.pow(params["gamma"]).clamp(0.0, 1.0)
        return inpt


class PlanckianJitter(Transform):
    """Simulate color temperature shift along the Planckian locus.

    Models the visual effect of different light sources (LED vs fluorescent vs
    daylight) by applying physically-motivated per-channel scaling. More accurate
    than arbitrary hue shift for lighting variation.

    Reference: Zini et al., "Planckian Jitter", CVPR 2022 Workshop.

    Args:
        strength: Range (min, max) for per-channel scale factor.
    """

    def __init__(self, strength: Sequence[float] = (0.85, 1.15)) -> None:
        super().__init__()
        self.strength = strength

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"scale": torch.empty(3).uniform_(self.strength[0], self.strength[1]).tolist()}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        scale = torch.tensor(params["scale"], device=inpt.device, dtype=inpt.dtype)
        if inpt.dim() == 3:
            return (inpt * scale.view(3, 1, 1)).clamp(0.0, 1.0)
        return (inpt * scale.view(1, 3, 1, 1)).clamp(0.0, 1.0)


# Custom transform registry for make_transform_from_config
_CUSTOM_TRANSFORMS: dict[str, type] = {}


def _register_custom_transforms() -> None:
    """Register all custom transforms defined in this module."""
    _CUSTOM_TRANSFORMS.update(
        {
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
    )


_register_custom_transforms()


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


def make_transform_from_config(cfg: ImageTransformConfig):
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

        self.weights = []
        self.transforms = {}
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
