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
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
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
    if cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)

    if cfg.type == "OpenCVColorJitter":
        return OpenCVColorJitter(**cfg.kwargs)

    transform_cls = getattr(v2, cfg.type, None)
    if isinstance(transform_cls, type) and issubclass(transform_cls, Transform):
        return transform_cls(**cfg.kwargs)

    raise ValueError(
        f"Transform '{cfg.type}' is not valid. It must be a class in "
        "torchvision.transforms.v2 or one of: 'OpenCVColorJitter', 'SharpnessJitter'."
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


class OpenCVColorJitter(Transform):
    """Apply Isaac-GR00T/Albumentations-compatible color jitter with OpenCV.

    The four arguments are non-negative jitter magnitudes. Brightness,
    contrast, and saturation factors are sampled uniformly from
    ``[max(0, 1 - magnitude), 1 + magnitude]``; hue is sampled uniformly from
    ``[-hue, hue]`` and must not exceed ``0.5``.

    A single set of factors and one random operation order are sampled per
    call and applied across every frame in the input. Inputs must be CPU
    ``torch.uint8`` tensors shaped ``(..., 3, H, W)``. This keeps temporal
    frames photometrically consistent while retaining the byte-level behavior
    of Albumentations 1.4.18 on RGB uint8 images.
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ) -> None:
        super().__init__()
        self.brightness = self._check_magnitude("brightness", brightness)
        self.contrast = self._check_magnitude("contrast", contrast)
        self.saturation = self._check_magnitude("saturation", saturation)
        self.hue = self._check_magnitude("hue", hue, maximum=0.5)

    @staticmethod
    def _check_magnitude(name: str, value: float, maximum: float | None = None) -> float:
        try:
            amount = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"OpenCVColorJitter {name} must be a float, got {value!r}") from exc
        if not np.isfinite(amount) or amount < 0:
            raise ValueError(f"OpenCVColorJitter {name} must be finite and non-negative, got {amount}")
        if maximum is not None and amount > maximum:
            raise ValueError(f"OpenCVColorJitter {name} must be <= {maximum}, got {amount}")
        return amount

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        del flat_inputs
        params: dict[str, Any] = {
            "brightness": random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness),
            "contrast": random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast),
            "saturation": random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation),
            "hue": random.uniform(-self.hue, self.hue),
        }
        order = np.arange(4)
        # Albumentations 1.4.18 seeds a NumPy RandomState from Python random
        # before shuffling the operation order. Preserve that exact sequence.
        np.random.RandomState(random.randint(0, (1 << 32) - 1)).shuffle(order)
        params["order"] = order.tolist()
        return params

    @staticmethod
    def _lut(image: np.ndarray, factor: float, value: float = 0.0) -> np.ndarray:
        lut = np.arange(256, dtype=np.float32) * factor + value
        return cv2.LUT(image, np.clip(lut, 0, 255).astype(np.uint8))

    @classmethod
    def apply_rgb_image(cls, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Apply already-sampled parameters to one HWC RGB uint8 image."""

        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                "OpenCVColorJitter expects an HWC RGB uint8 image, "
                f"got shape={image.shape}, dtype={image.dtype}"
            )
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)

        def adjust_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
            if factor == 0:
                return np.zeros_like(frame)
            if factor == 1:
                return frame
            return cls._lut(frame, factor)

        def adjust_contrast(frame: np.ndarray, factor: float) -> np.ndarray:
            if factor == 1:
                return frame
            mean = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).mean()
            if factor == 0:
                return np.full_like(frame, int(mean + 0.5), dtype=frame.dtype)
            return cls._lut(frame, factor, mean * (1 - factor))

        def adjust_saturation(frame: np.ndarray, factor: float) -> np.ndarray:
            if factor == 1:
                return frame
            grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
            if factor == 0:
                return grayscale
            adjusted = cv2.addWeighted(frame, factor, grayscale, 1 - factor, gamma=0)
            return np.clip(adjusted, 0, 255).astype(frame.dtype)

        def adjust_hue(frame: np.ndarray, factor: float) -> np.ndarray:
            if factor == 0:
                return frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            lut = np.arange(256, dtype=np.int16)
            lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
            hsv[..., 0] = cv2.LUT(hsv[..., 0], lut)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        transforms = (adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue)
        factors = (
            params["brightness"],
            params["contrast"],
            params["saturation"],
            params["hue"],
        )
        for index in params["order"]:
            image = transforms[index](image, factors[index])
        return image

    def transform(self, inpt: Any, params: dict[str, Any]) -> torch.Tensor:
        if not torch.is_tensor(inpt):
            raise TypeError(f"OpenCVColorJitter expects torch.Tensor inputs, got {type(inpt).__name__}")
        if inpt.device.type != "cpu":
            raise ValueError(f"OpenCVColorJitter expects CPU tensors, got device={inpt.device}")
        if inpt.dtype != torch.uint8:
            raise ValueError(f"OpenCVColorJitter expects uint8 tensors, got dtype={inpt.dtype}")
        if inpt.ndim < 3 or inpt.shape[-3] != 3:
            raise ValueError(
                "OpenCVColorJitter expects CHW RGB tensors shaped (..., 3, H, W), "
                f"got shape={tuple(inpt.shape)}"
            )
        if inpt.numel() == 0:
            return inpt.clone()

        height, width = inpt.shape[-2:]
        frames = inpt.detach().contiguous().reshape(-1, 3, height, width).permute(0, 2, 3, 1).numpy()
        transformed = np.stack([self.apply_rgb_image(frame, params) for frame in frames])
        return torch.from_numpy(transformed).permute(0, 3, 1, 2).reshape(inpt.shape).contiguous()

    def extra_repr(self) -> str:
        return (
            f"brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue}"
        )
