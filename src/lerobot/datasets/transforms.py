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
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Resize, Transform
from torchvision.transforms.v2 import functional as F  # noqa: N812


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
        if isinstance(sharpness, (int, float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpnesss values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


class ResizeWithPad(Transform):
    """Resize an image while maintaining aspect ratio, then pad to target size.

    This transform resizes an image to fit within (width, height) while keeping the aspect ratio
    and pads with a specified value.

    Args:
        size (int | tuple[int, int]): Target size (height, width).
        pad_value (float, optional): Value to pad the image with (default: 0).
    """

    def __init__(
        self, size: int | tuple[int, int], pad_value: float = 0, padding_side: str = "top_left"
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, collections.abc.Sequence) and len(size) == 2:
            self.size = tuple(size)
        else:
            raise ValueError(f"`size` should be an int or a tuple (h, w), but got {size}")

        self.pad_value = pad_value
        self.padding_side = padding_side

    def transform(self, inpt: torch.Tensor, params: dict[str, any] = None) -> torch.Tensor:
        """Resize and pad a single image.

        Expects input of shape (C, H, W).
        """
        original_ndim = inpt.ndim
        if inpt.ndim == 4 and inpt.shape[0] == 1:
            inpt = inpt.squeeze(0)

        if not isinstance(inpt, torch.Tensor) or inpt.ndim != 3:
            raise ValueError(f"Expected input of shape (C, H, W), but got {inpt.shape}")

        channels, cur_height, cur_width = inpt.shape
        target_height, target_width = self.size

        # Maintain aspect ratio
        ratio = max(cur_width / target_width, cur_height / target_height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)

        resized_img = torch.nn.functional.interpolate(
            inpt.unsqueeze(0), size=(resized_height, resized_width), mode="bilinear", align_corners=False
        ).squeeze(0)

        # Compute padding (pad left & top)
        pad_height = max(0, target_height - resized_height)
        pad_width = max(0, target_width - resized_width)
        if "left" in self.padding_side:
            padded_img = torch.nn.functional.pad(
                resized_img, (pad_width, 0, pad_height, 0), value=self.pad_value
            )
        else:
            padded_img = torch.nn.functional.pad(
                resized_img, (0, pad_width, 0, pad_height), value=self.pad_value
            )
        if original_ndim == 4:
            padded_img = padded_img[None]
        return padded_img


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


# FIXME(mshukor): this should be passsed as input to train.py
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
    transform_version: int = 0
    image_size: int = 256
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "resize_with_pad": ImageTransformConfig(
                weight=1.0,
                type="ResizeWithPad",
                kwargs={"size": (256, 256)},  # FIXME(mshukor): this should be passed as input to train.py
            ),
            # "resize": ImageTransformConfig(
            #     weight=1.0,
            #     type="Resize",
            #     kwargs={"size": (256, 256)},
            # ),
            # "brightness": ImageTransformConfig(
            #     weight=1.0,
            #     type="ColorJitter",
            #     kwargs={"brightness": (0.8, 1.2)},
            # ),
            # "contrast": ImageTransformConfig(
            #     weight=1.0,
            #     type="ColorJitter",
            #     kwargs={"contrast": (0.8, 1.2)},
            # ),
            # "saturation": ImageTransformConfig(
            #     weight=1.0,
            #     type="ColorJitter",
            #     kwargs={"saturation": (0.5, 1.5)},
            # ),
            # "hue": ImageTransformConfig(
            #     weight=1.0,
            #     type="ColorJitter",
            #     kwargs={"hue": (-0.05, 0.05)},
            # ),
            # "sharpness": ImageTransformConfig(
            #     weight=1.0,
            #     type="SharpnessJitter",
            #     kwargs={"sharpness": (0.5, 1.5)},
            # ),
        }
    )
    def __post_init__(self):
        self.tfs = self._get_transforms_by_version(self.transform_version)

    def _get_transforms_by_version(self, version: int) -> dict[str, ImageTransformConfig]:
        if version == 0:
            return {
                "resize_with_pad": ImageTransformConfig(
                    weight=1.0,
                    type="ResizeWithPad",
                    kwargs={"size": (256, 256)},
                )
            }
        elif version == 1:
            return {
                "resize_with_pad": ImageTransformConfig(
                    weight=1.0,
                    type="ResizeWithPad",
                    kwargs={"size": (self.image_size, self.image_size)},
                )
            }
        elif version == 2:
            return {
                "random_resize_crop": ImageTransformConfig(
                    weight=1.0,
                    type="RandomResizedCrop",
                    kwargs={"size": (int(self.image_size * 0.95), int(self.image_size * 0.95))},
                ),
                "resize": ImageTransformConfig(
                    weight=1.0,
                    type="Resize",
                    kwargs={"size": (self.image_size, self.image_size)},
                ),
                "rotation": ImageTransformConfig(
                    weight=1.0,
                    type="RandomRotation",
                    kwargs={"degrees": 5},
                ),
                "color_jitter": ImageTransformConfig(
                    weight=1.0,
                    type="ColorJitter",
                    kwargs={"brightness": 0.3, "contrast": 0.4, "saturation": 0.5},
                ),
            }
        elif version == 3:
            return {
                "resize_with_pad": ImageTransformConfig(
                    weight=1.0,
                    type="ResizeWithPad",
                    kwargs={"size": (self.image_size, self.image_size)},
                ),
                "random_resize_crop": ImageTransformConfig(
                    weight=1.0,
                    type="RandomResizedCrop",
                    kwargs={"size": (int(self.image_size * 0.95), int(self.image_size * 0.95))},
                ),
                "resize": ImageTransformConfig(
                    weight=1.0,
                    type="Resize",
                    kwargs={"size": (self.image_size, self.image_size)},
                ),
                "rotation": ImageTransformConfig(
                    weight=1.0,
                    type="RandomRotation",
                    kwargs={"degrees": 5},
                ),
                "color_jitter": ImageTransformConfig(
                    weight=1.0,
                    type="ColorJitter",
                    kwargs={"brightness": 0.3, "contrast": 0.4, "saturation": 0.5},
                ),
            }
        elif version == 4:
            return {
                "rotation": ImageTransformConfig(
                    weight=1.0,
                    type="RandomRotation",
                    kwargs={"degrees": 5},
                ),
                "color_jitter": ImageTransformConfig(
                    weight=1.0,
                    type="ColorJitter",
                    kwargs={"brightness": 0.3, "contrast": 0.4, "saturation": 0.5},
                ),
                "resize_with_pad": ImageTransformConfig(
                    weight=1.0,
                    type="ResizeWithPad",
                    kwargs={"size": (self.image_size, self.image_size)},
                ),
            }
        elif version == 5:
            return {
                "translate": ImageTransformConfig(
                    weight=1.0,
                    type="RandomAffine",
                    kwargs={"degrees": 0, "translate": (0.9, 0.9)},
                ),
                "rotation": ImageTransformConfig(
                    weight=1.0,
                    type="RandomRotation",
                    kwargs={"degrees": 5},
                ),
                "color_jitter": ImageTransformConfig(
                    weight=1.0,
                    type="ColorJitter",
                    kwargs={"brightness": 0.3, "contrast": 0.4, "saturation": 0.5},
                ),
                "resize_with_pad": ImageTransformConfig(
                    weight=1.0,
                    type="ResizeWithPad",
                    kwargs={"size": (self.image_size, self.image_size)},
                ),
            }
        elif version == 6:
            return {
                "translate": ImageTransformConfig(
                    weight=1.0,
                    type="RandomAffine",
                    kwargs={"degrees": 0, "translate": (0.9, 0.9)},
                ),
                "rotation": ImageTransformConfig(
                    weight=1.0,
                    type="RandomRotation",
                    kwargs={"degrees": 5},
                ),
                "resize_with_pad": ImageTransformConfig(
                    weight=1.0,
                    type="ResizeWithPad",
                    kwargs={"size": (self.image_size, self.image_size)},
                ),
            }
        elif version == 7:
            return {
                "flip": ImageTransformConfig(
                    weight=1.0,
                    type="RandomHorizontalFlip",
                    kwargs={"p": 0.3},
                ),
                "translate": ImageTransformConfig(
                    weight=1.0,
                    type="RandomAffine",
                    kwargs={"degrees": 0, "translate": (0.9, 0.9)},
                ),
                "rotation": ImageTransformConfig(
                    weight=1.0,
                    type="RandomRotation",
                    kwargs={"degrees": 5},
                ),
                "resize_with_pad": ImageTransformConfig(
                    weight=1.0,
                    type="ResizeWithPad",
                    kwargs={"size": (self.image_size, self.image_size)},
                ),
            }
        elif version == 8:
            return {
                "flip": ImageTransformConfig(
                    weight=1.0,
                    type="RandomHorizontalFlip",
                    kwargs={"p": 0.3},
                ),
                "color_jitter": ImageTransformConfig(
                    weight=1.0,
                    type="ColorJitter",
                    kwargs={"brightness": 0.3, "contrast": 0.4, "saturation": 0.5},
                ),
                "resize_with_pad": ImageTransformConfig(
                    weight=1.0,
                    type="ResizeWithPad",
                    kwargs={"size": (self.image_size, self.image_size)},
                ),
            }
        elif version == 9:
            return {
                "color_jitter": ImageTransformConfig(
                    weight=1.0,
                    type="ColorJitter",
                    kwargs={"brightness": 0.3, "contrast": 0.4, "saturation": 0.5},
                ),
                "resize_with_pad": ImageTransformConfig(
                    weight=1.0,
                    type="ResizeWithPad",
                    kwargs={"size": (self.image_size, self.image_size)},
                ),
            }
        else:
            raise ValueError(f"Unknown transform_version: {version}")



def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    elif cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    elif cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    elif cfg.type == "Resize":
        return Resize(**cfg.kwargs)
    elif cfg.type == "ResizeWithPad":
        return ResizeWithPad(**cfg.kwargs)
    elif cfg.type == "RandomRotation":
        return v2.RandomRotation(**cfg.kwargs)
    elif cfg.type == "RandomResizedCrop":
        return v2.RandomResizedCrop(**cfg.kwargs)
    elif cfg.type == "RandomAffine":
        return v2.RandomAffine(**cfg.kwargs)
    elif cfg.type == "RandomHorizontalFlip":
        return v2.RandomHorizontalFlip(**cfg.kwargs)
    else:
        raise ValueError(f"Transform '{cfg.type}' is not valid.")


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
