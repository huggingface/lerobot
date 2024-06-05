from typing import Any, Callable, Dict, Sequence

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F  # noqa: N812


class RandomSubsetApply(Transform):
    """
    Apply a random subset of N transformations from a list of transformations in a random order.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        N (int): number of transformations to apply
    """

    def __init__(self, transforms: Sequence[Callable], n_subset: int) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if not (0 <= n_subset <= len(transforms)):
            raise ValueError(f"N should be in the interval [0, {len(transforms)}]")

        self.transforms = transforms
        self.N = n_subset

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        # Randomly pick N transforms
        selected_transforms = torch.randperm(len(self.transforms))[: self.N]

        # Apply selected transforms in random order
        for idx in selected_transforms:
            transform = self.transforms[idx]
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        format_string = [f"N={self.N}"]
        for t in self.transforms:
            format_string.append(f"    {t}")
        return "\n".join(format_string)


class RangeRandomSharpness(Transform):
    """Similar to RandomAdjustSharpness but with p=1 and a sharpness_factor sampled randomly
    each time in [range_min, range_max].

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    """

    def __init__(self, range_min: float, range_max) -> None:
        super().__init__()
        self.range_min, self.range_max = self._check_input(range_min, range_max)

    def _check_input(self, range_min, range_max):
        if range_min < 0:
            raise ValueError("range_min must be non negative.")
        if range_min > range_max:
            raise ValueError("range_max must greater or equal to range_min")
        return range_min, range_max

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        sharpness_factor = self.range_min + (self.range_max - self.range_min) * torch.rand(1)
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


def make_transforms(cfg):
    image_transforms = [
        v2.ColorJitter(brightness=(cfg.brightness.min, cfg.brightness.max)),
        v2.ColorJitter(contrast=(cfg.contrast.min, cfg.contrast.max)),
        v2.ColorJitter(saturation=(cfg.saturation.min, cfg.saturation.max)),
        v2.ColorJitter(hue=(cfg.hue.min, cfg.hue.max)),
        RangeRandomSharpness(cfg.sharpness.min, cfg.sharpness.max),
    ]
    # WIP
    return v2.Compose(
        [RandomSubsetApply(image_transforms, n_subset=cfg.n_subset), v2.ToDtype(torch.float32, scale=True)]
    )
