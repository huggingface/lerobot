from typing import Any, Callable, Dict, Sequence

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F  # noqa: N812


class RandomSubsetApply(Transform):
    """
    Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (list of floats or None, optional): probability of each transform being picked.
            If ``p`` doesn't sum to 1, it is automatically normalized. If ``None``
            (default), all transforms have the same probability.
        n_subset (int or None): number of transformations to apply. If ``None``,
            all transforms are applied.
        random_order (bool): apply transformations in a random order
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
        elif not (0 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [0, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in selected_transforms:
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


class RangeRandomSharpness(Transform):
    """Similar to v2.RandomAdjustSharpness but with p=1 and a sharpness_factor sampled randomly
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
        sharpness_factor = self.range_min + (self.range_max - self.range_min) * torch.rand(1).item()
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


def make_image_transforms(cfg, to_dtype: torch.dtype = torch.float32):
    transforms_list = [
        v2.ColorJitter(brightness=(cfg.brightness.min, cfg.brightness.max)),
        v2.ColorJitter(contrast=(cfg.contrast.min, cfg.contrast.max)),
        v2.ColorJitter(saturation=(cfg.saturation.min, cfg.saturation.max)),
        v2.ColorJitter(hue=(cfg.hue.min, cfg.hue.max)),
        RangeRandomSharpness(cfg.sharpness.min, cfg.sharpness.max),
    ]
    transforms_weights = [
        cfg.brightness.weight,
        cfg.contrast.weight,
        cfg.saturation.weight,
        cfg.hue.weight,
        cfg.sharpness.weight,
    ]

    transforms = RandomSubsetApply(
        transforms_list, p=transforms_weights, n_subset=cfg.max_num_transforms, random_order=cfg.random_order
    )

    # return transforms
    # return v2.Compose([transforms, v2.ToDtype(to_dtype, scale=True)])
    return v2.Compose([transforms, v2.ToDtype(to_dtype, scale=False)])
