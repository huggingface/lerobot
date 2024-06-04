from typing import Any, Callable, Sequence

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform


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


def make_transforms(cfg):
    image_transforms = []
    if 'jit' in cfg.image_transform.list:
        image_transforms.append(v2.ColorJitter(brightness=cfg.colorjitter_range, contrast=cfg.colorjitter_range))
    if 'sharpness' in cfg.image_transform.list:
        image_transforms.append(v2.RandomAdjustSharpness(cfg.sharpness_factor, p=cfg.sharpness_p))
    if 'blur' in cfg.image_transform.list:
        image_transforms.append(v2.RandomAdjustSharpness(cfg.blur_factor, p=cfg.blur_p))

    return v2.Compose(RandomSubsetApply(image_transforms, n_subset=cfg.n_subset), v2.ToDtype(torch.float32, scale=True))
