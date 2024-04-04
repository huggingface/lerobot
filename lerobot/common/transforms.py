import torch
from torchvision.transforms.v2 import Compose, Transform


def apply_inverse_transform(item, transform):
    transforms = transform.transforms if isinstance(transform, Compose) else [transform]
    for tf in transforms[::-1]:
        if tf.invertible:
            item = tf.inverse_transform(item)
        else:
            raise ValueError(f"Inverse transform called on a non invertible transform ({tf}).")
    return item


class Prod(Transform):
    invertible = True

    def __init__(self, in_keys: list[str], prod: float):
        super().__init__()
        self.in_keys = in_keys
        self.prod = prod
        self.original_dtypes = {}

    def forward(self, item):
        for key in self.in_keys:
            if key not in item:
                continue
            self.original_dtypes[key] = item[key].dtype
            item[key] = item[key].type(torch.float32) * self.prod
        return item

    def inverse_transform(self, item):
        for key in self.in_keys:
            if key not in item:
                continue
            item[key] = (item[key] / self.prod).type(self.original_dtypes[key])
        return item

    # def transform_observation_spec(self, obs_spec):
    #     for key in self.in_keys:
    #         if obs_spec.get(key, None) is None:
    #             continue
    #         obs_spec[key].space.high = obs_spec[key].space.high.type(torch.float32) * self.prod
    #         obs_spec[key].space.low = obs_spec[key].space.low.type(torch.float32) * self.prod
    #         obs_spec[key].dtype = torch.float32
    #     return obs_spec


class NormalizeTransform(Transform):
    invertible = True

    def __init__(
        self,
        stats: dict,
        in_keys: list[str] = None,
        out_keys: list[str] | None = None,
        in_keys_inv: list[str] | None = None,
        out_keys_inv: list[str] | None = None,
        mode="mean_std",
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = in_keys if out_keys is None else out_keys
        self.in_keys_inv = self.out_keys if in_keys_inv is None else in_keys_inv
        self.out_keys_inv = self.in_keys if out_keys_inv is None else out_keys_inv
        self.stats = stats
        assert mode in ["mean_std", "min_max"]
        self.mode = mode

    def forward(self, item):
        for inkey, outkey in zip(self.in_keys, self.out_keys, strict=False):
            if inkey not in item:
                continue
            if self.mode == "mean_std":
                mean = self.stats[f"{inkey}.mean"]
                std = self.stats[f"{inkey}.std"]
                item[outkey] = (item[inkey] - mean) / (std + 1e-8)
            else:
                min = self.stats[f"{inkey}.min"]
                max = self.stats[f"{inkey}.max"]
                # normalize to [0,1]
                item[outkey] = (item[inkey] - min) / (max - min)
                # normalize to [-1, 1]
                item[outkey] = item[outkey] * 2 - 1
        return item

    def inverse_transform(self, item):
        for inkey, outkey in zip(self.in_keys_inv, self.out_keys_inv, strict=False):
            if inkey not in item:
                continue
            if self.mode == "mean_std":
                mean = self.stats[f"{inkey}.mean"]
                std = self.stats[f"{inkey}.std"]
                item[outkey] = item[inkey] * std + mean
            else:
                min = self.stats[f"{inkey}.min"]
                max = self.stats[f"{inkey}.max"]
                item[outkey] = (item[inkey] + 1) / 2
                item[outkey] = item[outkey] * (max - min) + min
        return item
