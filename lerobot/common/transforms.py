from typing import Sequence

import torch
from tensordict import TensorDictBase
from tensordict.nn import dispatch
from tensordict.utils import NestedKey
from torchrl.envs.transforms import ObservationTransform, Transform


class Prod(ObservationTransform):
    invertible = True

    def __init__(self, in_keys: Sequence[NestedKey], prod: float):
        super().__init__()
        self.in_keys = in_keys
        self.prod = prod
        self.original_dtypes = {}

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        # _reset is called once when the environment reset to normalize the first observation
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def _call(self, td):
        for key in self.in_keys:
            if td.get(key, None) is None:
                continue
            self.original_dtypes[key] = td[key].dtype
            td[key] = td[key].type(torch.float32) * self.prod
        return td

    def _inv_call(self, td: TensorDictBase) -> TensorDictBase:
        for key in self.in_keys:
            if td.get(key, None) is None:
                continue
            td[key] = (td[key] / self.prod).type(self.original_dtypes[key])
        return td

    def transform_observation_spec(self, obs_spec):
        for key in self.in_keys:
            if obs_spec.get(key, None) is None:
                continue
            obs_spec[key].space.high = obs_spec[key].space.high.type(torch.float32) * self.prod
            obs_spec[key].space.low = obs_spec[key].space.low.type(torch.float32) * self.prod
            obs_spec[key].dtype = torch.float32
        return obs_spec


class NormalizeTransform(Transform):
    invertible = True

    def __init__(
        self,
        stats: TensorDictBase,
        in_keys: Sequence[NestedKey] = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        mode="mean_std",
    ):
        if out_keys is None:
            out_keys = in_keys
        if in_keys_inv is None:
            in_keys_inv = out_keys
        if out_keys_inv is None:
            out_keys_inv = in_keys
        super().__init__(
            in_keys=in_keys, out_keys=out_keys, in_keys_inv=in_keys_inv, out_keys_inv=out_keys_inv
        )
        self.stats = stats
        assert mode in ["mean_std", "min_max"]
        self.mode = mode

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        # _reset is called once when the environment reset to normalize the first observation
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        for inkey, outkey in zip(self.in_keys, self.out_keys, strict=False):
            # TODO(rcadene): don't know how to do `inkey not in td`
            if td.get(inkey, None) is None:
                continue
            if self.mode == "mean_std":
                mean = self.stats[inkey]["mean"]
                std = self.stats[inkey]["std"]
                td[outkey] = (td[inkey] - mean) / (std + 1e-8)
            else:
                min = self.stats[inkey]["min"]
                max = self.stats[inkey]["max"]
                # normalize to [0,1]
                td[outkey] = (td[inkey] - min) / (max - min)
                # normalize to [-1, 1]
                td[outkey] = td[outkey] * 2 - 1
        return td

    def _inv_call(self, td: TensorDictBase) -> TensorDictBase:
        for inkey, outkey in zip(self.in_keys_inv, self.out_keys_inv, strict=False):
            # TODO(rcadene): don't know how to do `inkey not in td`
            if td.get(inkey, None) is None:
                continue
            if self.mode == "mean_std":
                mean = self.stats[inkey]["mean"]
                std = self.stats[inkey]["std"]
                td[outkey] = td[inkey] * std + mean
            else:
                min = self.stats[inkey]["min"]
                max = self.stats[inkey]["max"]
                td[outkey] = (td[inkey] + 1) / 2
                td[outkey] = td[outkey] * (max - min) + min
        return td
