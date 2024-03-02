from typing import Sequence

from tensordict import TensorDictBase
from tensordict.nn import dispatch
from tensordict.utils import NestedKey
from torchrl.envs.transforms import ObservationTransform, Transform


class Prod(ObservationTransform):
    def __init__(self, in_keys: Sequence[NestedKey], prod: float):
        super().__init__()
        self.in_keys = in_keys
        self.prod = prod

    def _call(self, td):
        for key in self.in_keys:
            td[key] *= self.prod
        return td

    def transform_observation_spec(self, obs_spec):
        for key in self.in_keys:
            obs_spec[key].space.high *= self.prod
        return obs_spec


class NormalizeTransform(Transform):
    invertible = True

    def __init__(
        self,
        mean_std: TensorDictBase,
        in_keys: Sequence[NestedKey] = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
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
        self.mean_std = mean_std

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        for inkey, outkey in zip(self.in_keys, self.out_keys, strict=False):
            # TODO(rcadene): don't know how to do `inkey not in td`
            if td.get(inkey, None) is None:
                continue
            mean = self.mean_std[inkey]["mean"]
            std = self.mean_std[inkey]["std"]
            td[outkey] = (td[inkey] - mean) / (std + 1e-8)
        return td

    def _inv_call(self, td: TensorDictBase) -> TensorDictBase:
        for inkey, outkey in zip(self.in_keys_inv, self.out_keys_inv, strict=False):
            # TODO(rcadene): don't know how to do `inkey not in td`
            if td.get(inkey, None) is None:
                continue
            mean = self.mean_std[inkey]["mean"]
            std = self.mean_std[inkey]["std"]
            td[outkey] = td[inkey] * std + mean
        return td
