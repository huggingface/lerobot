from typing import Sequence

from tensordict.utils import NestedKey
from torchrl.envs.transforms import ObservationTransform


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
