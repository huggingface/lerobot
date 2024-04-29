import torch
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution while still passing gradients through."""

    default_sample_shape = torch.Size()

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        # TODO(now): Hm looks like this is designed to pass gradients through!
        # TODO(now): Understand what this eps is for.
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=default_sample_shape):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
