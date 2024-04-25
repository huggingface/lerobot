import re

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

DEFAULT_ACT_FN = nn.Mish()


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        # TODO(rcadene, aliberts): issue with strict=False
        # for p, p_target in zip(m.parameters(), m_target.parameters(), strict=False):
        #     p_target.data.lerp_(p.data, tau)
        m_params_iter = iter(m.parameters())
        m_target_params_iter = iter(m_target.parameters())

        while True:
            try:
                p = next(m_params_iter)
                p_target = next(m_target_params_iter)
                p_target.data.lerp_(p.data, tau)
            except StopIteration:
                # If any iterator is exhausted, exit the loop
                break


def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution while still passing gradients through.
    TODO(now): consider simplifying the hell out of this but only once you understand what self.eps is for.
    """

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


def mlp(in_dim: int, mlp_dim: int | tuple[int, int], out_dim: int, act_fn=DEFAULT_ACT_FN) -> nn.Sequential:
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]),
        nn.LayerNorm(mlp_dim[0]),
        act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]),
        nn.LayerNorm(mlp_dim[1]),
        act_fn,
        nn.Linear(mlp_dim[1], out_dim),
    )


def dynamics(in_dim, mlp_dim, out_dim, act_fn=DEFAULT_ACT_FN) -> nn.Sequential:
    """Returns a dynamics network.

    TODO(now): this needs a better name. It's also an MLP...
    """
    return nn.Sequential(
        mlp(in_dim, mlp_dim, out_dim, act_fn),
        nn.LayerNorm(out_dim),
        nn.Sigmoid(),
    )


def q(cfg) -> nn.Sequential:
    """Returns a Q-function that uses Layer Normalization."""
    action_dim = cfg.action_dim
    return nn.Sequential(
        nn.Linear(cfg.latent_dim + action_dim, cfg.mlp_dim),
        nn.LayerNorm(cfg.mlp_dim),
        nn.Tanh(),
        nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
        nn.ELU(),
        nn.Linear(cfg.mlp_dim, 1),
    )


def v(cfg):
    """Returns a state value function that uses Layer Normalization."""
    return nn.Sequential(
        nn.Linear(cfg.latent_dim, cfg.mlp_dim),
        nn.LayerNorm(cfg.mlp_dim),
        nn.Tanh(),
        nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
        nn.ELU(),
        nn.Linear(cfg.mlp_dim, 1),
    )


def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final, start, end = (float(g) for g in match.groups())
            mix = np.clip((step - start) / (end - start), 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            # TODO(now): Looks like the original tdmpc code uses this with
            # `horizon_schedule: linear(1, ${horizon}, 25000)`
            init, final, duration = (float(g) for g in match.groups())
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)
