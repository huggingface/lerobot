import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness="different", **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs)

    def __repr__(self):
        return "Vectorized " + self._repr


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def soft_cross_entropy(pred, target, cfg):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)


@torch.jit.script
def log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


@torch.jit.script
def _gaussian_residual(eps, log_std):
    return -0.5 * eps.pow(2) - log_std


@torch.jit.script
def _gaussian_logprob(residual):
    return residual - 0.5 * torch.log(2 * torch.pi)


def gaussian_logprob(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
    if size is None:
        size = eps.size(-1)
    return _gaussian_logprob(residual) * size


@torch.jit.script
def _squash(pi):
    return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi -= _squash(pi).sum(-1, keepdim=True)
    return mu, pi, log_pi


@torch.jit.script
def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""

    # x shape [horizon, num_features]
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()  # shape [num_features]
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx.float()).unsqueeze(-1)  # shape [num_features , 1]
    soft_two_hot = torch.zeros(
        *x.shape, cfg.num_bins, device=x.device
    )  # shape [horizon, num_features, num_bins]
    soft_two_hot.scatter_(2, bin_idx.unsqueeze(-1), 1 - bin_offset)
    soft_two_hot.scatter_(2, (bin_idx.unsqueeze(-1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x, bins):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    num_bins = bins.shape[0]
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symexp(x)

    x = F.softmax(x, dim=-1)
    x = torch.sum(x * bins, dim=-1, keepdim=True)
    return symexp(x)
