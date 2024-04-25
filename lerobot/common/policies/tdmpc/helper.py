import os
import pickle
import re
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

DEFAULT_ACT_FN = nn.Mish()


def __REDUCE__(b):  # noqa: N802, N807
    return "mean" if b else "none"


def l1(pred, target, reduce=False):
    """Computes the L1-loss between predictions and targets."""
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def l2_expectile(diff, expectile=0.7, reduce=False):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    loss = weight * (diff**2)
    reduction = __REDUCE__(reduce)
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    return loss


def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def gaussian_logprob(eps, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * eps.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * eps.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


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


class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0)


class Flatten(nn.Module):
    """Flattens its input to a (batched) vector."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def enc(cfg) -> Callable[[dict[str, Tensor] | Tensor], dict[str, Tensor] | Tensor]:
    """
    Creates encoders for pixel and/or state modalities.
    TODO(now): Consolidate this into just working with a dict even if there is just one modality.
    TODO(now): Use the observation. keys instead of these ones.
    """

    obs_shape = {
        "observation.image": (3, cfg.img_size, cfg.img_size),
        "observation.state": (cfg.state_dim,),
    }

    """Returns a TOLD encoder."""
    pixels_enc_layers, state_enc_layers = None, None
    if cfg.modality in {"pixels", "all"}:
        C = int(3 * cfg.frame_stack)  # noqa: N806
        pixels_enc_layers = [
            # TODO(now): Leave this to the env / data loader
            NormalizeImg(),
            nn.Conv2d(C, cfg.num_channels, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
            nn.ReLU(),
        ]
        out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), pixels_enc_layers)
        pixels_enc_layers.extend(
            [
                Flatten(),
                nn.Linear(np.prod(out_shape), cfg.latent_dim),
                nn.LayerNorm(cfg.latent_dim),
                nn.Sigmoid(),
            ]
        )
        if cfg.modality == "pixels":
            return ConvExt(nn.Sequential(*pixels_enc_layers))
    if cfg.modality in {"state", "all"}:
        state_dim = obs_shape[0] if cfg.modality == "state" else obs_shape["observation.state"][0]
        state_enc_layers = [
            nn.Linear(state_dim, cfg.enc_dim),
            nn.ELU(),
            nn.Linear(cfg.enc_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.Sigmoid(),
        ]
        if cfg.modality == "state":
            return nn.Sequential(*state_enc_layers)
    else:
        raise NotImplementedError

    encoders = {}
    for k in obs_shape:
        if k == "observation.state":
            encoders[k] = nn.Sequential(*state_enc_layers)
        elif k == "observation.image":
            encoders[k] = ConvExt(nn.Sequential(*pixels_enc_layers))
        else:
            raise NotImplementedError
    return Multiplexer(nn.ModuleDict(encoders))


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


def aug(cfg):
    obs_shape = {
        "observation.image": (3, cfg.img_size, cfg.img_size),
        "observation.state": (4,),
    }

    """Multiplex augmentation"""
    if cfg.modality == "state":
        return nn.Identity()
    elif cfg.modality == "pixels":
        return RandomShiftsAug(cfg)
    else:
        augs = {}
        for k in obs_shape:
            if k == "observation.state":
                augs[k] = nn.Identity()
            elif k == "observation.image":
                augs[k] = RandomShiftsAug(cfg)
            else:
                raise NotImplementedError
        return Multiplexer(nn.ModuleDict(augs))


class ConvExt(nn.Module):
    """Helper to deal with arbitrary dimensions (B, *, C, H, W) for the input images."""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        if x.ndim > 4:
            # x has some has shape (B, * , C, H, W) so we first flatten (B, *) into the first dim, run the
            # layers, then unflatten to return the result.
            batch_shape = x.shape[:-3]
            out = self.conv(x.view(-1, *x.shape[-3:]))
            out = out.view(*batch_shape, *out.shape[1:])
        else:
            # x has shape (B, C, H, W).
            out = self.conv(x)
        return out


class Multiplexer(nn.Module):
    """Model multiplexer"""

    def __init__(self, choices: nn.ModuleDict):
        super().__init__()
        self.choices = choices

    def forward(self, x, key=None):
        if isinstance(x, dict):
            if key is not None:
                return self.choices[key](x)
            return {k: self.choices[k](_x) for k, _x in x.items()}
        return self.choices(x)


class RandomShiftsAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.modality in {"pixels", "all"}
        self.pad = int(cfg.img_size / 21)

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=torch.float32,
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=torch.float32,
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def get_dataset_dict(cfg, env, return_reward_normalizer=False):
    """Construct a dataset for env"""
    required_keys = [
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "dones",
        "masks",
    ]

    if cfg.task.startswith("xarm"):
        dataset_path = os.path.join(cfg.dataset_dir, "buffer.pkl")
        print(f"Using offline dataset '{dataset_path}'")
        with open(dataset_path, "rb") as f:
            dataset_dict = pickle.load(f)
        for k in required_keys:
            if k not in dataset_dict and k[:-1] in dataset_dict:
                dataset_dict[k] = dataset_dict.pop(k[:-1])
    elif cfg.task.startswith("legged"):
        dataset_path = os.path.join(cfg.dataset_dir, "buffer.pkl")
        print(f"Using offline dataset '{dataset_path}'")
        with open(dataset_path, "rb") as f:
            dataset_dict = pickle.load(f)
        dataset_dict["actions"] /= env.unwrapped.clip_actions
        print(f"clip_actions={env.unwrapped.clip_actions}")
    else:
        import d4rl

        dataset_dict = d4rl.qlearning_dataset(env)
        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(dataset_dict["observations"][i + 1] - dataset_dict["next_observations"][i])
                > 1e-6
                or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        dones[-1] = True

        dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        del dataset_dict["terminals"]

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones

    if cfg.is_data_clip:
        lim = 1 - cfg.data_clip_eps
        dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)
    reward_normalizer = get_reward_normalizer(cfg, dataset_dict)
    dataset_dict["rewards"] = reward_normalizer(dataset_dict["rewards"])

    for key in required_keys:
        assert key in dataset_dict, f"Missing `{key}` in dataset."

    if return_reward_normalizer:
        return dataset_dict, reward_normalizer
    return dataset_dict


def get_trajectory_boundaries_and_returns(dataset):
    """
    Split dataset into trajectories and compute returns
    """
    episode_starts = [0]
    episode_ends = []

    episode_return = 0
    episode_returns = []

    n_transitions = len(dataset["rewards"])

    for i in range(n_transitions):
        episode_return += dataset["rewards"][i]

        if dataset["dones"][i]:
            episode_returns.append(episode_return)
            episode_ends.append(i + 1)
            if i + 1 < n_transitions:
                episode_starts.append(i + 1)
            episode_return = 0.0

    return episode_starts, episode_ends, episode_returns


def normalize_returns(dataset, scaling=1000):
    """
    Normalize returns in the dataset
    """
    (_, _, episode_returns) = get_trajectory_boundaries_and_returns(dataset)
    dataset["rewards"] /= np.max(episode_returns) - np.min(episode_returns)
    dataset["rewards"] *= scaling
    return dataset


def get_reward_normalizer(cfg, dataset):
    """
    Get a reward normalizer for the dataset
    TODO(now): Leave this to the dataloader/env
    """
    if cfg.task.startswith("xarm"):
        return lambda x: x
    elif "maze" in cfg.task:
        return lambda x: x - 1.0
    elif cfg.task.split("-")[0] in ["hopper", "halfcheetah", "walker2d"]:
        (_, _, episode_returns) = get_trajectory_boundaries_and_returns(dataset)
        return lambda x: x / (np.max(episode_returns) - np.min(episode_returns)) * 1000.0
    elif hasattr(cfg, "reward_scale"):
        return lambda x: x * cfg.reward_scale
    return lambda x: x


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
