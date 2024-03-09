import os
import pickle
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
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
    """Utility class implementing the truncated normal distribution."""

    default_sample_shape = torch.Size()

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
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


def enc(cfg):
    obs_shape = {
        "rgb": (3, cfg.img_size, cfg.img_size),
        "state": (cfg.state_dim,),
    }

    """Returns a TOLD encoder."""
    pixels_enc_layers, state_enc_layers = None, None
    if cfg.modality in {"pixels", "all"}:
        C = int(3 * cfg.frame_stack)  # noqa: N806
        pixels_enc_layers = [
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
        state_dim = obs_shape[0] if cfg.modality == "state" else obs_shape["state"][0]
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
        if k == "state":
            encoders[k] = nn.Sequential(*state_enc_layers)
        elif k.endswith("rgb"):
            encoders[k] = ConvExt(nn.Sequential(*pixels_enc_layers))
        else:
            raise NotImplementedError
    return Multiplexer(nn.ModuleDict(encoders))


def mlp(in_dim, mlp_dim, out_dim, act_fn=DEFAULT_ACT_FN):
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


def dynamics(in_dim, mlp_dim, out_dim, act_fn=DEFAULT_ACT_FN):
    """Returns a dynamics network."""
    return nn.Sequential(
        mlp(in_dim, mlp_dim, out_dim, act_fn),
        nn.LayerNorm(out_dim),
        nn.Sigmoid(),
    )


def q(cfg):
    action_dim = cfg.action_dim
    """Returns a Q-function that uses Layer Normalization."""
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
        "rgb": (3, cfg.img_size, cfg.img_size),
        "state": (4,),
    }

    """Multiplex augmentation"""
    if cfg.modality == "state":
        return nn.Identity()
    elif cfg.modality == "pixels":
        return RandomShiftsAug(cfg)
    else:
        augs = {}
        for k in obs_shape:
            if k == "state":
                augs[k] = nn.Identity()
            elif k.endswith("rgb"):
                augs[k] = RandomShiftsAug(cfg)
            else:
                raise NotImplementedError
        return Multiplexer(nn.ModuleDict(augs))


class ConvExt(nn.Module):
    """Auxiliary conv net accommodating high-dim input"""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        if x.ndim > 4:
            batch_shape = x.shape[:-3]
            out = self.conv(x.view(-1, *x.shape[-3:]))
            out = out.view(*batch_shape, *out.shape[1:])
        else:
            out = self.conv(x)
        return out


class Multiplexer(nn.Module):
    """Model multiplexer"""

    def __init__(self, choices):
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


# TODO(aliberts): remove class
# class Episode:
#     """Storage object for a single episode."""

#     def __init__(self, cfg, init_obs):
#         action_dim = cfg.action_dim

#         self.cfg = cfg
#         self.device = torch.device(cfg.buffer_device)
#         if cfg.modality in {"pixels", "state"}:
#             dtype = torch.float32 if cfg.modality == "state" else torch.uint8
#             self.obses = torch.empty(
#                 (cfg.episode_length + 1, *init_obs.shape),
#                 dtype=dtype,
#                 device=self.device,
#             )
#             self.obses[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
#         elif cfg.modality == "all":
#             self.obses = {}
#             for k, v in init_obs.items():
#                 assert k in {"rgb", "state"}
#                 dtype = torch.float32 if k == "state" else torch.uint8
#                 self.obses[k] = torch.empty(
#                     (cfg.episode_length + 1, *v.shape), dtype=dtype, device=self.device
#                 )
#                 self.obses[k][0] = torch.tensor(v, dtype=dtype, device=self.device)
#         else:
#             raise ValueError
#         self.actions = torch.empty((cfg.episode_length, action_dim), dtype=torch.float32, device=self.device)
#         self.rewards = torch.empty((cfg.episode_length,), dtype=torch.float32, device=self.device)
#         self.dones = torch.empty((cfg.episode_length,), dtype=torch.bool, device=self.device)
#         self.masks = torch.empty((cfg.episode_length,), dtype=torch.float32, device=self.device)
#         self.cumulative_reward = 0
#         self.done = False
#         self.success = False
#         self._idx = 0

#     def __len__(self):
#         return self._idx

#     @classmethod
#     def from_trajectory(cls, cfg, obses, actions, rewards, dones=None, masks=None):
#         """Constructs an episode from a trajectory."""

#         if cfg.modality in {"pixels", "state"}:
#             episode = cls(cfg, obses[0])
#             episode.obses[1:] = torch.tensor(obses[1:], dtype=episode.obses.dtype, device=episode.device)
#         elif cfg.modality == "all":
#             episode = cls(cfg, {k: v[0] for k, v in obses.items()})
#             for k in obses:
#                 episode.obses[k][1:] = torch.tensor(
#                     obses[k][1:], dtype=episode.obses[k].dtype, device=episode.device
#                 )
#         else:
#             raise NotImplementedError
#         episode.actions = torch.tensor(actions, dtype=episode.actions.dtype, device=episode.device)
#         episode.rewards = torch.tensor(rewards, dtype=episode.rewards.dtype, device=episode.device)
#         episode.dones = (
#             torch.tensor(dones, dtype=episode.dones.dtype, device=episode.device)
#             if dones is not None
#             else torch.zeros_like(episode.dones)
#         )
#         episode.masks = (
#             torch.tensor(masks, dtype=episode.masks.dtype, device=episode.device)
#             if masks is not None
#             else torch.ones_like(episode.masks)
#         )
#         episode.cumulative_reward = torch.sum(episode.rewards)
#         episode.done = True
#         episode._idx = cfg.episode_length
#         return episode

#     @property
#     def first(self):
#         return len(self) == 0

#     def __add__(self, transition):
#         self.add(*transition)
#         return self

#     def add(self, obs, action, reward, done, mask=1.0, success=False):
#         """Add a transition into the episode."""
#         if isinstance(obs, dict):
#             for k, v in obs.items():
#                 self.obses[k][self._idx + 1] = torch.tensor(
#                     v, dtype=self.obses[k].dtype, device=self.obses[k].device
#                 )
#         else:
#             self.obses[self._idx + 1] = torch.tensor(obs, dtype=self.obses.dtype, device=self.obses.device)
#         self.actions[self._idx] = action
#         self.rewards[self._idx] = reward
#         self.dones[self._idx] = done
#         self.masks[self._idx] = mask
#         self.cumulative_reward += reward
#         self.done = done
#         self.success = self.success or success
#         self._idx += 1


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
            init, final, duration = (float(g) for g in match.groups())
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)
