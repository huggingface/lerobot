import os
import pickle
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

__REDUCE__ = lambda b: "mean" if b else "none"


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
    return (
        (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x)
        .squeeze(0)
        .shape
    )


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
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
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
        "state": (4,),
    }

    """Returns a TOLD encoder."""
    pixels_enc_layers, state_enc_layers = None, None
    if cfg.modality in {"pixels", "all"}:
        C = int(3 * cfg.frame_stack)
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


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.Mish()):
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


def dynamics(in_dim, mlp_dim, out_dim, act_fn=nn.Mish()):
    """Returns a dynamics network."""
    return nn.Sequential(
        mlp(in_dim, mlp_dim, out_dim, act_fn),
        nn.LayerNorm(out_dim),
        nn.Sigmoid(),
    )


def q(cfg):
    action_dim = 4
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


class Episode(object):
    """Storage object for a single episode."""

    def __init__(self, cfg, init_obs):
        action_dim = 4

        self.cfg = cfg
        self.device = torch.device(cfg.buffer_device)
        if cfg.modality in {"pixels", "state"}:
            dtype = torch.float32 if cfg.modality == "state" else torch.uint8
            self.obses = torch.empty(
                (cfg.episode_length + 1, *init_obs.shape),
                dtype=dtype,
                device=self.device,
            )
            self.obses[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
        elif cfg.modality == "all":
            self.obses = {}
            for k, v in init_obs.items():
                assert k in {"rgb", "state"}
                dtype = torch.float32 if k == "state" else torch.uint8
                self.obses[k] = torch.empty(
                    (cfg.episode_length + 1, *v.shape), dtype=dtype, device=self.device
                )
                self.obses[k][0] = torch.tensor(v, dtype=dtype, device=self.device)
        else:
            raise ValueError
        self.actions = torch.empty(
            (cfg.episode_length, action_dim), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.empty(
            (cfg.episode_length,), dtype=torch.float32, device=self.device
        )
        self.dones = torch.empty(
            (cfg.episode_length,), dtype=torch.bool, device=self.device
        )
        self.masks = torch.empty(
            (cfg.episode_length,), dtype=torch.float32, device=self.device
        )
        self.cumulative_reward = 0
        self.done = False
        self.success = False
        self._idx = 0

    def __len__(self):
        return self._idx

    @classmethod
    def from_trajectory(cls, cfg, obses, actions, rewards, dones=None, masks=None):
        """Constructs an episode from a trajectory."""

        if cfg.modality in {"pixels", "state"}:
            episode = cls(cfg, obses[0])
            episode.obses[1:] = torch.tensor(
                obses[1:], dtype=episode.obses.dtype, device=episode.device
            )
        elif cfg.modality == "all":
            episode = cls(cfg, {k: v[0] for k, v in obses.items()})
            for k, v in obses.items():
                episode.obses[k][1:] = torch.tensor(
                    obses[k][1:], dtype=episode.obses[k].dtype, device=episode.device
                )
        else:
            raise NotImplementedError
        episode.actions = torch.tensor(
            actions, dtype=episode.actions.dtype, device=episode.device
        )
        episode.rewards = torch.tensor(
            rewards, dtype=episode.rewards.dtype, device=episode.device
        )
        episode.dones = (
            torch.tensor(dones, dtype=episode.dones.dtype, device=episode.device)
            if dones is not None
            else torch.zeros_like(episode.dones)
        )
        episode.masks = (
            torch.tensor(masks, dtype=episode.masks.dtype, device=episode.device)
            if masks is not None
            else torch.ones_like(episode.masks)
        )
        episode.cumulative_reward = torch.sum(episode.rewards)
        episode.done = True
        episode._idx = cfg.episode_length
        return episode

    @property
    def first(self):
        return len(self) == 0

    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, action, reward, done, mask=1.0, success=False):
        """Add a transition into the episode."""
        if isinstance(obs, dict):
            for k, v in obs.items():
                self.obses[k][self._idx + 1] = torch.tensor(
                    v, dtype=self.obses[k].dtype, device=self.obses[k].device
                )
        else:
            self.obses[self._idx + 1] = torch.tensor(
                obs, dtype=self.obses.dtype, device=self.obses.device
            )
        self.actions[self._idx] = action
        self.rewards[self._idx] = reward
        self.dones[self._idx] = done
        self.masks[self._idx] = mask
        self.cumulative_reward += reward
        self.done = done
        self.success = self.success or success
        self._idx += 1


class ReplayBuffer:
    """
    Storage and sampling functionality.
    """

    def __init__(self, cfg, dataset=None):
        action_dim = 4
        obs_shape = {"rgb": (3, cfg.img_size, cfg.img_size), "state": (4,)}

        self.cfg = cfg
        self.device = torch.device(cfg.buffer_device)
        print("Replay buffer device: ", self.device)

        if dataset is not None:
            self.capacity = max(dataset["rewards"].shape[0], cfg.max_buffer_size)
        else:
            self.capacity = min(cfg.train_steps, cfg.max_buffer_size)

        if cfg.modality in {"pixels", "state"}:
            dtype = torch.float32 if cfg.modality == "state" else torch.uint8
            # Note self.obs_shape always has single frame, which is different from cfg.obs_shape
            self.obs_shape = (
                obs_shape if cfg.modality == "state" else (3, *obs_shape[-2:])
            )
            self._obs = torch.zeros(
                (self.capacity + cfg.horizon - 1, *self.obs_shape),
                dtype=dtype,
                device=self.device,
            )
            self._next_obs = torch.zeros(
                (self.capacity + cfg.horizon - 1, *self.obs_shape),
                dtype=dtype,
                device=self.device,
            )
        elif cfg.modality == "all":
            self.obs_shape = {}
            self._obs, self._next_obs = {}, {}
            for k, v in obs_shape.items():
                assert k in {"rgb", "state"}
                dtype = torch.float32 if k == "state" else torch.uint8
                self.obs_shape[k] = v if k == "state" else (3, *v[-2:])
                self._obs[k] = torch.zeros(
                    (self.capacity + cfg.horizon - 1, *self.obs_shape[k]),
                    dtype=dtype,
                    device=self.device,
                )
                self._next_obs[k] = self._obs[k].clone()
        else:
            raise ValueError

        self._action = torch.zeros(
            (self.capacity + cfg.horizon - 1, action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._reward = torch.zeros(
            (self.capacity + cfg.horizon - 1,), dtype=torch.float32, device=self.device
        )
        self._mask = torch.zeros(
            (self.capacity + cfg.horizon - 1,), dtype=torch.float32, device=self.device
        )
        self._done = torch.zeros(
            (self.capacity + cfg.horizon - 1,), dtype=torch.bool, device=self.device
        )
        self._priorities = torch.ones(
            (self.capacity + cfg.horizon - 1,), dtype=torch.float32, device=self.device
        )
        self._eps = 1e-6
        self._full = False
        self.idx = 0
        if dataset is not None:
            self.init_from_offline_dataset(dataset)

        self._aug = aug(cfg)

    def init_from_offline_dataset(self, dataset):
        """Initialize the replay buffer from an offline dataset."""
        assert self.idx == 0 and not self._full
        n_transitions = int(len(dataset["rewards"]) * self.cfg.data_first_percent)

        def copy_data(dst, src, n):
            assert isinstance(dst, dict) == isinstance(src, dict)
            if isinstance(dst, dict):
                for k in dst:
                    copy_data(dst[k], src[k], n)
            else:
                dst[:n] = torch.from_numpy(src[:n])

        copy_data(self._obs, dataset["observations"], n_transitions)
        copy_data(self._next_obs, dataset["next_observations"], n_transitions)
        copy_data(self._action, dataset["actions"], n_transitions)
        copy_data(self._reward, dataset["rewards"], n_transitions)
        copy_data(self._mask, dataset["masks"], n_transitions)
        copy_data(self._done, dataset["dones"], n_transitions)
        self.idx = (self.idx + n_transitions) % self.capacity
        self._full = n_transitions >= self.capacity

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        """Add an episode to the replay buffer."""
        if self.idx + len(episode) > self.capacity:
            print("Warning: episode got truncated")
        ep_len = min(len(episode), self.capacity - self.idx)
        idxs = slice(self.idx, self.idx + ep_len)
        assert self.idx + ep_len <= self.capacity
        if self.cfg.modality in {"pixels", "state"}:
            self._obs[idxs] = (
                episode.obses[:ep_len]
                if self.cfg.modality == "state"
                else episode.obses[:ep_len, -3:]
            )
            self._next_obs[idxs] = (
                episode.obses[1 : ep_len + 1]
                if self.cfg.modality == "state"
                else episode.obses[1 : ep_len + 1, -3:]
            )
        elif self.cfg.modality == "all":
            for k, v in episode.obses.items():
                assert k in {"rgb", "state"}
                assert k in self._obs
                assert k in self._next_obs
                if k == "rgb":
                    self._obs[k][idxs] = episode.obses[k][:ep_len, -3:]
                    self._next_obs[k][idxs] = episode.obses[k][1 : ep_len + 1, -3:]
                else:
                    self._obs[k][idxs] = episode.obses[k][:ep_len]
                    self._next_obs[k][idxs] = episode.obses[k][1 : ep_len + 1]
        self._action[idxs] = episode.actions[:ep_len]
        self._reward[idxs] = episode.rewards[:ep_len]
        self._mask[idxs] = episode.masks[:ep_len]
        self._done[idxs] = episode.dones[:ep_len]
        self._done[self.idx + ep_len - 1] = True  # in case truncated
        if self._full:
            max_priority = (
                self._priorities[: self.capacity].max().to(self.device).item()
            )
        else:
            max_priority = (
                1.0
                if self.idx == 0
                else self._priorities[: self.idx].max().to(self.device).item()
            )
        new_priorities = torch.full((ep_len,), max_priority, device=self.device)
        self._priorities[idxs] = new_priorities
        self.idx = (self.idx + ep_len) % self.capacity
        self._full = self._full or self.idx == 0

    def update_priorities(self, idxs, priorities):
        """Update priorities for Prioritized Experience Replay (PER)"""
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

    def _get_obs(self, arr, idxs):
        """Retrieve observations by indices"""
        if isinstance(arr, dict):
            return {k: self._get_obs(v, idxs) for k, v in arr.items()}
        if arr.ndim <= 2:  # if self.cfg.modality == 'state':
            return arr[idxs].cuda()
        obs = torch.empty(
            (self.cfg.batch_size, 3 * self.cfg.frame_stack, *arr.shape[-2:]),
            dtype=arr.dtype,
            device=torch.device("cuda"),
        )
        obs[:, -3:] = arr[idxs].cuda()
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        for i in range(1, self.cfg.frame_stack):
            mask[_idxs % self.cfg.episode_length == 0] = False
            _idxs[mask] -= 1
            obs[:, -(i + 1) * 3 : -i * 3] = arr[_idxs].cuda()
        return obs.float()

    def sample(self):
        """Sample transitions from the replay buffer."""
        probs = (
            self._priorities[: self.capacity]
            if self._full
            else self._priorities[: self.idx]
        ) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(
            np.random.choice(
                total,
                self.cfg.batch_size,
                p=probs.cpu().numpy(),
                replace=not self._full,
            )
        ).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()

        idxs_in_horizon = torch.stack([idxs + t for t in range(self.cfg.horizon)])

        obs = self._aug(self._get_obs(self._obs, idxs))
        next_obs = [
            self._aug(self._get_obs(self._next_obs, _idxs)) for _idxs in idxs_in_horizon
        ]
        if isinstance(next_obs[0], dict):
            next_obs = {k: torch.stack([o[k] for o in next_obs]) for k in next_obs[0]}
        else:
            next_obs = torch.stack(next_obs)
        action = self._action[idxs_in_horizon]
        reward = self._reward[idxs_in_horizon]
        mask = self._mask[idxs_in_horizon]
        done = self._done[idxs_in_horizon]

        if not action.is_cuda:
            action, reward, mask, done, idxs, weights = (
                action.cuda(),
                reward.cuda(),
                mask.cuda(),
                done.cuda(),
                idxs.cuda(),
                weights.cuda(),
            )

        return (
            obs,
            next_obs,
            action,
            reward.unsqueeze(2),
            mask.unsqueeze(2),
            done.unsqueeze(2),
            idxs,
            weights,
        )

    def save(self, path):
        """Save the replay buffer to path"""
        print(f"saving replay buffer to '{path}'...")
        sz = self.capacity if self._full else self.idx
        dataset = {
            "observations": (
                {k: v[:sz].cpu().numpy() for k, v in self._obs.items()}
                if isinstance(self._obs, dict)
                else self._obs[:sz].cpu().numpy()
            ),
            "next_observations": (
                {k: v[:sz].cpu().numpy() for k, v in self._next_obs.items()}
                if isinstance(self._next_obs, dict)
                else self._next_obs[:sz].cpu().numpy()
            ),
            "actions": self._action[:sz].cpu().numpy(),
            "rewards": self._reward[:sz].cpu().numpy(),
            "dones": self._done[:sz].cpu().numpy(),
            "masks": self._mask[:sz].cpu().numpy(),
        }
        with open(path, "wb") as f:
            pickle.dump(dataset, f)
        return dataset


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
        dataset_path = os.path.join(cfg.dataset_dir, f"buffer.pkl")
        print(f"Using offline dataset '{dataset_path}'")
        with open(dataset_path, "rb") as f:
            dataset_dict = pickle.load(f)
        for k in required_keys:
            if k not in dataset_dict and k[:-1] in dataset_dict:
                dataset_dict[k] = dataset_dict.pop(k[:-1])
    elif cfg.task.startswith("legged"):
        dataset_path = os.path.join(cfg.dataset_dir, f"buffer.pkl")
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
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
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
        assert key in dataset_dict.keys(), f"Missing `{key}` in dataset."

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
        return (
            lambda x: x / (np.max(episode_returns) - np.min(episode_returns)) * 1000.0
        )
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
            init, final, start, end = [float(g) for g in match.groups()]
            mix = np.clip((step - start) / (end - start), 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)
