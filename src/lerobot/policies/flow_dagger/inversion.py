# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turn a DAgger dataset into FlowDAgger ``(features, noise)`` training pairs.

DAgger rollouts (``lerobot-rollout --strategy.type=dagger``) tag every frame with an
``intervention`` flag. Two kinds of frames become training samples:

* **intervention** frames, whose action chunk is the operator's correction ``a*``;
* **autonomous** frames well before any intervention, whose action chunk is what the base
  policy did when nothing needed correcting. This stands in for the paper's "successful
  autonomous rollouts" buffer, because DAgger datasets carry no success label.

Each selected chunk is cut to its *segment*: the leading steps that stay in the same episode
and the same intervention state. Steps past it are replaced by the segment's last action,
because the frames that follow in the dataset come from a different controller after a pause
of unknown length. Each chunk is then preprocessed exactly as for training and inverted to
the noise the frozen base policy decodes into it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lerobot.processor.rename_processor import rename_batch_keys
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)

INTERVENTION_KEY = "intervention"


class InvertiblePolicy(Protocol):
    """What FlowDAgger needs from a base policy (implemented by pi05 and smolvla)."""

    config: Any

    def invert_action_chunk(
        self, batch: dict[str, Tensor], *, num_fixed_point_iters: int = 5
    ) -> tuple[Tensor, Tensor]: ...

    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def segment_run_lengths(episode_index: np.ndarray, intervention: np.ndarray) -> np.ndarray:
    """For each frame, how many frames from it onward share its episode and intervention state."""
    n = len(episode_index)
    run: np.ndarray = np.zeros(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        same_segment = (
            i + 1 < n and episode_index[i + 1] == episode_index[i] and intervention[i + 1] == intervention[i]
        )
        run[i] = run[i + 1] + 1 if same_segment else 1
    return run


def frames_to_next_intervention(episode_index: np.ndarray, intervention: np.ndarray) -> np.ndarray:
    """For each frame, the frame distance to the next intervention frame in its episode (inf if none)."""
    n = len(episode_index)
    dist = np.full(n, np.inf)
    next_dist = np.inf
    for i in range(n - 1, -1, -1):
        if i + 1 < n and episode_index[i + 1] != episode_index[i]:
            next_dist = np.inf
        if intervention[i]:
            next_dist = 0.0
        dist[i] = next_dist
        next_dist += 1.0
    return dist


@dataclass
class SampleSelection:
    """Row positions selected from one dataset, with the usable chunk length of each."""

    intervention_positions: np.ndarray
    autonomous_positions: np.ndarray
    valid_len: np.ndarray  # per row of the dataset: leading chunk steps inside the frame's segment

    @property
    def positions(self) -> np.ndarray:
        return np.concatenate([self.intervention_positions, self.autonomous_positions])

    @property
    def is_intervention(self) -> np.ndarray:
        return np.concatenate(
            [
                np.ones(len(self.intervention_positions), dtype=bool),
                np.zeros(len(self.autonomous_positions), dtype=bool),
            ]
        )


def select_samples(
    episode_index: np.ndarray,
    intervention: np.ndarray,
    *,
    chunk_size: int,
    min_valid_steps: int,
    exclude_pre_intervention_frames: int,
    autonomous_stride: int = 1,
    use_autonomous: bool = True,
) -> SampleSelection:
    """Pick the frames that become FlowDAgger training samples.

    Args:
        episode_index: Per-row episode index, rows in dataset order (episodes contiguous).
        intervention: Per-row intervention flag.
        chunk_size: Policy action chunk length.
        min_valid_steps: Minimum number of leading chunk steps that must lie in the frame's
            segment for it to be used.
        exclude_pre_intervention_frames: Autonomous frames this close (in frames) to a later
            intervention in the same episode are dropped: the policy was drifting there.
        autonomous_stride: Keep every n-th eligible autonomous frame.
        use_autonomous: Set False to keep only intervention samples.
    """
    episode_index = np.asarray(episode_index).reshape(-1)
    intervention = np.asarray(intervention).reshape(-1).astype(bool)
    if episode_index.shape != intervention.shape:
        raise ValueError("episode_index and intervention must have the same length")
    if autonomous_stride < 1:
        raise ValueError(f"autonomous_stride must be >= 1, got {autonomous_stride}")
    min_valid_steps = max(1, min(min_valid_steps, chunk_size))

    valid_len = np.minimum(segment_run_lengths(episode_index, intervention), chunk_size)
    long_enough = valid_len >= min_valid_steps

    intervention_positions = np.flatnonzero(intervention & long_enough)

    if use_autonomous:
        far_from_intervention = frames_to_next_intervention(episode_index, intervention) > max(
            exclude_pre_intervention_frames, 0
        )
        eligible = np.flatnonzero(~intervention & long_enough & far_from_intervention)
        autonomous_positions = eligible[::autonomous_stride]
    else:
        autonomous_positions = np.zeros(0, dtype=np.int64)

    return SampleSelection(
        intervention_positions=intervention_positions,
        autonomous_positions=autonomous_positions,
        valid_len=valid_len,
    )


def hold_last_beyond(actions: Tensor, valid_len: Tensor) -> Tensor:
    """Replace chunk steps ``>= valid_len`` with the step at ``valid_len - 1``. ``actions``: ``(B, H, D)``."""
    horizon = actions.shape[1]
    steps = torch.arange(horizon, device=actions.device)
    last = (valid_len.to(actions.device).clamp(1, horizon) - 1).long()
    src = torch.minimum(steps[None, :], last[:, None])  # (B, H)
    return torch.gather(actions, 1, src[..., None].expand(-1, -1, actions.shape[-1]))


# ---------------------------------------------------------------------------
# Inversion over a dataset
# ---------------------------------------------------------------------------


class _PositionDataset(Dataset):
    """Yields ``dataset[pos]`` for selected row positions, tagged with the position."""

    def __init__(self, dataset: Any, positions: Sequence[int]):
        self.dataset = dataset
        self.positions = [int(p) for p in positions]

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int) -> dict:
        pos = self.positions[i]
        item = dict(self.dataset[pos])
        item["_flow_dagger_pos"] = torch.tensor(pos, dtype=torch.long)
        return item


@dataclass
class InversionResult:
    features: Tensor  # (N, D) float32, pooled prefix features
    noise: Tensor  # (N, chunk_size, max_action_dim) float32, inverted noise
    is_intervention: Tensor  # (N,) bool
    reconstruction_mse: Tensor  # (N,) float32, decode(noise) vs target on valid steps; NaN if skipped

    def __len__(self) -> int:
        return self.features.shape[0]

    def to_dict(self) -> dict[str, Tensor]:
        return {
            "features": self.features.contiguous(),
            "noise": self.noise.contiguous(),
            "is_intervention": self.is_intervention.to(torch.uint8).contiguous(),
            "reconstruction_mse": self.reconstruction_mse.contiguous(),
        }

    @classmethod
    def from_dict(cls, tensors: dict[str, Tensor]) -> InversionResult:
        return cls(
            features=tensors["features"],
            noise=tensors["noise"],
            is_intervention=tensors["is_intervention"].bool(),
            reconstruction_mse=tensors["reconstruction_mse"],
        )

    @classmethod
    def concatenate(cls, results: Sequence[InversionResult]) -> InversionResult:
        if not results:
            raise ValueError("No inversion results to concatenate")
        non_empty = [r for r in results if len(r) > 0]
        if not non_empty:
            return results[0]
        results = non_empty
        return cls(
            features=torch.cat([r.features for r in results]),
            noise=torch.cat([r.noise for r in results]),
            is_intervention=torch.cat([r.is_intervention for r in results]),
            reconstruction_mse=torch.cat([r.reconstruction_mse for r in results]),
        )


def read_selection_columns(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row ``(episode_index, intervention)`` of a DAgger dataset, in row order."""
    if INTERVENTION_KEY not in dataset.features:
        raise ValueError(
            f"Dataset {getattr(dataset, 'repo_id', '?')} has no '{INTERVENTION_KEY}' feature. FlowDAgger "
            "needs a dataset recorded with `lerobot-rollout --strategy.type=dagger`."
        )
    columns = dataset.select_columns(["episode_index", INTERVENTION_KEY]).with_format("numpy")
    episode_index = np.asarray(columns["episode_index"]).reshape(-1)
    intervention = np.asarray(columns[INTERVENTION_KEY]).reshape(-1).astype(bool)
    return episode_index, intervention


def invert_dataset(
    policy: InvertiblePolicy,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    dataset: Any,
    selection: SampleSelection,
    *,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 4,
    num_fixed_point_iters: int = 5,
    check_reconstruction: bool = True,
    rename_map: dict[str, str] | None = None,
    camera_keys: Sequence[str] = (),
    progress: bool = True,
) -> InversionResult:
    """Invert every selected chunk of ``dataset`` with the frozen ``policy``.

    ``dataset`` must be built with the policy's ``delta_timestamps`` so that each item holds
    the full action chunk (see ``resolve_delta_timestamps``), and ``preprocessor`` must be
    the base policy's own pretrained preprocessor, so that the actions are normalised with
    the statistics the base policy was trained with.
    """
    positions = selection.positions
    is_intervention = torch.from_numpy(selection.is_intervention)
    if len(positions) == 0:
        return InversionResult(
            features=torch.zeros(0, 0),
            noise=torch.zeros(0, policy.config.chunk_size, policy.config.max_action_dim),
            is_intervention=is_intervention,
            reconstruction_mse=torch.zeros(0),
        )

    valid_len_all = torch.from_numpy(selection.valid_len)
    loader = DataLoader(
        _PositionDataset(dataset, positions.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lerobot_collate_fn,
        pin_memory=device.type == "cuda",
    )
    action_dim = policy.config.output_features[ACTION].shape[0]
    renamed_camera_keys = [(rename_map or {}).get(key, key) for key in camera_keys]

    features, noises, mses = [], [], []
    iterator: Any = loader
    if progress:
        from tqdm import tqdm

        iterator = tqdm(loader, desc=f"Inverting {getattr(dataset, 'repo_id', 'dataset')}")
    for batch in iterator:
        pos = batch.pop("_flow_dagger_pos")
        valid_len = valid_len_all[pos]
        # Canonicalise dataset keys (and their `_is_pad` companions) before touching them.
        batch = rename_batch_keys(batch, rename_map)
        # Keep each chunk inside its segment before normalisation / relative-action transforms.
        batch[ACTION] = hold_last_beyond(batch[ACTION], valid_len)
        batch.pop(f"{ACTION}_is_pad", None)
        for cam_key in renamed_camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)

        noise, feats = policy.invert_action_chunk(batch, num_fixed_point_iters=num_fixed_point_iters)
        features.append(feats.float().cpu())
        noises.append(noise.float().cpu())

        if check_reconstruction:
            target = batch[ACTION][..., :action_dim].float()
            decoded = policy.predict_action_chunk(batch, noise=noise).float()
            policy.reset()
            steps = torch.arange(target.shape[1], device=target.device)
            mask = (steps[None, :] < valid_len.to(target.device)[:, None]).float()[..., None]
            sq = ((decoded - target) ** 2 * mask).sum(dim=(1, 2))
            mses.append((sq / (mask.sum(dim=(1, 2)) * action_dim).clamp(min=1)).cpu())
        else:
            mses.append(torch.full((noise.shape[0],), float("nan")))

    return InversionResult(
        features=torch.cat(features),
        noise=torch.cat(noises),
        is_intervention=is_intervention,
        reconstruction_mse=torch.cat(mses),
    )


def summarize_inversion(result: InversionResult) -> dict[str, float]:
    """Counts and inversion-quality statistics per buffer, for logging."""
    stats: dict[str, float] = {}
    for name, mask in (("intervention", result.is_intervention), ("autonomous", ~result.is_intervention)):
        stats[f"{name}/num_samples"] = float(mask.sum())
        mse = result.reconstruction_mse[mask]
        mse = mse[~torch.isnan(mse)]
        if mse.numel() > 0:
            stats[f"{name}/recon_mse_mean"] = float(mse.mean())
            stats[f"{name}/recon_mse_p95"] = float(mse.quantile(0.95))
        noise = result.noise[mask]
        if noise.numel() > 0:
            stats[f"{name}/noise_std"] = float(noise.std())
    return stats
