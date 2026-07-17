#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_nn_functional

from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_dm05 import ACTION_MODE_CHOICES, resolve_dm05_action_mode
from .utils import (
    infer_dm05_non_delta_indices,
)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _wait_for_path(path: Path, *, wait_seconds: int, poll_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + wait_seconds
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for DM05 norm stats: {path}")
        time.sleep(poll_seconds)


class DM05Normalizer:
    def __init__(
        self,
        norm_stats: dict[str, Any],
        *,
        clip: bool = False,
        use_quantiles: bool = True,
        eps: float = 1e-6,
    ) -> None:
        self._payload = _to_jsonable(norm_stats)
        payload = norm_stats.get("norm_stats", norm_stats)
        self.clip = bool(clip)
        self.use_quantiles = bool(use_quantiles)
        self.eps = float(eps)
        self._stats: dict[str, dict[str, torch.Tensor]] = {}
        for key in ("state", "action"):
            if key not in payload:
                continue
            feature = payload[key]
            if "min" not in feature or "max" not in feature:
                raise KeyError(f"DM05 norm stats for {key!r} must contain min and max fields.")
            self._stats[key] = {
                "min": torch.as_tensor(feature["min"], dtype=torch.float32).reshape(-1),
                "max": torch.as_tensor(feature["max"], dtype=torch.float32).reshape(-1),
                "mean": torch.as_tensor(feature.get("mean", feature["min"]), dtype=torch.float32).reshape(-1),
                "std": torch.as_tensor(feature.get("std", feature["max"]), dtype=torch.float32).reshape(-1),
            }
        if not self._stats:
            raise ValueError("DM05 norm stats must contain at least state or action statistics.")

    @classmethod
    def from_path(
        cls,
        norm_stats_path: str | Path,
        *,
        clip: bool = False,
        use_quantiles: bool = True,
    ) -> DM05Normalizer:
        path = Path(norm_stats_path)
        if not path.exists():
            raise FileNotFoundError(f"DM05 norm stats file does not exist: {path}")
        return cls(json.loads(path.read_text(encoding="utf-8")), clip=clip, use_quantiles=use_quantiles)

    def normalize_state(self, state: Any) -> torch.Tensor:
        return self.normalize(state, "state")

    def normalize_action(self, action: Any) -> torch.Tensor:
        return self.normalize(action, "action")

    def denormalize_action(self, action: Any) -> torch.Tensor:
        return self.denormalize(action, "action")

    def to_payload(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._payload))

    def normalize(self, value: Any, key: str) -> torch.Tensor:
        tensor = _as_float_tensor(value)
        stats = self._stats.get(key)
        if stats is None:
            return tensor
        if self.use_quantiles:
            return self._apply_quantile_min_max(tensor, stats, inverse=False)
        return self._apply_mean_std(tensor, stats, inverse=False)

    def denormalize(self, value: Any, key: str) -> torch.Tensor:
        tensor = _as_float_tensor(value)
        stats = self._stats.get(key)
        if stats is None:
            return tensor
        if self.use_quantiles:
            return self._apply_quantile_min_max(tensor, stats, inverse=True)
        return self._apply_mean_std(tensor, stats, inverse=True)

    def _apply_quantile_min_max(
        self,
        tensor: torch.Tensor,
        stats: dict[str, torch.Tensor],
        *,
        inverse: bool,
    ) -> torch.Tensor:
        out = tensor.clone().float()
        lo = stats["min"].to(device=out.device, dtype=out.dtype)
        hi = stats["max"].to(device=out.device, dtype=out.dtype)
        dim = min(int(out.shape[-1]), int(lo.numel()))
        if dim == 0:
            return out
        lo = lo[:dim]
        hi = hi[:dim]
        scale = hi - lo
        zero_mask = (lo == 0) & (hi == 0)
        if inverse:
            out[..., :dim] = (out[..., :dim] + 1.0) * 0.5 * (scale + self.eps) + lo
        else:
            clipped = out[..., :dim].clamp(min=lo, max=hi)
            out[..., :dim] = (clipped - lo) / (scale + self.eps) * 2.0 - 1.0
            if self.clip:
                out[..., :dim] = out[..., :dim].clamp(-1.0, 1.0)
        out[..., :dim] = torch.where(zero_mask, torch.zeros_like(out[..., :dim]), out[..., :dim])
        if out.shape[-1] > dim:
            out[..., dim:] = 0.0
        return out

    def _apply_mean_std(
        self, tensor: torch.Tensor, stats: dict[str, torch.Tensor], *, inverse: bool
    ) -> torch.Tensor:
        out = tensor.clone().float()
        mean = stats["mean"].to(device=out.device, dtype=out.dtype)
        std = stats["std"].to(device=out.device, dtype=out.dtype)
        dim = min(int(out.shape[-1]), int(mean.numel()))
        if dim == 0:
            return out
        mean = mean[:dim]
        std = std[:dim]
        if inverse:
            out[..., :dim] = out[..., :dim] * (std + self.eps) + mean
        else:
            out[..., :dim] = (out[..., :dim] - mean) / (std + self.eps)
        if out.shape[-1] > dim:
            out[..., dim:] = 0.0
        return out


def prepare_dm05_norm_stats_from_dataset(
    *,
    dataset_repo_id: str,
    dataset_root: str | Path,
    norm_stats_path: str | Path,
    video_backend: str = "pyav",
    norm_clip: bool = False,
    norm_use_quantiles: bool = True,
    action_mode: str = "absolute",
    chunk_size: int = 50,
    n_action_steps: int | None = None,
    norm_non_delta_indices: tuple[int, ...] | None = None,
    action_target_offset: int = 0,
    required: bool = True,
) -> DM05Normalizer | None:
    from lerobot.configs.policies import PolicyFeature
    from lerobot.configs.types import FeatureType
    from lerobot.datasets import LeRobotDataset

    from .configuration_dm05 import DM05Config

    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=dataset_root,
        video_backend=video_backend,
    )
    config = DM05Config(
        norm_stats_path=str(norm_stats_path),
        norm_clip=norm_clip,
        norm_use_quantiles=norm_use_quantiles,
        action_mode=action_mode,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps or chunk_size,
        norm_non_delta_indices=norm_non_delta_indices,
        action_target_offset=action_target_offset,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=dataset.meta.features[OBS_STATE]["shape"])
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=dataset.meta.features[ACTION]["shape"])
    }
    return resolve_dm05_normalizer(config, dataset=dataset, required=required)


def resolve_dm05_normalizer(
    config: Any,
    *,
    dataset: Any | None = None,
    required: bool = False,
) -> DM05Normalizer | None:
    norm_kwargs = {
        "clip": getattr(config, "norm_clip", False),
        "use_quantiles": getattr(config, "norm_use_quantiles", True),
    }
    cached_payload = getattr(config, "_dm05_norm_stats_payload", None)
    if cached_payload is not None:
        return DM05Normalizer(cached_payload, **norm_kwargs)

    norm_path_value = getattr(config, "norm_stats_path", None)
    non_delta_indices = infer_dm05_non_delta_indices(config)
    action_mode = resolve_dm05_action_mode(config)
    overwrite = bool(getattr(config, "norm_stats_overwrite", False))
    compute_kwargs = {
        "chunk_size": getattr(config, "chunk_size", 50),
        "action_target_offset": getattr(config, "action_target_offset", 0),
        "drop_n_last_frames": getattr(config, "drop_n_last_frames", 0),
        "non_delta_indices": non_delta_indices,
        "action_mode": action_mode,
        "max_samples": getattr(config, "norm_max_samples", None),
        "fallback_max_samples": getattr(config, "norm_fallback_max_samples", 20000),
        "overwrite": overwrite,
    }
    wait_seconds = int(getattr(config, "norm_stats_wait_seconds", 3600))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = int(torch.distributed.get_world_size())
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    checkpoint_dir = getattr(config, "_dm05_checkpoint_dir", None)
    norm_path = None
    if norm_path_value is not None:
        norm_path = Path(norm_path_value).expanduser()
        if not norm_path.is_absolute() and checkpoint_dir is not None:
            norm_path = Path(checkpoint_dir).expanduser() / norm_path
    output_path = norm_path

    def load(path: Path) -> DM05Normalizer:
        return DM05Normalizer.from_path(path, **norm_kwargs)

    def compute(path: Path | None) -> dict[str, Any]:
        return compute_dm05_norm_stats_from_lerobot_dataset(dataset, output_path=path, **compute_kwargs)

    if norm_path is not None and norm_path.exists() and (not overwrite or dataset is None):
        return load(norm_path)

    if dataset is None:
        missing_path = norm_path or output_path
        if required and missing_path is not None and world_size > 1 and _rank() != 0:
            _wait_for_path(missing_path, wait_seconds=wait_seconds)
            config.norm_stats_path = str(missing_path)
            return load(missing_path)
        if required:
            if missing_path is not None:
                raise FileNotFoundError(f"DM05 norm stats file does not exist: {missing_path}")
            raise ValueError(
                "DM05 training requires norm stats. Set policy.norm_stats_path or "
                "call resolve_dm05_normalizer(config, dataset=...) before constructing the policy."
            )
        return None

    if output_path is not None:
        if _rank() == 0:
            if overwrite or not output_path.exists():
                compute(output_path)
        else:
            _wait_for_path(output_path, wait_seconds=wait_seconds)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        _wait_for_path(output_path, wait_seconds=wait_seconds)
        config.norm_stats_path = str(output_path)
        return load(output_path)

    if world_size > 1:
        payload = compute(None) if _rank() == 0 else None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            objects = [payload]
            torch.distributed.broadcast_object_list(objects, src=0)
            payload = objects[0]
        if payload is None:
            raise RuntimeError("Rank 0 did not produce DM05 norm stats payload.")
    else:
        payload = compute(None)
    config._dm05_norm_stats_payload = payload
    return DM05Normalizer(payload, **norm_kwargs)


class DM05RunningStats:
    def __init__(self, num_quantile_bins: int = 5000) -> None:
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = int(num_quantile_bins)

    def update(self, value: Any) -> None:
        batch = _as_float_tensor(value).numpy()
        if batch.size == 0:
            return
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(
                    self._min[i] - 1e-10,
                    self._max[i] + 1e-10,
                    self._num_quantile_bins + 1,
                )
                for i in range(vector_length)
            ]
        else:
            if self._mean is None or vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_min = np.min(batch, axis=0)
            new_max = np.max(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._min = np.minimum(self._min, new_min)
            self._max = np.maximum(self._max, new_max)
            if max_changed or min_changed:
                for i in range(len(self._histograms)):
                    old_edges = self._bin_edges[i]
                    new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
                    new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
                    self._histograms[i] = new_hist
                    self._bin_edges[i] = new_edges

        self._count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def get_statistics(self) -> dict[str, list[float]]:
        if self._count < 2:
            raise ValueError("Cannot compute DM05 norm stats from fewer than two vectors.")
        variance = self._mean_of_squares - self._mean**2
        std = np.sqrt(np.maximum(0, variance))
        quantile_results = []
        for q in (0.01, 0.99):
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = int(np.searchsorted(cumsum, target_count))
                idx = min(idx, len(edges) - 2)
                q_values.append(edges[idx])
            quantile_results.append(np.asarray(q_values))
        q01, q99 = quantile_results
        return {
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "min": self._min.tolist(),
            "max": self._max.tolist(),
        }


def _as_float_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def command_action_from_relative(
    state: torch.Tensor,
    action_chunk: torch.Tensor,
    *,
    action_dim: int,
    non_delta_indices: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    state = state.to(device=action_chunk.device, dtype=action_chunk.dtype)
    if state.dim() == action_chunk.dim() - 1:
        state = state.unsqueeze(1)
    if state.shape[-1] < action_chunk.shape[-1]:
        state = torch_nn_functional.pad(state, (0, action_chunk.shape[-1] - state.shape[-1]))
    elif state.shape[-1] > action_chunk.shape[-1]:
        state = state[..., : action_chunk.shape[-1]]

    absolute = action_chunk + state
    if non_delta_indices:
        valid = [int(idx) for idx in non_delta_indices if 0 <= int(idx) < absolute.shape[-1]]
        if valid:
            absolute[..., valid] = action_chunk[..., valid]
    return absolute[..., :action_dim]


def _finalize_action_target_chunk(
    *,
    raw_state: torch.Tensor,
    action_chunk: torch.Tensor,
    action_mode: str,
    non_delta_indices: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    if action_mode not in ACTION_MODE_CHOICES:
        raise ValueError(f"DM05 action_mode must be 'absolute' or 'relative', got {action_mode!r}.")
    if action_mode == "relative":
        state = _as_float_tensor(raw_state)
        action_chunk = _as_float_tensor(action_chunk)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action_chunk.ndim == 2:
            state = state.squeeze(0)
            relative = action_chunk - state.unsqueeze(0)
        elif action_chunk.ndim == 3:
            relative = action_chunk - state.unsqueeze(1)
        else:
            raise ValueError(f"Expected action chunk shape [T,D] or [B,T,D], got {tuple(action_chunk.shape)}")
        if non_delta_indices:
            valid = [idx for idx in non_delta_indices if 0 <= int(idx) < relative.shape[-1]]
            if valid:
                relative[..., valid] = action_chunk[..., valid]
        action_chunk = relative
    return action_chunk


def build_dm05_action_target_chunk(
    *,
    raw_state: torch.Tensor,
    raw_action: torch.Tensor,
    chunk_size: int,
    action_mode: str,
    non_delta_indices: list[int] | tuple[int, ...] | None = None,
    action_target_offset: int = 0,
) -> torch.Tensor:
    """Build a DM05 target chunk from an already selected LeRobot action sequence.

    LeRobotDataset applies ``config.action_delta_indices`` before samples reach
    this helper. For raw full-episode arrays, use
    ``build_dm05_action_target_chunks_from_episode`` so offset and end padding
    are handled in one place.
    """

    action_chunk = _as_float_tensor(raw_action)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.unsqueeze(0)
    if action_chunk.ndim != 2:
        raise ValueError(
            f"Expected raw action/state sequence with shape [T, D], got {tuple(action_chunk.shape)}"
        )
    if action_chunk.shape[0] >= chunk_size:
        action_chunk = action_chunk[:chunk_size]
    else:
        pad = action_chunk[-1:].repeat(chunk_size - action_chunk.shape[0], 1)
        action_chunk = torch.cat([action_chunk, pad], dim=0)
    return _finalize_action_target_chunk(
        raw_state=raw_state,
        action_chunk=action_chunk,
        action_mode=action_mode,
        non_delta_indices=non_delta_indices,
    )


def build_dm05_action_target_chunks_from_episode(
    *,
    raw_state: torch.Tensor,
    episode_actions: torch.Tensor,
    frame_indices: torch.Tensor,
    chunk_size: int,
    action_mode: str,
    non_delta_indices: list[int] | tuple[int, ...] | None = None,
    action_target_offset: int = 0,
) -> torch.Tensor:
    episode_actions = _as_float_tensor(episode_actions)
    if episode_actions.ndim != 2:
        raise ValueError(f"Expected episode actions with shape [T, D], got {tuple(episode_actions.shape)}")
    if episode_actions.shape[0] == 0:
        raise ValueError("Cannot build DM05 action targets from an empty episode.")

    frames = torch.as_tensor(frame_indices, dtype=torch.long)
    if frames.ndim == 0:
        frames = frames.reshape(1)
    if int(chunk_size) <= 0:
        raise ValueError(f"DM05 chunk_size must be positive, got {chunk_size!r}.")

    offsets = int(action_target_offset) + torch.arange(int(chunk_size), dtype=torch.long)
    action_indices = torch.clamp(
        frames.reshape(-1, 1) + offsets.reshape(1, -1),
        max=episode_actions.shape[0] - 1,
    )
    action_chunk = episode_actions[action_indices]
    return _finalize_action_target_chunk(
        raw_state=raw_state,
        action_chunk=action_chunk,
        action_mode=action_mode,
        non_delta_indices=non_delta_indices,
    )


def _build_norm_payload(state_stats: DM05RunningStats, action_stats: DM05RunningStats) -> dict[str, Any]:
    def pack(stats: dict[str, list[float]]) -> dict[str, list[float]]:
        return {"min": stats["q01"], "max": stats["q99"], "mean": stats["mean"], "std": stats["std"]}

    return {
        "norm_stats": {
            "default": {"min": -1, "max": 1},
            "state": pack(state_stats.get_statistics()),
            "action": pack(action_stats.get_statistics()),
        }
    }


def _write_norm_payload(
    payload: dict[str, Any], output_path: str | Path | None, *, overwrite: bool
) -> dict[str, Any]:
    if output_path is None:
        return payload
    path = Path(output_path)
    if path.exists() and not overwrite:
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return payload


def _compute_dm05_norm_stats_from_lerobot_parquet(
    dataset: Any,
    *,
    chunk_size: int,
    state_key: str,
    action_key: str,
    action_target_offset: int,
    drop_n_last_frames: int,
    non_delta_indices: list[int] | tuple[int, ...] | None,
    action_mode: str,
    max_samples: int | None,
) -> dict[str, Any] | None:
    children = getattr(dataset, "_datasets", None)
    if children:
        roots = [Path(root) for child in children if (root := getattr(child, "root", None)) is not None]
    else:
        root = getattr(dataset, "root", None)
        roots = [Path(root)] if root is not None else []
    if not roots:
        return None

    state_stats = DM05RunningStats()
    action_stats = DM05RunningStats()
    samples_seen = 0
    used_roots = []
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None

    for root in roots:
        parquet_files = sorted((root / "data").glob("**/*.parquet"))
        if not parquet_files:
            return None

        columns = [state_key, action_key, "episode_index", "frame_index"]
        arrays = {column: [] for column in columns}
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file, columns=columns)
            for column in columns:
                dtype = np.float32 if column in {state_key, action_key} else np.int64
                arrays[column].append(
                    np.asarray(table.column(column).combine_chunks().to_pylist(), dtype=dtype)
                )

        if not arrays[state_key]:
            return None
        states = np.concatenate(arrays[state_key], axis=0)
        actions = np.concatenate(arrays[action_key], axis=0)
        episodes = np.concatenate(arrays["episode_index"], axis=0)
        frames = np.concatenate(arrays["frame_index"], axis=0)
        order = np.lexsort((frames, episodes))
        states = states[order]
        actions = actions[order]
        episodes = episodes[order]
        _, starts, counts = np.unique(episodes, return_index=True, return_counts=True)

        for start, count in zip(starts, counts, strict=True):
            if max_samples is not None and samples_seen >= max_samples:
                break
            take = max(0, int(count) - int(drop_n_last_frames))
            if max_samples is not None:
                take = min(take, int(max_samples) - samples_seen)
            if take <= 0:
                continue

            end = start + count
            state_batch = states[start : start + take]
            action_chunks = build_dm05_action_target_chunks_from_episode(
                raw_state=torch.from_numpy(state_batch),
                episode_actions=torch.from_numpy(actions[start:end]),
                frame_indices=torch.arange(take, dtype=torch.long),
                chunk_size=chunk_size,
                action_mode=action_mode,
                non_delta_indices=non_delta_indices,
                action_target_offset=action_target_offset,
            ).numpy()

            state_stats.update(state_batch)
            action_stats.update(action_chunks.reshape(-1, action_chunks.shape[-1]))
            samples_seen += take
        used_roots.append(str(root))
        if max_samples is not None and samples_seen >= max_samples:
            break

    if samples_seen < 2:
        return None
    return _build_norm_payload(state_stats, action_stats) | {
        "dm05_norm_metadata": {
            "method": "lerobot_parquet_numeric",
            "dataset_roots": used_roots,
            "num_samples": samples_seen,
            "chunk_size": int(chunk_size),
            "action_target_offset": int(action_target_offset),
            "action_mode": action_mode,
            "non_delta_indices": [int(idx) for idx in (non_delta_indices or ())],
        }
    }


def compute_dm05_norm_stats_from_lerobot_dataset(
    dataset: Any,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 50,
    state_key: str = OBS_STATE,
    action_key: str = ACTION,
    action_target_offset: int = 0,
    drop_n_last_frames: int = 1,
    non_delta_indices: list[int] | tuple[int, ...] | None = None,
    action_mode: str = "absolute",
    max_samples: int | None = None,
    fallback_max_samples: int | None = 20000,
    overwrite: bool = False,
) -> dict[str, Any]:
    action_mode = str(action_mode or "absolute")
    if action_mode not in ACTION_MODE_CHOICES:
        raise ValueError("DM05 action_mode must be one of {'absolute', 'relative'}")

    if output_path is not None:
        path = Path(output_path)
        if path.exists() and not overwrite:
            return json.loads(path.read_text(encoding="utf-8"))

    payload = _compute_dm05_norm_stats_from_lerobot_parquet(
        dataset,
        chunk_size=chunk_size,
        state_key=state_key,
        action_key=action_key,
        action_target_offset=action_target_offset,
        drop_n_last_frames=drop_n_last_frames,
        non_delta_indices=non_delta_indices,
        action_mode=action_mode,
        max_samples=max_samples,
    )
    if payload is not None:
        return _write_norm_payload(payload, output_path, overwrite=overwrite)

    effective_max_samples = max_samples
    if effective_max_samples is None:
        effective_max_samples = fallback_max_samples
        if effective_max_samples is not None:
            warnings.warn(
                "DM05 norm stats fell back to dataset[idx]; limiting scan to "
                f"{effective_max_samples} samples. Set policy.norm_max_samples "
                "or policy.norm_fallback_max_samples to override.",
                RuntimeWarning,
                stacklevel=2,
            )

    from lerobot.datasets.sampler import EpisodeAwareSampler

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=drop_n_last_frames,
        shuffle=False,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    state_stats = DM05RunningStats()
    action_stats = DM05RunningStats()
    total = len(sampler) if effective_max_samples is None else min(len(sampler), int(effective_max_samples))
    for sample_idx, idx in enumerate(sampler):
        if sample_idx >= total:
            break
        sample = dataset[idx]
        if state_key not in sample or action_key not in sample:
            raise KeyError(f"DM05 norm computation requires `{state_key}` and `{action_key}` in samples.")
        state = _as_float_tensor(sample[state_key]).reshape(-1)
        action_chunk = build_dm05_action_target_chunk(
            raw_state=state,
            raw_action=sample[action_key],
            chunk_size=chunk_size,
            action_mode=action_mode,
            non_delta_indices=non_delta_indices,
            action_target_offset=action_target_offset,
        )
        state_stats.update(state)
        action_stats.update(action_chunk)

    payload = _build_norm_payload(state_stats, action_stats) | {
        "dm05_norm_metadata": {
            "method": "lerobot_dataset_getitem",
            "num_samples": int(total),
            "chunk_size": int(chunk_size),
            "action_target_offset": int(action_target_offset),
            "action_mode": action_mode,
            "fallback_max_samples": fallback_max_samples,
        }
    }
    return _write_norm_payload(payload, output_path, overwrite=overwrite)


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare DM05 norm stats from a LeRobotDataset.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--norm-clip", type=_parse_bool, default=False)
    parser.add_argument("--norm-use-quantiles", type=_parse_bool, default=True)
    parser.add_argument("--action-mode", choices=sorted(ACTION_MODE_CHOICES), default="absolute")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--norm-non-delta-indices", default=None)
    parser.add_argument("--action-target-offset", type=int, default=0)
    args = parser.parse_args()

    kwargs = vars(args)
    kwargs["norm_stats_path"] = kwargs.pop("norm_stats")
    non_delta_indices = kwargs["norm_non_delta_indices"]
    if non_delta_indices in {None, "", "None", "none", "null"}:
        kwargs["norm_non_delta_indices"] = None
    else:
        text = str(non_delta_indices).strip()
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
            text = text[1:-1]
        kwargs["norm_non_delta_indices"] = tuple(
            int(part.strip()) for part in text.split(",") if part.strip()
        )
    prepare_dm05_norm_stats_from_dataset(**kwargs, required=True)


if __name__ == "__main__":
    main()
