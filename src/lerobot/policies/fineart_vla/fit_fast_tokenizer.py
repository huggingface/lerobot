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

"""Fit and cache a FAST tokenizer for a dataset's action distribution.

Training invokes this automatically when FAST loss and automatic fitting are enabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from huggingface_hub import snapshot_download

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    create_transition,
    decode_fast_actions,
    encode_fast_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _datasets_available, _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
# pyarrow ships with the dataset extra; tokenizer fitting reads training parquet shards.
if TYPE_CHECKING or _datasets_available:
    import pyarrow as pa
    import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ``ProcessorMixin.save_pretrained`` writes this shared cache sentinel.
_CACHE_SENTINEL = "processor_config.json"


def _is_global_leader() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _dataset_signature(
    dataset_repo_id: str,
    base_tokenizer_name: str,
    n_samples: int,
    chunk_size: int,
    normalization_mode: str,
    dataset_revision: str | None = None,
    episodes: list[int] | None = None,
    exclude_episodes: list[int] | None = None,
    action_stats: dict | None = None,
    use_relative_actions: bool = False,
    relative_exclude_joints: list[str] | None = None,
    action_names: list[str] | None = None,
    validation_samples: int = 256,
    max_reconstruction_rmse: float = 0.10,
    max_dim_rmse: float = 0.20,
    seed: int = 42,
    dataset_root: str | Path | None = None,
) -> str:
    """Hash every input that changes the fitted action distribution."""
    payload = {
        "dataset_repo_id": dataset_repo_id,
        "dataset_revision": dataset_revision,
        "base_tokenizer_name": base_tokenizer_name,
        "n_samples": n_samples,
        "chunk_size": chunk_size,
        "normalization_mode": normalization_mode,
        "episodes": episodes,
        "exclude_episodes": exclude_episodes,
        "action_stats": action_stats,
        "use_relative_actions": use_relative_actions,
        "relative_exclude_joints": relative_exclude_joints,
        "action_names": action_names,
        "validation_samples": validation_samples,
        "max_reconstruction_rmse": max_reconstruction_rmse,
        "max_dim_rmse": max_dim_rmse,
        "seed": seed,
        "dataset_root": str(Path(dataset_root).expanduser().resolve()) if dataset_root is not None else None,
    }
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _select_episode_indices(
    available_episodes: list[int],
    episodes: list[int] | None,
    exclude_episodes: list[int] | None,
) -> list[int]:
    allowed = set(episodes) if episodes is not None else set(available_episodes)
    excluded = set(exclude_episodes or [])
    return [episode for episode in available_episodes if episode in allowed and episode not in excluded]


def _validate_fast_reconstruction(
    tokenizer: Any,
    actions: np.ndarray,
    max_reconstruction_rmse: float,
    max_dim_rmse: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Decode held-out chunks and reject tokenizers with excessive quantization error."""
    tokens = encode_fast_actions(tokenizer, actions)
    decoded = decode_fast_actions(tokenizer, tokens, actions.shape[-1], time_horizon=actions.shape[1])
    decoded = decoded.astype(np.float32)
    if decoded.shape != actions.shape:
        raise RuntimeError(
            f"FAST tokenizer reconstruction shape mismatch: expected {actions.shape}, got {decoded.shape}."
        )
    if not np.isfinite(decoded).all():
        raise RuntimeError("FAST tokenizer reconstruction contains non-finite values.")

    squared_error = np.square(decoded - actions)
    rmse = float(np.sqrt(squared_error.mean()))
    dim_rmse = np.sqrt(squared_error.mean(axis=(0, 1)))
    nonconstant_dims = np.ptp(actions, axis=(0, 1)) > 1e-8
    max_observed_dim_rmse = float(dim_rmse[nonconstant_dims].max(initial=0.0))
    report = {
        "num_validation_chunks": int(actions.shape[0]),
        "reconstruction_rmse": rmse,
        "max_dim_rmse": max_observed_dim_rmse,
        "dim_rmse": dim_rmse.tolist(),
        "max_reconstruction_rmse": max_reconstruction_rmse,
        "max_allowed_dim_rmse": max_dim_rmse,
    }
    if rmse > max_reconstruction_rmse or max_observed_dim_rmse > max_dim_rmse:
        raise RuntimeError(
            "FAST tokenizer reconstruction error exceeds the configured limit: "
            f"rmse={rmse:.4f} (max {max_reconstruction_rmse:.4f}), "
            f"max_dim_rmse={max_observed_dim_rmse:.4f} (max {max_dim_rmse:.4f})."
        )
    return report, decoded


def _load_fast_fitter(base_tokenizer_name: str) -> Any:
    """Load FAST's fitting implementation without requiring its universal BPE weights."""
    try:
        return AutoProcessor.from_pretrained(base_tokenizer_name, trust_remote_code=True)
    except ValueError as error:
        if base_tokenizer_name != "physical-intelligence/fast":
            raise
        logger.warning(
            "Could not load the universal FAST tokenizer backend; loading its fitting class directly: %s",
            error,
        )
        return get_class_from_dynamic_module(
            "processing_action_tokenizer.UniversalActionProcessor",
            base_tokenizer_name,
        )


def fit_fast_tokenizer(
    *,
    dataset_repo_id: str,
    cache_dir: str | Path,
    base_tokenizer_name: str = "physical-intelligence/fast",
    n_samples: int = 1024,
    chunk_size: int = 50,
    seed: int = 42,
    dataset_root: str | Path | None = None,
    dataset_revision: str | None = None,
    episodes: list[int] | None = None,
    exclude_episodes: list[int] | None = None,
    normalization_mode: str = "QUANTILES",
    action_stats: dict | None = None,
    use_relative_actions: bool = False,
    relative_exclude_joints: list[str] | None = None,
    action_names: list[str] | None = None,
    validation_samples: int = 256,
    max_reconstruction_rmse: float = 0.10,
    max_dim_rmse: float = 0.20,
) -> str:
    """Fit a FAST tokenizer on a LeRobot dataset's action distribution.

    Args:
        dataset_repo_id: HF Hub repo id of the LeRobotDataset to fit on.
        cache_dir: Directory under which to save (and look up) fitted
            tokenizers. The actual save path is
            ``{cache_dir}/{signature}``.
        base_tokenizer_name: HF identifier for the base FAST tokenizer
            to finetune from. ``physical-intelligence/fast`` is the
            universal one.
        n_samples: Number of action chunks to sample for the fit. The
            FAST paper uses a few thousand; ``1024`` is a good default
            for medium datasets.
        chunk_size: Length of each action chunk (matches
            ``policy.chunk_size``). The FAST tokenizer is fit on
            sequences of this length.
        seed: RNG seed for sample selection.
        normalization_mode: Action normalization mode of the training preprocessor.
        action_stats: Action statistics of the training preprocessor. Without them,
            actions stay unnormalized, as in training.
        use_relative_actions: Convert actions to offsets from the chunk's first state.
        relative_exclude_joints: Joint names kept absolute, as in
            ``RelativeActionsProcessorStep``.
        action_names: Action dimension names used to match ``relative_exclude_joints``.

    Returns:
        The local path to the fitted tokenizer. Passed directly to
        ``--policy.action_tokenizer_name`` for the training run.

    Raises:
        ImportError: If the ``transformers`` library doesn't expose
            ``AutoProcessor`` or the FAST tokenizer doesn't have a
            ``.fit()`` method (then you're on an older FAST snapshot —
            update to the current published model).
        FileNotFoundError: If the dataset can't be loaded.
    """
    cache_dir = Path(cache_dir)
    normalization_mode = getattr(normalization_mode, "value", normalization_mode).upper()
    sig = _dataset_signature(
        dataset_repo_id,
        base_tokenizer_name,
        n_samples,
        chunk_size,
        normalization_mode,
        dataset_revision,
        episodes,
        exclude_episodes,
        action_stats,
        use_relative_actions,
        relative_exclude_joints,
        action_names,
        validation_samples,
        max_reconstruction_rmse,
        max_dim_rmse,
        seed,
        dataset_root,
    )
    out_dir = cache_dir / sig

    if out_dir.exists() and (out_dir / _CACHE_SENTINEL).exists():
        logger.info(
            "FAST tokenizer cache hit: %s — re-using fitted tokenizer for dataset=%s base=%s n_samples=%d",
            out_dir,
            dataset_repo_id,
            base_tokenizer_name,
            n_samples,
        )
        return str(out_dir)

    # One global rank populates the shared cache; every other rank waits for the atomic publish.
    is_leader = _is_global_leader()
    if not is_leader:
        timeout_s = 1800.0  # 30 min — covers ~1024-sample fits on cold caches
        start = time.monotonic()
        while not (out_dir / _CACHE_SENTINEL).exists():
            if time.monotonic() - start > timeout_s:
                raise RuntimeError(
                    f"FAST tokenizer fit: non-leader rank timed out after "
                    f"{timeout_s:.0f}s waiting for {out_dir / _CACHE_SENTINEL}. "
                    "Leader rank likely crashed during the fit."
                )
            time.sleep(2.0)
        logger.info("FAST tokenizer ready (leader populated cache): %s", out_dir)
        return str(out_dir)

    logger.info(
        "FAST tokenizer cache miss — fitting on dataset=%s base=%s n_samples=%d chunk_size=%d → %s",
        dataset_repo_id,
        base_tokenizer_name,
        n_samples,
        chunk_size,
        out_dir,
    )

    # Read action columns directly to avoid video decoding and bound memory to sampled episodes.
    rng = np.random.default_rng(seed)
    actions_buf: list[np.ndarray] = []

    # Read v3 parquet shards directly to avoid split lookup failures and repeated metadata parsing.
    require_package("pyarrow", extra="dataset")

    if dataset_root is not None:
        snap = Path(dataset_root)
    else:
        snap = Path(
            snapshot_download(repo_id=dataset_repo_id, repo_type="dataset", revision=dataset_revision)
        )
    data_files = sorted((snap / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise RuntimeError(f"FAST fit: no ``data/chunk-*/file-*.parquet`` shards found under {snap!s}.")

    columns = ["episode_index", "action"]
    if use_relative_actions:
        columns.append("observation.state")
    tables = [pq.read_table(f, columns=columns) for f in data_files]
    table = pa.concat_tables(tables)
    eps = table["episode_index"].to_numpy()
    acts_col = table["action"]
    # Normalize Arrow action representations into an (N, D) array.
    try:
        acts = np.stack(acts_col.to_numpy(zero_copy_only=False)).astype(np.float32)
    except Exception:  # noqa: BLE001
        # Fallback path for nested-list types: flatten via to_pylist().
        acts = np.asarray(acts_col.to_pylist(), dtype=np.float32)
    if acts.ndim != 2:
        raise RuntimeError(f"FAST fit: expected ``action`` rows to be 1-D vectors; got shape {acts.shape}.")
    states = None
    if use_relative_actions:
        try:
            states = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
        except Exception:  # noqa: BLE001
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if states.ndim != 2:
            raise RuntimeError(
                f"FAST fit: expected ``observation.state`` rows to be 1-D vectors; got {states.shape}."
            )

    # Sort once because episode order is only guaranteed within each shard.
    order = np.argsort(eps, kind="stable")
    eps_sorted = eps[order]
    boundaries = np.asarray(np.searchsorted(eps_sorted, np.arange(int(eps_sorted.max()) + 2)))
    ep_to_slice: dict[int, tuple[int, int]] = {
        int(ep): (int(boundaries[ep]), int(boundaries[ep + 1]))
        for ep in range(len(boundaries) - 1)
        if boundaries[ep] < boundaries[ep + 1]
    }
    # ``acts`` is in original (un-sorted-by-episode) row order; reorder
    # so per-episode slices are contiguous.
    acts = acts[order]
    if states is not None:
        states = states[order]

    ep_indices = _select_episode_indices(list(ep_to_slice), episodes, exclude_episodes)
    if not ep_indices:
        raise RuntimeError("FAST fit: episode selection is empty after applying exclusions.")
    # Split the sample budget only across episodes with at least ``chunk_size`` frames.
    long_episodes = [ep for ep in ep_indices if ep_to_slice[ep][1] - ep_to_slice[ep][0] >= chunk_size]
    if not long_episodes:
        raise RuntimeError(
            f"FAST fit collected zero action chunks from {dataset_repo_id!r}: "
            f"all {len(ep_indices)} selected episodes are shorter than chunk_size={chunk_size}. "
            "Lower ``chunk_size`` to match your episode lengths."
        )
    total_samples = n_samples + validation_samples
    samples_per_episode = max(1, (total_samples + len(long_episodes) - 1) // len(long_episodes))
    collected = 0
    eps_visited = 0
    states_buf: list[np.ndarray] = []
    for ep_idx in rng.permutation(long_episodes):
        if collected >= total_samples:
            break
        start, stop = ep_to_slice[int(ep_idx)]
        ep_actions = acts[start:stop]
        starts = rng.integers(0, ep_actions.shape[0] - chunk_size + 1, size=samples_per_episode)
        for s in starts:
            actions_buf.append(ep_actions[int(s) : int(s) + chunk_size])
            if states is not None:
                states_buf.append(states[start + int(s)])
            collected += 1
            if collected >= total_samples:
                break
        eps_visited += 1

    actions = np.stack(actions_buf, axis=0).astype(np.float32)  # (N, H, D)
    logger.info(
        "FAST fit: collected %d chunks of shape %s from %d episodes",
        actions.shape[0],
        actions.shape[1:],
        eps_visited,
    )

    # Fit on what the model trains on: the training preprocessor's relative-action and
    # normalization steps, built from the same settings and applied in the same order.
    relative_step = RelativeActionsProcessorStep(
        enabled=use_relative_actions,
        exclude_joints=list(relative_exclude_joints or []),
        action_names=action_names,
    )
    normalizer = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(actions.shape[-1],))},
        norm_map={FeatureType.ACTION: NormalizationMode(normalization_mode)},
        stats={ACTION: action_stats} if action_stats is not None else None,
    )
    observation = {OBS_STATE: torch.from_numpy(np.stack(states_buf))} if states is not None else None
    transition = create_transition(observation=observation, action=torch.from_numpy(actions))
    actions = cast(torch.Tensor, normalizer(relative_step(transition))[TransitionKey.ACTION]).numpy()

    base = _load_fast_fitter(base_tokenizer_name)
    if not hasattr(base, "fit"):
        raise ImportError(
            f"Base FAST tokenizer {base_tokenizer_name!r} has no ``.fit()`` "
            "method — your transformers / model snapshot is too old. Update "
            "to the current ``physical-intelligence/fast`` revision."
        )

    fit_actions = actions[:n_samples]
    validation_actions = actions[n_samples:total_samples]
    fitted = base.fit(fit_actions)
    validation_report, decoded_actions = _validate_fast_reconstruction(
        fitted,
        validation_actions,
        max_reconstruction_rmse,
        max_dim_rmse,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = cache_dir / f".{sig}.tmp-{os.getpid()}"
    shutil.rmtree(staging_dir, ignore_errors=True)
    fitted.save_pretrained(str(staging_dir))
    (staging_dir / "reconstruction_validation.json").write_text(
        json.dumps(validation_report, indent=2) + "\n"
    )
    np.savez_compressed(
        staging_dir / "reconstruction_examples.npz",
        original=validation_actions[:8],
        decoded=decoded_actions[:8],
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging_dir.replace(out_dir)
    logger.info("FAST fit: saved fitted tokenizer to %s", out_dir)
    return str(out_dir)


def resolve_fast_tokenizer(
    config: Any,
    dataset_repo_id: str | None,
    dataset_root: str | Path | None = None,
    dataset_stats: dict | None = None,
    dataset_revision: str | None = None,
    episodes: list[int] | None = None,
    exclude_episodes: list[int] | None = None,
) -> str:
    """Return the configured tokenizer, fitting a cached dataset-specific one when requested."""
    if not getattr(config, "auto_fit_fast_tokenizer", False):
        return config.action_tokenizer_name
    if dataset_repo_id is None:
        logger.warning(
            "FAST tokenizer auto-fit is enabled but no dataset source was provided; using %s "
            "unchanged. Training through lerobot-train fits it on the training dataset.",
            config.action_tokenizer_name,
        )
        return config.action_tokenizer_name

    fit_kwargs = {
        "dataset_repo_id": dataset_repo_id,
        "cache_dir": Path(config.fast_tokenizer_cache_dir).expanduser(),
        "base_tokenizer_name": config.action_tokenizer_name,
        "n_samples": config.fast_tokenizer_fit_samples,
        "chunk_size": config.chunk_size,
        "dataset_root": dataset_root,
        "dataset_revision": dataset_revision,
        "episodes": episodes,
        "exclude_episodes": exclude_episodes,
        "normalization_mode": config.normalization_mapping.get("ACTION", "QUANTILES"),
        "action_stats": (dataset_stats or {}).get("action"),
        "use_relative_actions": getattr(config, "use_relative_actions", False),
        "relative_exclude_joints": getattr(config, "relative_exclude_joints", []),
        "action_names": getattr(config, "action_feature_names", None),
    }
    validation_fields = {
        "validation_samples": "fast_tokenizer_validation_samples",
        "max_reconstruction_rmse": "fast_tokenizer_max_reconstruction_rmse",
        "max_dim_rmse": "fast_tokenizer_max_dim_rmse",
    }
    fit_kwargs.update(
        {
            argument: getattr(config, attribute)
            for argument, attribute in validation_fields.items()
            if hasattr(config, attribute)
        }
    )
    return fit_fast_tokenizer(**fit_kwargs)
