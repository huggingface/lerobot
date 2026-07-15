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
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ``ProcessorMixin.save_pretrained`` writes this shared cache sentinel.
_CACHE_SENTINEL = "processor_config.json"


def _is_local_leader() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _dataset_signature(
    dataset_repo_id: str,
    base_tokenizer_name: str,
    n_samples: int,
    chunk_size: int,
) -> str:
    """Deterministic short hash for naming the cache directory.

    Keys on (dataset, base tokenizer, sample count, chunk size) so any
    of those changing re-runs the fit. ``chunk_size`` matters because
    the tokenizer is fit on chunks of that length.
    """
    h = hashlib.sha256()
    h.update(dataset_repo_id.encode("utf-8"))
    h.update(b"\0")
    h.update(base_tokenizer_name.encode("utf-8"))
    h.update(b"\0")
    h.update(str(n_samples).encode("utf-8"))
    h.update(b"\0")
    h.update(str(chunk_size).encode("utf-8"))
    return h.hexdigest()[:16]


def fit_fast_tokenizer(
    *,
    dataset_repo_id: str,
    cache_dir: str | Path,
    base_tokenizer_name: str = "physical-intelligence/fast",
    n_samples: int = 1024,
    chunk_size: int = 50,
    seed: int = 42,
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
    sig = _dataset_signature(dataset_repo_id, base_tokenizer_name, n_samples, chunk_size)
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

    # Each node fits its node-local cache once; its other local ranks wait.
    is_leader = _is_local_leader()
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

    from transformers import AutoProcessor  # noqa: PLC0415

    # Read action columns directly to avoid video decoding and bound memory to sampled episodes.
    rng = np.random.default_rng(seed)
    actions_buf: list[np.ndarray] = []

    # Read v3 parquet shards directly to avoid split lookup failures and repeated metadata parsing.
    import pyarrow as _pa  # noqa: PLC0415
    import pyarrow.parquet as _pq  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    snap = Path(snapshot_download(repo_id=dataset_repo_id, repo_type="dataset"))
    data_files = sorted((snap / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise RuntimeError(f"FAST fit: no ``data/chunk-*/file-*.parquet`` shards found under {snap!s}.")

    # Load only episode indices and fixed-width actions across all shards.
    tables = [_pq.read_table(f, columns=["episode_index", "action"]) for f in data_files]
    table = _pa.concat_tables(tables)
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

    # Sort once because episode order is only guaranteed within each shard.
    order = np.argsort(eps, kind="stable")
    eps_sorted = eps[order]
    boundaries = np.searchsorted(eps_sorted, np.arange(int(eps_sorted.max()) + 2))
    ep_to_slice: dict[int, tuple[int, int]] = {
        int(ep): (int(boundaries[ep]), int(boundaries[ep + 1]))
        for ep in range(len(boundaries) - 1)
        if boundaries[ep] < boundaries[ep + 1]
    }
    num_episodes = len(ep_to_slice)
    # ``acts`` is in original (un-sorted-by-episode) row order; reorder
    # so per-episode slices are contiguous.
    acts = acts[order]

    samples_per_episode = max(1, n_samples // max(num_episodes, 1))
    collected = 0
    eps_visited = 0
    short_episodes = 0
    ep_indices = list(ep_to_slice.keys())
    for ep_idx in rng.permutation(ep_indices):
        if collected >= n_samples:
            break
        start, stop = ep_to_slice[int(ep_idx)]
        ep_actions = acts[start:stop]
        if ep_actions.shape[0] < chunk_size:
            short_episodes += 1
            continue
        starts = rng.integers(0, ep_actions.shape[0] - chunk_size + 1, size=samples_per_episode)
        for s in starts:
            actions_buf.append(ep_actions[int(s) : int(s) + chunk_size])
            collected += 1
            if collected >= n_samples:
                break
        eps_visited += 1

    if not actions_buf:
        raise RuntimeError(
            f"FAST fit collected zero action chunks from {dataset_repo_id!r}: "
            f"all {num_episodes} episodes were shorter than chunk_size="
            f"{chunk_size} ({short_episodes} too short) or had an unreadable "
            "``action`` column. Lower ``chunk_size`` to match your episode "
            "lengths."
        )

    actions = np.stack(actions_buf, axis=0).astype(np.float32)  # (N, H, D)
    logger.info(
        "FAST fit: collected %d chunks of shape %s from %d episodes",
        actions.shape[0],
        actions.shape[1:],
        eps_visited,
    )

    # Match training-time quantile normalization so FAST sees the same bounded action space.
    flat = actions.reshape(-1, actions.shape[-1])
    q01 = np.quantile(flat, 0.01, axis=0)
    q99 = np.quantile(flat, 0.99, axis=0)
    span = np.where((q99 - q01) > 1e-6, q99 - q01, 1.0)
    actions = np.clip((actions - q01) / span * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

    base = AutoProcessor.from_pretrained(base_tokenizer_name, trust_remote_code=True)
    if not hasattr(base, "fit"):
        raise ImportError(
            f"Base FAST tokenizer {base_tokenizer_name!r} has no ``.fit()`` "
            "method — your transformers / model snapshot is too old. Update "
            "to the current ``physical-intelligence/fast`` revision."
        )

    fitted = base.fit(actions)
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted.save_pretrained(str(out_dir))
    logger.info("FAST fit: saved fitted tokenizer to %s", out_dir)
    return str(out_dir)


def resolve_fast_tokenizer(config: Any, dataset_repo_id: str | None) -> str:
    """Return the configured tokenizer, fitting a cached dataset-specific one when requested."""
    if not getattr(config, "auto_fit_fast_tokenizer", False) or dataset_repo_id is None:
        return config.action_tokenizer_name

    return fit_fast_tokenizer(
        dataset_repo_id=dataset_repo_id,
        cache_dir=Path(config.fast_tokenizer_cache_dir).expanduser(),
        base_tokenizer_name=config.action_tokenizer_name,
        n_samples=config.fast_tokenizer_fit_samples,
        chunk_size=config.chunk_size,
    )
