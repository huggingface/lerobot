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

"""Dataset-specific FAST action tokenizer fitting.

The published ``physical-intelligence/fast`` tokenizer is a *universal*
codebook fitted on a heterogeneous mix of robot datasets. Per Pertsch
et al. 2025 (the FAST paper, [64] in the π0.5 paper) and §III.C of
π0.5 itself, the recommended practice is to **finetune the tokenizer on
your specific dataset's action distribution** before training the
policy — same way one would adapt a language tokenizer to a domain
corpus. Without this finetune step, action sequences from your robot
may require more tokens per chunk than necessary, lowering effective
compression and slowing convergence of the action-CE loss.

This module provides a single utility, :func:`fit_fast_tokenizer`,
that does the finetune. The training entry point invokes it
automatically when the policy's ``enable_fast_action_loss`` and
``auto_fit_fast_tokenizer`` flags are both ``True`` and no cached
fitted tokenizer is found at ``fast_tokenizer_cache_dir``.

The fitted tokenizer is saved to
``{cache_dir}/{dataset_hash}_{base_hash}/`` so successive training
runs over the same dataset re-use it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Marker file the cache-hit check looks for. ``ProcessorMixin.save_pretrained``
# writes ``processor_config.json`` (NOT ``preprocessor_config.json`` —
# that's the image / feature-extractor convention). Centralised here so
# the cache-hit check and the rank-N readiness wait agree on the same
# sentinel.
_CACHE_SENTINEL = "processor_config.json"


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
            "FAST tokenizer cache hit: %s — re-using fitted tokenizer for "
            "dataset=%s base=%s n_samples=%d",
            out_dir, dataset_repo_id, base_tokenizer_name, n_samples,
        )
        return str(out_dir)

    # DDP-safe fit: only the (local) main process actually fits + saves;
    # other ranks poll the cache sentinel until the leader is done.
    # Without this guard, all N ranks fit concurrently and race on
    # ``save_pretrained`` + ``AutoProcessor.from_pretrained`` (the latter
    # copies ``processing_action_tokenizer.py`` into ``HF_MODULES_CACHE``
    # and compiles a ``.pyc`` — concurrent writers occasionally produce
    # a stale / partial ``.pyc`` and the subsequent ``from .. import
    # UniversalActionProcessor`` raises ``AttributeError``.
    is_leader = (
        int(os.environ.get("RANK", "0")) == 0
        and int(os.environ.get("LOCAL_RANK", "0")) == 0
    )
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
        "FAST tokenizer cache miss — fitting on dataset=%s "
        "base=%s n_samples=%d chunk_size=%d → %s",
        dataset_repo_id, base_tokenizer_name, n_samples, chunk_size, out_dir,
    )

    from transformers import AutoProcessor  # noqa: PLC0415

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    # Stream a single episode's worth of action chunks at a time so
    # we don't blow memory on huge datasets. Random episode +
    # random start offset gives a reasonable spread.
    #
    # Actions are read straight from the underlying HF dataset's
    # ``action`` *column* — never via ``ds[i]``. ``ds[i]`` builds a full
    # training item (delta-timestamp expansion + video decode + image
    # transforms); a single bad video frame would then throw and, since
    # the failure was swallowed at debug level, silently starve the fit
    # of every chunk. The action column carries no video, so reading it
    # directly is both faster and immune to decode errors.
    rng = np.random.default_rng(seed)
    actions_buf: list[np.ndarray] = []

    # Load just the metadata first to know episode boundaries.
    ds_meta_only = LeRobotDataset(dataset_repo_id, episodes=[0])
    num_episodes = ds_meta_only.meta.total_episodes
    if "action" not in ds_meta_only.features:
        available = ", ".join(sorted(ds_meta_only.features)) or "<none>"
        raise RuntimeError(
            f"FAST fit: dataset {dataset_repo_id!r} has no ``action`` feature. "
            f"Available features: {available}."
        )
    del ds_meta_only

    samples_per_episode = max(1, n_samples // max(num_episodes, 1))
    collected = 0
    eps_visited = 0
    short_episodes = 0
    for ep_idx in rng.permutation(num_episodes):
        if collected >= n_samples:
            break
        ep_idx = int(ep_idx)
        try:
            ds = LeRobotDataset(dataset_repo_id, episodes=[ep_idx])
            ep_actions = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAST fit: skipping episode %d: %s", ep_idx, exc)
            continue
        if ep_actions.ndim != 2 or ep_actions.shape[0] < chunk_size:
            short_episodes += 1
            continue
        # Sample ``samples_per_episode`` contiguous chunks uniformly.
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
        actions.shape[0], actions.shape[1:], eps_visited,
    )

    # Quantile-normalise per dimension before fitting.
    #
    # The FAST tokenizer DCT-transforms actions, scales by ``scale`` and
    # rounds to integer tokens; the integer *range* must fit the
    # codebook (vocab_size, default 1024). Raw motor units (e.g. encoder
    # ticks) blow that range up — hence "Vocab size 1024 is too small".
    # More importantly, at training time ``ActionTokenizerProcessorStep``
    # runs *after* the QUANTILES ``NormalizerProcessorStep``, so it
    # encodes normalised actions. Fitting on raw actions would mismatch
    # that space. We replicate QUANTILES normalisation here (per-dim
    # [q01, q99] → [-1, 1], clipped) so the fit and the training-time
    # encode see the same distribution.
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
