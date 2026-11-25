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
"""A training sampler bound to one dataset: the rows every sample reads, by dataset key.

Every dataset reader (map-style, streaming, Lance) builds its samples from :class:`SampleWindows`,
so a policy's :class:`~lerobot.configs.sampling.TrainingSampler` reads the same frames everywhere.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from lerobot.configs.sampling import EpisodeTable, FrameSampler, SampleKey, TrainingSampler
from lerobot.configs.types import Deltas, FeatureType
from lerobot.utils.constants import ACTION, OBS_ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STR

from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import check_delta_timestamps, get_delta_indices
from .utils import delta_window

logger = logging.getLogger(__name__)

# Keys a sample can request that are read from another column, at their own offsets.
COLUMN_ALIASES = {OBS_ACTION: ACTION}


def column_of(key: str) -> str:
    """The dataset column that window ``key`` reads."""
    return COLUMN_ALIASES.get(key, key)


def episode_bounds(meta: LeRobotDatasetMetadata) -> tuple[np.ndarray, np.ndarray]:
    """The first dataset row and the number of frames of every episode of ``meta``."""
    if meta.episodes is None:  # metadata of a dataset without episodes yet
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    bounds = meta.episodes.select_columns(["dataset_from_index", "dataset_to_index"]).with_format(None)[:]
    starts = np.asarray(bounds["dataset_from_index"], dtype=np.int64)
    return starts, np.asarray(bounds["dataset_to_index"], dtype=np.int64) - starts


def _observation_type(policy_key: str, feature: Mapping) -> FeatureType | None:
    """The type a dataset feature has as the policy's observation ``policy_key``."""
    if feature.get("dtype") in ("image", "video") or policy_key.startswith(OBS_IMAGE):
        return FeatureType.VISUAL
    if policy_key == OBS_ENV_STATE:
        return FeatureType.ENV
    if policy_key.startswith(OBS_STR + "."):
        return FeatureType.STATE
    return None


class SampleWindows:
    """The samples of a dataset: how many each episode gives and the rows each one reads.

    Built from a policy's training ``sampler`` (offsets by feature type or policy key, resolved
    through ``rename_map``) or from ``delta_timestamps`` (seconds by dataset key, one sample per
    frame). Without either, every frame is one sample without windows.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        episodes: Sequence[int] | None,
        *,
        sampler: TrainingSampler | None = None,
        rename_map: Mapping[str, str] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
    ):
        if sampler is not None and delta_timestamps is not None:
            raise ValueError("Pass either a training sampler or delta_timestamps, not both.")
        self.starts, lengths = episode_bounds(meta)
        selected = (
            np.arange(len(lengths)) if episodes is None else np.unique(np.asarray(episodes, dtype=np.int64))
        )
        self.episodes = EpisodeTable(indices=selected, lengths=lengths)
        self.sampler = sampler if sampler is not None else FrameSampler()
        self._fixed: dict[str, tuple[int, ...]] | None = None
        if sampler is None:
            self._fixed = {}
            if delta_timestamps:
                check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
                indices = get_delta_indices(delta_timestamps, meta.fps)
                self._fixed = {key: tuple(offsets) for key, offsets in indices.items()}
        self._features = meta.features
        self._rename_map = rename_map
        self._cached: tuple[Deltas, dict[str, tuple[int, ...]]] | None = None

        counts = [self.sampler.count(self.episodes, int(ep)) for ep in self.episodes.indices]
        self.counts = np.asarray(counts, dtype=np.int64)
        self._ends = np.cumsum(self.counts)
        if not len(self) and len(self.episodes.indices):
            raise ValueError(f"{type(self.sampler).__name__} takes no sample from any selected episode.")
        if skipped := int((self.counts == 0).sum()):
            logger.warning(
                "%d of %d episodes give no sample to %s (too short?)",
                skipped,
                len(self.counts),
                type(self.sampler).__name__,
            )
        if len(self):
            # Resolve one sample now, so a window the dataset cannot supply fails here and not in a worker.
            self.windows(self.key_at(0))

    def __len__(self) -> int:
        return int(self._ends[-1]) if len(self._ends) else 0

    @property
    def sample_counts(self) -> dict[int, int]:
        """Samples per dataset episode, for the episodes that give any."""
        return {int(ep): int(n) for ep, n in zip(self.episodes.indices, self.counts, strict=True) if n}

    def key_at(self, position: int, epoch: int = 0) -> SampleKey:
        """The ``position``-th sample, episodes in ascending order (negative counts from the end)."""
        total = len(self)
        if position < 0:
            position += total
        if not 0 <= position < total:
            raise IndexError(f"Sample {position} is out of range for a dataset of {total} samples.")
        slot = int(np.searchsorted(self._ends, position, side="right"))
        first = int(self._ends[slot - 1]) if slot else 0
        return SampleKey(int(self.episodes.indices[slot]), position - first, epoch)

    def windows(self, key: SampleKey) -> tuple[int, dict[str, tuple[int, ...]]]:
        """The anchor of sample ``key`` in its episode, and its offsets by dataset key."""
        sample = self.sampler.sample(self.episodes, key)
        if self._fixed is not None:
            return sample.anchor, self._fixed
        # Read the cache once: streaming decode threads share it.
        cached = self._cached
        if cached is None or cached[0] is not sample.deltas:
            cached = (sample.deltas, resolve_windows(sample.deltas, self._features, self._rename_map))
            self._cached = cached
        return sample.anchor, cached[1]

    def rows(
        self, key: SampleKey, start: int | None = None
    ) -> tuple[int, dict[str, list[int]], dict[str, torch.Tensor]]:
        """The rows sample ``key`` reads, counted from row ``start`` of its episode.

        ``start`` defaults to the episode's first dataset row (absolute rows); streaming passes 0
        (rows within the episode). Returns the row of the sample's own values (its anchor, clamped
        to the episode), the rows of every window and their ``{key}_is_pad`` masks.
        """
        if start is None:
            start = int(self.starts[key.episode])
        end = start + self.episodes.length(key.episode)
        anchor, windows = self.windows(key)
        rows: dict[str, list[int]] = {}
        padding: dict[str, torch.Tensor] = {}
        for name, offsets in windows.items():
            rows[name], padding[f"{name}_is_pad"] = delta_window(start + anchor, offsets, start, end)
        return min(max(start + anchor, start), end - 1), rows, padding


def resolve_windows(
    deltas: Deltas, features: Mapping[str, Mapping], rename_map: Mapping[str, str] | None = None
) -> dict[str, tuple[int, ...]]:
    """The offsets of ``deltas`` by dataset key (see :class:`~lerobot.configs.types.Deltas`).

    ``features`` are the dataset's, named by dataset key; ``rename_map`` renames them to the
    policy keys ``deltas`` uses. ``observation.action`` reads the ``action`` column.
    """
    rename = rename_map or {}
    typed = {
        key: (rename.get(key, key), _observation_type(rename.get(key, key), feature))
        for key, feature in features.items()
    }
    windows: dict[str, tuple[int, ...]] = {}
    for key, (policy_key, feature_type) in typed.items():
        if (offsets := deltas.of(policy_key, feature_type)) is not None:
            windows[key] = offsets
    if (offsets := deltas.of(OBS_ACTION)) is not None:
        if ACTION not in features:
            raise ValueError("The training sampler reads observation.action but the dataset has no action.")
        windows[OBS_ACTION] = offsets
    # A camera history that no dataset key can supply would train on single frames silently.
    if len(deltas.observation.get(FeatureType.VISUAL, ())) > 1 and all(
        kind is not FeatureType.VISUAL for _, kind in typed.values()
    ):
        raise ValueError(
            "The training sampler requests several frames per camera, but no dataset feature maps to an image "
            f"key. Dataset features: {sorted(features)}. Image keys must be named `observation.image*` after "
            "applying `--rename_map`."
        )
    # A window declared for one feature that the dataset lacks would be dropped silently too.
    policy_keys = {policy_key for policy_key, _ in typed.values()}
    named = [key for key in deltas.observation if not isinstance(key, FeatureType)]
    if missing := sorted(key for key in named if key not in policy_keys):
        raise ValueError(
            f"The training sampler declares frames of {missing}, which no dataset feature maps to. "
            f"Dataset features: {sorted(features)}; rename them with `--rename_map`."
        )
    return windows
