"""Episode selection must not silently coerce invalid indices into different episodes."""

import numpy as np
import pytest
import torch

from lerobot.datasets.sampler import EpisodeAwareSampler


@pytest.mark.parametrize("selection", [[-1], [0, -1], [3], [1.9], ["1"], [True], [[0]], [2**64]])
def test_reject_invalid_episode_selection(selection: list) -> None:
    """Report invalid indices rather than selecting a different episode or leaking an indexing error."""
    with pytest.raises(ValueError, match="episode_indices_to_use"):
        EpisodeAwareSampler([0, 2, 5], [2, 5, 9], episode_indices_to_use=selection)


@pytest.mark.parametrize(
    "selection",
    [
        [2, 0],
        [0, 2, 2],
        np.array([0, 2], dtype=np.int32),
        np.array([0, 2], dtype=np.uint64),
        torch.tensor([0, 2]),
    ],
)
@pytest.mark.parametrize("shuffle", [False, True])
def test_valid_episode_selection_preserves_frames(selection: list, shuffle: bool) -> None:
    """Keep valid integer containers, deduplication, shuffling, and frame trimming intact."""
    sampler = EpisodeAwareSampler(
        [0, 2, 5], [2, 5, 9], episode_indices_to_use=selection, drop_n_last_frames=1, shuffle=shuffle, seed=23
    )
    assert sorted(sampler) == [0, 5, 6, 7]
    assert len(sampler) == 4


def test_empty_selection_retains_no_frames_error() -> None:
    """An empty selection keeps the existing no-valid-frames diagnostic."""
    with pytest.raises(ValueError, match="No valid frames remain"):
        EpisodeAwareSampler([0], [2], episode_indices_to_use=[])
