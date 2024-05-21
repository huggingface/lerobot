import random
from typing import Callable

import numpy as np
import pytest
import torch
from datasets import Dataset

from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
    reset_episode_index,
)
from lerobot.common.utils.utils import seeded_context, set_global_seed


@pytest.mark.parametrize(
    "rand_fn",
    (
        [
            random.random,
            np.random.random,
            lambda: torch.rand(1).item(),
        ]
        + [lambda: torch.rand(1, device="cuda")]
        if torch.cuda.is_available()
        else []
    ),
)
def test_seeding(rand_fn: Callable[[], int]):
    set_global_seed(0)
    a = rand_fn()
    with seeded_context(1337):
        c = rand_fn()
    b = rand_fn()
    set_global_seed(0)
    a_ = rand_fn()
    b_ = rand_fn()
    # Check that `set_global_seed` lets us reproduce a and b.
    assert a_ == a
    # Additionally, check that the `seeded_context` didn't interrupt the global RNG.
    assert b_ == b
    set_global_seed(1337)
    c_ = rand_fn()
    # Check that `seeded_context` and `global_seed` give the same reproducibility.
    assert c_ == c


def test_calculate_episode_data_index():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    assert torch.equal(episode_data_index["from"], torch.tensor([0, 2, 3]))
    assert torch.equal(episode_data_index["to"], torch.tensor([2, 3, 6]))


def test_reset_episode_index():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [10, 10, 11, 12, 12, 12],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    correct_episode_index = [0, 0, 1, 2, 2, 2]
    dataset = reset_episode_index(dataset)
    assert dataset["episode_index"] == correct_episode_index
