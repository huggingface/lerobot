import random
from typing import Callable

import numpy as np
import pytest
import torch

from lerobot.common.utils.utils import seeded_context, set_global_seed


@pytest.mark.parametrize(
    "rand_fn",
    [
        random.random,
        np.random.random,
        lambda: torch.rand(1).item(),
    ]
    + [lambda: torch.rand(1, device="cuda")]
    if torch.cuda.is_available()
    else [],
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
