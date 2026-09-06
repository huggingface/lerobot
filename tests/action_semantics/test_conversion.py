import numpy as np
import torch
from lerobot.action_semantics import (
    LIBERO_DATASET_ACTION_CONTRACT,
    LIBERO_SAFETY_DATASET_ACTION_CONTRACT,
    LIBERO_SAFETY_ENV_ACTION_CONTRACT,
    convert_action,
)


def test_libero_dataset_to_canonical():
    a = np.zeros(7, dtype=float)
    a[0] = 0.4
    canonical = convert_action(a, LIBERO_DATASET_ACTION_CONTRACT, LIBERO_DATASET_ACTION_CONTRACT)
    # identity roundtrip
    assert np.allclose(canonical[0], 0.4)


def test_libero_to_safety_roundtrip():
    a = np.zeros(7, dtype=float)
    a[0] = 0.4
    # libero dataset command -> canonical -> safety env command
    safety_cmd = convert_action(a, LIBERO_DATASET_ACTION_CONTRACT, LIBERO_SAFETY_ENV_ACTION_CONTRACT)
    # expected: canonical = 0.4 * 0.05 = 0.02 ; safety_cmd = 0.02 / 2.0 = 0.01
    assert np.isclose(safety_cmd[0], 0.01)


def test_torch_batch_shape():
    a = torch.zeros((2, 7), dtype=torch.float32)
    a[0, 0] = 0.4
    out = convert_action(a, LIBERO_DATASET_ACTION_CONTRACT, LIBERO_SAFETY_ENV_ACTION_CONTRACT)
    assert out.shape == (2, 7)
    assert torch.is_tensor(out)
