import numpy as np
import torch

from lerobot.scripts.lerobot_eval import _action_to_env_numpy


def test_action_to_env_numpy_casts_bfloat16_to_float32():
    action = torch.tensor([[0.5, -1.0]], dtype=torch.bfloat16)

    action_numpy = _action_to_env_numpy(action)

    assert action_numpy.shape == (1, 2)
    assert action_numpy.dtype == np.float32
    np.testing.assert_allclose(action_numpy, np.array([[0.5, -1.0]], dtype=np.float32))
