"""Tests for delta action transforms using a local dummy dataset."""

import numpy as np
import pytest
import torch

from lerobot.processor import TransitionKey, batch_to_transition
from lerobot.processor.delta_action_processor import (
    DeltaActionsProcessorStep,
    to_absolute_actions,
    to_delta_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE

ACTION_DIM = 14
STATE_DIM = 14


@pytest.fixture
def dataset(tmp_path, empty_lerobot_dataset_factory):
    features = {
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
    }
    ds = empty_lerobot_dataset_factory(root=tmp_path / "delta_test", features=features)
    for ep in range(2):
        for _ in range(5):
            ds.add_frame(
                {
                    "action": np.random.randn(ACTION_DIM).astype(np.float32),
                    "observation.state": np.random.randn(STATE_DIM).astype(np.float32),
                    "task": f"task_{ep}",
                }
            )
        ds.save_episode()
    ds.finalize()
    return ds


def _collate(dataset, indices):
    items = [dataset[i] for i in indices]
    batch = {}
    for key in items[0]:
        vals = [item[key] for item in items]
        if isinstance(vals[0], torch.Tensor):
            batch[key] = torch.stack(vals)
        else:
            batch[key] = vals
    return batch


def test_roundtrip_3d(dataset):
    """Delta then absolute on real data should recover original actions."""
    batch = _collate(dataset, range(4))
    actions = batch[ACTION].unsqueeze(1).expand(-1, 10, -1).clone()
    state = batch[OBS_STATE]
    mask = [True] * actions.shape[-1]

    delta = to_delta_actions(actions, state, mask)
    recovered = to_absolute_actions(delta, state, mask)
    torch.testing.assert_close(recovered, actions)


def test_roundtrip_2d(dataset):
    """Works with (B, action_dim) shaped actions too."""
    batch = _collate(dataset, range(4))
    actions = batch[ACTION]
    state = batch[OBS_STATE]
    mask = [True] * actions.shape[-1]

    delta = to_delta_actions(actions, state, mask)
    recovered = to_absolute_actions(delta, state, mask)
    torch.testing.assert_close(recovered, actions)


def test_delta_changes_all_dims(dataset):
    """All dims should change when mask is all True."""
    batch = _collate(dataset, range(4))
    actions = batch[ACTION].unsqueeze(1)
    state = batch[OBS_STATE]
    mask = [True] * actions.shape[-1]

    delta = to_delta_actions(actions, state, mask)
    assert (delta - actions).abs().sum() > 0


def test_no_mutation(dataset):
    """Original tensors should not be modified."""
    batch = _collate(dataset, range(2))
    actions = batch[ACTION].unsqueeze(1)
    original = actions.clone()
    state = batch[OBS_STATE]
    mask = [True] * actions.shape[-1]

    to_delta_actions(actions, state, mask)
    torch.testing.assert_close(actions, original)


def test_processor_step_roundtrip(dataset):
    """DeltaActionsProcessorStep applies delta; to_absolute_actions recovers original."""
    batch = _collate(dataset, range(4))
    original_actions = batch[ACTION].clone()
    transition = batch_to_transition(batch)

    step = DeltaActionsProcessorStep(enabled=True)
    delta_transition = step(transition)

    delta_actions = delta_transition[TransitionKey.ACTION]
    assert not torch.allclose(delta_actions, original_actions)

    state = transition[TransitionKey.OBSERVATION][OBS_STATE]
    mask = [True] * original_actions.shape[-1]
    recovered = to_absolute_actions(delta_actions, state, mask)
    torch.testing.assert_close(recovered, original_actions)


def test_processor_step_disabled_is_noop(dataset):
    """enabled=False should be a no-op."""
    batch = _collate(dataset, range(2))
    original = batch[ACTION].clone()
    transition = batch_to_transition(batch)

    result = DeltaActionsProcessorStep(enabled=False)(transition)
    torch.testing.assert_close(result[TransitionKey.ACTION], original)
