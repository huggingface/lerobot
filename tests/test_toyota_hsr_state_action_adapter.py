#!/usr/bin/env python

import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.toyota_hsr.state_action_32_adapter_processor import StateAction32AdapterProcessorStep


def _make_transition(state: torch.Tensor, action: torch.Tensor | None) -> dict:
    return {
        TransitionKey.OBSERVATION: {"observation.state": state},
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: None,
        TransitionKey.DONE: None,
        TransitionKey.TRUNCATED: None,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }


def test_index_embedding_scatter():
    step = StateAction32AdapterProcessorStep(
        mode="index_embedding",
        apply_mean_std_normalization=False,
    )

    state = torch.arange(8, dtype=torch.float32)
    action = torch.arange(11, dtype=torch.float32)
    transition = _make_transition(state=state, action=action)

    output = step(transition)
    out_state = output[TransitionKey.OBSERVATION]["observation.state"]
    out_action = output[TransitionKey.ACTION]

    expected_state = torch.zeros(32, dtype=torch.float32)
    expected_state[step.state_index_map] = state

    expected_action = torch.zeros(32, dtype=torch.float32)
    expected_action[step.action_index_map] = action

    assert out_state.shape == (32,)
    assert out_action.shape == (32,)
    torch.testing.assert_close(out_state, expected_state)
    torch.testing.assert_close(out_action, expected_action)


def test_linear_projection_columns_are_orthonormal():
    step = StateAction32AdapterProcessorStep(
        mode="linear_projection",
        apply_mean_std_normalization=False,
    )

    assert step.state_projection_matrix is not None
    assert step.action_projection_matrix is not None

    state_projection = step.state_projection_matrix
    action_projection = step.action_projection_matrix

    state_identity = torch.eye(step.raw_state_dim, dtype=torch.float32)
    action_identity = torch.eye(step.raw_action_dim, dtype=torch.float32)

    torch.testing.assert_close(
        state_projection.transpose(0, 1) @ state_projection,
        state_identity,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        action_projection.transpose(0, 1) @ action_projection,
        action_identity,
        rtol=1e-5,
        atol=1e-5,
    )


def test_mean_std_normalization_applied_before_adaptation():
    stats = {
        "observation.state": {
            "mean": torch.arange(8, dtype=torch.float32),
            "std": torch.arange(1, 9, dtype=torch.float32),
        },
        "action": {
            "mean": torch.arange(11, dtype=torch.float32),
            "std": torch.full((11,), 2.0, dtype=torch.float32),
        },
    }

    state_input = stats["observation.state"]["mean"] + stats["observation.state"]["std"] * 2.0
    action_input = stats["action"]["mean"] + stats["action"]["std"] * 3.0

    step = StateAction32AdapterProcessorStep(
        mode="index_embedding",
        apply_mean_std_normalization=True,
        dataset_stats=stats,
    )

    output = step(_make_transition(state=state_input, action=action_input))
    out_state = output[TransitionKey.OBSERVATION]["observation.state"]
    out_action = output[TransitionKey.ACTION]

    expected_state = torch.zeros(32, dtype=torch.float32)
    expected_state[step.state_index_map] = 2.0

    expected_action = torch.zeros(32, dtype=torch.float32)
    expected_action[step.action_index_map] = 3.0

    torch.testing.assert_close(out_state, expected_state)
    torch.testing.assert_close(out_action, expected_action)


def test_action_time_dimension_shape_preserved():
    step = StateAction32AdapterProcessorStep(
        mode="index_embedding",
        apply_mean_std_normalization=False,
    )

    state = torch.randn(2, 8)
    action = torch.randn(2, 4, 11)

    output = step(_make_transition(state=state, action=action))
    out_state = output[TransitionKey.OBSERVATION]["observation.state"]
    out_action = output[TransitionKey.ACTION]

    assert out_state.shape == (2, 32)
    assert out_action.shape == (2, 4, 32)
