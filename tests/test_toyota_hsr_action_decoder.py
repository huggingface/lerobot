#!/usr/bin/env python

import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.toyota_hsr.action_32_decode_processor import Action32DecodeProcessorStep
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


def test_index_embedding_decoder_restores_raw_action():
    adapter = StateAction32AdapterProcessorStep(
        mode="index_embedding",
        apply_mean_std_normalization=False,
    )
    decoder = Action32DecodeProcessorStep(
        mode="index_embedding",
        action_index_map=list(adapter.action_index_map),
    )

    raw_action = torch.randn(3, 11)
    encoded_transition = adapter(_make_transition(state=torch.randn(3, 8), action=raw_action))
    encoded_action = encoded_transition[TransitionKey.ACTION]

    decoded_transition = decoder(_make_transition(state=torch.randn(3, 8), action=encoded_action))
    decoded_action = decoded_transition[TransitionKey.ACTION]

    assert decoded_action.shape == (3, 11)
    torch.testing.assert_close(decoded_action, raw_action)


def test_linear_projection_decoder_restores_raw_action():
    adapter = StateAction32AdapterProcessorStep(
        mode="linear_projection",
        apply_mean_std_normalization=False,
        projection_seed=7,
    )
    decoder = Action32DecodeProcessorStep(
        mode="linear_projection",
        projection_seed=7,
    )

    raw_action = torch.randn(2, 5, 11)
    encoded_transition = adapter(_make_transition(state=torch.randn(2, 8), action=raw_action))
    encoded_action = encoded_transition[TransitionKey.ACTION]

    decoded_transition = decoder(_make_transition(state=torch.randn(2, 8), action=encoded_action))
    decoded_action = decoded_transition[TransitionKey.ACTION]

    assert decoded_action.shape == (2, 5, 11)
    torch.testing.assert_close(decoded_action, raw_action, rtol=1e-5, atol=1e-5)


def test_decoder_disabled_keeps_32d_action():
    decoder = Action32DecodeProcessorStep(enabled=False)

    action_32 = torch.randn(4, 32)
    output = decoder(_make_transition(state=torch.randn(4, 8), action=action_32))

    assert output[TransitionKey.ACTION].shape == (4, 32)
    torch.testing.assert_close(output[TransitionKey.ACTION], action_32)
