# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.factory import get_policy_class
from lerobot.policies.g05 import G05Config, G05Policy, make_g05_pre_post_processors
from lerobot.policies.g05.action_tokenizer import (
    G05ActionCodecConfig,
    G05ActionCodecModel,
    G05ActionTokenizer,
)
from lerobot.policies.g05.processor_g05 import (
    G05LiberoActionStep,
    G05LiberoObservationStep,
    G05StepwiseActionUnnormalizerStep,
    G05StepwiseNormalizerStep,
)
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def _test_config() -> G05Config:
    return G05Config(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        device="cpu",
        camera_keys=["observation.images.cam"],
        image_size=(32, 32),
        internal_action_dim=4,
        internal_state_dim=4,
        action_indices=[1, 3],
        state_indices=[0, 1, 3],
        vocab_size=100,
        image_token_id=2,
        state_token_id=3,
        eov_token_id=4,
        eos_token_id=5,
        text_hidden_size=32,
        text_intermediate_size=64,
        text_num_layers=4,
        text_num_heads=2,
        text_num_kv_heads=1,
        text_head_dim=16,
        text_layer_types=["linear_attention"] * 3 + ["full_attention"],
        mrope_section=(2, 1, 1),
        vision_depth=1,
        vision_hidden_size=32,
        vision_intermediate_size=64,
        vision_num_heads=2,
        expert_hidden_size=32,
        expert_intermediate_size=64,
        expert_num_layers=4,
        expert_num_heads=2,
        expert_num_kv_heads=1,
        expert_head_dim=16,
        chunk_size=2,
        n_action_steps=1,
        num_inference_steps=2,
        dtype="float32",
    )


def test_policy_training_and_action_contract() -> None:
    torch.manual_seed(1337)
    config = _test_config()
    assert get_policy_class(config.type) is G05Policy
    assert config.observation_delta_indices == list(range(1 - config.n_obs_steps, 1))
    assert config.action_delta_indices == list(range(config.chunk_size))
    config.validate_features()

    policy = G05Policy(config)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.linspace(-1.0, 1.0, 3 * 32 * 32).reshape(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
        ACTION: torch.linspace(-0.75, 0.75, 8).reshape(1, config.chunk_size, config.internal_action_dim),
    }

    loss, logs = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(isinstance(value, float) for value in logs.values())
    assert any(parameter.grad is not None for parameter in policy.parameters())

    noise = torch.zeros(1, config.chunk_size, config.internal_action_dim)
    action_chunk = policy.predict_action_chunk(batch, noise=noise)
    action_dim = config.output_features[ACTION].shape[-1]
    assert action_chunk.shape == (1, config.chunk_size, action_dim)
    assert policy.select_action(batch, noise=noise).shape == (1, action_dim)


def test_processor_factory_supports_base_and_stepwise_stats(monkeypatch) -> None:
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    dataset_stats = {
        OBS_STATE: {"q01": torch.full((3,), -1.0), "q99": torch.full((3,), 1.0)},
        ACTION: {"q01": torch.full((2,), -1.0), "q99": torch.full((2,), 1.0)},
    }

    base = _test_config()
    preprocessor, postprocessor = make_g05_pre_post_processors(
        base,
        dataset_stats,
        tokenizer_path="artifact/processor",
    )
    assert any(isinstance(step, NormalizerProcessorStep) for step in preprocessor.steps)
    assert any(isinstance(step, UnnormalizerProcessorStep) for step in postprocessor.steps)

    finetuned = _test_config()
    finetuned.normalization_strategy = "g05_stepwise"
    finetuned.state_normalization = [
        {"mode": "q01/q99", "width": 3, "stats": {"q01": [-1.0] * 3, "q99": [1.0] * 3}}
    ]
    finetuned.action_normalization = [
        {
            "mode": "q01/q99",
            "width": 2,
            "stats": {"q01": [[-1.0] * 2] * 2, "q99": [[1.0] * 2] * 2},
        }
    ]
    preprocessor, postprocessor = make_g05_pre_post_processors(
        finetuned,
        tokenizer_path="artifact/processor",
    )
    assert any(isinstance(step, G05StepwiseNormalizerStep) for step in preprocessor.steps)
    assert any(isinstance(step, G05StepwiseActionUnnormalizerStep) for step in postprocessor.steps)


def test_action_codec_grouped_token_roundtrip() -> None:
    config = G05ActionCodecConfig(
        max_component_dim=3,
        horizon=4,
        horizon_patch_size=2,
        conv_in_action_kernel=2,
        encoder_channels=8,
        latent_dim=4,
        c_mults=[1],
        strides=[[1, 1]],
        transformer_depths=[1],
        num_heads=1,
        dim_heads=32,
        use_block_dct=True,
        block_dct_block_size=2,
        n_codebooks=1,
        codebook_size=16,
        codebook_dim=2,
        parts_meta={"control": 2, "gripper": 1},
        num_residuals=1,
    )
    tokenizer = G05ActionTokenizer(G05ActionCodecModel(config))
    action = torch.tensor([[[0.1, 0.2, -1.0], [0.3, 0.4, -1.0], [0.5, 0.6, 1.0], [0.7, 0.8, 1.0]]])

    token_ids = tokenizer.encode_action_indices(action)
    decoded = tokenizer.decode_action_indices(token_ids)

    assert token_ids.shape[0] == action.shape[0]
    assert token_ids[0, 0] == tokenizer.marker_indices["<control_0>"]
    assert token_ids[0, -2] == tokenizer.marker_indices["<gripper>"]
    torch.testing.assert_close(decoded[..., -1], action[..., -1])


def test_libero_environment_boundary() -> None:
    observation = {
        "observation.images.image": torch.arange(12).reshape(1, 1, 3, 4),
        "observation.robot_state": {
            "eef": {
                "pos": torch.tensor([[1.0, 2.0, 3.0]]),
                "quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            },
            "gripper": {"qpos": torch.tensor([[0.03, -0.03]])},
        },
    }
    processed = G05LiberoObservationStep().observation(observation)
    assert processed[OBS_STATE].shape[-1] == 7
    torch.testing.assert_close(
        processed["observation.images.image"],
        torch.flip(observation["observation.images.image"], dims=(-2, -1)),
    )

    transition = {TransitionKey.ACTION: torch.tensor([[0.0, 0.49], [0.0, 0.51]])}
    action = G05LiberoActionStep()(transition)[TransitionKey.ACTION]
    torch.testing.assert_close(action[..., -1], torch.tensor([1.0, -1.0]))
