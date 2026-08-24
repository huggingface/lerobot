#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features


def test_tactile_feature_typing():
    features = {
        "observation.state": {"dtype": "float32", "shape": (10,), "names": None},
        "observation.tactile.sensor_1": {"dtype": "int16", "shape": (6, 6), "names": ["rows", "columns"]},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    policy_features = dataset_to_policy_features(features)
    assert policy_features["observation.tactile.sensor_1"].type is FeatureType.TACTILE
    assert policy_features["observation.tactile.sensor_1"].shape == (6, 6)
    assert policy_features["observation.state"].type is FeatureType.STATE


def _tactile_input_features(shapes):
    feats = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    for key, shape in shapes.items():
        feats[key] = PolicyFeature(type=FeatureType.TACTILE, shape=shape)
    return feats


def test_tactile_config_property_and_validation():
    input_features = _tactile_input_features(
        {"observation.tactile.sensor_1": (6, 6), "observation.tactile.sensor_2": (4, 8)}
    )
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        use_tactile=True,
    )
    assert list(config.tactile_features) == ["observation.tactile.sensor_1", "observation.tactile.sensor_2"]
    assert config.normalization_mapping["TACTILE"].value == "MEAN_STD"

    with pytest.raises(ValueError):
        ACTConfig(
            input_features=input_features,
            output_features=output_features,
            use_tactile=True,
            tactile_encoder_type="bogus",
        )

    # use_tactile enabled but no tactile features present: construction succeeds
    # (features are attached from the dataset later); validate_features() is what raises.
    no_tactile = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
            f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        output_features=output_features,
        use_tactile=True,
    )
    with pytest.raises(ValueError):
        no_tactile.validate_features()

    # Tactile as the sole input modality must be accepted (no images / state / env_state).
    tactile_only = ACTConfig(
        input_features={
            "observation.tactile.sensor_1": PolicyFeature(type=FeatureType.TACTILE, shape=(6, 6))
        },
        output_features=output_features,
        use_tactile=True,
    )
    tactile_only.validate_features()  # must not raise
    assert list(tactile_only.tactile_features) == ["observation.tactile.sensor_1"]


_B = 2
_CHUNK = 5
_ACTION_DIM = 6
_STATE_DIM = 10
_IMG = (3, 96, 96)
_SHAPES = {
    "single": {"observation.tactile.sensor_1": (6, 6)},
    "multi": {"observation.tactile.sensor_1": (6, 6), "observation.tactile.sensor_2": (4, 8)},
    "tactile_only": {"observation.tactile.sensor_1": (6, 6)},
}


def _encoder_seq_len_hook(policy):
    captured = {}

    def hook(module, args, kwargs):
        captured["seq"] = args[0].shape[0]

    handle = policy.model.encoder.register_forward_pre_hook(hook, with_kwargs=True)
    return captured, handle


def _build_batch(shapes, with_image_state):
    batch = {
        ACTION: torch.randn(_B, _CHUNK, _ACTION_DIM),
        "action_is_pad": torch.zeros(_B, _CHUNK, dtype=torch.bool),
    }
    if with_image_state:
        batch[OBS_STATE] = torch.randn(_B, _STATE_DIM)
        batch[f"{OBS_IMAGES}.cam"] = torch.rand(_B, *_IMG)
    for key, shape in shapes.items():
        batch[key] = torch.randint(0, 4096, (_B, *shape), dtype=torch.int16)
    return batch


@pytest.mark.parametrize("encoder_type", ["cnn", "attention"])
@pytest.mark.parametrize("scenario", ["single", "multi", "tactile_only"])
@pytest.mark.parametrize("n_tactile_tokens", [1, 4])
def test_act_tactile_forward(encoder_type, scenario, n_tactile_tokens):
    shapes = _SHAPES[scenario]
    with_image_state = scenario != "tactile_only"
    use_vae = with_image_state  # no proprioceptive state in tactile_only -> skip VAE encoder

    input_features = {}
    if with_image_state:
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,))
        input_features[f"{OBS_IMAGES}.cam"] = PolicyFeature(type=FeatureType.VISUAL, shape=_IMG)
    for key, shape in shapes.items():
        input_features[key] = PolicyFeature(type=FeatureType.TACTILE, shape=shape)
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))}

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=_CHUNK,
        n_action_steps=_CHUNK,
        use_tactile=True,
        tactile_encoder_type=encoder_type,
        n_tactile_tokens=n_tactile_tokens,
        use_vae=use_vae,
    )
    policy = ACTPolicy(config)
    policy.train()

    batch = _build_batch(shapes, with_image_state)
    captured, handle = _encoder_seq_len_hook(policy)
    loss, loss_dict = policy.forward(batch)
    handle.remove()
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict
    seq_with_tactile = captured["seq"]

    policy.eval()
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    assert actions.shape == (_B, _CHUNK, _ACTION_DIM)

    if with_image_state:
        base_features = {k: v for k, v in input_features.items() if v.type is not FeatureType.TACTILE}
        base_config = ACTConfig(
            input_features=base_features,
            output_features=output_features,
            chunk_size=_CHUNK,
            n_action_steps=_CHUNK,
            use_tactile=False,
            use_vae=use_vae,
        )
        base_policy = ACTPolicy(base_config)
        base_policy.train()
        base_batch = {k: v for k, v in batch.items() if not k.startswith("observation.tactile")}
        base_captured, base_handle = _encoder_seq_len_hook(base_policy)
        base_policy.forward(base_batch)
        base_handle.remove()
        expected_extra = len(shapes) * n_tactile_tokens
        assert seq_with_tactile - base_captured["seq"] == expected_extra


def test_act_default_off_has_no_tactile():
    """Default ACTConfig with use_tactile=False has no tactile encoders."""
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        use_tactile=False,
    )
    policy = ACTPolicy(config)
    assert policy.model.tactile_encoder_keys == []
    assert not hasattr(policy.model, "tactile_encoders")


def test_tactile_int16_normalization_pipeline():
    """Test int16 tactile tensors are safely normalized via the preprocessing pipeline."""
    # Build config with tactile enabled
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "observation.tactile.sensor_1": PolicyFeature(type=FeatureType.TACTILE, shape=(6, 6)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        use_tactile=True,
        n_tactile_tokens=2,
    )

    # Construct dataset_stats with per-cell tactile stats
    tactile_mean = torch.randn(6, 6) * 10.0  # Non-trivial values
    tactile_std = torch.abs(torch.randn(6, 6)) + 0.1  # Positive, some close to 0
    tactile_std[0, 0] = 1e-8  # Very small std to exercise eps path
    
    dataset_stats = {
        OBS_STATE: {"mean": torch.zeros(4), "std": torch.ones(4)},
        "observation.tactile.sensor_1": {"mean": tactile_mean, "std": tactile_std},
        ACTION: {"mean": torch.zeros(2), "std": torch.ones(2)},
    }

    # Get preprocessor
    preprocessor, _ = make_act_pre_post_processors(config, dataset_stats)
    
    # Create batch with int16 tactile data
    batch_size = 3
    tactile_int16 = torch.randint(-1000, 1000, (batch_size, 6, 6), dtype=torch.int16)
    batch = {
        OBS_STATE: torch.randn(batch_size, 4),
        "observation.tactile.sensor_1": tactile_int16,
        ACTION: torch.randn(batch_size, 2),
    }
    
    # Process the batch
    processed_batch = preprocessor(batch)
    processed_tactile = processed_batch["observation.tactile.sensor_1"]
    
    # Assertions
    # (a) is floating dtype
    assert torch.is_floating_point(processed_tactile), f"Expected float, got {processed_tactile.dtype}"
    
    # (b) is finite everywhere  
    assert torch.isfinite(processed_tactile).all(), "Processed tactile should be finite"
    
    # (c) matches expected normalization formula
    eps = 1e-8  # Default eps from NormalizerProcessorStep
    expected = (tactile_int16.float() - tactile_mean) / (tactile_std + eps)
    torch.testing.assert_close(processed_tactile, expected, atol=1e-6, rtol=1e-4)
