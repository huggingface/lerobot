# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""VLA-JEPA's real preparation/processor contract without backbone downloads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.inference import ExecutionMode, FeatureSpec, ObservationSnapshot, PolicyRunner
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors
from lerobot.policies.vla_jepa.qwen_interface import Qwen3VLInterface
from lerobot.utils.constants import ACTION, OBS_STATE


class FixedBackend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qwen = SimpleNamespace(to_pixel_values=Qwen3VLInterface.to_pixel_values)
        self.inputs = None

    def predict_action(self, images, instructions, state):
        self.inputs = (images, instructions, state)
        return (
            torch.arange(self.config.chunk_size * self.config.action_dim, dtype=torch.float32).reshape(
                1, self.config.chunk_size, self.config.action_dim
            )
            / 100
        )


@pytest.fixture
def setup():
    names = tuple(
        f"{side}_{joint}.pos"
        for side in ("right", "left")
        for joint in (*[f"joint_{i}" for i in range(1, 8)], "gripper")
    )
    cameras = ("base", "right_wrist", "left_wrist")
    config = VLAJEPAConfig(
        device="cpu",
        enable_world_model=True,
        chunk_size=30,
        n_action_steps=9,
        action_dim=16,
        state_dim=16,
        use_relative_actions=True,
        action_feature_names=list(names),
        resize_images_to=(16, 16),
        input_features={
            **{
                f"observation.images.{key}": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)) for key in cameras
            },
            OBS_STATE: PolicyFeature(FeatureType.STATE, (16,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (16,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )
    policy = VLAJEPAPolicy.__new__(VLAJEPAPolicy)
    PreTrainedPolicy.__init__(policy, config)
    policy.model = FixedBackend(config)
    policy.reset()
    features = tuple(
        FeatureSpec(f"observation.images.{key}", (24 + i * 8, 32, 3), "uint8", kind="rgb", semantics=key)
        for i, key in enumerate(cameras)
    ) + (FeatureSpec(OBS_STATE, (16,), "float32", names=names, semantics="ordered joints"),)
    action = FeatureSpec(ACTION, (16,), "float32", names=names, semantics="ordered joints")
    stats = {
        OBS_STATE: {"mean": torch.ones(16), "std": torch.full((16,), 2.0)},
        ACTION: {"mean": torch.full((16,), 0.25), "std": torch.full((16,), 3.0)},
    }
    runner = PolicyRunner(
        policy,
        *make_vla_jepa_pre_post_processors(config, stats),
        action_interval=1 / 30,
        features=features,
        action_feature=action,
    )
    return policy, runner, stats


def observation(runner, state=10.0):
    return ObservationSnapshot(
        {
            feature.name: np.full(
                feature.shape, state if feature.kind == "tensor" else 127, dtype=feature.dtype
            )
            for feature in runner.capabilities.features
        },
        capture_time=10.0,
        task="Fold the T-shirt properly",
    )


def test_world_model_checkpoint_uses_current_frames_and_matches_local_processors(setup):
    policy, runner, stats = setup
    assert policy.config.observation_delta_indices == [0, 4, 8, 12, 16, 20, 24, 28]
    assert policy.config.enable_world_model
    assert runner.capabilities.prediction_steps == 30
    assert runner.capabilities.execution_steps == 9
    assert runner.capabilities.modes == policy.chunk_inference_spec().modes == (ExecutionMode.CHUNK,)
    assert not runner.capabilities.language
    source = observation(runner)
    local_pre, local_post = make_vla_jepa_pre_post_processors(policy.config, stats)
    local_batch = local_pre(
        prepare_observation_for_inference(
            {name: value.copy() for name, value in source.features.items()}, torch.device("cpu"), source.task
        )
    )
    local_inputs = policy._prepare_model_inputs(local_batch, training=False)
    assert "videos" not in local_inputs
    expected = local_post(policy.predict_action_chunk(local_batch))[0, :9]
    actual = runner.predict(source)
    torch.testing.assert_close(actual.canonical_actions, expected, rtol=0, atol=0)
    images, instructions, state = policy.model.inputs
    assert instructions == [source.task]
    assert state.shape == (1, 1, 16)
    for remote, local in zip(images[0], local_inputs["images"][0], strict=True):
        assert remote.shape == (3, 16, 16)
        torch.testing.assert_close(remote, local, rtol=0, atol=0)
    # Replanning uses the new raw-state anchor; excluded grippers remain absolute.
    next_actions = runner.predict(observation(runner, state=20.0)).canonical_actions
    delta = next_actions - actual.canonical_actions
    expected_delta = torch.full_like(delta, 10.0)
    expected_delta[:, [7, 15]] = 0
    torch.testing.assert_close(delta, expected_delta)
    runner.reset()
    torch.testing.assert_close(runner.predict(source).canonical_actions, actual.canonical_actions)
