# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""GR00T serving horizons use the same bounds as its direct prediction wrapper."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


class FixedBackend(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(action_horizon=horizon)

    def get_action(self, inputs):
        return {"action_pred": torch.zeros(inputs["state"].shape[0], self.config.action_horizon, 132)}


def policy_for(
    tmp_path,
    *,
    chunk_size,
    n_action_steps,
    native_horizon,
    checkpoint_horizon=None,
    embodiment="new_embodiment",
):
    if checkpoint_horizon is not None:
        (tmp_path / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_kwargs": {
                        "modality_configs": {
                            embodiment: {"action": {"delta_indices": list(range(checkpoint_horizon))}}
                        }
                    }
                }
            )
        )
    config = GrootConfig(
        device="cpu",
        use_bf16=False,
        base_model_path=str(tmp_path),
        embodiment_tag=embodiment,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (6,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (6,))},
    )
    policy = GrootPolicy.__new__(GrootPolicy)
    PreTrainedPolicy.__init__(policy, config)
    policy._groot_model = FixedBackend(native_horizon)
    return policy


@pytest.mark.parametrize(
    "chunk,steps,native,checkpoint,embodiment,prediction,execution",
    [
        (16, 8, 40, None, "new_embodiment", 8, 8),  # supplied Super Chatton checkpoint
        (40, 40, 40, 16, "libero_sim", 16, 8),
        (40, 40, 12, None, "new_embodiment", 12, 12),
    ],
)
def test_declared_horizon_matches_direct_chunk_and_local_playback(
    tmp_path, chunk, steps, native, checkpoint, embodiment, prediction, execution
):
    policy = policy_for(
        tmp_path,
        chunk_size=chunk,
        n_action_steps=steps,
        native_horizon=native,
        checkpoint_horizon=checkpoint,
        embodiment=embodiment,
    )
    spec = policy.chunk_inference_spec()
    assert spec.prediction_steps == prediction
    assert spec.execution_steps == execution
    actions = policy.predict_action_chunk({"state": torch.zeros(1, 1, 132)})
    assert actions.shape == (1, prediction, 6)
    assert spec.execution_steps == min(prediction, policy._resolve_action_queue_steps())


def test_groot_keeps_temporal_and_unknown_native_horizon_rejections(tmp_path):
    policy = policy_for(tmp_path, chunk_size=16, n_action_steps=8, native_horizon=40)
    policy.config.n_obs_steps = 2
    with pytest.raises(ValueError, match="n_obs_steps=1"):
        policy.chunk_inference_spec()
    policy.config.n_obs_steps = 1
    policy._groot_model.config.action_horizon = None
    with pytest.raises(ValueError, match="positive native action_horizon"):
        policy.chunk_inference_spec()
