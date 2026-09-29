# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Processor boundaries, normalization alignment, camera preparation and saved state."""

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.flux3 import make_flux3_pre_post_processors
from lerobot.policies.flux3.processor_flux3 import (
    PAST_ACTIONS,
    ActionTargetNormalizerProcessorStep,
    CameraResizeProcessorStep,
    ObservationHistoryNormalizerProcessorStep,
    connect_history_processors,
)
from lerobot.processor.pipeline import ActionProcessorStep, ObservationProcessorStep, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.flux3.helpers import rich_task_config, single_config


def test_saved_frame_processors_keep_overrides_and_reconnect_relative_actions(tmp_path, monkeypatch):
    cfg = single_config(
        use_relative_actions=True,
        action_feature_names=["j0", "j1", "j2", "j3", "j4", "gripper"],
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )
    stats = {ACTION: {"mean": torch.zeros(6), "std": torch.ones(6)}}
    pre, post = make_flux3_pre_post_processors(cfg, stats)
    pre.save_pretrained(tmp_path, config_filename="input.json")
    post.save_pretrained(tmp_path, config_filename="output.json")
    load = PolicyProcessorPipeline.from_pretrained
    revisions = []

    def tracked_load(**kwargs):
        revisions.append(kwargs.get("revision"))
        return load(**kwargs)

    monkeypatch.setattr(PolicyProcessorPipeline, "from_pretrained", tracked_load)
    stats = {ACTION: {"mean": torch.ones(6), "std": torch.full((6,), 2.0)}}
    pre_overrides = {"normalizer_processor": {"stats": stats}}
    post_overrides = {"unnormalizer_processor": {"stats": stats}}
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=tmp_path,
        pretrained_revision="test-revision",
        preprocessor_config_filename="input.json",
        postprocessor_config_filename="output.json",
        preprocessor_overrides=pre_overrides,
        postprocessor_overrides=post_overrides,
    )
    assert revisions == ["test-revision", "test-revision"]
    assert pre_overrides == {"normalizer_processor": {"stats": stats}}
    assert post_overrides == {"unnormalizer_processor": {"stats": stats}}
    state = torch.arange(6.0)
    actions = state[None].repeat(32, 1) + 2
    batch = {
        OBS_STATE: state[None],
        ACTION: actions[None],
        "observation.images.top": torch.rand(1, 3, 64, 96),
    }
    prepared = pre(batch)
    torch.testing.assert_close(prepared[ACTION][..., :5], torch.full((1, 32, 5), 0.5))
    torch.testing.assert_close(post(prepared[ACTION]).reshape_as(actions), actions)


@pytest.mark.parametrize("history", [1, 8])
@pytest.mark.parametrize("representation", ["absolute", "delta"])
def test_history_steps_change_only_their_own_fields(history, representation):
    cfg = rich_task_config(
        n_obs_steps=history,
        history_snapshots=1,
        action_representation=representation,
        delta_absolute_dims=[-1] if representation == "delta" else [],
    )
    pre, _ = make_flux3_pre_post_processors(cfg)
    observation_step = next(s for s in pre.steps if isinstance(s, ObservationHistoryNormalizerProcessorStep))
    action_step = next(s for s in pre.steps if isinstance(s, ActionTargetNormalizerProcessorStep))
    assert isinstance(observation_step, ObservationProcessorStep)
    assert isinstance(action_step, ActionProcessorStep)
    commands = torch.arange(history + cfg.chunk_size).float()[None, :, None].repeat(2, 1, 6) / 4
    states = torch.full((2, history, 6), 0.4)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: states},
        TransitionKey.ACTION: commands,
        TransitionKey.REWARD: 7.0,
    }
    observed = observation_step(transition)
    assert observed[TransitionKey.ACTION] is commands
    assert transition[TransitionKey.OBSERVATION][OBS_STATE] is states and torch.all(states == 0.4)
    assert PAST_ACTIONS not in transition[TransitionKey.OBSERVATION]
    processed = action_step(observed)
    assert processed[TransitionKey.OBSERVATION] is observed[TransitionKey.OBSERVATION]
    assert processed[TransitionKey.REWARD] == 7.0
    expected = commands[:, 1:].clone()
    if representation == "delta":
        expected[..., :-1] -= commands[:, :-1, :-1]
    # The saved [-2, 2] quantiles map values to value / 2, with clipping at +/-6.
    expected = (expected / 2).clamp(-6, 6)
    torch.testing.assert_close(processed[TransitionKey.ACTION], expected[:, history - 1 :], rtol=0, atol=0)
    past = torch.cat([torch.zeros(2, 1, 6), expected[:, : history - 1]], 1)
    torch.testing.assert_close(observed[TransitionKey.OBSERVATION][PAST_ACTIONS], past, rtol=0, atol=1e-7)
    changed = commands.clone()
    changed[:, history:] += 100
    changed_observations = observation_step({**transition, TransitionKey.ACTION: changed})
    assert torch.equal(changed_observations[TransitionKey.OBSERVATION][PAST_ACTIONS], past)
    inference = {TransitionKey.OBSERVATION: {OBS_STATE: states}}
    assert action_step(inference)[TransitionKey.OBSERVATION] is inference[TransitionKey.OBSERVATION]


@pytest.mark.parametrize("mismatch", ["order", "quantiles", "length"])
def test_history_reconnect_rejects_inconsistent_steps(mismatch):
    cfg = rich_task_config()
    pre, post = make_flux3_pre_post_processors(cfg)
    action = next(s for s in pre.steps if isinstance(s, ActionTargetNormalizerProcessorStep))
    if mismatch == "order":
        pre.steps.remove(action)
        pre.steps.insert(0, action)
        match = "before action targets"
    elif mismatch == "quantiles":
        state = action.state_dict()
        state["action.q99"] += 1
        action.load_state_dict(state)
        match = "quantiles disagree"
    else:
        action.n_obs_steps += 1
        match = "settings disagree"
    with pytest.raises(ValueError, match=match):
        connect_history_processors(cfg, pre, post)


@pytest.mark.parametrize("frames", [None, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_grid_resize_preserves_values_and_saved_pipeline(tmp_path, frames, dtype):
    camera_shapes = {"observation.images.a": (3, 16, 32), "observation.images.b": (3, 32, 16)}
    cfg = single_config(
        camera_layout="grid",
        input_features={
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=shape)
                for key, shape in camera_shapes.items()
            },
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
    )
    pre, post = make_flux3_pre_post_processors(cfg)
    batch = {OBS_STATE: torch.ones(2, 6), ACTION: torch.ones(2, 32, 6)}
    for key, shape in camera_shapes.items():
        prefix = (2,) if frames is None else (2, frames)
        batch[key] = torch.full((*prefix, *shape), 127 if dtype == torch.uint8 else 0.5, dtype=dtype)
    result = pre(batch)
    for key in camera_shapes:
        assert result[key].shape == (*batch[key].shape[:-2], 32, 32)
        assert result[key].dtype == dtype and result[key].device == batch[key].device
        assert torch.all(result[key] == batch[key].flatten()[0])
        assert batch[key].shape[-2:] == camera_shapes[key][-2:]
    assert torch.equal(result[ACTION], batch[ACTION]) and torch.equal(result[OBS_STATE], batch[OBS_STATE])
    resize = next(s for s in pre.steps if isinstance(s, CameraResizeProcessorStep))
    features = {PipelineFeatureType.OBSERVATION: cfg.input_features}
    updated = resize.transform_features(features)[PipelineFeatureType.OBSERVATION]
    assert all(updated[key].shape == (3, 32, 32) for key in camera_shapes)
    assert all(cfg.input_features[key].shape == shape for key, shape in camera_shapes.items())
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    restored, _ = make_pre_post_processors(cfg, pretrained_path=tmp_path)
    torch.testing.assert_close(restored(batch), result, rtol=0, atol=0)
    cfg.camera_keys = ["observation.images.a"]
    restored, _ = make_pre_post_processors(cfg, pretrained_path=tmp_path)
    subset = restored({OBS_STATE: batch[OBS_STATE], cfg.camera_keys[0]: batch[cfg.camera_keys[0]]})
    assert torch.equal(subset[cfg.camera_keys[0]], batch[cfg.camera_keys[0]])


def test_grid_resize_leaves_equal_images_and_rejects_inconsistent_times():
    step = CameraResizeProcessorStep(["a", "b"], "grid")
    image = torch.rand(2, 3, 8, 16)
    transition = {TransitionKey.OBSERVATION: {"a": image, "b": image[:, None]}}
    result = step(transition)[TransitionKey.OBSERVATION]
    assert result["a"] is image and result["b"] is transition[TransitionKey.OBSERVATION]["b"]
    with pytest.raises(ValueError, match="share"):
        step({TransitionKey.OBSERVATION: {"a": image, "b": torch.rand(2, 2, 3, 8, 16)}})
