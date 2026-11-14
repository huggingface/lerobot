#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""The bimanual (YAM) rig: three views, two arms, the praxis bimanual session, bit for bit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.praxis.configuration_praxis import (
    BIMANUAL_CAMERA_KEYS,
    PRAXIS_ACTION_DIM,
    PRAXIS_COMMAND_DIM,
)
from lerobot.policies.praxis.praxis_core import bimanual, single_arm
from lerobot.policies.praxis.praxis_core.checkpoint import validate_config
from lerobot.policies.praxis.processor_praxis import droid7_to_pose8, pose12_grip_to_droid7
from lerobot.utils.constants import ACTION, OBS_STATE

from .test_praxis import _ReferencePipeline
from .tiny_model import (
    BIMANUAL_IMAGE_SHAPE,
    TILE,
    fake_bimanual_observation,
    make_tiny_bimanual_policy,
    tiny_bimanual_lerobot_config,
    tiny_bimanual_praxis_config,
)

pytest.importorskip("transformers", reason="praxis requires the praxis extra (transformers)")

CONFIGS = Path(__file__).parent / "checkpoint_configs"


def _reference_obs(observations: list[dict]) -> dict:
    """The praxis bimanual observation stack of a list of LeRobot observations (the last 25)."""
    rows = observations[-25:]
    obs: dict = {}
    for view, key in zip(bimanual.VIEWS, BIMANUAL_CAMERA_KEYS, strict=True):
        obs[bimanual.VIEW_KEY_CANDIDATES[view][0]] = np.stack(
            [(o[key][0] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy() for o in rows]
        )
    state = np.stack([o[OBS_STATE][0].numpy() for o in rows])
    for i, side in enumerate(bimanual.SIDES):
        pose8 = droid7_to_pose8(state[:, 7 * i : 7 * i + 7])
        obs[bimanual.pose_keys(side)[0]] = pose8[:, :7]
        obs[bimanual.grip_keys(side)[0]] = pose8[:, 7]
    obs["task"] = rows[-1]["task"][0]
    return obs


def _commands_to_droid14(commands: list[dict]) -> list[np.ndarray]:
    return [
        np.concatenate(
            [
                pose12_grip_to_droid7(c[f"robot_command.{s}"]["pose"], c[f"target_grip.{s}"])
                for s in bimanual.SIDES
            ]
        )
        for c in commands
    ]


def test_bimanual_config_dims_and_features() -> None:
    config = tiny_bimanual_lerobot_config()
    assert config.embodiment == "bimanual" and config.n_arms == 2
    assert (config.state_dim, config.pose_dim, config.command_dim) == (14, 16, 26)
    assert config.image_keys == list(BIMANUAL_CAMERA_KEYS)
    with pytest.raises(ValueError, match="camera inputs"):
        bad = tiny_bimanual_lerobot_config()
        bad.input_features.pop("observation.images.wrist_right")
        bad.validate_features()
    with pytest.raises(ValueError, match="14 values"):
        tiny_bimanual_lerobot_config(
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        ).validate_features()


def test_yam_checkpoint_config_validates() -> None:
    yam = json.loads((CONFIGS / "praxis-1-yam.json").read_text())
    validate_config(yam)
    assert yam["embodiment"] == "bimanual" and bimanual.canvas_size(yam) == (256, 3 * 416)


def test_tile_step_follows_the_yam_fit_rule() -> None:
    """The preprocessor's tiles are the bimanual canvas of the praxis session, bit for bit."""
    config = tiny_bimanual_lerobot_config()
    pre, _ = make_pre_post_processors(config, None)
    obs = fake_bimanual_observation(3)
    tiles = pre(obs)
    canvas = bimanual.canvas_clip(
        bimanual.view_stacks(_reference_obs([obs])), TILE[0], TILE[1], 0.03, "cpu", True
    )  # (1, 3, 1, h, 3w)
    for v, key in enumerate(BIMANUAL_CAMERA_KEYS):
        got = (tiles[key][0] * 255).round()
        want = canvas[0, :, 0, :, v * TILE[1] : (v + 1) * TILE[1]]
        assert got.shape == (3, *TILE)
        assert torch.equal(got, want), key
    # a 14-value state becomes the 16-value pose rows, each arm through the DROID conversion
    state = tiles[OBS_STATE][0].numpy()
    want = np.concatenate(
        [droid7_to_pose8(obs[OBS_STATE][0].numpy()[:7]), droid7_to_pose8(obs[OBS_STATE][0].numpy()[7:])]
    )
    np.testing.assert_array_equal(state, want)


def test_select_action_matches_the_praxis_bimanual_session_bitwise() -> None:
    """The two-arm glue (three views, pose16, the bimanual session, seeds) gives the praxis commands."""
    pipeline = pytest.importorskip("praxis.pipeline", reason="the praxis package gives the reference Session")
    policy = make_tiny_bimanual_policy(seed=1, n_action_steps=24, max_context_chunks=1)
    reference_pipe = _ReferencePipeline(policy)
    reference_pipe.embodiment = "bimanual"
    reference = pipeline.Session(
        reference_pipe,
        prompt="",
        seed=policy.config.seed,
        max_context_chunks=1,
        control_mode="none",
        exec_horizon=24,
    )  # type: ignore[arg-type]
    pre, post = make_pre_post_processors(policy.config, None)
    observations: list[dict] = []
    policy.reset()
    actions: list[torch.Tensor] = []
    for step in range(22):
        obs = fake_bimanual_observation(step)
        observations.append(obs)
        command = policy.select_action(pre(obs))
        assert command.shape == (1, 2 * PRAXIS_COMMAND_DIM)
        actions.append(post(command)[0])
    res0 = reference.step(_reference_obs(observations[:1]))
    res1 = reference.step(_reference_obs(observations[:22]))
    assert len(res0.commands) == 21 and len(res1.commands) == 24 and res1.depth == 1
    expected = _commands_to_droid14(res0.commands) + _commands_to_droid14(res1.commands[:1])
    for step, (got, want) in enumerate(zip(actions, expected, strict=True)):
        assert got.shape == (2 * PRAXIS_ACTION_DIM,)
        np.testing.assert_array_equal(got.numpy(), want, err_msg=f"step {step}")
    assert policy.sessions()[0].state.stream.depth == 1


def test_bimanual_raw_frames_match_the_pipeline_path() -> None:
    """Without the pipelines the policy makes the canvas itself and reads the 14-value state."""
    policy = make_tiny_bimanual_policy(seed=2)
    pre, post = make_pre_post_processors(policy.config, None)
    policy.reset()
    through_pipeline = [post(policy.select_action(pre(fake_bimanual_observation(s))))[0] for s in range(3)]
    policy.reset()
    raw = []
    for s in range(3):
        cmd = policy.select_action(fake_bimanual_observation(s))[0].numpy()
        raw.append(
            np.concatenate(
                [pose12_grip_to_droid7(cmd[13 * i : 13 * i + 12], float(cmd[13 * i + 12])) for i in range(2)]
            )
        )
    for a, b in zip(through_pipeline, raw, strict=True):
        np.testing.assert_array_equal(a.numpy(), b)


def test_bimanual_chunk_api_and_prediction_split() -> None:
    policy = make_tiny_bimanual_policy(seed=3, return_prediction=True)
    policy.reset()
    chunk = policy.predict_action_chunk(fake_bimanual_observation(0))
    assert chunk.shape == (1, 24, 2 * PRAXIS_COMMAND_DIM)
    policy.reset()
    out = policy.select_action(fake_bimanual_observation(0))
    assert isinstance(out, dict)
    frames = out["prediction"]["observation"]
    assert set(frames) == set(BIMANUAL_CAMERA_KEYS)
    assert all(f.shape == (1, 3, *TILE) for f in frames.values())


def test_rel20_to_pose12_composes_each_arm_with_its_own_anchor() -> None:
    """Each arm's rows are the single-arm composition against that arm's pose, bit for bit."""
    rng = np.random.default_rng(5)
    rel = (0.1 * rng.standard_normal((24, 20))).astype(np.float32)
    rel[:, [9, 19]] = rng.uniform(0.0, 1.0, (24, 2)).astype(np.float32)
    quats = rng.standard_normal((2, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    pose16 = np.concatenate([[0.4, 0.2, 0.3], quats[0], [0.1], [0.4, -0.2, 0.3], quats[1], [0.9]])
    arms = bimanual.rel20_to_pose12(rel, pose16.astype(np.float32))
    assert len(arms) == len(bimanual.SIDES)
    for i, (poses, grips) in enumerate(arms):
        assert poses.shape == (24, 12) and grips.shape == (24,)
        assert poses.dtype == grips.dtype == np.float32
        want_poses, want_grips = single_arm.rel10_to_pose12(
            rel[:, 10 * i : 10 * i + 10], pose16[8 * i : 8 * i + 8].astype(np.float32)
        )
        np.testing.assert_array_equal(poses, want_poses)
        np.testing.assert_array_equal(grips, want_grips)
    with pytest.raises(ValueError, match="rel actions"):
        bimanual.rel20_to_pose12(rel[:, :10], pose16)


def test_bimanual_training_is_refused_with_a_message() -> None:
    policy = make_tiny_bimanual_policy(seed=4)
    policy.train()
    with pytest.raises(NotImplementedError, match="single-arm"):
        policy.forward({})


def test_bimanual_tiny_config_validates() -> None:
    validate_config(tiny_bimanual_praxis_config())
    assert BIMANUAL_IMAGE_SHAPE[1:] == (40, 80)
