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

"""CPU tests of the SO-100 / SO-101 rig (``embodiment: so101``) on a tiny random model.

The parity tests drive the LeRobot policy and the praxis repository's ``Session`` with its
``So101ServeSession`` (the ``praxis`` package, when it is installed) with the same weights, fake
frozen parts and observations, and require bitwise equal joint targets.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.praxis.configuration_praxis import SO101_JOINT_DIM, SO101_JOINT_NAMES, PraxisConfig
from lerobot.policies.praxis.praxis_core import so101
from lerobot.policies.praxis.praxis_core.checkpoint import CheckpointError, model_variant, validate_config
from lerobot.policies.praxis.processor_praxis import (
    PraxisCommandProcessorStep,
    PraxisPoseProcessorStep,
    PraxisTileViewsProcessorStep,
    tile_views_so101,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .test_praxis import _ReferencePipeline
from .tiny_model import (
    SO101_CAMERA_KEYS,
    SO101_REST,
    fake_so101_observation,
    make_tiny_so101_policy,
    tiny_so101_lerobot_config,
)

pytest.importorskip("transformers", reason="praxis requires the praxis extra (transformers)")

CONFIG_DIR = Path(__file__).parent / "checkpoint_configs"
SO101_CONFIG = CONFIG_DIR / "praxis-1-so101.json"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _uint8_hwc(frame: torch.Tensor) -> np.ndarray:
    """A LeRobot image ``(3, H, W)`` in [0, 1] as the uint8 ``(H, W, 3)`` bytes of the camera."""
    return (frame * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


def _reference_obs(observations: list[dict], n_views: int) -> dict:
    """The praxis so101 observation stack of a list of LeRobot observations (the last 25)."""
    rows = observations[-25:]
    obs: dict = {}
    for view, key in zip(so101.VIEWS, SO101_CAMERA_KEYS[:n_views], strict=False):
        obs[so101.VIEW_KEYS[view]] = np.stack([_uint8_hwc(o[key][0]) for o in rows])
    obs[so101.STATE_KEY] = np.stack([o[OBS_STATE][0].numpy() for o in rows])
    obs["task"] = rows[-1]["task"][0]
    return obs


def _reference_session(policy, n_views: int):  # noqa: ANN001, ANN202 (the praxis package is optional)
    """The praxis repository's ``Session`` on the so101 rig with the policy's parts, or skip."""
    pipeline = pytest.importorskip("praxis.pipeline", reason="the praxis package gives the reference Session")
    ref_so101 = pytest.importorskip(
        "praxis.so101", reason="the praxis package has no so101 rig before afef2e1"
    )
    reference_pipe = _ReferencePipeline(policy)
    reference_pipe.embodiment = "so101"
    grip = ref_so101.GripRange(float(policy.config.grip_closed), float(policy.config.grip_open))
    return pipeline.Session(
        reference_pipe,
        prompt="",
        seed=policy.config.seed,
        max_context_chunks=1,
        control_mode="none",
        exec_horizon=24,
        rig_options={"n_views": n_views, "grip": grip},
    )  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Checkpoint config and features
# ---------------------------------------------------------------------------


def test_so101_checkpoint_config_validates() -> None:
    """The so101 checkpoint's config.json loads unchanged: the 6-D joint head and the fit-rule tile."""
    config = json.loads(SO101_CONFIG.read_text())
    validate_config(config)
    variant = model_variant(config)
    assert variant.tile_size == so101.TILE_SIZE and variant.antialias is True
    assert so101.canvas_size(config, 2) == (288, 832) and so101.canvas_size(config, 1) == (288, 416)
    assert config["normalization"]["domain"] == "so101"
    policy_config = PraxisConfig(model_config=config)
    assert policy_config.embodiment == "so101" and policy_config.model_config == config
    assert (policy_config.state_dim, policy_config.pose_dim, policy_config.command_dim) == (6, 6, 6)


def test_so101_config_refusals() -> None:
    config = json.loads(SO101_CONFIG.read_text())
    with pytest.raises(CheckpointError, match="pose_frame 'flange' \\(the so101 rig has 'joints'\\)"):
        validate_config({**config, "pose_frame": "flange"})
    with pytest.raises(CheckpointError, match="action_space.dim 6"):
        validate_config({**config, "action_space": {**config["action_space"], "dim": 10}})
    video = {k: v for k, v in config["video"].items() if k != "tile_bounds_rows_cols"}
    with pytest.raises(CheckpointError, match="the so101 rig has the views"):
        validate_config({**config, "video": {**video, "views": ["wrist", "exterior"]}})


@pytest.mark.parametrize("n_views", [1, 2])
def test_so101_features(n_views: int) -> None:
    config = tiny_so101_lerobot_config(n_views=n_views)
    config.validate_features()
    assert config.image_keys == list(SO101_CAMERA_KEYS[:n_views])
    assert len(SO101_JOINT_NAMES) == SO101_JOINT_DIM == 6


def test_so101_feature_refusals() -> None:
    three = (*SO101_CAMERA_KEYS, "observation.images.top")
    config = tiny_so101_lerobot_config(
        camera_keys=list(three),
        input_features={
            **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for k in three},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
    )
    with pytest.raises(ValueError, match="reads 1 or 2 views"):
        config.validate_features()
    config = tiny_so101_lerobot_config(
        input_features={
            **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for k in SO101_CAMERA_KEYS},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
    )
    with pytest.raises(ValueError, match=r"observation.state must have shape \(6,\)"):
        config.validate_features()
    with pytest.raises(ValueError, match="must differ by"):
        tiny_so101_lerobot_config(grip_closed=20.0, grip_open=20.5)
    # A gripper whose reading grows as it closes is allowed.
    assert tiny_so101_lerobot_config(grip_closed=42.0, grip_open=2.0).grip_closed == 42.0


# ---------------------------------------------------------------------------
# Units and canvas
# ---------------------------------------------------------------------------


def test_joint_units_round_trip() -> None:
    grip = so101.GripRange(closed=1.4, open=42.1)
    state = np.array([[0.0, -100.0, 95.0, 70.0, 0.0, 1.4], [10.0, -20.0, 30.0, -40.0, 50.0, 42.1]])
    rows = so101.lerobot_to_model(state, grip)
    np.testing.assert_allclose(rows[:, :5], np.deg2rad(state[:, :5]))
    np.testing.assert_array_equal(rows[:, 5], [1.0, 0.0])  # closed -> closure 1, open -> 0
    np.testing.assert_allclose(so101.model_to_lerobot(rows, grip), state)
    anchor = rows[0]
    np.testing.assert_allclose(so101.rel_to_abs(so101.abs_to_rel(rows, anchor), anchor), rows)


def test_state_must_be_degrees() -> None:
    """A raw motor tick (not ``use_degrees=True``) is refused with the pointer to the setting."""
    policy = make_tiny_so101_policy(seed=2, n_views=1)
    policy.reset()
    obs = fake_so101_observation(0, n_views=1)
    obs[OBS_STATE] = torch.tensor([[2048.0, 1024.0, 3000.0, 2048.0, 2048.0, 30.0]])
    with pytest.raises(ValueError, match="use_degrees=True"):
        policy.select_action(obs)


@pytest.mark.parametrize(
    ("sizes", "tiles"),
    [
        ([(480, 640), (480, 640)], [(288, 416), (288, 416)]),  # the native views: 312 x 416, trimmed
        ([(480, 640)], [(288, 416)]),
        ([(720, 1280), (480, 640)], [(224, 416), (224, 288)]),  # one common height, then the 32 px grid
        ([(240, 320)], [(224, 320)]),  # never scaled up
    ],
)
def test_tile_step_is_the_fit_rule_and_its_tiles_pass_through(
    sizes: list[tuple[int, int]], tiles: list[tuple[int, int]]
) -> None:
    """The tile step gives the bytes of the session's canvas, and its tiles are a fixed point of the rule."""
    rng = np.random.default_rng(3)
    raw = [rng.integers(0, 256, (2, h, w, 3), dtype=np.uint8) for h, w in sizes]
    keys = list(SO101_CAMERA_KEYS[: len(sizes)])
    frames = {
        k: torch.from_numpy(r).permute(0, 3, 1, 2).float() / 255 for k, r in zip(keys, raw, strict=True)
    }
    out = tile_views_so101(frames)
    assert [tuple(out[k].shape[-2:]) for k in keys] == tiles
    assert so101.fit_tile_sizes(sizes) == tiles and so101.is_fit_canvas(tiles)
    want = so101.canvas_clip(raw)  # (1, 3, T, H, W) float with uint8 levels
    got = torch.cat([out[k] for k in keys], dim=-1) * 255
    assert torch.equal(got.round().permute(1, 0, 2, 3).unsqueeze(0), want)
    # The tiles go through the rule unchanged, as float frames or as the session's uint8 tensors.
    again = tile_views_so101(out)
    assert all(torch.equal(again[k], out[k]) for k in keys)
    as_tiles = [(out[k] * 255).round().to(torch.uint8) for k in keys]
    assert torch.equal(so101.canvas_clip(as_tiles), want)


def test_canvas_refuses_tensors_that_are_not_tiles() -> None:
    with pytest.raises(ValueError, match="not tiles of the fit rule"):
        so101.canvas_clip([torch.zeros(1, 3, 300, 416, dtype=torch.uint8)])
    with pytest.raises(ValueError, match="must all be prepared tiles or all raw frames"):
        so101.canvas_clip(
            [torch.zeros(1, 3, 288, 416, dtype=torch.uint8), np.zeros((1, 288, 416, 3), np.uint8)]
        )


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def test_so101_processors_skip_the_pose_and_command_steps() -> None:
    """The so101 state and actions are the follower's joints in LeRobot's units: no pose or command step."""
    config = tiny_so101_lerobot_config(n_views=2)
    pre, post = make_pre_post_processors(config, None)
    assert not any(isinstance(s, PraxisPoseProcessorStep | PraxisCommandProcessorStep) for s in pre.steps)
    assert not any(isinstance(s, PraxisCommandProcessorStep) for s in post.steps)
    step = next(s for s in pre.steps if isinstance(s, PraxisTileViewsProcessorStep))
    assert (step.fit_rule, step.tile_height, step.tile_width) == ("so101", 288, 416)
    targets = torch.tensor([[1.0, -99.0, 94.0, 69.0, 0.5, 20.0]])
    assert torch.equal(post(targets), targets)


@pytest.mark.parametrize("n_views", [2, 1])
def test_select_action_matches_the_praxis_so101_session_bitwise(n_views: int) -> None:
    """The LeRobot glue (tiles, joints, history, seeds) gives the praxis joint targets, bit for bit."""
    policy = make_tiny_so101_policy(seed=1, n_views=n_views)
    reference = _reference_session(policy, n_views)
    pre, post = make_pre_post_processors(policy.config, None)
    observations: list[dict] = []
    policy.reset()
    actions: list[torch.Tensor] = []
    for step in range(22):  # 21 rows of the first chunk, then the first row of the second
        obs = fake_so101_observation(step, n_views=n_views)
        observations.append(obs)
        command = post(policy.select_action(pre(obs)))
        assert command.shape == (1, SO101_JOINT_DIM)
        actions.append(command[0])
    res0 = reference.step(_reference_obs(observations[:1], n_views))
    res1 = reference.step(_reference_obs(observations[:22], n_views))
    assert len(res0.commands) == 21 and len(res1.commands) == 24 and res1.depth == 1
    expected = [c["joints"] for c in res0.commands] + [res1.commands[0]["joints"]]
    for step, (got, want) in enumerate(zip(actions, expected, strict=True)):
        np.testing.assert_array_equal(got.numpy(), want, err_msg=f"step {step}")
    assert policy.sessions()[0].state.stream.depth == 1


def test_so101_raw_frames_match_the_pipeline_path() -> None:
    """Called without the pipeline, on raw 640 x 480 frames, the policy gives the pipeline's targets."""
    with_pipe = make_tiny_so101_policy(seed=4)
    raw = make_tiny_so101_policy(seed=4)
    pre, post = make_pre_post_processors(with_pipe.config, None)
    with_pipe.reset()
    raw.reset()
    for step in range(22):
        obs = fake_so101_observation(step, seed=1)
        a = post(with_pipe.select_action(pre(obs)))
        b = raw.select_action(obs)
        assert torch.equal(a, b), f"step {step}"
    # The preprocessor's tiles stayed tensors; the raw frames became the bytes of the canvas rule.
    assert isinstance(with_pipe.sessions()[0].history[-1][0][0], torch.Tensor)
    assert isinstance(raw.sessions()[0].history[-1][0][0], np.ndarray)


def test_so101_chunk_api_and_prediction_split() -> None:
    policy = make_tiny_so101_policy(seed=5)
    policy.reset()
    chunk = policy.predict_action_chunk(fake_so101_observation(0))
    assert chunk.shape == (1, 24, SO101_JOINT_DIM)
    assert torch.equal(chunk[0, 20], chunk[0, 23])  # 21 real rows, padded with the last one
    # The predicted frames split at the width of each view: a 1280 x 720 and a 640 x 480 camera.
    shapes = {SO101_CAMERA_KEYS[0]: (3, 720, 1280), SO101_CAMERA_KEYS[1]: (3, 480, 640)}
    policy = make_tiny_so101_policy(seed=5, return_prediction=True)
    policy.reset()
    obs = fake_so101_observation(0)
    rng = np.random.default_rng(0)
    for key, shape in shapes.items():
        obs[key] = torch.from_numpy(rng.integers(0, 256, shape, dtype=np.uint8)).float()[None] / 255
    out = policy.select_action(obs)
    assert isinstance(out, dict) and tuple(out[ACTION].shape) == (1, SO101_JOINT_DIM)
    frames = out["prediction"]["observation"]
    assert [tuple(frames[k].shape[-2:]) for k in SO101_CAMERA_KEYS] == [(224, 416), (224, 288)]


def test_so101_training_is_refused_with_a_message() -> None:
    policy = make_tiny_so101_policy(seed=6)
    batch = {OBS_STATE: torch.zeros(1, 93, 6), ACTION: torch.zeros(1, 93, 6), "task": ["x"]}
    with pytest.raises(NotImplementedError, match="no so101 training recipe"):
        policy.train().forward(batch)


def test_so101_targets_start_at_the_rest_pose_units() -> None:
    """The targets are absolute joints in degrees and a gripper.pos between the calibrated readings."""
    policy = make_tiny_so101_policy(seed=7, n_views=1, grip_closed=2.0, grip_open=40.0)
    policy.reset()
    chunk = policy.predict_action_chunk(fake_so101_observation(0, n_views=1))[0].numpy()
    assert np.all((chunk[:, 5] >= 2.0) & (chunk[:, 5] <= 40.0))
    # The joints are the anchor (the rest pose) plus a bounded move: the q99 range of the head.
    q01 = np.array(policy.config.model_config["normalization"]["action_q01"][:5])
    q99 = np.array(policy.config.model_config["normalization"]["action_q99"][:5])
    move = np.deg2rad(chunk[:, :5] - np.array(SO101_REST[:5]))
    assert np.all(move >= q01 - 1e-5) and np.all(move <= q99 + 1e-5)


# ---------------------------------------------------------------------------
# The converter
# ---------------------------------------------------------------------------


def _converter():
    path = REPO_ROOT / "examples" / "praxis" / "convert_checkpoint.py"
    spec = importlib.util.spec_from_file_location("praxis_convert_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("camera_keys", [None, ["observation.images.front"]])
def test_converter_writes_the_so101_features_and_serve_settings(camera_keys: list[str] | None) -> None:
    config = _converter().build_config(
        json.loads(SO101_CONFIG.read_text()), camera_keys=camera_keys, device="cpu"
    )
    keys = camera_keys or list(SO101_CAMERA_KEYS)
    assert config.image_keys == keys
    assert {k: tuple(v.shape) for k, v in config.image_features.items()} == dict.fromkeys(keys, (3, 480, 640))
    assert tuple(config.robot_state_feature.shape) == (6,) and tuple(config.action_feature.shape) == (6,)
    # The SO-101 serve preset of the praxis repository (configs/serve_so101.yaml).
    assert (config.num_steps, config.t_shift, config.cfg_scale, config.cfg_video_scale) == (4, 12.5, 1.0, 4.0)
    assert config.max_context_chunks == 1
