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

"""Tests for the SONIC whole-body controller.

No robot and no network: ``load_policy`` is stubbed with a fake ONNX session, so these
cover the parts that are easy to get silently wrong — the 994-D decoder input layout, the
IsaacLab/MuJoCo reordering, the token hold/neutral-seed logic, and the joint-target math.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from lerobot.robots.unitree_g1.controllers import sonic_whole_body as swb  # noqa: E402
from lerobot.robots.unitree_g1.g1_utils import (  # noqa: E402
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    NUM_MOTORS,
    G1_29_JointIndex,
    get_gravity_orientation,
)

HISTORY_LEN = 10

# Offsets of each block inside the 994-D decoder input, mirroring the assembly in run_step:
# token, then 10 frames of [angular velocity, joint pos, joint vel, last action], then gravity.
TOKEN_SLICE = slice(0, swb.TOKEN_DIM)
ANG_START = swb.TOKEN_DIM
Q_START = ANG_START + HISTORY_LEN * 3
DQ_START = Q_START + HISTORY_LEN * NUM_MOTORS
ACT_START = DQ_START + HISTORY_LEN * NUM_MOTORS
GRAVITY_START = ACT_START + HISTORY_LEN * NUM_MOTORS

# Each block stores oldest -> newest, so the newest frame sits in the final slot.
NEWEST_ANG = slice(Q_START - 3, Q_START)
NEWEST_Q = slice(DQ_START - NUM_MOTORS, DQ_START)
NEWEST_DQ = slice(ACT_START - NUM_MOTORS, ACT_START)
NEWEST_ACT = slice(GRAVITY_START - NUM_MOTORS, GRAVITY_START)
NEWEST_GRAVITY = slice(swb.DECODER_INPUT_DIM - 3, swb.DECODER_INPUT_DIM)


class FakeDecoder:
    """Minimal stand-in for an ``onnxruntime.InferenceSession``."""

    def __init__(self, action: np.ndarray | None = None):
        self.action = np.zeros(NUM_MOTORS, np.float32) if action is None else action
        self.inputs: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="obs", shape=[1, swb.DECODER_INPUT_DIM])]

    def get_outputs(self):
        return [SimpleNamespace(name="action", shape=[1, NUM_MOTORS])]

    def run(self, _output_names, feed):
        obs = feed["obs"]
        assert obs.shape == (1, swb.DECODER_INPUT_DIM), f"unexpected decoder input {obs.shape}"
        self.inputs.append(obs[0].copy())
        return [self.action.reshape(1, NUM_MOTORS).astype(np.float32)]

    @property
    def last_input(self) -> np.ndarray:
        return self.inputs[-1]


KP = np.full(NUM_MOTORS, 100.0, np.float32)
KD = np.full(NUM_MOTORS, 2.0, np.float32)
DEFAULT_ANGLES = np.arange(NUM_MOTORS, dtype=np.float32) * 0.01
ACTION_SCALE = np.full(NUM_MOTORS, 0.25, np.float32)
NEUTRAL_TOKEN = np.full(swb.TOKEN_DIM, 0.5, np.float32)


@pytest.fixture
def make_controller(monkeypatch):
    """Build a controller backed by a fake decoder, bypassing the Hub download."""

    def _factory(action: np.ndarray | None = None):
        decoder = FakeDecoder(action)
        monkeypatch.setattr(
            swb,
            "load_policy",
            lambda **_kwargs: (
                decoder,
                KP.copy(),
                KD.copy(),
                DEFAULT_ANGLES.copy(),
                ACTION_SCALE.copy(),
                NEUTRAL_TOKEN.copy(),
            ),
        )
        return swb.SonicWholeBodyController(), decoder

    return _factory


def make_lowstate(q=None, dq=None, quat=(1.0, 0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0)):
    """Build a duck-typed lowstate matching what run_step reads."""
    q = np.zeros(NUM_MOTORS, np.float32) if q is None else np.asarray(q, np.float32)
    dq = np.zeros(NUM_MOTORS, np.float32) if dq is None else np.asarray(dq, np.float32)
    motors = [SimpleNamespace(q=float(q[i]), dq=float(dq[i])) for i in range(NUM_MOTORS)]
    return SimpleNamespace(
        motor_state=motors,
        imu_state=SimpleNamespace(quaternion=list(quat), gyroscope=list(gyro)),
    )


def token_action(token: np.ndarray) -> dict[str, float]:
    return {f"{swb.TOKEN_ACTION_PREFIX}.{i}.pos": float(v) for i, v in enumerate(token)}


class TestConstants:
    def test_decoder_input_dim_matches_block_layout(self):
        """994 = token + 10 frames of (ang, q, dq, last action) + 10 gravity vectors."""
        expected = swb.TOKEN_DIM + HISTORY_LEN * (3 + 3 * NUM_MOTORS) + HISTORY_LEN * 3
        assert expected == swb.DECODER_INPUT_DIM == 994

    def test_gravity_block_ends_the_vector(self):
        assert GRAVITY_START + HISTORY_LEN * 3 == swb.DECODER_INPUT_DIM

    def test_control_dt_is_50hz(self):
        assert swb.CONTROL_DT == 0.02
        assert swb.SonicWholeBodyController.control_dt == swb.CONTROL_DT

    def test_token_dim(self):
        assert swb.TOKEN_DIM == 64

    def test_policy_files_expose_both_decoders(self):
        assert set(swb.POLICY_FILES) == {"default", "low_latency"}


class TestFeatureSchemas:
    def test_action_features_are_the_64d_token(self, make_controller):
        controller, _ = make_controller()
        assert len(controller.action_ft) == swb.TOKEN_DIM
        assert set(controller.action_ft) == {
            f"{swb.TOKEN_ACTION_PREFIX}.{i}.pos" for i in range(swb.TOKEN_DIM)
        }
        assert all(v is float for v in controller.action_ft.values())

    def test_observation_ft_is_token_echo(self, make_controller):
        controller, _ = make_controller()
        assert len(controller.observation_ft) == swb.TOKEN_DIM
        assert set(controller.observation_ft) == {
            f"{swb.TOKEN_STATE_PREFIX}.{i}.pos" for i in range(swb.TOKEN_DIM)
        }

    def test_action_and_observation_keys_are_disjoint(self, make_controller):
        """A shared prefix would make the action token and its echo collide in a dataset frame."""
        controller, _ = make_controller()
        assert not set(controller.action_ft) & set(controller.observation_ft)

    def test_gains_and_home_pose_are_exposed(self, make_controller):
        """UnitreeG1.connect() reads these off the controller."""
        controller, _ = make_controller()
        assert controller.kp.shape == (NUM_MOTORS,)
        assert controller.kd.shape == (NUM_MOTORS,)
        assert controller.default_angles.shape == (NUM_MOTORS,)


class TestReset:
    def test_history_buffers_are_initialised(self, make_controller):
        controller, _ = make_controller()
        for buf, width in (
            (controller.h_q_mj, NUM_MOTORS),
            (controller.h_dq_mj, NUM_MOTORS),
            (controller.h_act_mj, NUM_MOTORS),
            (controller.h_ang, 3),
            (controller.h_quat, 4),
        ):
            assert len(buf) == HISTORY_LEN
            assert all(frame.shape == (width,) for frame in buf)

    def test_quaternion_history_starts_upright(self, make_controller):
        controller, _ = make_controller()
        for frame in controller.h_quat:
            np.testing.assert_allclose(frame, [1.0, 0.0, 0.0, 0.0])

    def test_token_starts_unset(self, make_controller):
        controller, _ = make_controller()
        assert controller._last_token is None
        np.testing.assert_allclose(controller.last_action_mj, np.zeros(NUM_MOTORS))

    def test_reset_clears_state_after_stepping(self, make_controller):
        controller, _ = make_controller(action=np.ones(NUM_MOTORS, np.float32))
        controller.run_step(token_action(np.ones(swb.TOKEN_DIM, np.float32)), make_lowstate())
        assert controller._last_token is not None

        controller.reset()
        assert controller._last_token is None
        np.testing.assert_allclose(controller.last_action_mj, np.zeros(NUM_MOTORS))
        for frame in controller.h_q_mj:
            np.testing.assert_allclose(frame, np.zeros(NUM_MOTORS))


class TestTokenHandling:
    def test_first_tick_seeds_the_neutral_token(self, make_controller):
        """With no token in the action, the decoder must still see a valid idle latent."""
        controller, decoder = make_controller()
        controller.run_step({}, make_lowstate())
        np.testing.assert_allclose(decoder.last_input[TOKEN_SLICE], NEUTRAL_TOKEN)

    def test_token_from_action_reaches_the_decoder(self, make_controller):
        controller, decoder = make_controller()
        token = np.linspace(-1.0, 1.0, swb.TOKEN_DIM, dtype=np.float32)
        controller.run_step(token_action(token), make_lowstate())
        np.testing.assert_allclose(decoder.last_input[TOKEN_SLICE], token, atol=1e-6)

    def test_partial_token_is_ignored_and_previous_held(self, make_controller):
        """All 64 keys are required; a partial chunk must not be spliced into the latent."""
        controller, decoder = make_controller()
        token = np.linspace(-1.0, 1.0, swb.TOKEN_DIM, dtype=np.float32)
        controller.run_step(token_action(token), make_lowstate())

        partial = token_action(np.zeros(swb.TOKEN_DIM, np.float32))
        partial.pop(f"{swb.TOKEN_ACTION_PREFIX}.7.pos")
        controller.run_step(partial, make_lowstate())

        np.testing.assert_allclose(decoder.last_input[TOKEN_SLICE], token, atol=1e-6)

    def test_unrelated_action_keys_hold_the_token(self, make_controller):
        """Joystick-only actions (the pre-policy case) must not disturb the latent."""
        controller, decoder = make_controller()
        token = np.full(swb.TOKEN_DIM, 0.25, np.float32)
        controller.run_step(token_action(token), make_lowstate())
        controller.run_step({"remote.lx": 1.0, "remote.ly": -1.0}, make_lowstate())
        np.testing.assert_allclose(decoder.last_input[TOKEN_SLICE], token, atol=1e-6)

    def test_state_echoes_last_token(self, make_controller):
        controller, _ = make_controller()
        token = np.linspace(0.0, 1.0, swb.TOKEN_DIM, dtype=np.float32)
        controller.run_step(token_action(token), make_lowstate())

        state = controller.observation_state()
        assert len(state) == swb.TOKEN_DIM
        echoed = np.array(
            [state[f"{swb.TOKEN_STATE_PREFIX}.{i}.pos"] for i in range(swb.TOKEN_DIM)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(echoed, token, atol=1e-6)

    def test_observation_state_is_zeros_before_first_tick(self, make_controller):
        controller, _ = make_controller()
        state = controller.observation_state()
        assert len(state) == swb.TOKEN_DIM
        assert all(v == 0.0 for v in state.values())


class TestDecoderInputAssembly:
    def test_proprioception_lands_in_the_newest_slots(self, make_controller):
        controller, decoder = make_controller()
        q = np.linspace(0.1, 2.9, NUM_MOTORS, dtype=np.float32)
        dq = np.linspace(-1.0, 1.0, NUM_MOTORS, dtype=np.float32)
        gyro = (0.1, 0.2, 0.3)
        controller.run_step({}, make_lowstate(q=q, dq=dq, gyro=gyro))

        obs = decoder.last_input
        np.testing.assert_allclose(obs[NEWEST_ANG], gyro, atol=1e-6)
        np.testing.assert_allclose(
            obs[NEWEST_Q], q[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB], atol=1e-6
        )
        np.testing.assert_allclose(obs[NEWEST_DQ], dq[MUJOCO_TO_ISAACLAB], atol=1e-6)

    def test_older_frames_remain_empty_on_first_tick(self, make_controller):
        """Only the newest slot of each block is filled one tick after a reset."""
        controller, decoder = make_controller()
        q = np.full(NUM_MOTORS, 1.0, np.float32)
        controller.run_step({}, make_lowstate(q=q, dq=q, gyro=(1.0, 1.0, 1.0)))

        obs = decoder.last_input
        np.testing.assert_allclose(obs[ANG_START : Q_START - 3], 0.0)
        np.testing.assert_allclose(obs[Q_START : DQ_START - NUM_MOTORS], 0.0)
        np.testing.assert_allclose(obs[DQ_START : ACT_START - NUM_MOTORS], 0.0)

    def test_gravity_block_uses_the_imu_quaternion(self, make_controller):
        controller, decoder = make_controller()
        quat = (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)  # 90 deg about x
        controller.run_step({}, make_lowstate(quat=quat))

        normalised = np.array(quat, np.float32) / (np.linalg.norm(quat) + 1e-8)
        expected = get_gravity_orientation(normalised)
        obs = decoder.last_input
        np.testing.assert_allclose(obs[NEWEST_GRAVITY], expected, atol=1e-6)
        # The nine older frames still hold the upright quaternion seeded by reset().
        np.testing.assert_allclose(obs[GRAVITY_START : GRAVITY_START + 3], [0.0, 0.0, -1.0], atol=1e-6)

    def test_previous_action_is_fed_back(self, make_controller):
        """The decoder sees its own previous output in the action-history block."""
        action = np.linspace(-2.0, 2.0, NUM_MOTORS, dtype=np.float32)
        controller, decoder = make_controller(action=action)

        controller.run_step({}, make_lowstate())
        np.testing.assert_allclose(decoder.last_input[NEWEST_ACT], 0.0)

        controller.run_step({}, make_lowstate())
        np.testing.assert_allclose(decoder.last_input[NEWEST_ACT], action, atol=1e-6)

    def test_history_length_is_bounded(self, make_controller):
        controller, decoder = make_controller()
        for i in range(HISTORY_LEN + 5):
            controller.run_step({}, make_lowstate(q=np.full(NUM_MOTORS, float(i), np.float32)))

        assert len(controller.h_q_mj) == HISTORY_LEN
        assert len(controller.h_quat) == HISTORY_LEN
        assert decoder.last_input.shape == (swb.DECODER_INPUT_DIM,)

    def test_input_is_finite(self, make_controller):
        controller, decoder = make_controller(action=np.ones(NUM_MOTORS, np.float32))
        controller.run_step(token_action(np.ones(swb.TOKEN_DIM, np.float32)), make_lowstate())
        assert np.all(np.isfinite(decoder.last_input))

    def test_zero_quaternion_does_not_divide_by_zero(self, make_controller):
        """The 1e-8 epsilon in the normalisation guards a dropped IMU frame."""
        controller, decoder = make_controller()
        controller.run_step({}, make_lowstate(quat=(0.0, 0.0, 0.0, 0.0)))
        assert np.all(np.isfinite(decoder.last_input))


class TestJointTargets:
    def test_targets_are_default_angles_plus_scaled_residual(self, make_controller):
        action = np.arange(NUM_MOTORS, dtype=np.float32)
        controller, _ = make_controller(action=action)

        targets = controller.run_step({}, make_lowstate())

        expected = DEFAULT_ANGLES + action[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        for joint in G1_29_JointIndex:
            assert targets[f"{joint.name}.q"] == pytest.approx(expected[joint.value], abs=1e-6)

    def test_returns_one_key_per_joint(self, make_controller):
        controller, _ = make_controller()
        targets = controller.run_step({}, make_lowstate())
        assert set(targets) == {f"{joint.name}.q" for joint in G1_29_JointIndex}
        assert len(targets) == NUM_MOTORS

    def test_targets_are_plain_floats(self, make_controller):
        """publish_lowcmd writes these straight into the SDK message."""
        controller, _ = make_controller(action=np.ones(NUM_MOTORS, np.float32))
        targets = controller.run_step({}, make_lowstate())
        assert all(type(v) is float for v in targets.values())

    def test_zero_residual_holds_the_home_pose(self, make_controller):
        controller, _ = make_controller(action=np.zeros(NUM_MOTORS, np.float32))
        targets = controller.run_step({}, make_lowstate())
        for joint in G1_29_JointIndex:
            assert targets[f"{joint.name}.q"] == pytest.approx(DEFAULT_ANGLES[joint.value], abs=1e-6)

    def test_residual_is_reordered_not_passed_through(self, make_controller):
        """A non-identity permutation must actually be applied to the decoder output."""
        action = np.arange(NUM_MOTORS, dtype=np.float32)
        controller, _ = make_controller(action=action)
        targets = controller.run_step({}, make_lowstate())

        produced = np.array([targets[f"{j.name}.q"] for j in G1_29_JointIndex], dtype=np.float32)
        unpermuted = DEFAULT_ANGLES + action * ACTION_SCALE
        assert not np.allclose(produced, unpermuted)


class TestLoadPolicy:
    def test_unknown_policy_type_raises(self):
        with pytest.raises(ValueError, match="Unknown policy type"):
            swb.load_policy(policy_type="does_not_exist")
