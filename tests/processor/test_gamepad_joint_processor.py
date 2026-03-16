"""Tests for MapGamepadToJointPositionsStep and make_default_processors factory."""

import pytest

from lerobot.configs.types import FeatureType, PipelineFeatureType
from lerobot.processor.converters import robot_action_observation_to_transition
from lerobot.processor.delta_action_processor import MapGamepadToJointPositionsStep
from lerobot.processor.factory import (
    SO_MOTOR_NAMES,
    make_default_processors,
)

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _make_observation(**overrides: float) -> dict[str, float]:
    """Create a fake observation dict with motor positions."""
    obs = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 45.0,
        "elbow_flex.pos": -30.0,
        "wrist_flex.pos": 10.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 50.0,
    }
    for k, v in overrides.items():
        obs[k] = v
    return obs


def _make_gamepad_action(
    delta_x: float = 0.0,
    delta_y: float = 0.0,
    delta_z: float = 0.0,
    delta_wx: float = 0.0,
    delta_wz: float = 0.0,
    gripper: float = 1.0,
) -> dict[str, float]:
    """Create a gamepad action dict."""
    return {
        "delta_x": delta_x,
        "delta_y": delta_y,
        "delta_z": delta_z,
        "delta_wx": delta_wx,
        "delta_wz": delta_wz,
        "gripper": gripper,
    }


def _run_step(
    step: MapGamepadToJointPositionsStep,
    action: dict[str, float],
    observation: dict[str, float],
) -> dict[str, float]:
    """Run a processor step through the transition machinery."""
    transition = robot_action_observation_to_transition((action, observation))
    result_transition = step(transition)
    return result_transition["action"]


# ---------------------------------------------------------------------------
# MapGamepadToJointPositionsStep tests
# ---------------------------------------------------------------------------


class TestMapGamepadToJointPositionsStep:
    def test_no_input_preserves_positions(self):
        """When all deltas are zero and gripper=stay, positions should be unchanged."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES)
        obs = _make_observation()
        action = _make_gamepad_action()  # all zeros, gripper=stay(1)

        result = _run_step(step, action, obs)

        for name in MOTOR_NAMES:
            assert result[f"{name}.pos"] == pytest.approx(obs[f"{name}.pos"])

    def test_left_stick_y_moves_shoulder_lift(self):
        """delta_x (left stick Y) should move shoulder_lift."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=3.0)
        obs = _make_observation()
        action = _make_gamepad_action(delta_x=1.0)

        result = _run_step(step, action, obs)

        assert result["shoulder_lift.pos"] == pytest.approx(45.0 + 3.0)
        # Other joints should be unchanged
        assert result["shoulder_pan.pos"] == pytest.approx(0.0)
        assert result["elbow_flex.pos"] == pytest.approx(-30.0)

    def test_left_stick_x_moves_shoulder_pan(self):
        """delta_y (left stick X) should move shoulder_pan."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=3.0)
        obs = _make_observation()
        action = _make_gamepad_action(delta_y=-0.5)

        result = _run_step(step, action, obs)

        assert result["shoulder_pan.pos"] == pytest.approx(0.0 + (-0.5 * 3.0))

    def test_right_stick_y_moves_elbow_flex(self):
        """delta_z (right stick Y) should move elbow_flex."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=2.0)
        obs = _make_observation()
        action = _make_gamepad_action(delta_z=0.8)

        result = _run_step(step, action, obs)

        assert result["elbow_flex.pos"] == pytest.approx(-30.0 + 0.8 * 2.0)

    def test_right_stick_x_moves_wrist_flex(self):
        """delta_wx (right stick X) should move wrist_flex."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=3.0)
        obs = _make_observation()
        action = _make_gamepad_action(delta_wx=-1.0)

        result = _run_step(step, action, obs)

        assert result["wrist_flex.pos"] == pytest.approx(10.0 + (-1.0 * 3.0))

    def test_bumpers_move_wrist_roll(self):
        """delta_wz (LB/RB) should move wrist_roll."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=3.0)
        obs = _make_observation()

        # RB pressed (+1)
        result = _run_step(step, _make_gamepad_action(delta_wz=1.0), obs)
        assert result["wrist_roll.pos"] == pytest.approx(0.0 + 3.0)

        # LB pressed (-1)
        result = _run_step(step, _make_gamepad_action(delta_wz=-1.0), obs)
        assert result["wrist_roll.pos"] == pytest.approx(0.0 - 3.0)

    def test_gripper_open(self):
        """Gripper action 0 (CLOSE) should decrease position."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES, gripper_step_size=5.0)
        obs = _make_observation()
        action = _make_gamepad_action(gripper=0.0)

        result = _run_step(step, action, obs)

        assert result["gripper.pos"] == pytest.approx(50.0 - 5.0)

    def test_gripper_close(self):
        """Gripper action 2 (OPEN) should increase position."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES, gripper_step_size=5.0)
        obs = _make_observation()
        action = _make_gamepad_action(gripper=2.0)

        result = _run_step(step, action, obs)

        assert result["gripper.pos"] == pytest.approx(50.0 + 5.0)

    def test_gripper_stay(self):
        """Gripper action 1 (STAY) should not change position."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES)
        obs = _make_observation()
        action = _make_gamepad_action(gripper=1.0)

        result = _run_step(step, action, obs)

        assert result["gripper.pos"] == pytest.approx(50.0)

    def test_gripper_clamps_at_zero(self):
        """Gripper should not go below 0."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES, gripper_step_size=10.0)
        obs = _make_observation(**{"gripper.pos": 3.0})
        action = _make_gamepad_action(gripper=0.0)

        result = _run_step(step, action, obs)

        assert result["gripper.pos"] == pytest.approx(0.0)

    def test_gripper_clamps_at_hundred(self):
        """Gripper should not go above 100."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES, gripper_step_size=10.0)
        obs = _make_observation(**{"gripper.pos": 97.0})
        action = _make_gamepad_action(gripper=2.0)

        result = _run_step(step, action, obs)

        assert result["gripper.pos"] == pytest.approx(100.0)

    def test_multiple_axes_simultaneously(self):
        """Multiple axes should be applied independently."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=2.0)
        obs = _make_observation()
        action = _make_gamepad_action(delta_x=1.0, delta_y=-0.5, delta_z=0.3)

        result = _run_step(step, action, obs)

        assert result["shoulder_lift.pos"] == pytest.approx(45.0 + 1.0 * 2.0)
        assert result["shoulder_pan.pos"] == pytest.approx(0.0 + (-0.5) * 2.0)
        assert result["elbow_flex.pos"] == pytest.approx(-30.0 + 0.3 * 2.0)

    def test_outputs_all_motor_names(self):
        """Result should have a .pos key for every motor name."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES)
        obs = _make_observation()
        action = _make_gamepad_action()

        result = _run_step(step, action, obs)

        for name in MOTOR_NAMES:
            assert f"{name}.pos" in result

    def test_transform_features_removes_deltas_adds_motor_pos(self, policy_feature_factory):
        """transform_features should remove delta keys and add motor.pos keys."""
        step = MapGamepadToJointPositionsStep(motor_names=MOTOR_NAMES)
        features = {
            PipelineFeatureType.ACTION: {
                "delta_x": policy_feature_factory(FeatureType.ACTION, (1,)),
                "delta_y": policy_feature_factory(FeatureType.ACTION, (1,)),
                "delta_z": policy_feature_factory(FeatureType.ACTION, (1,)),
                "delta_wx": policy_feature_factory(FeatureType.ACTION, (1,)),
                "delta_wz": policy_feature_factory(FeatureType.ACTION, (1,)),
                "gripper": policy_feature_factory(FeatureType.ACTION, (1,)),
            },
            PipelineFeatureType.OBSERVATION: {},
        }

        out = step.transform_features(features)

        # Deltas should be removed
        for key in ["delta_x", "delta_y", "delta_z", "delta_wx", "delta_wz", "gripper"]:
            assert key not in out[PipelineFeatureType.ACTION]

        # Motor positions should be added
        for name in MOTOR_NAMES:
            assert f"{name}.pos" in out[PipelineFeatureType.ACTION]


# ---------------------------------------------------------------------------
# make_processors factory tests
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal config stub with a .type property."""

    def __init__(self, config_type: str):
        self._type = config_type

    @property
    def type(self) -> str:
        return self._type


class TestMakeProcessors:
    def test_gamepad_so_follower_returns_joint_pipeline(self):
        """Gamepad + SO follower should return the direct joint control pipeline."""
        teleop_cfg = _FakeConfig("gamepad")
        robot_cfg = _FakeConfig("so101_follower")

        teleop_proc, robot_proc, obs_proc = make_default_processors(teleop_cfg, robot_cfg)

        # The teleop processor should contain a MapGamepadToJointPositionsStep
        assert len(teleop_proc.steps) == 1
        assert isinstance(teleop_proc.steps[0], MapGamepadToJointPositionsStep)

    def test_keyboard_ee_so_follower_returns_joint_pipeline(self):
        """keyboard_ee + SO follower should also trigger the joint control pipeline."""
        teleop_cfg = _FakeConfig("keyboard_ee")
        robot_cfg = _FakeConfig("so100_follower")

        teleop_proc, _, _ = make_default_processors(teleop_cfg, robot_cfg)

        assert isinstance(teleop_proc.steps[0], MapGamepadToJointPositionsStep)

    def test_leader_follower_returns_default_pipeline(self):
        """Leader-follower teleop should return default (identity) processors."""
        teleop_cfg = _FakeConfig("so101_leader")
        robot_cfg = _FakeConfig("so101_follower")

        result = make_default_processors(teleop_cfg, robot_cfg)
        default = make_default_processors()

        # Both should have a single IdentityProcessorStep in each pipeline
        assert len(result[0].steps) == len(default[0].steps)
        assert type(result[0].steps[0]) is type(default[0].steps[0])

    def test_gamepad_non_so_robot_returns_default_pipeline(self):
        """Gamepad + non-SO robot should return default processors."""
        teleop_cfg = _FakeConfig("gamepad")
        robot_cfg = _FakeConfig("koch_follower")

        result = make_default_processors(teleop_cfg, robot_cfg)
        default = make_default_processors()

        assert type(result[0].steps[0]) is type(default[0].steps[0])

    def test_so_motor_names_constant(self):
        """SO_MOTOR_NAMES should contain the expected 6 motors in order."""
        assert SO_MOTOR_NAMES == [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

    def test_joint_pipeline_end_to_end(self):
        """Full end-to-end test: gamepad action + observation -> motor positions."""
        teleop_cfg = _FakeConfig("gamepad")
        robot_cfg = _FakeConfig("so101_follower")

        teleop_proc, robot_proc, _ = make_default_processors(teleop_cfg, robot_cfg)

        obs = _make_observation()
        action = _make_gamepad_action(delta_x=1.0, gripper=2.0)

        # Run through the teleop processor (which is the only non-identity one)
        teleop_result = teleop_proc((action, obs))

        # Should have motor.pos keys
        assert "shoulder_lift.pos" in teleop_result

        # Robot processor is identity, should pass through
        robot_result = robot_proc((teleop_result, obs))
        assert robot_result == teleop_result
