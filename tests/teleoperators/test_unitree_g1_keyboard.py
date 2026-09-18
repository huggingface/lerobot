"""G1 keyboard contracts, without a display, hardware, or simulator."""

from types import SimpleNamespace

import pytest

from lerobot.robots.unitree_g1 import UnitreeG1Config
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.teleoperators.unitree_g1 import keyboard_g1 as module
from lerobot.teleoperators.utils import make_teleoperator_from_config


@pytest.fixture
def keyboard(monkeypatch, tmp_path):
    clock = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    listener = SimpleNamespace(stop=lambda: None, is_alive=lambda: True)
    monkeypatch.setattr(module, "create_key_listener", lambda *a, **k: listener)
    teleop = make_teleoperator_from_config(module.UnitreeG1KeyboardConfig(calibration_dir=tmp_path))
    teleop.connect()
    teleop.send_feedback(dict.fromkeys(teleop.action_features, 0.0))
    yield teleop, clock
    teleop.disconnect()


def test_measured_initialization_and_enable(keyboard):
    teleop, _ = keyboard
    teleop.disconnect()
    teleop.connect()
    teleop._on_key("enter")
    teleop._on_key("+")
    with pytest.raises(RuntimeError, match="feedback"):
        teleop.get_action()
    pose = dict.fromkeys(teleop.action_features, 0.2)
    teleop.send_feedback(pose)
    assert teleop.get_action() == pose
    teleop._on_key("+")
    assert teleop.get_action() == pose
    teleop._on_key("enter")
    teleop._on_key("+")
    assert teleop.get_action()["kLeftShoulderPitch.q"] == pytest.approx(0.22)


@pytest.mark.parametrize("side", ["l", "r"])
@pytest.mark.parametrize("joint", range(1, 8))
@pytest.mark.parametrize("direction", ["+", "-"])
def test_each_arm_joint_only(keyboard, side, joint, direction):
    teleop, _ = keyboard
    teleop._on_key("enter")
    teleop._on_key(side)
    teleop._on_key(str(joint))
    teleop._on_key(direction)
    action = teleop.get_action()
    changed = [key for key, value in action.items() if value != 0]
    expected = teleop.joints[joint - 1 + (7 if side == "r" else 0)] + ".q"
    assert changed == [expected]
    assert action[expected] == pytest.approx(0.02 if direction == "+" else -0.02)


def test_repeat_rate_no_backlog_and_no_time_integration(keyboard):
    teleop, clock = keyboard
    teleop._on_key("enter")
    for _ in range(100):
        teleop._on_key("+")
    assert teleop.get_action()["kLeftShoulderPitch.q"] == pytest.approx(0.02)
    clock[0] += 0.2
    assert teleop.get_action()["kLeftShoulderPitch.q"] == pytest.approx(0.02)
    teleop._on_key("+")
    assert teleop.get_action()["kLeftShoulderPitch.q"] == pytest.approx(0.04)


def test_hold_reengage_and_escape(keyboard):
    teleop, _ = keyboard
    teleop._on_key("enter")
    teleop._on_key("+")
    pose = dict.fromkeys(teleop.action_features, 0.1)
    teleop.send_feedback(pose)
    teleop._on_key("space")
    teleop._on_key("-")
    assert teleop.get_action() == pose
    teleop._on_key("enter")
    assert teleop.get_action() == pose
    teleop._on_key("esc")
    with pytest.raises(KeyboardInterrupt):
        teleop.get_action()


@pytest.mark.parametrize("bad", [{}, {"kLeftShoulderPitch.q": float("nan")}, {"kLeftShoulderPitch.q": 20}])
def test_invalid_feedback_fails_closed(keyboard, bad):
    teleop, _ = keyboard
    pose = dict.fromkeys(teleop.action_features, 0.0) if bad else {}
    pose.update(bad)
    with pytest.raises(ValueError, match="observations"):
        teleop.send_feedback(pose)
    with pytest.raises(RuntimeError, match="feedback"):
        teleop.get_action()


def test_stale_feedback_and_listener_failure(keyboard):
    teleop, clock = keyboard
    clock[0] += 0.6
    with pytest.raises(RuntimeError, match="stale"):
        teleop.get_action()
    teleop.send_feedback(dict.fromkeys(teleop.action_features, 0.0))
    teleop.listener.is_alive = lambda: False
    with pytest.raises(RuntimeError, match="listener"):
        teleop.get_action()


def test_joint_limit_and_target_lead(keyboard):
    teleop, clock = keyboard
    pose = dict.fromkeys(teleop.action_features, 0.0)
    teleop._on_key("enter")
    for _ in range(20):
        clock[0] += 0.11
        teleop.send_feedback(pose)
        teleop._on_key("+")
    assert teleop.get_action()["kLeftShoulderPitch.q"] == pytest.approx(0.15)
    pose["kLeftShoulderPitch.q"] = teleop.limits["kLeftShoulderPitch.q"][1] - 0.001
    teleop.send_feedback(pose)
    teleop._on_key("enter")
    clock[0] += 0.11
    teleop._on_key("+")
    assert teleop.get_action()["kLeftShoulderPitch.q"] == teleop.limits["kLeftShoulderPitch.q"][1]


@pytest.mark.parametrize("step", [0, -1, 0.051, float("nan"), float("inf")])
def test_invalid_step(step):
    with pytest.raises(ValueError):
        module.UnitreeG1KeyboardConfig(step_rad=step)


def test_no_keyboard_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "create_key_listener", lambda *a, **k: None)
    teleop = module.UnitreeG1Keyboard(module.UnitreeG1KeyboardConfig(calibration_dir=tmp_path))
    with pytest.raises(RuntimeError, match="terminal"):
        teleop.connect()
    teleop.disconnect()


def test_cli_rejects_hardware_and_competing_controller():
    for config in (
        UnitreeG1Config(is_simulation=False),
        UnitreeG1Config(controller="SonicWholeBodyController"),
    ):
        with pytest.raises(ValueError, match="simulation without"):
            TeleoperateConfig(robot=config, teleop=module.UnitreeG1KeyboardConfig())
    TeleoperateConfig(robot=UnitreeG1Config(), teleop=module.UnitreeG1KeyboardConfig())
