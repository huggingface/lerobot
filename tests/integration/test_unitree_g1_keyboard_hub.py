"""Opt-in real upstream Hub/DDS simulation through the standard teleop loop."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(not os.getenv("G1_KEYBOARD_HUB_TESTS"), reason="Requires cached Hub model and DDS")
def test_keyboard_drives_upstream_hub_simulator():
    result = subprocess.run(
        [sys.executable, __file__],
        capture_output=True,
        text=True,
        timeout=90,
        env=dict(os.environ, MUJOCO_GL="egl"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS upstream keyboard arms" in result.stdout


def run_hub():
    import time
    from types import SimpleNamespace

    import numpy as np

    from lerobot.processor import make_default_processors
    from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
    from lerobot.scripts.lerobot_teleoperate import teleop_loop
    from lerobot.teleoperators.unitree_g1 import keyboard_g1

    keyboard_g1.create_key_listener = lambda *a, **k: SimpleNamespace(
        stop=lambda: None, is_alive=lambda: True
    )
    config = UnitreeG1Config(sim_publish_images=False, sim_onscreen=False, cameras={})
    config.sim_env.hub_path = "lerobot/unitree-g1-mujoco@68459ed68f6f68e1f661091dfcb6ebce44681aec"
    robot = UnitreeG1(config)
    teleop = keyboard_g1.UnitreeG1Keyboard(keyboard_g1.UnitreeG1KeyboardConfig())
    teleop.connect()
    try:
        robot.connect()
        time.sleep(0.5)
        measured_before = robot.get_observation()
        original_get = teleop.get_action
        actions = []
        observations = []
        started = time.monotonic()

        def scripted_keys():
            elapsed = time.monotonic() - started
            if not actions:
                teleop._on_key("enter")
            teleop._on_key("l" if elapsed < 1.5 else "r")
            teleop._on_key("4")
            if elapsed < 3:
                teleop._on_key("+")
            action = original_get()
            actions.append(action)
            observations.append(robot.get_observation())
            return action

        teleop.get_action = scripted_keys
        teleop_loop(teleop, robot, 30, *make_default_processors(), duration=4)
        after = robot.get_observation()
        assert set(actions[-1]) == set(teleop.action_features)
        for side in ("Left", "Right"):
            key = f"k{side}Elbow.q"
            assert actions[-1][key] - actions[0][key] > 0.1, (side, actions[-1])
            assert after[key] - measured_before[key] > 0.05, (side, measured_before[key], after[key])
        # The held base and untouched joints must not receive keyboard targets.
        assert all("Hip" not in key and "Waist" not in key for key in actions[-1])
        assert all(np.isfinite(list(action.values())).all() for action in actions)
        print("PASS upstream keyboard arms", flush=True)
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    run_hub()
