from unittest.mock import MagicMock

from lerobot.motors.damiao import DamiaoMotorsBus


def test_sync_write_metal_delegates():
    bus = DamiaoMotorsBus.__new__(DamiaoMotorsBus)  # skip __init__/hardware
    bus._mit_control_batch = MagicMock()
    cmds = {"shoulder_pan": (0.0, 0.5, 10.0, 0.0, 1.2)}
    bus.sync_write_metal(cmds)
    bus._mit_control_batch.assert_called_once_with(cmds)
