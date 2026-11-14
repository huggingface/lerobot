from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.teleoperators.openarm_leader.openarm_leader import OpenArmLeader

pytestmark = pytest.mark.hardware


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("allow_calibration", [False, True])
def test_connect_only_zeros_during_requested_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calibrated: bool, allow_calibration: bool
) -> None:
    arm = object.__new__(OpenArmLeader)
    arm.id = "test"
    arm.config = SimpleNamespace(port="mock-can", manual_control=True)
    arm.calibration = {}
    arm.calibration_fpath = tmp_path / "calibration.json"
    arm._save_calibration = Mock()
    arm.bus = Mock(is_connected=False, is_calibrated=calibrated, motors={})
    arm.bus.write_calibration.side_effect = lambda _: setattr(arm.bus, "is_calibrated", True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")

    arm.connect(calibrate=allow_calibration)

    expected = int(not calibrated and allow_calibration)
    assert arm.bus.set_zero_position.call_count == expected
    assert len(prompts) == expected
    arm.bus.connect.assert_called_once_with()
    assert arm.bus.disable_torque.called
