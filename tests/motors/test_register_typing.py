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

import runpy
import textwrap
from importlib.metadata import version
from pathlib import Path
from typing import get_args

import pytest
from mypy import api

from lerobot.motors import CustomRegister, Motor, MotorNormMode
from lerobot.motors.registers import RegisterName
from tests.mocks.mock_motors_bus import DummyDefinition, MockMotorsBus


def test_register_types_in_a_user_program(tmp_path, monkeypatch):
    """Check real imported signatures, including the connection decorators."""
    source = textwrap.dedent("""\
        from lerobot.motors import CustomRegister, Register, RegisterName, SerialMotorsBus
        from lerobot.motors.motors_bus import MotorsBusBase
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.dynamixel import DynamixelMotorsBus
        from lerobot.motors.damiao.damiao import DamiaoMotorsBus
        from lerobot.motors.robstride.robstride import RobstrideMotorsBus

        def serial(bus: SerialMotorsBus, dynamic_name: str) -> None:
            bus.read("Present_Position", "joint", normalize=False)
            bus.read("present_position", "joint")
            bus.read("PRESENT_POSITION", "joint")
            bus.write("Goal_Position", "joint", 0)
            bus.sync_read("Present_Position", ["joint", 1], num_retry=1)
            bus.sync_write("Goal_Position", {"joint": 0}, normalize=False)
            name: RegisterName = "Present_Position"
            bus.read(name, "joint")
            custom = CustomRegister(dynamic_name)
            bus.read(custom, "joint")
            bus.write(custom, "joint", 0)
            bus.sync_read(custom)
            bus.sync_write(custom, {"joint": 0})
            bus.read("Presnet_Position", "joint")  # type: ignore[arg-type]
            bus.write("Gaol_Position", "joint", 0)  # type: ignore[arg-type]
            bus.sync_read("Presnet_Position")  # type: ignore[arg-type]
            bus.sync_write("Gaol_Position", {"joint": 0})  # type: ignore[arg-type]
            bus.read(dynamic_name, "joint")  # type: ignore[arg-type]
            bus.read("Present_Position", 123)  # type: ignore[arg-type]
            bus.read("Present_Position", "joint", unknown_option=True)  # type: ignore[call-arg]
            bus.write("Goal_Position", "joint", "invalid")  # type: ignore[arg-type]

        def aliases(feetech: FeetechMotorsBus, dynamixel: DynamixelMotorsBus) -> None:
            feetech.read("Presnet_Position", "joint")  # type: ignore[arg-type]
            dynamixel.write("Gaol_Position", "joint", 0)  # type: ignore[arg-type]

        def generic(bus: MotorsBusBase[Register], custom: CustomRegister) -> None:
            bus.read(custom, "joint")
            bus.read("Presnet_Position", "joint")  # type: ignore[arg-type]

        def can_and_custom(bus: MotorsBusBase[str], damiao: DamiaoMotorsBus,
                           robstride: RobstrideMotorsBus, dynamic: str) -> None:
            bus.read(dynamic, "joint")
            damiao.read("Temperature_MOS", "joint")
            damiao.write("Kp", "joint", 1)
            robstride.sync_read(dynamic)
            robstride.sync_write("Kd", {"joint": 1})
    """)
    program = tmp_path / "custom_robot.py"
    program.write_text(source)
    monkeypatch.setenv("MYPYPATH", str(Path(__file__).resolve().parents[2] / "src"))
    stdout, stderr, status = api.run(
        [
            "--config-file=" + str(Path(__file__).resolve().parents[2] / "pyproject.toml"),
            "--follow-imports=silent",
            "--ignore-missing-imports",
            "--no-incremental",
            "--warn-unused-ignores",
            str(program),
        ]
    )
    assert status == 0, stdout + stderr


def test_register_names_match_servo_definitions():
    rustypot = pytest.importorskip("rustypot")
    names = set()
    for controller in vars(rustypot).values():
        models = getattr(controller, "models", None)
        if isinstance(controller, type) and callable(models):
            for model in models():
                found = rustypot.find_model(model)
                assert found is not None
                names.update(register.name for register in found[1].registers())
    assert names
    expected = {value for name in names for value in (name, name.upper(), name.title())}
    actual = set(get_args(RegisterName.__value__))
    assert actual <= expected  # Newer rustypot releases can add names.
    if version("rustypot") == "1.11.0":
        assert actual == expected


def test_generated_source_matches_snapshot(capsys):
    pytest.importorskip("rustypot")
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root / "scripts/generate_motor_registers.py"), run_name="__main__")
    source = capsys.readouterr().out
    assert source
    if version("rustypot") == "1.11.0":
        assert source == (root / "src/lerobot/motors/registers.py").read_text()


@pytest.mark.parametrize(
    "register",
    [
        "Present_Position",
        "present_position",
        "PRESENT_POSITION",
        CustomRegister("pReSeNt_PoSiTiOn"),
        CustomRegister("Vendor_Gain"),
    ],
)
def test_register_names_preserve_runtime_access(register):
    class CustomBus(MockMotorsBus):
        @staticmethod
        def _find_model(model):
            return 1234, DummyDefinition("CUSTOM", {model: 1234}, ("Present_Position", "Vendor_Gain"))

    bus = CustomBus("/dev/dummy-port", {"joint": Motor(1, "model_1", MotorNormMode.RANGE_M100_100)})
    bus.connect(handshake=False)
    transport = bus.opened[-1]
    try:
        bus.write(register, "joint", 123, normalize=False)
        assert bus.read(register, "joint", normalize=False) == 123
        bus.sync_write(register, {"joint": 456}, normalize=False)
        assert bus.sync_read(register, normalize=False) == {"joint": 456}
        assert transport.reads[-1] == (1, register.lower())
        assert transport.sync_writes[-1] == ([1], register.lower(), [456])
    finally:
        bus.disconnect()
