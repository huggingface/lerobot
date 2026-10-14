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

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from lerobot.motors import CustomRegister, Motor, MotorNormMode
from tests.mocks.mock_motors_bus import DummyDefinition, MockMotorsBus


def test_register_types_in_a_user_program(tmp_path):
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
            bus.read("Presnet_Position", "joint")  # error: arg-type
            bus.write("Gaol_Position", "joint", 0)  # error: arg-type
            bus.sync_read("Presnet_Position")  # error: arg-type
            bus.sync_write("Gaol_Position", {"joint": 0})  # error: arg-type
            bus.read(dynamic_name, "joint")  # error: arg-type
            bus.read("Present_Position", 123)  # error: arg-type
            bus.read("Present_Position", "joint", unknown_option=True)  # error: call-arg
            bus.write("Goal_Position", "joint", "invalid")  # error: arg-type

        def aliases(feetech: FeetechMotorsBus, dynamixel: DynamixelMotorsBus) -> None:
            feetech.read("Present_Position", "joint")
            dynamixel.write("Goal_Position", "joint", 0)
            feetech.read("Presnet_Position", "joint")  # error: arg-type
            dynamixel.write("Gaol_Position", "joint", 0)  # error: arg-type

        def generic(bus: MotorsBusBase[Register], custom: CustomRegister) -> None:
            bus.read("Present_Position", "joint")
            bus.write("Goal_Position", "joint", 0)
            bus.sync_read("Present_Position")
            bus.sync_write("Goal_Position", {"joint": 0})
            bus.read(custom, "joint")
            bus.write(custom, "joint", 0)
            bus.sync_read(custom)
            bus.sync_write(custom, {"joint": 0})
            bus.read("Presnet_Position", "joint")  # error: arg-type

        def can_and_custom(bus: MotorsBusBase[str], damiao: DamiaoMotorsBus,
                           robstride: RobstrideMotorsBus, dynamic: str) -> None:
            bus.read(dynamic, "joint")
            bus.write(dynamic, "joint", 0)
            bus.sync_read(dynamic)
            bus.sync_write(dynamic, {"joint": 0})
            damiao.read("Temperature_MOS", "joint")
            damiao.write("Kp", "joint", 1)
            damiao.sync_read(dynamic)
            damiao.sync_write("Kd", {"joint": 1})
            robstride.read("Present_Torque", "joint")
            robstride.write("Kp", "joint", 1)
            robstride.sync_read(dynamic)
            robstride.sync_write("Kd", {"joint": 1})
    """)
    program = tmp_path / "custom_robot.py"
    program.write_text(source)
    config = tmp_path / "mypy.ini"
    # Follow the public API and its decorators. Skip unrelated utility implementations.
    config.write_text(
        textwrap.dedent("""\
        [mypy]
        python_version = 3.12
        follow_imports = silent
        ignore_missing_imports = True

        [mypy-lerobot.utils.*]
        follow_imports = skip

        [mypy-lerobot.utils.decorators]
        follow_imports = silent

        [mypy-numpy.*]
        follow_imports = skip

        [mypy-rustypot.*]
        follow_imports = skip
    """)
    )
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(config), "--no-incremental", str(program)],
        cwd=repository,
        env={**os.environ, "MYPYPATH": str(repository / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    expected = set()
    for number, line in enumerate(source.splitlines(), 1):
        if match := re.search(r"# error: ([\w-]+)", line):
            expected.add((str(number), match.group(1)))
    actual = set(re.findall(r":(\d+): error: .* \[([\w-]+)\]", result.stdout))
    assert result.returncode == 1, result.stdout + result.stderr
    assert actual == expected, result.stdout + result.stderr
    assert result.stdout.count(": error:") == len(expected), result.stdout + result.stderr


@pytest.mark.parametrize(
    "register",
    ["Present_Position", "present_position", "PRESENT_POSITION", CustomRegister("pReSeNt_PoSiTiOn")],
)
def test_register_case_and_custom_marker_preserve_runtime_access(register):
    bus = MockMotorsBus("/dev/dummy-port", {"joint": Motor(1, "model_1", MotorNormMode.RANGE_M100_100)})
    bus.connect(handshake=False)
    transport = bus.opened[-1]
    try:
        bus.write(register, "joint", 123, normalize=False)
        assert bus.read(register, "joint", normalize=False) == 123
        bus.sync_write(register, {"joint": 456}, normalize=False)
        assert bus.sync_read(register, normalize=False) == {"joint": 456}
        assert transport.reads[-1] == (1, "present_position")
        assert transport.sync_writes[-1] == ([1], "present_position", [456])
    finally:
        bus.disconnect()


def test_custom_register_outside_the_generated_names():
    class CustomBus(MockMotorsBus):
        @staticmethod
        def _find_model(model):
            return 1234, DummyDefinition("CUSTOM", {model: 1234}, ("Vendor_Gain",))

    bus = CustomBus("/dev/dummy-port", {"joint": Motor(1, "custom", MotorNormMode.RANGE_M100_100)})
    bus.connect(handshake=False)
    register = CustomRegister("Vendor_Gain")
    try:
        bus.write(register, "joint", 123, normalize=False)
        assert bus.read(register, "joint", normalize=False) == 123
        bus.sync_write(register, {"joint": 456}, normalize=False)
        assert bus.sync_read(register, normalize=False) == {"joint": 456}
    finally:
        bus.disconnect()
