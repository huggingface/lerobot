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

"""Generate register-name types from rustypot and write the Python source to stdout."""

import json
from importlib.metadata import version
from pathlib import Path

import rustypot


def generate_source() -> str:
    """Return static register types from rustypot, without opening a serial port."""
    names: set[str] = set()
    for controller in vars(rustypot).values():
        registers = getattr(controller, "registers", None)
        if (
            isinstance(controller, type)
            and callable(registers)
            and callable(getattr(controller, "models", None))
        ):
            names.update(register.name for register in registers())
    if not names:
        raise RuntimeError("rustypot exports no controller register names")
    entries = "".join(
        "    " + ", ".join(json.dumps(value) for value in (name, name.upper(), name.title())) + ",\n"
        for name in sorted(names)
    )
    header = Path(__file__).read_text().split('"""', 1)[0]
    return (
        header
        + f'''"""Generated from rustypot {version("rustypot")}. See scripts/generate_motor_registers.py."""

from typing import Literal, NewType

# fmt: off
type RegisterName = Literal[
{entries}]
# fmt: on

# Explicit access for dynamic names, custom registers, or other case patterns.
CustomRegister = NewType("CustomRegister", str)
type Register = RegisterName | CustomRegister
'''
    )


if __name__ == "__main__":
    print(generate_source(), end="")
