"""Lightweight type aliases that do not require torch.

These are the most commonly imported names from the processor package.
Keeping them in a separate module allows robot/teleoperator code to
import them without pulling in torch or any other heavy dependency.
"""

from __future__ import annotations

from typing import Any

RobotAction = dict[str, Any]
RobotObservation = dict[str, Any]
