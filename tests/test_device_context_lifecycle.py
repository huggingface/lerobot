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

from unittest.mock import MagicMock

import pytest

from lerobot.robots.robot import Robot
from lerobot.teleoperators.teleoperator import Teleoperator


class FakeRobotContext:
    __enter__ = Robot.__enter__
    __exit__ = Robot.__exit__

    def __init__(self, connect_error=None, disconnect_error=None):
        self.connect = MagicMock(side_effect=connect_error)
        self.disconnect = MagicMock(side_effect=disconnect_error)

    def __str__(self):
        return "fake robot"


class FakeTeleoperatorContext:
    __enter__ = Teleoperator.__enter__
    __exit__ = Teleoperator.__exit__

    def __init__(self, connect_error=None, disconnect_error=None):
        self.connect = MagicMock(side_effect=connect_error)
        self.disconnect = MagicMock(side_effect=disconnect_error)

    def __str__(self):
        return "fake teleoperator"


@pytest.mark.parametrize("device_cls", [FakeRobotContext, FakeTeleoperatorContext])
def test_context_connects_on_entry_and_disconnects_on_exit(device_cls):
    device = device_cls()

    with device as entered:
        assert entered is device
        device.connect.assert_called_once_with()
        device.disconnect.assert_not_called()

    device.disconnect.assert_called_once_with()


@pytest.mark.parametrize("device_cls", [FakeRobotContext, FakeTeleoperatorContext])
def test_context_entry_propagates_a_connect_failure_without_extra_cleanup(device_cls):
    # `connect()` is responsible for releasing what it acquired (see `idempotent_connect`),
    # so the context manager does not disconnect on top of it.
    connect_error = RuntimeError("connect failed")
    device = device_cls(connect_error=connect_error)

    with pytest.raises(RuntimeError, match="connect failed") as exc_info, device:
        pass

    assert exc_info.value is connect_error
    device.disconnect.assert_not_called()


@pytest.mark.parametrize("device_cls", [FakeRobotContext, FakeTeleoperatorContext])
def test_context_exit_reports_a_cleanup_failure(device_cls):
    disconnect_error = OSError("cleanup failed")
    device = device_cls(disconnect_error=disconnect_error)

    with pytest.raises(OSError, match="cleanup failed") as exc_info, device:
        pass

    assert exc_info.value is disconnect_error
    assert any("while exiting the context" in note for note in exc_info.value.__notes__)


@pytest.mark.parametrize("device_cls", [FakeRobotContext, FakeTeleoperatorContext])
def test_context_exit_reports_body_and_cleanup_failures(device_cls):
    body_error = RuntimeError("body failed")
    disconnect_error = OSError("cleanup failed")
    device = device_cls(disconnect_error=disconnect_error)

    with pytest.raises(BaseExceptionGroup) as exc_info, device:
        raise body_error

    assert exc_info.value.exceptions == (body_error, disconnect_error)
