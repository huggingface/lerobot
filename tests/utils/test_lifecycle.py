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

import logging

import pytest

from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.lifecycle import Cleanup, idempotent_connect


class FakeResource:
    """A sub-device honouring the motor bus / camera contract: disconnecting twice raises."""

    def __init__(self, *, connect_error: Exception | None = None, disconnect_error: Exception | None = None):
        self.is_connected = False
        self.connect_error = connect_error
        self.disconnect_error = disconnect_error
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True
        if self.connect_error is not None:
            raise self.connect_error

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("already released")
        self.disconnect_calls += 1
        if self.disconnect_error is not None:
            raise self.disconnect_error
        self.is_connected = False


class FakeDevice:
    def __init__(self, *resources: FakeResource):
        self.resources = list(resources)
        self.configure_error: Exception | None = None
        self.configure_calls = 0

    def __str__(self) -> str:
        return "fake device"

    @property
    def is_connected(self) -> bool:
        return all(resource.is_connected for resource in self.resources)

    @idempotent_connect
    def connect(self, calibrate: bool = True) -> None:
        for resource in self.resources:
            if not resource.is_connected:
                resource.connect()
        self.configure_calls += 1
        if self.configure_error is not None:
            raise self.configure_error

    def disconnect(self) -> None:
        with Cleanup(self) as cleanup:
            for index, resource in enumerate(self.resources):
                with cleanup.step(f"resource {index}"):
                    resource.disconnect()


def test_connect_and_disconnect_are_idempotent():
    device = FakeDevice(FakeResource(), FakeResource())

    device.connect()
    device.connect()

    assert all(resource.connect_calls == 1 for resource in device.resources)
    assert device.configure_calls == 1

    device.disconnect()
    device.disconnect()

    assert all(resource.disconnect_calls == 1 for resource in device.resources)


def test_connect_resumes_a_partial_connection():
    first, second = FakeResource(), FakeResource()
    first.connect()
    device = FakeDevice(first, second)

    device.connect()

    assert device.is_connected
    assert (first.connect_calls, second.connect_calls) == (1, 1)


def test_connect_releases_everything_when_the_body_fails():
    device = FakeDevice(FakeResource(), FakeResource())
    configure_error = RuntimeError("configuration failed")
    device.configure_error = configure_error

    with pytest.raises(RuntimeError, match="configuration failed") as exc_info:
        device.connect()

    assert exc_info.value is configure_error
    assert any("while connecting fake device" in note for note in exc_info.value.__notes__)
    assert not any(resource.is_connected for resource in device.resources)


def test_connect_reports_body_and_rollback_failures_together():
    connect_error = RuntimeError("second failed")
    rollback_error = OSError("first stuck")
    device = FakeDevice(
        FakeResource(disconnect_error=rollback_error), FakeResource(connect_error=connect_error)
    )

    with pytest.raises(ExceptionGroup) as exc_info:
        device.connect()

    assert exc_info.value.exceptions == (connect_error, rollback_error)


def test_connect_passes_arguments_through():
    seen = []

    class Device:
        is_connected = False

        def __str__(self):
            return "device"

        @idempotent_connect
        def connect(self, calibrate: bool = True) -> None:
            seen.append(calibrate)

        def disconnect(self) -> None:
            pass

    Device().connect(calibrate=False)

    assert seen == [False]


def test_cleanup_releases_remaining_resources_after_a_failure():
    disconnect_error = OSError("first stuck")
    first, second = FakeResource(disconnect_error=disconnect_error), FakeResource()
    device = FakeDevice(first, second)
    device.connect()

    with pytest.raises(OSError, match="first stuck") as exc_info:
        device.disconnect()

    assert exc_info.value is disconnect_error
    assert any("while releasing resource 0 of fake device" in note for note in exc_info.value.__notes__)
    assert second.disconnect_calls == 1


def test_cleanup_groups_several_failures():
    first_error, second_error = OSError("first"), RuntimeError("second")
    device = FakeDevice(
        FakeResource(disconnect_error=first_error), FakeResource(disconnect_error=second_error)
    )
    device.connect()

    with pytest.raises(ExceptionGroup) as exc_info:
        device.disconnect()

    assert exc_info.value.exceptions == (first_error, second_error)


def test_cleanup_step_stops_at_the_first_failing_statement():
    released = []

    with pytest.raises(RuntimeError, match="boom"), Cleanup("owner") as cleanup:
        with cleanup.step("the handle"):
            raise RuntimeError("boom")
            released.append("must not run")  # noqa: B012 - unreachable on purpose
        with cleanup.step("the other handle"):
            released.append("other")

    assert released == ["other"]


def test_cleanup_ignores_the_configured_exceptions_only():
    with pytest.raises(ValueError, match="not ignored"), Cleanup("owner", ignore=(KeyError,)) as cleanup:
        with cleanup.step("a"):
            raise KeyError("ignored")
        with cleanup.step("b"):
            raise ValueError("not ignored")


def test_cleanup_does_not_swallow_errors_raised_outside_a_step():
    with pytest.raises(KeyError), Cleanup("owner"):
        raise KeyError("programming error")


def test_cleanup_logs_only_when_something_was_released(caplog):
    device = FakeDevice(FakeResource())

    with caplog.at_level(logging.INFO, logger="lerobot.utils.lifecycle"):
        device.disconnect()
        assert "disconnected" not in caplog.text

        device.connect()
        device.disconnect()

    assert caplog.text.count("fake device disconnected.") == 1
