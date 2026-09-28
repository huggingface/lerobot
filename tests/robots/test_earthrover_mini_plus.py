#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""The EarthRover Mini Plus reaches the SDK the caller configured.

``EarthRoverMiniPlusConfig.sdk_url`` is the only field that says which SDK
server this robot talks to. Every HTTP call the robot makes -- the connect
probe, the telemetry read, the two camera endpoints and the drive command --
has to be built from that value, or a caller who points the robot at a remote
SDK is silently served by whatever answers on the default port instead.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from lerobot.robots.earthrover_mini_plus import EarthRoverMiniPlus, EarthRoverMiniPlusConfig
from lerobot.utils.errors import DeviceNotConnectedError

REMOTE_URL = "http://scout.local:8002"
DEFAULT_URL = "http://localhost:8000"


# A /data body carrying every key ``get_observation`` reads. The four sensor
# arrays hold ``[values..., timestamp]`` entries, matching the SDK burst format
# the robot documents.
TELEMETRY = {
    "speed": 0.4,
    "battery": 88,
    "orientation": 137.0,
    "latitude": 52.37,
    "longitude": 4.89,
    "gps_signal": 71.0,
    "signal_level": 4,
    "vibration": 0.02,
    "lamp": 0,
    "accels": [[0.01, 0.02, 9.81, 1000]],
    "gyros": [[0.0, 0.0, 0.1, 1000]],
    "mags": [[12.0, -7.0, 41.0, 1000]],
    "rpms": [[10, 11, 12, 13, 1000]],
}


def _ok_response(payload: dict | None = None) -> MagicMock:
    """A 200 response whose JSON body is a plausible SDK reply."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = TELEMETRY if payload is None else payload
    return response


class TestTheConfiguredSdkUrlIsHonoured:
    """The configured URL reaches the attribute every request is built from."""

    def test_a_configured_url_is_adopted(self):
        robot = EarthRoverMiniPlus(EarthRoverMiniPlusConfig(sdk_url=REMOTE_URL))
        assert robot.sdk_base_url == REMOTE_URL

    def test_the_default_url_is_unchanged(self):
        """Control: a caller who configures nothing still gets the documented default.

        This is what keeps the fix from being a behaviour change for the local
        workflow the setup guide describes.
        """
        robot = EarthRoverMiniPlus(EarthRoverMiniPlusConfig())
        assert robot.sdk_base_url == DEFAULT_URL


class TestEveryRequestTargetsTheConfiguredSdk:
    """No endpoint keeps a URL of its own.

    Asserting on the attribute alone would still pass if a single call site
    interpolated a hardcoded host, so this drives the robot through every
    endpoint it owns and reads back the URLs it asked for.
    """

    def _urls(self, robot) -> list[str]:
        with (
            patch("lerobot.robots.earthrover_mini_plus.robot_earthrover_mini_plus.requests.get") as get,
            patch("lerobot.robots.earthrover_mini_plus.robot_earthrover_mini_plus.requests.post") as post,
        ):
            get.return_value = _ok_response()
            post.return_value = _ok_response({"ok": True})

            robot.connect(calibrate=False)
            robot.get_observation()
            robot.send_action({"linear_velocity": 0.5, "angular_velocity": -0.5})

            calls = list(get.call_args_list) + list(post.call_args_list)
        return [call.args[0] for call in calls if call.args]

    def test_every_url_is_built_from_the_configured_base(self):
        urls = self._urls(EarthRoverMiniPlus(EarthRoverMiniPlusConfig(sdk_url=REMOTE_URL)))
        assert urls, "expected the robot to issue at least one request"
        assert all(url.startswith(REMOTE_URL) for url in urls), urls

    def test_no_request_reaches_the_default_host(self):
        """The failure this pins is a request served by the wrong rover."""
        urls = self._urls(EarthRoverMiniPlus(EarthRoverMiniPlusConfig(sdk_url=REMOTE_URL)))
        assert not any(url.startswith(DEFAULT_URL) for url in urls), urls

    def test_the_drive_command_reaches_the_configured_sdk(self):
        """A control command sent to the wrong host is the safety-relevant case."""
        urls = self._urls(EarthRoverMiniPlus(EarthRoverMiniPlusConfig(sdk_url=REMOTE_URL)))
        assert f"{REMOTE_URL}/control" in urls, urls


class TestAnUnreachableSdkIsNamedByItsConfiguredUrl:
    """A refusal has to name the server the caller asked for.

    Reporting the default URL sends the reader to check a host they never
    configured.
    """

    def test_the_connect_refusal_names_the_configured_url(self):
        robot = EarthRoverMiniPlus(EarthRoverMiniPlusConfig(sdk_url=REMOTE_URL))
        with patch("lerobot.robots.earthrover_mini_plus.robot_earthrover_mini_plus.requests.get") as get:
            get.side_effect = requests.RequestException("refused")
            with pytest.raises(DeviceNotConnectedError) as excinfo:
                robot.connect(calibrate=False)
        message = str(excinfo.value)
        assert REMOTE_URL in message, message
        assert DEFAULT_URL not in message, message
