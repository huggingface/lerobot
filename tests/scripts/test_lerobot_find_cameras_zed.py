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

# Example of running a specific test:
# ```bash
# pytest tests/scripts/test_lerobot_find_cameras_zed.py::test_find_all_zed_cameras_lists_stereo_and_monocular
# ```

# The ZED branch of `lerobot-find-cameras` glues discovery of two SDK classes together and must not
# take the whole tool down when the ZED SDK is missing. Only the glue is tested here; the camera
# backends have their own suites.

import sys

import pytest

from lerobot.cameras import zed as zed_pkg
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.zed import ZedCamera, ZedOneCamera
from lerobot.scripts import lerobot_find_cameras as lfc

STEREO = {"name": "ZED X", "type": "zed", "id": 1, "serial_number": 1}
MONO = {"name": "ZED X One GS", "type": "zed_one", "id": 2, "serial_number": 2}


@pytest.fixture(name="fake_finders")
def fixture_fake_finders(monkeypatch):
    monkeypatch.setattr(ZedCamera, "find_cameras", staticmethod(lambda: [STEREO]))
    monkeypatch.setattr(ZedOneCamera, "find_cameras", staticmethod(lambda: [MONO]))


@pytest.fixture(name="no_other_cameras")
def fixture_no_other_cameras(monkeypatch):
    monkeypatch.setattr(lfc, "find_all_opencv_cameras", lambda: [])
    monkeypatch.setattr(lfc, "find_all_realsense_cameras", lambda: [])


class _SpyCamera:
    """Stands in for a camera class: records the config it was built with, connects instantly."""

    instances: list["_SpyCamera"] = []

    def __init__(self, config):
        self.config = config
        self.is_connected = False
        _SpyCamera.instances.append(self)

    def connect(self, warmup=True):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False


@pytest.fixture(name="spy")
def fixture_spy():
    _SpyCamera.instances = []
    yield _SpyCamera
    _SpyCamera.instances = []


# --- discovery -------------------------------------------------------------------------------------


def test_find_all_zed_cameras_lists_stereo_and_monocular(fake_finders):
    assert lfc.find_all_zed_cameras() == [STEREO, MONO]


def test_find_all_zed_cameras_survives_a_broken_backend(monkeypatch, caplog):
    def boom():
        raise RuntimeError("SDK exploded")

    monkeypatch.setattr(ZedCamera, "find_cameras", staticmethod(boom))
    monkeypatch.setattr(ZedOneCamera, "find_cameras", staticmethod(lambda: [MONO]))

    assert lfc.find_all_zed_cameras() == [MONO]
    assert any("stereo discovery failed" in r.getMessage() for r in caplog.records)


def test_find_all_zed_cameras_returns_empty_without_the_backend(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "lerobot.cameras.zed", None)  # makes the import raise

    assert lfc.find_all_zed_cameras() == []
    assert any("skipping ZED discovery" in r.getMessage() for r in caplog.records)


def test_zed_filter_runs_only_zed_discovery(monkeypatch, fake_finders):
    def not_expected():
        raise AssertionError("other backends must not be probed under the zed filter")

    monkeypatch.setattr(lfc, "find_all_opencv_cameras", not_expected)
    monkeypatch.setattr(lfc, "find_all_realsense_cameras", not_expected)

    assert lfc.find_and_print_cameras("zed") == [STEREO, MONO]


def test_zed_filter_is_case_insensitive(fake_finders, no_other_cameras):
    assert lfc.find_and_print_cameras("ZED") == [STEREO, MONO]


def test_unfiltered_discovery_includes_zed(fake_finders, no_other_cameras):
    assert lfc.find_and_print_cameras(None) == [STEREO, MONO]


# --- instantiation ---------------------------------------------------------------------------------


def test_create_camera_instance_builds_a_stereo_camera(monkeypatch, spy):
    monkeypatch.setattr(zed_pkg, "ZedCamera", spy)
    monkeypatch.setattr(zed_pkg, "ZedOneCamera", None)  # must not be touched

    result = lfc.create_camera_instance(STEREO, warmup_s=0)

    assert result == {"instance": spy.instances[0], "meta": STEREO}
    config = spy.instances[0].config
    assert isinstance(config, zed_pkg.ZedCameraConfig)
    assert config.serial_number == 1
    assert config.color_mode is ColorMode.RGB  # the script saves RGB via PIL
    assert config.warmup_s == 0
    assert spy.instances[0].is_connected


def test_create_camera_instance_routes_zed_one_to_the_monocular_backend(monkeypatch, spy):
    monkeypatch.setattr(zed_pkg, "ZedOneCamera", spy)
    monkeypatch.setattr(zed_pkg, "ZedCamera", None)  # must not be touched

    result = lfc.create_camera_instance(MONO)

    assert result is not None
    config = spy.instances[0].config
    assert isinstance(config, zed_pkg.ZedOneCameraConfig)
    assert config.serial_number == 2
    assert config.color_mode is ColorMode.RGB


def test_create_camera_instance_returns_none_when_the_camera_fails_to_open(monkeypatch, spy, caplog):
    def failing_connect(self, warmup=True):
        raise ConnectionError("Failed to open")

    monkeypatch.setattr(spy, "connect", failing_connect)
    monkeypatch.setattr(zed_pkg, "ZedCamera", spy)

    assert lfc.create_camera_instance(STEREO) is None
    assert any("Failed to connect" in r.getMessage() for r in caplog.records)
