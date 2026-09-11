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

"""Stand-in for the `pyzed.sl` module used by the ZED camera tests.

`pyzed` ships with the ZED SDK and is not installable from PyPI, so it cannot be a CI dependency.
`lerobot.cameras.zed` imports it behind a try/except; the tests replace the module handle with the
namespace built by `make_fake_sl`, which mimics the small subset of the SDK the backends touch. The
fake cameras never assert: they record every call so a test can check them on the main thread.
"""

import contextlib
import time
from enum import Enum, IntEnum
from types import SimpleNamespace

import numpy as np

DEFAULT_SERIAL = 12345678
STEREO_SERIAL = 87654321
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 60
# Synthetic left-view pixel, in the BGRA order the ZED SDK returns for sl.VIEW.LEFT.
BGRA_PIXEL = (10, 20, 30, 255)
# MEASURE.DEPTH_U16_MM hands back uint16 millimetres with 0 marking an invalid measurement.
DEPTH_MM = 1500
INVALID_DEPTH = 0
# The rectified and raw calibrations differ, as on real hardware.
RECTIFIED_FX = 700.0
UNRECTIFIED_FX = 530.0


class FakeResolution(Enum):
    AUTO = 0
    HD2K = 1
    HD1200 = 2
    HD1080 = 3
    HD720 = 4
    SVGA = 5
    VGA = 6
    LAST = 7


class FakeDepthMode(Enum):
    # Mirrors sl.DEPTH_MODE in SDK 5.4; PERFORMANCE, QUALITY and ULTRA are deprecated there.
    NONE = 0
    PERFORMANCE = 1
    QUALITY = 2
    ULTRA = 3
    NEURAL_LIGHT = 4
    NEURAL = 5
    NEURAL_PLUS = 6
    CUSTOM = 7
    LAST = 8


class FakeErrorCode(IntEnum):
    # IntEnum, like sl.ERROR_CODE: negative values are warnings, positive values are errors.
    SENSOR_CONFIGURATION_CHANGED = -6
    POTENTIAL_CALIBRATION_ISSUE = -5
    CONFIGURATION_FALLBACK = -4
    CORRUPTED_FRAME = -2
    CAMERA_REBOOTING = -1
    SUCCESS = 0
    FAILURE = 1
    CAMERA_NOT_DETECTED = 2
    INVALID_RESOLUTION = 3


class FakeUnit(Enum):
    MILLIMETER = 0
    CENTIMETER = 1
    METER = 2


class FakeView(Enum):
    LEFT = 0
    RIGHT = 1
    LEFT_UNRECTIFIED = 2


class FakeMeasure(Enum):
    DEPTH = 0
    XYZ = 1
    DEPTH_U16_MM = 2


class FakeTimeReference(Enum):
    IMAGE = 0
    CURRENT = 1


class FakeMat:
    """Stand-in for `sl.Mat`: a reusable buffer the SDK writes each frame into."""

    def __init__(self):
        self.data = None

    def get_data(self):
        return self.data


class FakeTimestamp:
    def __init__(self, nanoseconds: int):
        self._nanoseconds = nanoseconds

    def get_nanoseconds(self) -> int:
        return self._nanoseconds


class FakeInitParameters:
    """Stand-in for `sl.InitParameters`, the *stereo* open parameters."""

    def __init__(self):
        self.camera_resolution = FakeResolution.AUTO
        self.camera_fps = 0
        self.depth_mode = FakeDepthMode.NEURAL
        # the remaining defaults mirror pyzed 5.4 (they differ from the C++ header for validity_check)
        self.coordinate_units = FakeUnit.MILLIMETER
        self.sdk_verbose = 1
        self.enable_image_validity_check = 0
        self.serial_number = None

    def set_from_serial_number(self, serial_number):
        self.serial_number = int(serial_number)


class FakeInitParametersOne:
    """Stand-in for `sl.InitParametersOne`: like `sl.InitParameters` but with no `depth_mode`."""

    def __init__(self):
        self.camera_resolution = FakeResolution.AUTO
        self.camera_fps = 0
        self.coordinate_units = FakeUnit.MILLIMETER
        self.sdk_verbose = 1
        self.serial_number = None

    def set_from_serial_number(self, serial_number):
        self.serial_number = int(serial_number)


class FakeRuntimeParameters:
    """Exists on the real `sl` module (for `sl.Camera.grab`), but `sl.CameraOne.grab` takes none."""


def fake_device(camera_model="ZED X", serial_number=DEFAULT_SERIAL, path="/dev/i2c-14"):
    """Stand-in for `sl.DeviceProperties`, as returned by the SDK device lists."""
    return SimpleNamespace(camera_model=camera_model, serial_number=serial_number, path=path)


class _FakeCameraBase:
    """Behaviour shared by the stereo and monocular fakes: a BGRA image served off one buffer."""

    def __init__(self, width, height, fps, open_status, grab_status, grab_delay_s, served_size):
        self.width = width
        self.height = height
        self.fps = fps
        self.open_status = open_status
        self.grab_status = grab_status
        self.grab_delay_s = grab_delay_s

        self.init_parameters = None
        self.opened = False
        self.close_count = 0
        self.grab_count = 0
        # Recorded rather than asserted: an assert inside a fake would fire on the read thread.
        self.requested_views = []
        self.requested_measures = []
        self.time_references = []

        # The SDK hands back a view on an internal buffer, so allocate these once. `served_size`
        # lets a test hand out frames that do not match the mode the camera reports.
        served_w, served_h = served_size or (width, height)
        self._color = np.zeros((served_h, served_w, 4), dtype=np.uint8)
        self._color[:, :] = BGRA_PIXEL
        # The SDK collapses occluded / too-far / too-close pixels to 0 in DEPTH_U16_MM.
        self._depth = np.full((served_h, served_w), DEPTH_MM, dtype=np.uint16)
        self._depth[0, 0:3] = INVALID_DEPTH

    def open(self, init_parameters):
        self.init_parameters = init_parameters
        self.opened = self.open_status <= FakeErrorCode.SUCCESS
        return self.open_status

    def is_opened(self):
        return self.opened

    def _grab(self):
        self.grab_count += 1
        time.sleep(self.grab_delay_s)  # keep the background read thread from spinning flat out
        return self.grab_status

    def retrieve_image(self, mat, view):
        self.requested_views.append(view)
        mat.data = self._color
        return FakeErrorCode.SUCCESS

    def get_timestamp(self, time_reference):
        self.time_references.append(time_reference)
        return FakeTimestamp(1_700_000_000_000_000_000 + self.grab_count)

    def close(self):
        self.opened = False
        self.close_count += 1


class FakeCamera(_FakeCameraBase):
    """Stand-in for `sl.Camera` (stereo): BGRA colour plus uint16 millimetre depth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One entry per grab() call, holding the RuntimeParameters it received.
        self.grab_runtime_parameters = []

    def get_camera_information(self):
        return SimpleNamespace(
            camera_configuration=SimpleNamespace(
                resolution=SimpleNamespace(width=self.width, height=self.height),
                fps=self.fps,
                calibration_parameters=SimpleNamespace(
                    left_cam=SimpleNamespace(fx=RECTIFIED_FX, fy=701.0, cx=self.width / 2, cy=self.height / 2)
                ),
                calibration_parameters_raw=SimpleNamespace(
                    left_cam=SimpleNamespace(
                        fx=UNRECTIFIED_FX, fy=531.0, cx=self.width / 2 + 9, cy=self.height / 2 + 7
                    )
                ),
            ),
        )

    def grab(self, runtime_parameters):
        self.grab_runtime_parameters.append(runtime_parameters)
        return self._grab()

    def retrieve_measure(self, mat, measure):
        self.requested_measures.append(measure)
        mat.data = self._depth
        return FakeErrorCode.SUCCESS


class FakeCameraOne(_FakeCameraBase):
    """Stand-in for `sl.CameraOne` (single sensor): BGRA colour only, flat calibration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One entry per grab() call, holding how many positional arguments it received: the
        # monocular API has no RuntimeParameters, so these must all be 0.
        self.grab_arg_counts = []

    def get_camera_information(self):
        # `CameraOne` exposes a *flat* calibration (no `.left_cam`): there is a single sensor.
        return SimpleNamespace(
            camera_configuration=SimpleNamespace(
                resolution=SimpleNamespace(width=self.width, height=self.height),
                fps=self.fps,
                calibration_parameters=SimpleNamespace(
                    fx=RECTIFIED_FX, fy=701.0, cx=self.width / 2, cy=self.height / 2
                ),
                calibration_parameters_raw=SimpleNamespace(
                    fx=UNRECTIFIED_FX, fy=531.0, cx=self.width / 2 + 9, cy=self.height / 2 + 7
                ),
            )
        )

    def grab(self, *args):
        self.grab_arg_counts.append(len(args))
        return self._grab()


def make_fake_sl(
    width=CAPTURE_WIDTH,
    height=CAPTURE_HEIGHT,
    fps=CAPTURE_FPS,
    open_status=FakeErrorCode.SUCCESS,
    grab_status=FakeErrorCode.SUCCESS,
    grab_delay_s=0.002,
    served_size=None,
    stereo_device_list=None,
    mono_device_list=None,
):
    """Build a stand-in for the `pyzed.sl` module.

    The returned namespace also exposes `cameras`, the list of fake `sl.Camera` / `sl.CameraOne`
    objects the backend instantiated, so tests can assert on what was passed to the SDK.
    """
    fake = SimpleNamespace(
        RESOLUTION=FakeResolution,
        DEPTH_MODE=FakeDepthMode,
        ERROR_CODE=FakeErrorCode,
        UNIT=FakeUnit,
        VIEW=FakeView,
        MEASURE=FakeMeasure,
        TIME_REFERENCE=FakeTimeReference,
        Mat=FakeMat,
        InitParameters=FakeInitParameters,
        InitParametersOne=FakeInitParametersOne,
        RuntimeParameters=FakeRuntimeParameters,
        cameras=[],
    )
    stereo_devices = [fake_device()] if stereo_device_list is None else stereo_device_list
    mono_devices = [fake_device("ZED X One GS")] if mono_device_list is None else mono_device_list
    camera_kwargs = {
        "width": width,
        "height": height,
        "fps": fps,
        "open_status": open_status,
        "grab_status": grab_status,
        "grab_delay_s": grab_delay_s,
        "served_size": served_size,
    }

    class Camera(FakeCamera):
        def __init__(self):
            super().__init__(**camera_kwargs)
            fake.cameras.append(self)

        @staticmethod
        def get_device_list():
            return stereo_devices

    class CameraOne(FakeCameraOne):
        def __init__(self):
            super().__init__(**camera_kwargs)
            fake.cameras.append(self)

        @staticmethod
        def get_device_list():
            return mono_devices

    fake.Camera = Camera
    fake.CameraOne = CameraOne
    return fake


class ZedTestHarness:
    """Patches the backend's `pyzed.sl` handle and hands out cameras that always get cleaned up.

    `modules` are the backend modules whose `sl` global must be replaced: the shared machinery
    resolves `sl` in `base_zed`, the model-specific calls in `camera_zed` / `camera_zed_one`.
    """

    def __init__(self, monkeypatch, modules, camera_cls, config_cls):
        self._monkeypatch = monkeypatch
        self._modules = modules
        self._camera_cls = camera_cls
        self._config_cls = config_cls
        self._cameras = []
        self.sl = None
        self.patch_sl()

    def patch_sl(self, **kwargs):
        """(Re-)install a fake `sl` module, optionally tweaking how the fake camera behaves."""
        self.sl = make_fake_sl(**kwargs)
        for module in self._modules:
            self._monkeypatch.setattr(module, "sl", self.sl)
        self._monkeypatch.setattr(self._modules[0], "_pyzed_import_error", None)
        return self.sl

    def config(self, **kwargs):
        return self._config_cls(**{"serial_number": DEFAULT_SERIAL, "warmup_s": 0, **kwargs})

    def camera(self, **kwargs):
        camera = self._camera_cls(self.config(**kwargs))
        self._cameras.append(camera)
        return camera

    @property
    def last_camera(self):
        """The most recent fake SDK camera the backend instantiated."""
        return self.sl.cameras[-1]

    def cleanup(self):
        # Stop every read thread before monkeypatch restores the real `sl`.
        for camera in self._cameras:
            with contextlib.suppress(Exception):
                camera.disconnect()
