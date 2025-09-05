import sys
import types
from unittest.mock import MagicMock


def _install_reachy2_sdk_stub():
    sdk = types.ModuleType("reachy2_sdk")
    sdk.__path__ = []
    sdk.ReachySDK = MagicMock(name="ReachySDK")

    media = types.ModuleType("reachy2_sdk.media")
    media.__path__ = []
    camera = types.ModuleType("reachy2_sdk.media.camera")
    camera.CameraView = MagicMock(name="CameraView")
    camera_manager = types.ModuleType("reachy2_sdk.media.camera_manager")
    camera_manager.CameraManager = MagicMock(name="CameraManager")

    sdk.media = media
    media.camera = camera
    media.camera_manager = camera_manager

    # Register in sys.modules
    sys.modules.setdefault("reachy2_sdk", sdk)
    sys.modules.setdefault("reachy2_sdk.media", media)
    sys.modules.setdefault("reachy2_sdk.media.camera", camera)
    sys.modules.setdefault("reachy2_sdk.media.camera_manager", camera_manager)


def pytest_sessionstart(session):
    _install_reachy2_sdk_stub()
