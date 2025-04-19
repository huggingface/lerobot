# ruff: noqa: N802,N803
import abc
from typing import Optional, Tuple

import numpy as np


# --- Interface Definition ---
class IVideoCapture(abc.ABC):
    """Interface for the cv2.VideoCapture class."""

    @abc.abstractmethod
    def __init__(self, index: int | str, backend: Optional[int] = None):
        pass

    @abc.abstractmethod
    def isOpened(self) -> bool:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass

    @abc.abstractmethod
    def set(self, propId: int, value: float) -> bool:
        pass

    @abc.abstractmethod
    def get(self, propId: int) -> float:
        pass

    @abc.abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass


class IOpenCVSDK(abc.ABC):
    """Interface defining the contract for OpenCV SDK interactions."""

    # --- Constants ---
    CAP_PROP_FPS: int
    CAP_PROP_FRAME_WIDTH: int
    CAP_PROP_FRAME_HEIGHT: int
    COLOR_BGR2RGB: int
    ROTATE_90_COUNTERCLOCKWISE: int
    ROTATE_90_CLOCKWISE: int
    ROTATE_180: int
    CAP_V4L2: int
    CAP_DSHOW: int
    CAP_AVFOUNDATION: int
    CAP_ANY: int

    # --- Inner Class Type Hint ---
    VideoCapture: type[IVideoCapture]

    # --- Methods ---
    @abc.abstractmethod
    def setNumThreads(self, nthreads: int) -> None:
        pass

    @abc.abstractmethod
    def cvtColor(self, src: np.ndarray, code: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def rotate(self, src: np.ndarray, rotateCode: int) -> np.ndarray:
        pass


# --- Real SDK Adapter ---
class OpenCVSDKAdapter(IOpenCVSDK):
    """Adapts the real cv2 library to the IOpenCVSDK interface."""

    _cv2 = None

    def __init__(self):
        try:
            import cv2

            OpenCVSDKAdapter._cv2 = cv2
        except ImportError as e:
            raise ImportError(
                "OpenCV (cv2) is not installed. Please install it to use the real camera."
            ) from e

        # --- Constants ---
        self.CAP_PROP_FPS = self._cv2.CAP_PROP_FPS
        self.CAP_PROP_FRAME_WIDTH = self._cv2.CAP_PROP_FRAME_WIDTH
        self.CAP_PROP_FRAME_HEIGHT = self._cv2.CAP_PROP_FRAME_HEIGHT
        self.COLOR_BGR2RGB = self._cv2.COLOR_BGR2RGB
        self.ROTATE_90_COUNTERCLOCKWISE = self._cv2.ROTATE_90_COUNTERCLOCKWISE
        self.ROTATE_90_CLOCKWISE = self._cv2.ROTATE_90_CLOCKWISE
        self.ROTATE_180 = self._cv2.ROTATE_180
        self.CAP_V4L2 = self._cv2.CAP_V4L2
        self.CAP_DSHOW = self._cv2.CAP_DSHOW
        self.CAP_AVFOUNDATION = self._cv2.CAP_AVFOUNDATION
        self.CAP_ANY = self._cv2.CAP_ANY

        # --- Inner Class Implementation ---
        class RealVideoCapture(IVideoCapture):
            def __init__(self, index: int | str, backend: Optional[int] = None):
                self._cap = OpenCVSDKAdapter._cv2.VideoCapture(index, backend)

            def isOpened(self) -> bool:
                return self._cap.isOpened()

            def release(self) -> None:
                self._cap.release()

            def set(self, propId: int, value: float) -> bool:
                return self._cap.set(propId, value)

            def get(self, propId: int) -> float:
                return self._cap.get(propId)

            def read(self) -> Tuple[bool, Optional[np.ndarray]]:
                return self._cap.read()

            def __del__(self):
                if hasattr(self, "_cap") and self._cap and self._cap.isOpened():
                    self._cap.release()

        self.VideoCapture = RealVideoCapture

    # --- Methods ---
    def setNumThreads(self, nthreads: int) -> None:
        self._cv2.setNumThreads(nthreads)

    def cvtColor(self, src: np.ndarray, code: int) -> np.ndarray:
        return self._cv2.cvtColor(src, code)

    def rotate(self, src: np.ndarray, rotateCode: int) -> np.ndarray:
        return self._cv2.rotate(src, rotateCode)


# Emulates the cheap USB camera
VALID_INDICES = {0, 1, 2, "/dev/video0", "/dev/video1", "/dev/video2"}
DEFAULT_FPS = 30.0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


# --- Fake SDK Adapter ---
class FakeOpenCVSDKAdapter(IOpenCVSDK):
    """Implements the IOpenCVSDK interface with fake behavior for testing."""

    # --- Constants ---
    CAP_PROP_FPS = DEFAULT_FPS
    CAP_PROP_FRAME_WIDTH = DEFAULT_WIDTH
    CAP_PROP_FRAME_HEIGHT = DEFAULT_HEIGHT
    COLOR_BGR2RGB = 99
    ROTATE_90_COUNTERCLOCKWISE = -90
    ROTATE_90_CLOCKWISE = 90
    ROTATE_180 = 180
    CAP_V4L2 = 91
    CAP_DSHOW = 92
    CAP_AVFOUNDATION = 93
    CAP_ANY = 90

    _cameras_opened: dict[int | str, bool] = {}
    _camera_properties: dict[tuple[int | str, int], float] = {}
    _simulated_image: np.ndarray = np.random.randint(
        0, 256, (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8
    )
    _simulated_fps: float = DEFAULT_FPS
    _image_read_count: int = 0
    _fail_read_after: Optional[int] = None  # Simulate read failure

    @classmethod
    def init_configure_fake(
        cls,
        simulated_image: Optional[np.ndarray] = None,
        simulated_fps: Optional[float] = None,
        fail_read_after: Optional[int] = None,
    ):
        if simulated_image is not None:
            cls._simulated_image = simulated_image
        if simulated_fps is not None:
            cls._simulated_fps = simulated_fps
        cls._fail_read_after = fail_read_after
        cls._image_read_count = 0
        cls._cameras_opened = {}
        cls._camera_properties = {}

    @classmethod
    def configure_fake_simulated_image(cls, simulated_image: Optional[np.ndarray] = None):
        if simulated_image is not None:
            cls._simulated_image = simulated_image

    @classmethod
    def configure_fail_read_after(cls, fail_read_after: Optional[int] = None):
        cls._fail_read_after = fail_read_after

    @classmethod
    def configure_fake_simulated_fps(cls, simulated_fps: Optional[float] = None):
        if simulated_fps is not None:
            cls._simulated_fps = simulated_fps

    # --- Inner Class Implementation ---
    class FakeVideoCapture(IVideoCapture):
        def __init__(self, index: int | str, backend: Optional[int] = None):
            self.index = index
            self.backend = backend
            valid_indices = VALID_INDICES
            if self.index in valid_indices:
                FakeOpenCVSDKAdapter._cameras_opened[self.index] = True
                print(f"[FAKE SDK] Opened camera {self.index}")
                # Set some default fake properties
                FakeOpenCVSDKAdapter._camera_properties[(self.index, FakeOpenCVSDKAdapter.CAP_PROP_FPS)] = (
                    DEFAULT_FPS
                )
                FakeOpenCVSDKAdapter._camera_properties[
                    (self.index, FakeOpenCVSDKAdapter.CAP_PROP_FRAME_WIDTH)
                ] = float(FakeOpenCVSDKAdapter._simulated_image.shape[1])
                FakeOpenCVSDKAdapter._camera_properties[
                    (self.index, FakeOpenCVSDKAdapter.CAP_PROP_FRAME_HEIGHT)
                ] = float(FakeOpenCVSDKAdapter._simulated_image.shape[0])
            else:
                FakeOpenCVSDKAdapter._cameras_opened[self.index] = False
                print(f"[FAKE SDK] Failed to open camera {self.index}")

        def isOpened(self) -> bool:
            return FakeOpenCVSDKAdapter._cameras_opened.get(self.index, False)

        def release(self) -> None:
            if self.index in FakeOpenCVSDKAdapter._cameras_opened:
                FakeOpenCVSDKAdapter._cameras_opened[self.index] = False
                print(f"[FAKE SDK] Released camera {self.index}")
                # Clear properties on release
                props_to_remove = [k for k in FakeOpenCVSDKAdapter._camera_properties if k[0] == self.index]
                for k in props_to_remove:
                    del FakeOpenCVSDKAdapter._camera_properties[k]

        def set(self, propId: int, value: float) -> bool:
            if not self.isOpened():
                return False
            print(
                f"[FAKE SDK] Ignoring set property {propId} = {value} for camera {self.index} to preserve state."
            )
            # FakeOpenCVSDKAdapter._camera_properties[(self.index, propId)] = value
            # Simulate failure for specific unrealistic settings if needed
            return True

        def get(self, propId: int) -> float:
            if not self.isOpened():
                return 0.0  # Or raise error? Mimic cv2 behavior
            val = FakeOpenCVSDKAdapter._camera_properties.get((self.index, propId))
            print(f"[FAKE SDK] Get property {propId} for camera {self.index} -> {val}")
            return val

        def read(self) -> Tuple[bool, Optional[np.ndarray]]:
            if not self.isOpened():
                print(f"[FAKE SDK] Read failed: Camera {self.index} not open.")
                return False, None

            FakeOpenCVSDKAdapter._image_read_count += 1
            if (
                FakeOpenCVSDKAdapter._fail_read_after is not None
                and FakeOpenCVSDKAdapter._image_read_count > FakeOpenCVSDKAdapter._fail_read_after
            ):
                print(
                    f"[FAKE SDK] Simulated read failure for camera {self.index} after {FakeOpenCVSDKAdapter._fail_read_after} reads."
                )
                return False, None

            print(
                f"[FAKE SDK] Read image from camera {self.index} (read #{FakeOpenCVSDKAdapter._image_read_count})"
            )
            # Return a copy to prevent modification issues if the caller changes it
            return True, FakeOpenCVSDKAdapter._simulated_image.copy()

        def __del__(self):
            # Ensure cleanup if garbage collected
            self.release()

    VideoCapture = FakeVideoCapture  # Assign inner class

    # --- Methods ---
    def setNumThreads(self, nthreads: int) -> None:
        print(f"[FAKE SDK] setNumThreads({nthreads}) called.")
        # No actual behavior needed in fake

    def cvtColor(self, src: np.ndarray, code: int) -> np.ndarray:
        print(f"[FAKE SDK] cvtColor called with code {code}.")
        # Just return the source image, or simulate channel swap if needed
        if code == self.COLOR_BGR2RGB and src.shape[2] == 3:
            print("[FAKE SDK] Simulating BGR -> RGB conversion.")
            return src[..., ::-1]
        return src.copy()

    def rotate(self, src: np.ndarray, rotateCode: int) -> np.ndarray:
        print(f"[FAKE SDK] rotate called with code {rotateCode}.")
        if rotateCode == self.ROTATE_90_COUNTERCLOCKWISE:
            print("[FAKE SDK] Simulating 90 degree counter-clockwise rotation.")
            rotated_img = np.rot90(np.rot90(np.rot90(src.copy())))
            return rotated_img
        elif rotateCode == self.ROTATE_90_CLOCKWISE:
            print("[FAKE SDK] Simulating 90 degree clockwise rotation.")
            rotated_img = np.rot90(src.copy())
            return rotated_img
        elif rotateCode == self.ROTATE_180:
            print("[FAKE SDK] Simulating 180 degree rotation.")
            rotated_img = np.rot90(np.rot90(src.copy()))
            return rotated_img
        return src.copy()
