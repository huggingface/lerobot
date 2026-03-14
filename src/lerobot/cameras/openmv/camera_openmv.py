import cv2
import numpy as np
import serial
import threading
import time
from typing import Any
from numpy.typing import NDArray

from ..camera import Camera
from .configuration_openmv import OpenMVCameraConfig

JPEG_START = b'\xff\xd8'
JPEG_END = b'\xff\xd9'

class OpenMVCamera(Camera):
    def __init__(self, config: OpenMVCameraConfig):
        super().__init__(config)
        self.config = config
        self.port = config.port
        self._width = config.width or 640
        self._height = config.height or 480
        self._fps = config.fps or 30
        self.ser = None
        self._frame = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = False
        self._thread = None
        self._is_connected = False
        self._frame_count = 0

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    def connect(self, warmup: bool = True) -> None:
        self.ser = serial.Serial(self.port, baudrate=115200, timeout=0.1)
        time.sleep(3)
        self.ser.reset_input_buffer()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        for _ in range(100):
            if self._frame is not None:
                break
            time.sleep(0.1)
        self._is_connected = True
        print(f"OpenMVCamera connected to: {self.port}")

    def _read_loop(self):
        buf = b""
        last_frame_time = time.time()
        while self._running:
            try:
                chunk = self.ser.read(4096)
                if chunk:
                    buf += chunk

                # Buffer cok buyuduyse temizle
                if len(buf) > 100_000:
                    last_start = buf.rfind(JPEG_START)
                    buf = buf[last_start:] if last_start > 0 else b""

                while True:
                    start = buf.find(JPEG_START)
                    if start == -1:
                        buf = buf[-2:]
                        break
                    end = buf.find(JPEG_END, start + 2)
                    if end == -1:
                        buf = buf[start:]
                        break
                    jpeg_data = buf[start:end + 2]
                    buf = buf[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    frame = cv2.resize(frame, (self._width, self._height))
                    if self.config.color_mode.value == "rgb":
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._frame = frame
                        self._frame_count += 1
                        if self._frame_count % 30 == 0:
                            fps = 30 / (time.time() - last_frame_time)
                            last_frame_time = time.time()
                            print(f"OpenMV frame: {self._frame_count} ({fps:.1f} fps)", flush=True)
                    self._event.set()

            except Exception as e:
                print(f"OpenMV error: {e}", flush=True)
                if self._running:
                    buf = b""
                    time.sleep(0.1)

    def read(self) -> NDArray[Any]:
        with self._lock:
            if self._frame is None:
                return np.zeros((self._height, self._width, 3), dtype=np.uint8)
            return self._frame.copy()

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        self._event.wait(timeout=timeout_ms / 1000)
        self._event.clear()
        return self.read()

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        return self.read()

    def disconnect(self) -> None:
        self._running = False
        self._is_connected = False
        if self.ser:
            self.ser.close()
        print("OpenMVCamera disconnected.")
