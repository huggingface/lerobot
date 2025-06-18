import threading
import time
from fractions import Fraction
from typing import Optional

import av
import av.video.stream
import cv2
import numpy as np
import rerun as rr


class VideoLogger:
    def __init__(
        self,
        stream_name: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: int = 30,
        codec: str = "libx264",
        preset: str = "ultrafast",
        tune: str = "zerolatency",
    ):
        """
        Initializes the VideoLogger to log video frames to a Rerun stream.

        Be sure to call `close()` when done to finalize the video stream, or use it within a `with` statement to ensure proper cleanup.

        :param stream_name: The name of the Rerun stream to log video frames to.
        :param width: The width of the video frames. If None, it will be determined from the first frame logged.
        :param height: The height of the video frames. If None, it will be determined from the first frame logged.
        :param fps: Frames per second for the video stream.
        :param codec: The codec to use for encoding the video. Default is "libx264".
        :param preset: The encoding preset to use. Default is "ultrafast".
        :param tune: The tuning option for the codec. Default is "zerolatency".
        """
        self.stream_name = stream_name
        self._width = width
        self._height = height
        self.fps = fps
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: Optional[float] = None
        self._frame_interval = 1.0 / fps
        self._lock = threading.Lock()
        self._encoder_initialized = False
        self._codec = codec
        self._preset = preset
        self._tune = tune
        rr.log(stream_name, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

    def _open_encoder(self, width: int, height: int):
        self.container = av.open("/dev/null", "w", format="h264")
        stream = self.container.add_stream(self._codec, rate=self.fps)
        if not isinstance(stream, av.video.stream.VideoStream):
            raise RuntimeError("Failed to create a video stream for encoding.")
        stream.width = width
        stream.height = height
        stream.rate = Fraction(self.fps, 1)
        stream.max_b_frames = 0
        stream.options = {  # type: ignore[attr-defined]
            "preset": self._preset,
            "tune": self._tune,
            "vbv_bufsize": "1",
            "vbv_maxrate": str(self.fps * width * height * 3),
        }
        self.stream = stream
        self._encoder_initialized = True

    def log_frame(self, img: np.ndarray):
        """
        Encodes and logs a single frame as video to the Rerun stream.

        :param img: The image frame to log, expected to be in HWC format (height, width, channels).
        :raises RuntimeError: If the video stream is not initialized or if the image shape does not match the expected dimensions.
        :return: None
        """
        now = time.perf_counter()
        with self._lock:
            if self._last_frame_time is not None and (now - self._last_frame_time) < self._frame_interval:
                return  # Skip frame: too soon for target FPS
            self._last_frame_time = now

            if self._width is None or self._height is None:
                self._height, self._width = img.shape[:2]

            if not self._encoder_initialized:
                self._open_encoder(self._width, self._height)
            
            if img.shape != (self._height, self._width, 3):
                img = cv2.resize(img, (self._width, self._height))

            if self._last_frame is not None and np.array_equal(img, self._last_frame):
                return  # Skip identical frame

            self._last_frame = img.copy()
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in self.stream.encode(frame):
                if packet.pts is not None and packet.time_base is not None:
                    rr.set_time(self.stream_name, duration=float(packet.pts * packet.time_base))
                rr.log(self.stream_name, rr.VideoStream.from_fields(sample=bytes(packet)))

    def close(self):
        """
        Finalizes the video stream by flushing any remaining frames and closing the container.
        """
        with self._lock:
            if self._encoder_initialized:
                for packet in self.stream.encode():
                    if packet.pts is not None and packet.time_base is not None:
                        rr.set_time(self.stream_name, duration=float(packet.pts * packet.time_base))
                    rr.log(self.stream_name, rr.VideoStream.from_fields(sample=bytes(packet)))
                self.container.close()
                self._encoder_initialized = False

    def __repr__(self):
        return f"VideoLogger(stream_name={self.stream_name}, width={self._width}, height={self._height}, fps={self.fps})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
