import enum

import numpy as np


class MockStream(enum.Enum):
    color = 0
    depth = 1


class MockFormat(enum.Enum):
    rgb8 = 0
    z16 = 1


class MockConfig:
    def enable_device(self, device_id: str):
        self.device_enabled = device_id

    def enable_stream(
        self, stream_type: MockStream, width=None, height=None, color_format: MockFormat = None, fps=None
    ):
        self.stream_type = stream_type
        # Overwrite default values when possible
        self.width = 848 if width is None else width
        self.height = 480 if height is None else height
        self.color_format = MockFormat.rgb8 if color_format is None else color_format
        self.fps = 30 if fps is None else fps


class MockColorProfile:
    def __init__(self, config: MockConfig):
        self.config = config

    def fps(self):
        return self.config.fps

    def width(self):
        return self.config.width

    def height(self):
        return self.config.height


class MockColorStream:
    def __init__(self, config: MockConfig):
        self.config = config

    def as_video_stream_profile(self):
        return MockColorProfile(self.config)


class MockProfile:
    def __init__(self, config: MockConfig):
        self.config = config

    def get_stream(self, color_format: MockFormat):
        del color_format  # unused
        return MockColorStream(self.config)


class MockPipeline:
    def __init__(self):
        self.started = False
        self.config = None

    def start(self, config: MockConfig):
        self.started = True
        self.config = config
        return MockProfile(self.config)

    def stop(self):
        if not self.started:
            raise RuntimeError("You need to start the camera before stop.")
        self.started = False
        self.config = None

    def wait_for_frames(self, timeout_ms=50000):
        del timeout_ms  # unused
        return MockFrames(self.config)


class MockFrames:
    def __init__(self, config: MockConfig):
        self.config = config

    def get_color_frame(self):
        return MockColorFrame(self.config)

    def get_depth_frame(self):
        return MockDepthFrame(self.config)


class MockColorFrame:
    def __init__(self, config: MockConfig):
        self.config = config

    def get_data(self):
        data = np.ones((self.config.height, self.config.width, 3), dtype=np.uint8)
        # Create a difference between rgb and bgr
        data[:, :, 0] = 2
        return data


class MockDepthFrame:
    def __init__(self, config: MockConfig):
        self.config = config

    def get_data(self):
        return np.ones((self.config.height, self.config.width), dtype=np.uint16)


class MockDevice:
    def __init__(self):
        pass

    def get_info(self, camera_info) -> str:
        del camera_info  # unused
        # return fake serial number
        return "123456789"


class MockContext:
    def __init__(self):
        pass

    def query_devices(self):
        return [MockDevice()]
