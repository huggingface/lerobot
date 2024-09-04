import time


def busy_wait(seconds):
    # Significantly more accurate than `time.sleep`, and mandatory for our use case,
    # but it consumes CPU cycles.
    # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass


class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)
