import abc


class MotorsBus(abc.ABC):
    """The main LeRobot class for implementing motors buses."""

    def __init__(
        self,
        motors: dict[str, tuple[int, str]],
    ):
        self.motors = motors

    def __len__(self):
        return len(self.motors)

    @abc.abstractmethod
    def connect(self):
        pass

    @abc.abstractmethod
    def reconnect(self):
        pass

    @abc.abstractmethod
    def set_calibration(self, calibration: dict[str, list]):
        pass

    @abc.abstractmethod
    def apply_calibration(self):
        pass

    @abc.abstractmethod
    def revert_calibration(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass

    @abc.abstractmethod
    def write(self):
        pass

    @abc.abstractmethod
    def disconnect(self):
        pass
