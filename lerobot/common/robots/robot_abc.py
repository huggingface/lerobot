import abc


class Robot(abc.ABC):
    robot_type: str
    features: dict

    @abc.abstractmethod
    def connect(self): ...

    @abc.abstractmethod
    def calibrate(self): ...

    @abc.abstractmethod
    def teleop_step(self, record_data=False): ...

    @abc.abstractmethod
    def capture_observation(self): ...

    @abc.abstractmethod
    def send_action(self, action): ...

    @abc.abstractmethod
    def disconnect(self): ...
