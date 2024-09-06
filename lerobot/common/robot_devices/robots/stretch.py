from stretch_body.gamepad_teleop import GamePadTeleop
from stretch_body.robot import Robot as Stretch


class LeRobotStretchTeleop(GamePadTeleop):
    """Wrapper of stretch_body.gamepad_teleop.GamePadTeleop"""

    def __init__(self):
        super().__init__()


class LeRobotStretch(Stretch):
    """Wrapper of stretch_body.robot.Robot"""

    def __init__(self, teleoperate: bool = False, **kwargs):
        super().__init__()
        self.robot_type = "stretch"
        self.is_connected = False
        self.teleop = GamePadTeleop(robot_instance=False) if teleoperate else None

    def connect(self):
        self.is_connected = self.startup()

    def run_calibration(self):
        if not self.is_homed():
            self.home()

    def teleop_step(self, record_data=False):
        if self.teleop is None:
            raise ValueError
        ...

    def capture_observation(self): ...

    def send_action(self, action): ...

    def disconnect(self):
        self.stop()
        if self.teleop is not None:
            self.teleop.stop()

    def __del__(self):
        self.disconnect()
