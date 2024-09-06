from stretch_body.gamepad_teleop import GamePadTeleop
from stretch_body.robot import Robot as Stretch

# class LeRobotStretchTeleop(GamePadTeleop):
#     """Wrapper of stretch_body.gamepad_teleop.GamePadTeleop"""

#     def __init__(self):
#         super().__init__()


class StretchLeRobot(Stretch):
    """Wrapper of stretch_body.robot.Robot"""

    robot_type = "stretch"

    def __init__(self, **kwargs):
        super().__init__()
        self.is_connected = False
        self.teleop = None

    def connect(self):
        self.is_connected = self.startup()

    def run_calibration(self):
        if not self.is_homed():
            self.home()

    def teleop_step(self, record_data=False) -> None | tuple[dict, dict]:
        if self.teleop is None:
            self.teleop = GamePadTeleop(robot_instance=False)
            self.teleop.startup(robot=self)

        self.teleop.do_motion(robot=self)
        state = self._get_state()
        action = self.teleop.gamepad_controller.get_state()
        self.push_command()

        if record_data:
            # TODO(aliberts): get proper types (ndarrays)
            obs_dict, action_dict = {}, {}
            obs_dict["observation.state"] = state
            action_dict["action"] = action
            return obs_dict, action_dict

    def _get_state(self):
        status = self.get_status()
        return {
            "head_pan.pos": status["head"]["head_pan"]["pos"],
            "head_tilt.pos": status["head"]["head_tilt"]["pos"],
            "lift.pos": status["lift"]["pos"],
            "arm.pos": status["arm"]["pos"],
            "wrist_pitch.pos": status["end_of_arm"]["wrist_pitch"]["pos"],
            "wrist_roll.pos": status["end_of_arm"]["wrist_roll"]["pos"],
            "wrist_yaw.pos": status["end_of_arm"]["wrist_yaw"]["pos"],
            "base_x.vel": status["base"]["x_vel"],
            "base_y.vel": status["base"]["y_vel"],
            "base_theta.vel": status["base"]["theta_vel"],
        }

    def capture_observation(self): ...

    def send_action(self, action): ...

    def disconnect(self):
        self.stop()
        if self.teleop is not None:
            self.teleop.gamepad_controller.stop()
            self.teleop.stop()

    def __del__(self):
        self.disconnect()
