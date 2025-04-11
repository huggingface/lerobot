from lerobot.common.robot_devices.motors.feetech import (
    CalibrationMode,
    FeetechMotorsBus,
)
import yaml

class HopeJuniorRobot:
    def __init__(self):
        self.arm_port = "/dev/ttyUSB0"
        self.hand_port = "/dev/ttyACM1"
        self.arm_bus = FeetechMotorsBus(
            port = self.arm_port,
            motors={
                # "motor1": (1, "sts3250"),
                # "motor2": (2, "sts3250"),
                # "motor3": (3, "sts3250"),
                
                #"shoulder_pitch": [1, "sts3215"],
                "shoulder_pitch": [1, "sm8512bl"],
                "shoulder_yaw": [2, "sts3250"],  # TODO: sts3250
                "shoulder_roll": [3, "sts3250"],  # TODO: sts3250
                "elbow_flex": [4, "sts3250"],
                "wrist_roll": [5, "sts3215"],
                "wrist_yaw": [6, "sts3215"],
                "wrist_pitch": [7, "sts3215"],
            },
            protocol_version=0,
        )
        self.hand_bus = FeetechMotorsBus(
            port=self.hand_port,

        motors = {
            # Thumb
            "thumb_basel_rotation": [1, "scs0009"],
            "thumb_mcp": [3, "scs0009"],
            "thumb_pip": [4, "scs0009"],
            "thumb_dip": [13, "scs0009"],

            # Index
            "index_thumb_side": [5, "scs0009"],
            "index_pinky_side": [6, "scs0009"],
            "index_flexor": [16, "scs0009"],

            # Middle
            "middle_thumb_side": [8, "scs0009"],
            "middle_pinky_side": [9, "scs0009"],
            "middle_flexor": [2, "scs0009"],

            # Ring
            "ring_thumb_side": [11, "scs0009"],
            "ring_pinky_side": [12, "scs0009"],
            "ring_flexor": [7, "scs0009"],

            # Pinky
            "pinky_thumb_side": [14, "scs0009"],
            "pinky_pinky_side": [15, "scs0009"],
            "pinky_flexor": [10, "scs0009"],
        },
            protocol_version=1,#1
            group_sync_read=False,
        )

        self.arm_calib_dict = self.get_arm_calibration()
        self.hand_calib_dict = self.get_hand_calibration()


    def apply_arm_config(self, config_file):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        for param, value in config.get("robot", {}).get("arm_bus", {}).items():
            self.arm_bus.write(param, value)

    def apply_hand_config(config_file, robot):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        for param, value in config.get("robot", {}).get("hand_bus", {}).items():
            robot.arm_bus.write(param, value)

    def get_hand_calibration(self):
        homing_offset = [0] * len(self.hand_bus.motor_names)
        drive_mode = [0] * len(self.hand_bus.motor_names)
        
        start_pos = [
            750,  # thumb_basel_rotation
            100,  # thumb_mcp
            700,  # thumb_pip
            100,  # thumb_dip

            800,  # index_thumb_side
            950,  # index_pinky_side
            0,  # index_flexor

            250,  # middle_thumb_side
            850,  # middle_pinky_side
            0,  # middle_flexor

            850,  # ring_thumb_side
            900,  # ring_pinky_side
            0,  # ring_flexor

            00,  # pinky_thumb_side
            950,  # pinky_pinky_side
            0,  # pinky_flexor
        ]

        end_pos = [
            start_pos[0] - 550,  # thumb_basel_rotation
            start_pos[1] + 400,  # thumb_mcp
            start_pos[2] + 300,  # thumb_pip
            start_pos[3] + 200,  # thumb_dip

            start_pos[4] - 700,  # index_thumb_side
            start_pos[5] - 300,  # index_pinky_side
            start_pos[6] + 600,  # index_flexor

            start_pos[7] + 700,  # middle_thumb_side
            start_pos[8] - 400,  # middle_pinky_side
            start_pos[9] + 600,  # middle_flexor

            start_pos[10] - 600,  # ring_thumb_side
            start_pos[11] - 400,  # ring_pinky_side
            start_pos[12] + 600,  # ring_flexor

            start_pos[13] + 400,  # pinky_thumb_side
            start_pos[14] - 450,  # pinky_pinky_side
            start_pos[15] + 600,  # pinky_flexor
        ]


        

        calib_modes = [CalibrationMode.LINEAR.name] * len(self.hand_bus.motor_names)

        calib_dict = {
            "homing_offset": homing_offset,
            "drive_mode": drive_mode,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "calib_mode": calib_modes,
            "motor_names": self.hand_bus.motor_names,
        }
        return calib_dict
    
    def get_arm_calibration(self):

        homing_offset = [0] * len(self.arm_bus.motor_names)
        drive_mode = [0] * len(self.arm_bus.motor_names)

        start_pos = [
            1800,   # shoulder_up
            2800,  # shoulder_forward
            1800,  # shoulder_roll
            1200,  # bend_elbow
            700,  # wrist_roll
            1850,  # wrist_yaw
            1700,  # wrist_pitch
        ]

        end_pos = [
            2800,  # shoulder_up
            3150,  # shoulder_forward
            400,  #shoulder_roll
            2300,  # bend_elbow
            2300,  # wrist_roll
            2150,  # wrist_yaw
            2300,  # wrist_pitch
        ]

        calib_modes = [CalibrationMode.LINEAR.name] * len(self.arm_bus.motor_names)

        calib_dict = {
            "homing_offset": homing_offset,
            "drive_mode": drive_mode,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "calib_mode": calib_modes,
            "motor_names": self.arm_bus.motor_names,
        }
        return calib_dict

    def connect_arm(self):
        self.arm_bus.connect()

    def connect_hand(self):
        self.hand_bus.connect()