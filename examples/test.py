import threading
import time
from typing import Callable

import cv2
import numpy as np

# from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
# from qai_hub_models.models.mediapipe_hand.model import (
#     MediaPipeHand,
# )
# from qai_hub_models.utils.image_processing import (
#     app_to_net_image_inputs,
# )
from lerobot.common.robot_devices.motors.feetech import (
    CalibrationMode,
    FeetechMotorsBus,
)

LOWER_BOUND_LINEAR = -100
UPPER_BOUND_LINEAR = 200

import serial


class HomonculusGlove:
    def __init__(self):
        self.serial_port = "/dev/tty.usbmodem21401"
        self.baud_rate = 115200
        self.serial = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        self.thread = threading.Thread(target=self.async_read)
        self.thread.start()
        self.last_d = {
            "thumb_0": 100,
            "thumb_1": 100,
            "thumb_2": 100,
            "thumb_3": 100,
            "index_0": 100,
            "index_1": 100,
            "index_2": 100,
            "middle_0": 100,
            "middle_1": 100,
            "middle_2": 100,
            "ring_0": 100,
            "ring_1": 100,
            "ring_2": 100,
            "pinky_0": 100,
            "pinky_1": 100,
            "pinky_2": 100,
            "battery_voltage": 100,
        }
        self.calibration = None

    @property
    def joint_names(self):
        return list(self.last_d.keys())

    def read(self, motor_names: list[str] | None = None):
        if motor_names is None:
            motor_names = self.joint_names

        values = np.array([self.last_d[k] for k in motor_names])

        print(motor_names)
        print(values)

        if self.calibration is not None:
            values = self.apply_calibration(values, motor_names)
            print(values)
        return values

    def async_read(self):
        while True:
            if self.serial.in_waiting > 0:
                self.serial.flush()
                vals = self.serial.readline().decode("utf-8").strip()
                vals = vals.split(" ")
                if len(vals) != 17:
                    continue
                vals = [int(val) for val in vals]

                d = {
                    "thumb_0": vals[0],
                    "thumb_1": vals[1],
                    "thumb_2": vals[2],
                    "thumb_3": vals[3],
                    "index_0": vals[4],
                    "index_1": vals[5],
                    "index_2": vals[6],
                    "middle_0": vals[7],
                    "middle_1": vals[8],
                    "middle_2": vals[9],
                    "ring_0": vals[10],
                    "ring_1": vals[11],
                    "ring_2": vals[12],
                    "pinky_0": vals[13],
                    "pinky_1": vals[14],
                    "pinky_2": vals[15],
                    "battery_voltage": vals[16],
                }
                self.last_d = d
                # print(d.values())

    def run_calibration(self):
        print("\nMove arm to open position")
        input("Press Enter to continue...")
        open_pos_list = []
        for _ in range(300):
            open_pos = self.read()
            open_pos_list.append(open_pos)
            time.sleep(0.01)
        open_pos = np.array(open_pos_list)
        max_open_pos = open_pos.max(axis=0)
        min_open_pos = open_pos.min(axis=0)

        print(f"{max_open_pos=}")
        print(f"{min_open_pos=}")

        print("\nMove arm to closed position")
        input("Press Enter to continue...")
        closed_pos_list = []
        for _ in range(300):
            closed_pos = self.read()
            closed_pos_list.append(closed_pos)
            time.sleep(0.01)
        closed_pos = np.array(closed_pos_list)
        max_closed_pos = closed_pos.max(axis=0)
        closed_pos[closed_pos < 1000] = 60000
        min_closed_pos = closed_pos.min(axis=0)

        print(f"{max_closed_pos=}")
        print(f"{min_closed_pos=}")

        open_pos = np.array([max_open_pos, max_closed_pos]).max(axis=0)
        closed_pos = np.array([min_open_pos, min_closed_pos]).min(axis=0)

        # INVERTION
        # INVERTION
        # INVERTION
        # INVERTION
        # INVERTION
        # INVERTION
        # INVERTION
        for i, jname in enumerate(self.joint_names):
            if jname in ["thumb_0", "thumb_3", "index_2", "middle_2", "ring_2", "pinky_0", "pinky_2"]:
                tmp_pos = open_pos[i]
                open_pos[i] = closed_pos[i]
                closed_pos[i] = tmp_pos

        print()
        print(f"{open_pos=}")
        print(f"{closed_pos=}")

        homing_offset = [0] * len(self.joint_names)
        drive_mode = [0] * len(self.joint_names)
        calib_modes = [CalibrationMode.LINEAR.name] * len(self.joint_names)

        calib_dict = {
            "homing_offset": homing_offset,
            "drive_mode": drive_mode,
            "start_pos": open_pos,
            "end_pos": closed_pos,
            "calib_mode": calib_modes,
            "motor_names": self.joint_names,
        }
        # return calib_dict
        self.set_calibration(calib_dict)

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Convert from unsigned int32 joint position range [0, 2**32[ to the universal float32 nominal degree range ]-180.0, 180.0[ with
        a "zero position" at 0 degree.

        Note: We say "nominal degree range" since the motors can take values outside this range. For instance, 190 degrees, if the motor
        rotate more than a half a turn from the zero position. However, most motors can't rotate more than 180 degrees and will stay in this range.

        Joints values are original in [0, 2**32[ (unsigned int32). Each motor are expected to complete a full rotation
        when given a goal position that is + or - their resolution. For instance, feetech xl330-m077 have a resolution of 4096, and
        at any position in their original range, let's say the position 56734, they complete a full rotation clockwise by moving to 60830,
        or anticlockwise by moving to 52638. The position in the original range is arbitrary and might change a lot between each motor.
        To harmonize between motors of the same model, different robots, or even models of different brands, we propose to work
        in the centered nominal degree range ]-180, 180[.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the present position to a nominal range [0, 100] %,
                # useful for joints with linear motions like Aloha gripper
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                if (values[i] < LOWER_BOUND_LINEAR) or (values[i] > UPPER_BOUND_LINEAR):
                    if name == "pinky_1" and (values[i] < LOWER_BOUND_LINEAR):
                        values[i] = end_pos
                    else:
                        msg = (
                            f"Wrong motor position range detected for {name}. "
                            f"Expected to be in nominal range of [0, 100] % (a full linear translation), "
                            f"with a maximum range of [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}] % to account for some imprecision during calibration, "
                            f"but present value is {values[i]} %. "
                            "This might be due to a cable connection issue creating an artificial jump in motor values. "
                            "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                        )
                        print(msg)
                        # raise JointOutOfRangeError(msg)

        return values

    # def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
    #     """Inverse of `apply_calibration`."""
    #     if motor_names is None:
    #         motor_names = self.motor_names

    #     for i, name in enumerate(motor_names):
    #         calib_idx = self.calibration["motor_names"].index(name)
    #         calib_mode = self.calibration["calib_mode"][calib_idx]

    #         if CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
    #             start_pos = self.calibration["start_pos"][calib_idx]
    #             end_pos = self.calibration["end_pos"][calib_idx]

    #             # Convert from nominal lnear range of [0, 100] % to
    #             # actual motor range of values which can be arbitrary.
    #             values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

    #     values = np.round(values).astype(np.int32)
    #     return values


class HopeJuniorRobot:
    def __init__(self):
        self.arm_bus = FeetechMotorsBus(
            port="/dev/tty.usbmodem58760429571",
            motors={
                # "motor1": (2, "sts3250"),
                # "motor2": (1, "scs0009"),
                "shoulder_pitch": [1, "sts3250"],
                "shoulder_yaw": [2, "sts3215"],  # TODO: sts3250
                "shoulder_roll": [3, "sts3215"],  # TODO: sts3250
                "elbow_flex": [4, "sts3250"],
                "wrist_roll": [5, "sts3215"],
                "wrist_yaw": [6, "sts3215"],
                "wrist_pitch": [7, "sts3215"],
            },
            protocol_version=0,
        )
        self.hand_bus = FeetechMotorsBus(
            port="/dev/tty.usbmodem585A0077581",
            motors={
                "thumb_basel_rotation": [30, "scs0009"],
                "thumb_flexor": [27, "scs0009"],
                "thumb_pinky_side": [26, "scs0009"],
                "thumb_thumb_side": [28, "scs0009"],
                "index_flexor": [25, "scs0009"],
                "index_pinky_side": [31, "scs0009"],
                "index_thumb_side": [32, "scs0009"],
                "middle_flexor": [24, "scs0009"],
                "middle_pinky_side": [33, "scs0009"],
                "middle_thumb_side": [34, "scs0009"],
                "ring_flexor": [21, "scs0009"],
                "ring_pinky_side": [36, "scs0009"],
                "ring_thumb_side": [35, "scs0009"],
                "pinky_flexor": [23, "scs0009"],
                "pinky_pinky_side": [38, "scs0009"],
                "pinky_thumb_side": [37, "scs0009"],
            },
            protocol_version=1,
            group_sync_read=False,
        )

    def get_hand_calibration(self):
        homing_offset = [0] * len(self.hand_bus.motor_names)
        drive_mode = [0] * len(self.hand_bus.motor_names)

        start_pos = [
            500,
            900,
            1000,
            0,
            100,
            250,
            750,
            100,
            400,
            150,
            100,
            120,
            980,
            100,
            950,
            750,
        ]

        end_pos = [
            500 - 250,
            900 - 300,
            1000 - 550,
            0 + 550,
            1000,
            250 + 700,
            750 - 700,
            1000,
            400 + 700,
            150 + 700,
            1000,
            120 + 700,
            980 - 700,
            1000,
            950 - 700,
            750 - 700,
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

    def connect(self):
        self.arm_bus.connect()
        self.hand_bus.connect()


ESCAPE_KEY_ID = 27


def capture_and_display_processed_frames(
    frame_processor: Callable[[np.ndarray], np.ndarray],
    window_display_name: str,
    cap_device: int = 0,
) -> None:
    """
    Capture frames from the given input camera device, run them through
    the frame processor, and display the outputs in a window with the given name.

    User should press Esc to exit.

    Inputs:
        frame_processor: Callable[[np.ndarray], np.ndarray]
            Processes frames.
            Input and output are numpy arrays of shape (H W C) with BGR channel layout and dtype uint8 / byte.
        window_display_name: str
            Name of the window used to display frames.
        cap_device: int
            Identifier for the camera to use to capture frames.
    """
    cv2.namedWindow(window_display_name)
    capture = cv2.VideoCapture(cap_device)
    if not capture.isOpened():
        raise ValueError("Unable to open video capture.")

    frame_count = 0
    has_frame, frame = capture.read()
    while has_frame:
        assert isinstance(frame, np.ndarray)

        frame_count = frame_count + 1
        # mirror frame
        frame = np.ascontiguousarray(frame[:, ::-1, ::-1])

        # process & show frame
        processed_frame = frame_processor(frame)
        cv2.imshow(window_display_name, processed_frame[:, :, ::-1])

        has_frame, frame = capture.read()
        key = cv2.waitKey(1)
        if key == ESCAPE_KEY_ID:
            break

    capture.release()


def main():
    robot = HopeJuniorRobot()
    robot.connect()

    # robot.hand_bus.calibration = None

    # breakpoint()
    # print(robot.arm_bus.read("Present_Position"))
    robot.arm_bus.write("Torque_Enable", 1)
    robot.arm_bus.write("Acceleration", 20)
    robot.arm_bus.read("Acceleration")

    calibration = robot.get_hand_calibration()
    robot.hand_bus.write("Goal_Position", calibration["start_pos"])
    # robot.hand_bus.write("Goal_Position", calibration["end_pos"][:4], robot.hand_bus.motor_names[:4])
    robot.hand_bus.set_calibration(calibration)
    lol = 1

    # # print(motors_bus.write("Goal_Position", 500))
    # print(robot.hand_bus.read("Present_Position"))
    # # pos = hand_bus.read("Present_Position")
    # # hand_bus.write("Goal_Position", pos[0]+20, hand_bus.motor_names[0])
    # # hand_bus.write("Goal_Position", pos[i]+delta, hand_bus.motor_names[i])
    # robot.hand_bus.read("Acceleration")
    # robot.hand_bus.write("Acceleration", 10)

    # sleep = 1
    # # robot.hand_bus.write(
    # #     "Goal_Position", [glove.last_d['index_2']-1500,300,300], ["index_pinky_side", "index_flexor", "index_thumb_side"]
    # # )
    # #time.sleep(sleep)
    # time.sleep(sleep)
    # robot.hand_bus.write(
    #     "Goal_Position", [100, 100, 100], ["index_flexor", "index_pinky_side", "index_thumb_side"]
    # )
    # time.sleep(sleep)
    # robot.hand_bus.write(
    #     "Goal_Position", [100, 0, 0], ["middle_flexor", "middle_pinky_side", "middle_thumb_side"]
    # )
    # time.sleep(sleep)
    # robot.hand_bus.write(
    #     "Goal_Position", [200, 200, 0], ["ring_flexor", "ring_pinky_side", "ring_thumb_side"]
    # )
    # time.sleep(sleep)
    # robot.hand_bus.write(
    #     "Goal_Position", [200, 100, 600], ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"]
    # )
    # time.sleep(sleep)

    # breakpoint()

    glove = HomonculusGlove()
    glove.run_calibration()
    # while True:
    #     joint_names = ["index_1", "index_2"]
    #     joint_values = glove.read(joint_names)
    #     print(joint_values)

    input()
    while True:
        joint_names = []
        joint_names += ["thumb_0", "thumb_2", "thumb_3"]
        joint_names += ["index_1", "index_2"]
        joint_names += ["middle_1", "middle_2"]
        joint_names += ["ring_1", "ring_2"]
        joint_names += ["pinky_1", "pinky_2"]
        joint_values = glove.read(joint_names)
        joint_values = joint_values.round().astype(int)
        joint_dict = {k: v for k, v in zip(joint_names, joint_values, strict=False)}

        motor_values = []
        motor_names = []
        motor_names += ["thumb_basel_rotation", "thumb_flexor", "thumb_pinky_side", "thumb_thumb_side"]
        motor_values += [
            joint_dict["thumb_3"],
            joint_dict["thumb_0"],
            joint_dict["thumb_2"],
            joint_dict["thumb_2"],
        ]
        motor_names += ["index_flexor", "index_pinky_side", "index_thumb_side"]
        motor_values += [joint_dict["index_2"], joint_dict["index_1"], joint_dict["index_1"]]
        motor_names += ["middle_flexor", "middle_pinky_side", "middle_thumb_side"]
        motor_values += [joint_dict["middle_2"], joint_dict["middle_1"], joint_dict["middle_1"]]
        motor_names += ["ring_flexor", "ring_pinky_side", "ring_thumb_side"]
        motor_values += [joint_dict["ring_2"], joint_dict["ring_1"], joint_dict["ring_1"]]
        motor_names += ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"]

        motor_values += [joint_dict["pinky_2"], joint_dict["pinky_1"], joint_dict["pinky_1"]]

        motor_values = np.array(motor_values)
        motor_values = np.clip(motor_values, 0, 100)

        robot.hand_bus.write("Goal_Position", motor_values, motor_names)
        time.sleep(0.02)

    while True:
        # print(glove.read()['index_2']-1500)
        glove_index_flexor = glove.read()["index_2"] - 1500
        glove_index_subflexor = glove.read()["index_1"] - 1500
        glove_index_side = glove.read()["index_0"] - 2100

        vals = [glove_index_flexor, 1000 - (glove_index_subflexor), glove_index_subflexor]

        keys = ["index_flexor", "index_pinky_side", "index_thumb_side"]

        glove_middle_flexor = glove.read()["middle_2"] - 1500
        glove_middle_subflexor = 1000 - (glove.read()["middle_1"] - 1700)
        vals += [glove_middle_flexor, glove_middle_subflexor, glove_middle_subflexor - 200]
        keys += ["middle_flexor", "middle_pinky_side", "middle_thumb_side"]

        glove_ring_flexor = glove.read()["ring_2"] - 1300
        print(glove_ring_flexor)
        glove_ring_subflexor = glove.read()["ring_1"] - 1100

        vals += [glove_ring_flexor, 1000 - glove_ring_subflexor, glove_ring_subflexor]
        keys += ["ring_flexor", "ring_pinky_side", "ring_thumb_side"]

        glove_pinky_flexor = glove.read()["pinky_2"] - 1500
        glove_pinky_subflexor = glove.read()["pinky_1"] - 1300
        vals += [300 + glove_pinky_flexor, max(1000 - glove_pinky_subflexor - 100, 0), glove_pinky_subflexor]
        keys += ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"]

        robot.hand_bus.write("Goal_Position", vals, keys)
        time.sleep(0.1)

    time.sleep(3)

    def move_arm(loop=10):
        sleep = 1
        for i in range(loop):
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1195])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 2195])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1457, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 2357, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 974, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 2674, 1957, 1695])
            time.sleep(sleep + 2)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 1632, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 1369, 1632, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 1330, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [2381, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1681, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)

    def move_hand(loop=10):
        sleep = 0.5
        for i in range(loop):
            robot.hand_bus.write(
                "Goal_Position",
                [500, 1000, 0, 1000],
                ["thumb_basel_rotation", "thumb_flexor", "thumb_pinky_side", "thumb_thumb_side"],
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position", [100, 100, 100], ["index_flexor", "index_pinky_side", "index_thumb_side"]
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position", [100, 1000, 150], ["middle_flexor", "middle_pinky_side", "middle_thumb_side"]
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position", [200, 200, 0], ["ring_flexor", "ring_pinky_side", "ring_thumb_side"]
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position", [200, 100, 700], ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"]
            )
            time.sleep(sleep)

            robot.hand_bus.write(
                "Goal_Position",
                [500, 1000 - 250, 0 + 300, 1000 - 200],
                ["thumb_basel_rotation", "thumb_flexor", "thumb_pinky_side", "thumb_thumb_side"],
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position",
                [100 + 450, 100 + 400, 100 + 400],
                ["index_flexor", "index_pinky_side", "index_thumb_side"],
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position",
                [100 + 350, 1000 - 450, 150 + 450],
                ["middle_flexor", "middle_pinky_side", "middle_thumb_side"],
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position",
                [200 + 650, 200 + 350, 0 + 350],
                ["ring_flexor", "ring_pinky_side", "ring_thumb_side"],
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position",
                [200 + 450, 100 + 400, 700 - 400],
                ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"],
            )
            time.sleep(sleep)

    move_hand(3)

    move_arm(1)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        executor.submit(move_arm)
        executor.submit(move_hand)

    # initial position
    for i in range(3):
        robot.hand_bus.write(
            "Goal_Position", [500, 1000, 0, 1000, 100, 950, 100, 100, 1000, 150, 200, 200, 0, 200, 100, 700]
        )
        time.sleep(1)

    # for i in range(3):
    #     robot.hand_bus.write("Goal_Position", [500, 1000-150, 0+250, 1000-150,
    #                                             100+300, 950-250, 100+250,
    #                                             100+200, 1000-300, 150+300,
    #                                             200+500, 200+200, 0+200,
    #                                             200+300, 100+200, 700-200])
    #     time.sleep(1)

    # camera = 0
    # score_threshold = 0.95
    # iou_threshold = 0.3

    # app = MediaPipeHandApp(MediaPipeHand.from_pretrained(), score_threshold, iou_threshold)

    # def frame_processor(frame: np.ndarray) -> np.ndarray:
    #     # Input Prep
    #     NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(frame)

    #     # Run Bounding Box & Keypoint Detector
    #     batched_selected_boxes, batched_selected_keypoints = app._run_box_detector(NCHW_fp32_torch_frames)

    #     # The region of interest ( bounding box of 4 (x, y) corners).
    #     # list[torch.Tensor(shape=[Num Boxes, 4, 2])],
    #     # where 2 == (x, y)
    #     #
    #     # A list element will be None if there is no selected ROI.
    #     batched_roi_4corners = app._compute_object_roi(batched_selected_boxes, batched_selected_keypoints)

    #     # selected landmarks for the ROI (if any)
    #     # list[torch.Tensor(shape=[Num Selected Landmarks, K, 3])],
    #     # where K == number of landmark keypoints, 3 == (x, y, confidence)
    #     #
    #     # A list element will be None if there is no ROI.
    #     landmarks_out = app._run_landmark_detector(NHWC_int_numpy_frames, batched_roi_4corners)

    #     app._draw_predictions(
    #         NHWC_int_numpy_frames,
    #         batched_selected_boxes,
    #         batched_selected_keypoints,
    #         batched_roi_4corners,
    #         *landmarks_out,
    #     )

    #     return NHWC_int_numpy_frames[0]

    # capture_and_display_processed_frames(frame_processor, "QAIHM Mediapipe Hand Demo", camera)


if __name__ == "__main__":
    main()
