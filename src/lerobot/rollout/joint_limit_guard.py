
import math

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


class JointLimitError(ValueError):
    pass


class JointLimitGuard:

    def __init__(self, calibration, resolution=4096):
        if resolution <= 1:
            raise ValueError("Invalid motor resolution")

        self.limits = {}

        max_res = resolution - 1

        for joint in JOINTS:
            if joint not in calibration:
                raise JointLimitError(
                    f"Missing calibration: {joint}"
                )

            cal = calibration[joint]

            raw_min = cal["range_min"]
            raw_max = cal["range_max"]

            if not (
                0 <= raw_min < raw_max
            ):
                raise JointLimitError(
                    f"Invalid calibration range: {joint}"
                )

            mid = (raw_min + raw_max) / 2

            angle_min = (
                (raw_min - mid) * 360 / max_res
            )

            angle_max = (
                (raw_max - mid) * 360 / max_res
            )

            self.limits[joint + ".pos"] = (
                angle_min,
                angle_max,
            )

    def validate(self, action):

        if not isinstance(action, dict):
            raise JointLimitError("Action must be a dictionary")

        allowed = set(self.limits) | {"gripper.pos"}

        if not action:
            raise JointLimitError("Empty action")

        unknown = set(action) - allowed

        if unknown:
            raise JointLimitError(
                f"Unknown action joints: {sorted(unknown)}"
            )

        for joint, (minimum, maximum) in self.limits.items():

            if joint not in action:
                continue

            try:
                value = float(action[joint])
            except (TypeError, ValueError, OverflowError):
                raise JointLimitError(
                    f"Invalid joint value: {joint}"
                )

            if not math.isfinite(value):
                raise JointLimitError(
                    f"Non-finite joint value: {joint}"
                )

            if not minimum <= value <= maximum:
                raise JointLimitError(
                    f"{joint}: {value} outside "
                    f"[{minimum:.3f}, {maximum:.3f}]"
                )

        if "gripper.pos" in action:
            try:
                gripper = float(action["gripper.pos"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise JointLimitError(
                    "Invalid gripper value"
                ) from exc

            if not math.isfinite(gripper):
                raise JointLimitError(
                    "Non-finite gripper"
                )

            if not 0 <= gripper <= 100:
                raise JointLimitError(
                    "Gripper outside [0,100]"
                )

        return action
