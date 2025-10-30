# Piper SDK interface for LeRobot integration

import time
from typing import Any

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    print("Is the piper_sdk installed: pip install piper_sdk")
    C_PiperInterface_V2 = None  # For type checking and docs


class PiperSDKInterface:
    def __init__(self, port: str = "can0"):
        if C_PiperInterface_V2 is None:
            raise ImportError("piper_sdk is not installed. Please install it with `pip install piper_sdk`.")
        try:
            self.piper = C_PiperInterface_V2(port)
        except Exception as e:
            print(
                f"Failed to initialize Piper SDK: {e} Did you activate the can interface with `piper_sdk/can_activate.sh can0 1000000`"
            )
            self.piper = None
            return
        self.piper.ConnectPort()
        time.sleep(0.1)  # wait for connection to establish

        # reset the arm if it's not in idle state
        print(self.piper.GetArmStatus().arm_status.motion_status)
        if self.piper.GetArmStatus().arm_status.motion_status != 0:
            self.piper.EmergencyStop(0x02)  # resume

        if self.piper.GetArmStatus().arm_status.ctrl_mode == 2:
            print("The arm is in teaching mode, the light is green, press the button to exit teaching mode.")
            self.piper.EmergencyStop(0x02)  # resume

        while not self.piper.EnablePiper():
            time.sleep(0.01)

        # Set motion control to joint mode at 100% speed
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)

        # Get the min and max positions for each joint and gripper
        angel_status = self.piper.GetAllMotorAngleLimitMaxSpd()
        self.min_pos = [
            pos.min_angle_limit for pos in angel_status.all_motor_angle_limit_max_spd.motor[1:7]
        ] + [0]
        self.max_pos = [
            pos.max_angle_limit for pos in angel_status.all_motor_angle_limit_max_spd.motor[1:7]
        ] + [10]  # Gripper max position in mm

    def set_joint_positions(self, positions):
        # positions: list of 7 floats, first 6 are joint and 7 is gripper position
        # positions are in -100% to 100% range, we need to map them on the min and max positions
        # so -100% is min_pos and 100% is max_pos
        scaled_positions = [
            self.min_pos[i] + (self.max_pos[i] - self.min_pos[i]) * (pos + 100) / 200
            for i, pos in enumerate(positions[:6])
        ]
        scaled_positions = [100.0 * pos for pos in scaled_positions]  # Adjust factor

        # the gripper is from 0 to 100% range
        scaled_positions.append(self.min_pos[6] + (self.max_pos[6] - self.min_pos[6]) * positions[6] / 100)
        scaled_positions[6] = int(scaled_positions[6] * 10000)  # Convert to mm

        # joint 0, 3 and 5 are inverted
        joint_0 = int(-scaled_positions[0])
        joint_1 = int(scaled_positions[1])
        joint_2 = int(scaled_positions[2])
        joint_3 = int(-scaled_positions[3])
        joint_4 = int(scaled_positions[4])
        joint_5 = int(-scaled_positions[5])
        joint_6 = int(scaled_positions[6])

        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(joint_6, 1000, 0x01, 0)

    # --- LeRobot-friendly helpers (degrees/mm) ---
    def get_status_deg(self) -> dict[str, float]:
        """Return joints in degrees and gripper in mm."""
        js = self.piper.GetArmJointMsgs().joint_state
        g = self.piper.GetArmGripperMsgs()
        out = {
            "joint_1.pos": js.joint_1 / 1000.0,
            "joint_2.pos": js.joint_2 / 1000.0,
            "joint_3.pos": js.joint_3 / 1000.0,
            "joint_4.pos": js.joint_4 / 1000.0,
            "joint_5.pos": js.joint_5 / 1000.0,
            "joint_6.pos": js.joint_6 / 1000.0,
        }
        # Convert gripper back from SDK unit to mm (SDK used *10000 when sending)
        try:
            out["gripper.pos"] = g.gripper_state.grippers_angle / 10000.0
        except Exception:
            pass
        return out

    def set_joint_positions_deg(self, joints_deg: list[float], gripper_mm: float | None = None) -> None:
        """Send joints in degrees and optional gripper in mm."""
        j_ints = [int(round(d * 1000.0)) for d in joints_deg]
        self.piper.JointCtrl(*j_ints)
        if gripper_mm is not None:
            self.piper.GripperCtrl(int(round(gripper_mm * 10000.0)), 1000, 0x01, 0)

    def get_status(self) -> dict[str, Any]:
        joint_status = self.piper.GetArmJointMsgs()
        gripper = self.piper.GetArmGripperMsgs()

        joint_state = joint_status.joint_state
        obs_dict = {
            "joint_0.pos": joint_state.joint_1,
            "joint_1.pos": joint_state.joint_2,
            "joint_2.pos": joint_state.joint_3,
            "joint_3.pos": joint_state.joint_4,
            "joint_4.pos": joint_state.joint_5,
            "joint_5.pos": joint_state.joint_6,
        }
        obs_dict.update(
            {
                "joint_6.pos": gripper.gripper_state.grippers_angle,
            }
        )

        return obs_dict

    def disconnect(self):
        self.piper.JointCtrl(0, 0, 0, 0, 25000, 0)