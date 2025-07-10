# Piper SDK interface for LeRobot integration

from typing import Any, Dict
import time
try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    print('Is the piper_sdk installed: pip install piper_sdk')
    C_PiperInterface_V2 = None  # For type checking and docs

class PiperSDKInterface:
    def __init__(self, port: str = "can0"):
        if C_PiperInterface_V2 is None:
            raise ImportError("piper_sdk is not installed.")
        self.piper = C_PiperInterface_V2(port)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)

        # Get the min and max positions for each joint and gripper
        angel_status = self.piper.GetAllMotorAngleLimitMaxSpd()
        self.min_pos = [pos.min_angle_limit for pos in angel_status.all_motor_angle_limit_max_spd.motor[1:7]] + \
                  [0]
        self.max_pos = [pos.max_angle_limit for pos in angel_status.all_motor_angle_limit_max_spd.motor[1:7]] + \
                  [10]  # Gripper max position in mm
    def set_joint_positions(self, positions):
        # positions: list of 7 floats, first 6 are joint and 7 is gripper position 
        # postions are in -100% to 100% range, we need to map them on the min and max positions
        # so -100% is min_pos and 100% is max_pos
        scaled_positions = [self.min_pos[i] + (self.max_pos[i] - self.min_pos[i]) * (pos + 100) / 200 for i, pos in enumerate(positions[:6])]
        scaled_positions = [100.0*pos for pos in scaled_positions]  # Adjust factor

        # the gripper is from 0 to 100% range
        scaled_positions.append(self.min_pos[6] + (self.max_pos[6] - self.min_pos[6]) * positions[6] / 100)
        scaled_positions[6] = int(scaled_positions[6] * 10000)  # Convert to mm

        # joint 0, 3 and 5 are inverted
        joint_0 = int(-scaled_positions[0])
        joint_1 = int( scaled_positions[1])
        joint_2 = int( scaled_positions[2])
        joint_3 = int(-scaled_positions[3])
        joint_4 = int( scaled_positions[4])
        joint_5 = int(-scaled_positions[5]) 
        joint_6 = int( scaled_positions[6]) 

        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(joint_6, 1000, 0x01, 0)

    def get_status(self) -> Dict[str, Any]:
        joint_status = self.piper.GetArmJointMsgs()
        gripper = self.piper.GetArmGripperMsgs()
        gripper.gripper_state.grippers_angle
        
        joint_state = joint_status.joint_state
        obs_dict = {f"joint_0.pos": joint_state.joint_1,
                    f"joint_1.pos": joint_state.joint_2,
                    f"joint_2.pos": joint_state.joint_3,
                    f"joint_3.pos": joint_state.joint_4,
                    f"joint_4.pos": joint_state.joint_5,
                    f"joint_5.pos": joint_state.joint_6,
                    }
        obs_dict.update({
            "joint_6.pos": gripper.gripper_state.grippers_angle,
        })

        return obs_dict

    def disconnect(self):
        # No explicit disconnect
        pass
