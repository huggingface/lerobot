from dataclasses import dataclass

from piper_sdk import C_PiperInterface_V2


@dataclass
class PiperMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]


class PiperMotorsBus:
    """
    Master (Leader) 模式的 Piper SDK 封装。

    用于录制数据集时读取 Master 的控制指令 (operator intent)。
    此 bus 为只读模式：只 ConnectPort() 不使能，通过
    GetArmJointCtrl() / GetArmGripperCtrl() 读取 Master 发出的控制帧。
    """

    def __init__(self, config: PiperMotorsBusConfig):
        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()
        self._is_connected = False
        self.motors = config.motors
        # Converts from 0.001 degrees to radians: 1000 * 180 / 3.14
        self.joint_factor = 57324.840764

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def connect(self) -> None:
        """
        连接 CAN 端口（只读，不使能电机）。
        Master 通过 CAN 总线与 Follower 通信，PC 只需读取控制帧。
        """
        self._is_connected = True

    def set_calibration(self):
        return

    def revert_calibration(self):
        return

    def read(self) -> dict:
        """
        读取 Master 的控制指令（operator intent）。
        使用 GetArmJointCtrl() 和 GetArmGripperCtrl()。
        """
        joint_msg = self.piper.GetArmJointCtrl()
        joint_ctrl = joint_msg.joint_ctrl

        gripper_msg = self.piper.GetArmGripperCtrl()
        gripper_ctrl = gripper_msg.gripper_ctrl

        return {
            "joint_1": joint_ctrl.joint_1 / self.joint_factor,
            "joint_2": joint_ctrl.joint_2 / self.joint_factor,
            "joint_3": joint_ctrl.joint_3 / self.joint_factor,
            "joint_4": joint_ctrl.joint_4 / self.joint_factor,
            "joint_5": joint_ctrl.joint_5 / self.joint_factor,
            "joint_6": joint_ctrl.joint_6 / self.joint_factor,
            "gripper": gripper_ctrl.grippers_angle / 1_000_000.0,
        }

    def disconnect(self) -> None:
        """断开连接。"""
        self._is_connected = False
