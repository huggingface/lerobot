from __future__ import annotations

from typing import Dict, List

try:
    # Prefer a normal dependency (pip install piper_sdk)
    from piper_sdk import C_PiperInterface, LogLevel  # type: ignore
except Exception:  # pragma: no cover
    C_PiperInterface = None
    LogLevel = None


class PiperCanBus:
    def __init__(self, interface: str, bitrate: int, joint_names: List[str]):
        self.interface = interface
        self.bitrate = bitrate
        self.joint_names = joint_names
        self._p = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return bool(self._connected)

    def connect(self) -> None:
        if C_PiperInterface is None:
            raise ImportError("piper_sdk is not installed. Please `pip install piper_sdk` or `pip install -e ./piper_sdk`.")
        # Piper units: angles in 0.001 degrees
        self._p = C_PiperInterface(
            can_name=self.interface,
            judge_flag=False,          # allow non-official CAN modules
            can_auto_init=True,        # auto init bus
            dh_is_offset=1,            # new DH params for recent firmwares
            start_sdk_joint_limit=False,
            start_sdk_gripper_limit=False,
            logger_level=LogLevel.WARNING,
            log_to_file=False,
            log_file_path=None,
        )
        self._p.ConnectPort()
        # If needed to get feedback in slave mode:
        # self._p.MasterSlaveConfig(0xFC, 0, 0, 0)
        self._connected = True

    def disconnect(self) -> None:
        if self._p is not None:
            self._p.DisconnectPort()
        self._connected = False

    def read_positions(self) -> Dict[str, float]:
        """
        Returns joint positions in degrees as {joint_name: deg}.
        Piper feedback is int in 0.001 degrees; convert to float degrees.
        """
        msg = self._p.GetArmJointMsgs()
        js = msg.joint_state
        raw = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
        vals_deg = [v / 1000.0 for v in raw]
        return {name: val for name, val in zip(self.joint_names, vals_deg, strict=True)}

    def write_goal_positions(self, goals: Dict[str, float]) -> None:
        """
        Accepts positions in degrees; converts to 0.001 degrees ints; sends via JointCtrl.
        """
        ordered = [goals[name] for name in self.joint_names]
        ints = [int(round(d * 1000.0)) for d in ordered]
        self._p.JointCtrl(*ints)