from pathlib import Path
from attrs import field
import numpy as np
from dataclasses import dataclass
from typing import Any
from lerobot.teleoperators import Teleoperator

# Import your existing controller
from .controller import Controller, extract_deltas


@dataclass
class OmniConfig:
    device_name: str = "PhantomOmni"
    # scaling factors for translation (mm → robot mm) and rotation (deg → robot deg)
    scale_translation: float = 1.0
    scale_rotation: float = .5
    max_step: float = 2.0
    max_angle: float = 2.0
    id: int = 0  # device ID if multiple devices connected
    calibration_dir: Path = Path("~/.lerobot/calibration/omni").expanduser()



class OmniTeleoperator(Teleoperator):
    config_class = OmniConfig
    name = "omni"

    def __init__(self, config: OmniConfig):
        super().__init__(config)
        self.ctrl = Controller()
        self.connected = True
        self.prev_hand = None
        self.prev_R = None
        self.prev_buttons = 0
        self.motion_scale = 1.0
        self.gripper_state = False  # toggle state
        self.config = config
    # ------------------------
    # Required abstract stubs
    # ------------------------
    def configure(self) -> None:
        """No configuration needed for Omni."""
        pass

    def calibrate(self) -> None:
        """No calibration needed for Omni."""
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def feedback_features(self) -> dict:
        """No haptic feedback features exposed yet."""
        return {}

    # ------------------------
    # Core interface
    # ------------------------
    @property
    def action_features(self) -> dict[str, type]:
        return {
            "dx": float,
            "dy": float,
            "dz": float,
            "da": float,
            "db": float,
            "dg": float,
            "gripper": bool,
        }

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        if self.connected:
            try:
                self.ctrl.close()
            except Exception as e:
                print(f"⚠️ Error while closing haptic device: {e}")
            self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    def get_action(self) -> dict[str, Any]:
        state = self.ctrl.get_state()
        if not state:
            return {}
        pos, T, buttons = state

        # --- Button edge detection ---
        if buttons != self.prev_buttons:
            self.prev_buttons = buttons
            if buttons & 1:  # Button 1 toggles gripper
                self.gripper_state = not self.gripper_state
                print("Toggled gripper:", self.gripper_state)
            if buttons & 2:  # Button 2 toggles motion scaling
                self.motion_scale = 0.1 if self.motion_scale == 1 else 1
                print("Motion scale set to", self.motion_scale)

        # deltas
        deltas, self.prev_hand, self.prev_R = extract_deltas(
            T, self.prev_hand, self.prev_R,
            sr=self.config.scale_rotation,
            st=self.config.scale_translation,
            max_step=self.config.max_step,
            max_angle=self.config.max_angle,
        )
        dx, dy, dz, da, db, dg = deltas

        return {
            "dx": dx * self.motion_scale,
            "dy": dy * self.motion_scale,
            "dz": dz * self.motion_scale,
            "da": da * self.motion_scale,
            "db": db * self.motion_scale,
            "dg": dg * self.motion_scale,
            "gripper": self.gripper_state,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Optionally implement force feedback if pyOpenHaptics exposes it
        pass
