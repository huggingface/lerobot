"""
controller.py

Manages input from a haptic device (via pyOpenHaptics).
Provides a DeviceState object with latest transform + button state.
"""

import time
import numpy as np
from dataclasses import dataclass
from pyOpenHaptics.hd_device import HapticDevice
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd import get_transform, get_buttons
from scipy.spatial.transform import Rotation as R


# ------------------------
# Shared Device State
# ------------------------

@dataclass
class DeviceState:
    transform: object = None
    buttons: int = 0


device_state = DeviceState()  # global state container


# ------------------------
# Utilities
# ------------------------

def transform_to_numpy(matrix):
    """Convert transform matrix (4x4) into np.array."""
    return np.array([[matrix[i][j] for j in range(4)] for i in range(4)], dtype=float)


def extract_position(matrix): 
    T = np.array([[matrix[i][j] for j in range(4)] for i in range(4)]) 
    pos = T[3, :3] # If position is in the last row return pos, T
    return pos, T


def extract_deltas(matrix, prev_hand, prev_R, sr, st, max_step=10.0, max_angle=10.0):
    """Compute relative translation and orientation deltas from haptic transform."""
    T = np.array([[matrix[i][j] for j in range(4)] for i in range(4)])
    pos = T[3, :3]      # last row = position
    Rmat = T[:3, :3]    # 3x3 rotation
    
    if prev_hand is None or prev_R is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), pos.copy(), Rmat.copy()
    
    # --- translation delta ---
    dh = (pos - prev_hand) * st
    dx = dh[0]
    dy = dh[1]
    dz = dh[2]
    
    # --- orientation delta ---
    dR = Rmat @ prev_R.T
    rotvec = R.from_matrix(dR).as_rotvec(degrees=True) * sr
    dα, dβ, dγ = rotvec  # roll, pitch, yaw changes
    
    return (dx, dy, dz, dα, dβ, dγ), pos.copy(), Rmat.copy()



# ------------------------
# Callback
# ------------------------

@hd_callback
def device_callback():
    """Callback executed by pyOpenHaptics scheduler."""
    global device_state
    device_state.transform = get_transform()
    device_state.buttons = get_buttons()
    return True


# ------------------------
# Interface
# ------------------------

class Controller:
    """
    Wrapper around HapticDevice that provides latest state via .get_state().
    """

    def __init__(self):
        self.device = HapticDevice(callback=device_callback, scheduler_type="async")

    def get_state(self):
        """Return (position[3], transform[4x4], buttons)."""
        if device_state.transform is None:
            return None
        pos, T = extract_position(device_state.transform)
        return pos, T, device_state.buttons

    def close(self):
        self.device.close()

device = HapticDevice(callback=device_callback, scheduler_type="async")



# ------------------------
# Standalone test
# ------------------------

if __name__ == "__main__":
    ctrl = Controller()
    try:
        while True:
            state = ctrl.get_state()
            if state:
                pos, T, buttons = state
                print("Position:", pos)
                print("Transform:\n", T)
                print("Buttons:", buttons)
                print("-" * 40)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        ctrl.close()
