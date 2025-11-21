import json
import time
import math
from pathlib import Path

# ---- key → (section, name, id)
MAP = {
    # LEFT
    "kLeftShoulderPitch.pos": ("left",  "shoulder_pitch", 0),
    "kLeftShoulderYaw.pos":   ("left",  "shoulder_yaw",   1),
    "kLeftShoulderRoll.pos":  ("left",  "shoulder_roll",  2),
    "kLeftElbow.pos":         ("left",  "elbow_flex",     3),
    "kLeftWristRoll.pos":     ("left",  "wrist_roll",     4),
    "kLeftWristYaw.pos":      ("left",  "wrist_yaw",      5),
    "kLeftWristyaw.pos":      ("left",  "wrist_yaw",      5),  # tolerate casing variant
    "kLeftWristPitch.pos":    ("left",  "wrist_pitch",    6),

    # RIGHT
    "kRightShoulderPitch.pos": ("right", "shoulder_pitch", 0),
    "kRightShoulderYaw.pos":   ("right", "shoulder_yaw",   1),
    "kRightShoulderRoll.pos":  ("right", "shoulder_roll",  2),
    "kRightElbow.pos":         ("right", "elbow_flex",     3),
    "kRightWristRoll.pos":     ("right", "wrist_roll",     4),
    "kRightWristYaw.pos":      ("right", "wrist_yaw",      5),
    "kRightWristPitch.pos":    ("right", "wrist_pitch",    6),
}

# Output
CALIB_PATH = Path("calibration.json")
ROUND_TO_INT = False  # set True if you want int ranges

# Init tracker: tracker["left"]["shoulder_pitch"] = {...}
tracker = {"left": {}, "right": {}}
for sec, name, idx in MAP.values():
    if name not in tracker[sec]:
        tracker[sec][name] = {
            "id": idx,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": math.inf,
            "range_max": -math.inf,
        }

def _to_float(x):
    # unwrap numpy / torch scalars if present
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass
    return float(x)

def update_tracker(obs: dict):
    for k, v in obs.items():
        if k not in MAP:
            continue
        sec, name, _ = MAP[k]
        try:
            x = _to_float(v)
        except Exception:
            continue
        t = tracker[sec][name]
        if x < t["range_min"]:
            t["range_min"] = x
        if x > t["range_max"]:
            t["range_max"] = x

def dump_calibration(path: Path):
    out = {"left": {}, "right": {}}
    for sec in ("left", "right"):
        for name, d in tracker[sec].items():
            mn, mx = d["range_min"], d["range_max"]
            if ROUND_TO_INT:
                mn = None if mn is math.inf else int(round(mn))
                mx = None if mx is -math.inf else int(round(mx))
            else:
                mn = None if mn is math.inf else mn
                mx = None if mx is -math.inf else mx
            out[sec][name] = {
                "id": d["id"],
                "drive_mode": d["drive_mode"],
                "homing_offset": d["homing_offset"],
                "range_min": mn,
                "range_max": mx,
            }
    path.write_text(json.dumps(out, indent=4))
    print(f"Saved calibration to {path.resolve()}")

from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1, G1_29_JointIndex
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import time
config = UnitreeG1Config(
    motion_mode=False,
    simulation_mode=False
)

robot = UnitreeG1(config)
try:
    while True:
        observation = robot.get_observation()
        update_tracker(observation)
        robot.send_action(observation)  # mirror, if desired
        time.sleep(0.01)
except KeyboardInterrupt:
    dump_calibration(CALIB_PATH)
