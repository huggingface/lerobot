import os
import sys
import time
from threading import Event
from pynput import keyboard
import select
import termios
import tty

from lerobot.utils.import_utils import register_third_party_devices
from lerobot_robot_piper import Piper, PiperConfig

STEP_DEG = 5.0       # per key press tick (increase for clearer motion)
STEP_MM = 1.0        # gripper step per tick
FPS = 30
LOG_EVERY = 10  # print periodic debug every N cycles (when no key pressed)
VERBOSE = True

keymap = {
    'q': ('joint_1.pos', -1), 'w': ('joint_1.pos', +1),
    'a': ('joint_2.pos', -1), 's': ('joint_2.pos', +1),
    'z': ('joint_3.pos', -1), 'x': ('joint_3.pos', +1),
    'e': ('joint_4.pos', -1), 'r': ('joint_4.pos', +1),
    'd': ('joint_5.pos', -1), 'f': ('joint_5.pos', +1),
    'c': ('joint_6.pos', -1), 'v': ('joint_6.pos', +1),
    'g': ('gripper.pos', -1), 'h': ('gripper.pos', +1),
}

pressed = set()
stop_evt = Event()
USE_STDIN = False
_old_term_settings = None

# Prefer stdin-based keys on Wayland/headless where pynput may not capture
if ("DISPLAY" not in os.environ) or (os.environ.get("XDG_SESSION_TYPE") != "x11"):
    USE_STDIN = True

def on_press(k):
    try:
        ch = k.char.lower()
        if ch in keymap: pressed.add(ch)
    except AttributeError:
        if k == keyboard.Key.esc:
            stop_evt.set()
            return False

def on_release(k):
    try:
        ch = k.char.lower()
        pressed.discard(ch)
    except AttributeError:
        pass

register_third_party_devices()
robot = Piper(PiperConfig(can_interface="can0", bitrate=1_000_000, include_gripper=True))
robot.connect()

# Move to zero pose before teleoperation
print("Moving Piper to zero pose...")
try:
    # Optional: slow down the motion if SDK supports it
    try:
        robot._iface.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)  # 30% speed
    except Exception:
        pass

    # Build zero action at joint mid-ranges (avoids starting at limits)
    zero_action = {f"joint_{i+1}.pos": 0.0 for i in range(6)}
    if robot._iface is not None:
        mins = robot._iface.min_pos[:6]
        maxs = robot._iface.max_pos[:6]
        mid = []
        for i in range(6):
            tgt = 0.5 * (mins[i] + maxs[i])
            zero_action[f"joint_{i+1}.pos"] = tgt
            mid.append(tgt)
        if robot.config.include_gripper:
            g_min, g_max = robot._iface.min_pos[6], robot._iface.max_pos[6]
            zero_action["gripper.pos"] = 0.5 * (g_min + g_max)
        print("Zero targets (deg):", [round(v, 1) for v in mid], "grip:", round(zero_action.get("gripper.pos", -1), 2))

    # Log current pose before commanding
    try:
        obs_before = robot.get_observation()
        cur = [float(obs_before.get(f"joint_{i}.pos", 0.0)) for i in range(1, 7)]
        print("Current pose (deg): ", [round(v, 1) for v in cur], "grip:", round(obs_before.get("gripper.pos", -1.0), 2))
    except Exception as e:
        print("Warning: couldn't read pre-command observation:", e)

    robot.send_action(zero_action)

    # Wait until near target
    TOL_DEG = 1.0
    TOL_MM = 1.0
    deadline = time.time() + 15.0
    poll_idx = 0
    while time.time() < deadline:
        obs0 = robot.get_observation()
        at_target = True
        errs = []
        for i in range(6):
            k = f"joint_{i+1}.pos"
            err = float(obs0.get(k, 0.0)) - float(zero_action.get(k, 0.0))
            errs.append(err)
            if abs(err) > TOL_DEG:
                at_target = False
        if "gripper.pos" in zero_action:
            gerr = float(obs0.get("gripper.pos", 0.0)) - zero_action["gripper.pos"]
            if abs(gerr) > TOL_MM:
                at_target = False
        # Periodic debug during zeroing
        if poll_idx % 5 == 0:
            pose_now = [round(float(obs0.get(f"joint_{i}.pos", 0.0)), 1) for i in range(1, 7)]
            print(
                "Zeroing: pose=",
                pose_now,
                "err=",
                [round(e, 1) for e in errs],
            )
        if at_target:
            break
        poll_idx += 1
        time.sleep(0.05)
    print("Zero pose reached (or timeout). Starting teleop...")
except Exception as e:
    print(f"Warning: zeroing step encountered an issue: {e}. Continuing to teleop.")

# Print SDK and mapping info
try:
    if robot._iface is not None:
        s = robot._iface.piper.GetArmStatus().arm_status
        print(f"Arm status: motion_status={getattr(s,'motion_status',None)} ctrl_mode={getattr(s,'ctrl_mode',None)}")
        print("joint_signs:", robot.config.joint_signs)
        print(
            "limits_deg:",
            [
                (robot._iface.min_pos[i], robot._iface.max_pos[i])
                for i in range(6)
            ],
        )
        print("gripper_mm_limits:", (robot._iface.min_pos[6], robot._iface.max_pos[6]))
except Exception as e:
    print("Warning: could not read initial arm status/limits:", e)

if not USE_STDIN:
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
else:
    # put stdin in cbreak non-blocking mode
    fd = sys.stdin.fileno()
    _old_term_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

def _stdin_kbhit(timeout=0.0) -> bool:
    return bool(select.select([sys.stdin], [], [], timeout)[0])

def _stdin_getch() -> str:
    try:
        ch = sys.stdin.read(1)
        return ch
    except Exception:
        return ""

try:
    loop_idx = 0
    while not stop_evt.is_set():
        start = time.perf_counter()
        obs = robot.get_observation()

        # start from current pose (hold-by-default)
        action = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}

        # read stdin keys if in stdin mode
        if USE_STDIN:
            # drain stdin quickly; treat each key as a tap (discrete)
            while _stdin_kbhit(0.0):
                ch = _stdin_getch().lower()
                if ch == "\x1b":  # ESC
                    stop_evt.set()
                    break
                if ch in keymap:
                    pressed.add(ch)

        # apply step per pressed key
        for ch in list(pressed):
            name, sgn = keymap[ch]
            if name == "gripper.pos":
                action[name] = action.get(name, 0.0) + sgn * STEP_MM
            else:
                action[name] = action.get(name, 0.0) + sgn * STEP_DEG

        # if using stdin, treat keys as taps (clear after applying)
        if USE_STDIN:
            pressed.clear()

        # Build debug arrays
        if VERBOSE:
            try:
                obs_deg = [float(obs.get(f"joint_{i}.pos", 0.0)) for i in range(1, 7)]
                act_deg = [float(action.get(f"joint_{i}.pos", obs_deg[i-1])) for i in range(1, 7)]
                signs = robot.config.joint_signs
                hw_deg = [a * s for a, s in zip(act_deg, signs)]
                if robot._iface is not None:
                    mn = robot._iface.min_pos[:6]
                    mx = robot._iface.max_pos[:6]
                    hw_deg_clip = [max(mn[i], min(mx[i], hw_deg[i])) for i in range(6)]
                else:
                    hw_deg_clip = hw_deg
            except Exception:
                obs_deg = act_deg = hw_deg = hw_deg_clip = []

        robot.send_action(action)

        # Periodic or on-keypress logging
        if VERBOSE and ((pressed and not USE_STDIN) or loop_idx % LOG_EVERY == 0 or (USE_STDIN and len(pressed) == 0)):
            try:
                keys_list = list(pressed)
                if USE_STDIN:
                    # we clear pressed after applying, so just show last applied action deltas by comparing obs vs act
                    pass
                status_line = ""
                if robot._iface is not None:
                    try:
                        as_ = robot._iface.piper.GetArmStatus().arm_status
                        status_line = f" | ctrl_mode={getattr(as_,'ctrl_mode',None)} motion_status={getattr(as_,'motion_status',None)}"
                    except Exception:
                        status_line = ""
                print(
                    f"keys={keys_list} obs_deg={[round(v,1) for v in obs_deg]} act_deg={[round(v,1) for v in act_deg]} hw_deg={[round(v,1) for v in hw_deg]} hw_clip={[round(v,1) for v in hw_deg_clip]}" + status_line
                )
            except Exception:
                pass

        dt = time.perf_counter() - start
        time.sleep(max(0.0, 1.0 / FPS - dt))
        loop_idx += 1
finally:
    robot.disconnect()
    if not USE_STDIN:
        listener.stop()
    else:
        if _old_term_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _old_term_settings)