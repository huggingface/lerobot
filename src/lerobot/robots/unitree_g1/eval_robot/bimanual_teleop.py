# teleop_to_robot_bimanual_with_cams_and_pedal.py
import os
import re
import time
import threading
import traceback
import numpy as np
import cv2

from evdev import InputDevice, categorize, ecodes

from unitree_lerobot.lerobot.src.lerobot.teleoperators.homunculus import HomunculusArm, HomunculusArmConfig
from unitree_lerobot.eval_robot.robot_control.robot_arm_test import G1_29_ArmController, G1_29_JointIndex
from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK
from unitree_lerobot.eval_robot.make_robot import setup_image_client

# ------------ Config ------------
PEDAL_DEV = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
EPISODE_ROOT = "box_task"
DISPLAY_SCALE_HEAD = (1280, 960)  # ~2x per eye when binocular
DISPLAY_SCALE_WRIST = (960, 720)  # ~2x per wrist view
LOG_HZ = 30.0
CONTROL_HZ = 100.0
# --------------------------------

# LEFT ARM order: S_pitch, S_yaw, S_roll, Elbow_flex, Wrist_roll
def scale_to_joint_limits_left(u5: np.ndarray) -> np.ndarray:
    mins = np.array([-3.05, 0.00, -2.30, -1.00,  1.95], dtype=np.float32)
    maxs = np.array([ 1.65, 1.20,  1.30,  1.00, -1.00], dtype=np.float32)
    u = np.clip(u5.astype(np.float32), -1.0, 1.0)
    return mins + (u + 1.0) * 0.5 * (maxs - mins)

# RIGHT ARM order: S_pitch, S_yaw, S_roll, Elbow_flex, Wrist_roll
def scale_to_joint_limits_right(u5: np.ndarray) -> np.ndarray:
    # (min > max for some joints is intentional to mirror orientation)
    mins = np.array([-3.05, 0.00,  2.30,  2.00, -1.95], dtype=np.float32)
    maxs = np.array([ 1.65, -1.20, -1.30, -1.00,  1.00], dtype=np.float32)
    u = np.clip(u5.astype(np.float32), -1.0, 1.0)
    return mins + (u + 1.0) * 0.5 * (maxs - mins)

# ---- Episode numbering helpers ----
_num_re = re.compile(r"^\d+$")

def _ensure_root():
    os.makedirs(EPISODE_ROOT, exist_ok=True)

def _next_episode_id() -> int:
    _ensure_root()
    existing = [int(d) for d in os.listdir(EPISODE_ROOT)
                if _num_re.match(d) and os.path.isdir(os.path.join(EPISODE_ROOT, d))]
    return (max(existing) + 1) if existing else 1

def _episode_dir(ep_id: int) -> str:
    d = os.path.join(EPISODE_ROOT, str(ep_id))
    os.makedirs(d, exist_ok=True)
    return d

# ---- Pedal listener (runs in a thread) ----
class PedalState:
    def __init__(self):
        self.recording = False
        self.lock = threading.Lock()
        self.start_trigger = False
        self.stop_trigger = False

    def arm_start(self):
        with self.lock:
            self.start_trigger = True

    def arm_stop(self):
        with self.lock:
            self.stop_trigger = True

    def consume_triggers(self):
        with self.lock:
            s, t = self.start_trigger, self.stop_trigger
            self.start_trigger = False
            self.stop_trigger = False
        return s, t

def pedal_thread(dev_path: str, pstate: PedalState):
    try:
        dev = InputDevice(dev_path)
        print(f"[Pedal] Using {dev.path} ({dev.name})")
        for ev in dev.read_loop():
            if ev.type != ecodes.EV_KEY:
                continue
            key = categorize(ev)  # key.keystate: 1=down, 0=up, 2=hold
            code = getattr(key, "keycode", None)
            state = getattr(key, "keystate", None)
            if code == "KEY_A" and state == 1:
                pstate.arm_start()
            elif code == "KEY_B" and state == 1:
                pstate.arm_stop()
    except PermissionError as e:
        print(f"[Pedal] Permission error opening {dev_path}: {e}")
        print("  Try: sudo setfacl -m u:$USER:rw <device_path>")
    except Exception as e:
        print(f"[Pedal] Error: {e}")
        traceback.print_exc()

def main():
    # --- Teleop (Homunculus) ---
    left_exo  = HomunculusArm(HomunculusArmConfig("/dev/ttyACM0", id="unitree_left"))
    right_exo = HomunculusArm(HomunculusArmConfig("/dev/ttyACM1", id="unitree_right"))
    left_exo.connect(calibrate=True)
    right_exo.connect(calibrate=True)

    # --- Robot ---
    arm_ik = G1_29_ArmIK()
    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=False)

    # Zero non-arm joints ONCE
    arm_joint_indices = set(range(15, 29))  # 15..28 are arms
    for jid in G1_29_JointIndex:
        if jid.value not in arm_joint_indices:
            arm_ctrl.msg.motor_cmd[jid].mode = 1
            arm_ctrl.msg.motor_cmd[jid].q = 0.0
            arm_ctrl.msg.motor_cmd[jid].dq = 0.0
            arm_ctrl.msg.motor_cmd[jid].tau = 0.0

    # --- Camera client ---
    class _Cfg:
        sim = False
        arm = "G1_29"
        ee = "dex3"
        motion = False
    image_info = setup_image_client(_Cfg)
    tv_img_array    = image_info["tv_img_array"]
    wrist_img_array = image_info["wrist_img_array"]
    is_binocular    = image_info["is_binocular"]
    has_wrist_cam   = image_info["has_wrist_cam"]

    # --- Pedal listener thread ---
    pstate = PedalState()
    th = threading.Thread(target=pedal_thread, args=(PEDAL_DEV, pstate), daemon=True)
    th.start()

    # --- Logging state ---
    frames_rgb = []   # current episode frames (N, 640, 480, 3) RGB
    obs_states = []   # current episode obs (N, 14)
    recording = False
    ep_id = None
    log_dt = 1.0 / LOG_HZ
    next_log_t = time.perf_counter() + log_dt

    try:
        dt = 1.0 / CONTROL_HZ
        k = 0

        while True:
            t0 = time.perf_counter()

            # Handle pedal triggers
            start_trig, stop_trig = pstate.consume_triggers()
            if start_trig and not recording:
                ep_id = _next_episode_id()
                _episode_dir(ep_id)
                frames_rgb = []
                obs_states = []
                recording = True
                print(f"[Episode] START -> {ep_id}")

            if stop_trig and recording:
                # Save episode
                try:
                    ep_dir = _episode_dir(ep_id)
                    if len(frames_rgb) > 0:
                        frames_np = np.stack(frames_rgb, axis=0)
                        np.savez_compressed(os.path.join(ep_dir, "frames_rgb_640x480.npz"), frames_np)
                        print(f"[Episode] Saved {frames_np.shape} frames to {ep_dir}/frames_rgb_640x480.npz")
                    if len(obs_states) > 0:
                        obs_np = np.stack(obs_states, axis=0).astype(np.float32)
                        np.savez_compressed(os.path.join(ep_dir, "obs_state.npz"), obs_np)
                        print(f"[Episode] Saved {obs_np.shape} obs to {ep_dir}/obs_state.npz")
                except Exception:
                    traceback.print_exc()
                finally:
                    frames_rgb = []
                    obs_states = []
                    recording = False
                    print(f"[Episode] STOP  -> {ep_id}")
                    ep_id = None

            # --- LEFT teleop
            teleop_l = left_exo.get_action()
            vals_l   = list(teleop_l.values())[:5]
            vals_l   = [vals_l[0]] + [-v for v in vals_l[1:]]
            norm_l   = np.asarray(vals_l, dtype=np.float32) / 100.0
            q_left   = scale_to_joint_limits_left(norm_l)

            # --- RIGHT teleop
            teleop_r = right_exo.get_action()
            vals_r   = list(teleop_r.values())[:5]
            vals_r   = [vals_r[0]] + [-v for v in vals_r[1:]]
            norm_r   = np.asarray(vals_r, dtype=np.float32) / 100.0
            q_right  = scale_to_joint_limits_right(norm_r)

            # Build combined command (left: 0..4, right: 7..11)
            arm_cmd = np.zeros(14, dtype=np.float32)
            arm_cmd[0:5]  = q_left
            arm_cmd[7:12] = q_right

            # Send
            tau = arm_ik.solve_tau(arm_cmd)
            arm_ctrl.ctrl_dual_arm(arm_cmd, tau)

            # Readback for display/log
            q_curr = None
            try:
                q_curr = arm_ctrl.get_current_dual_arm_q()  # (14,)
                if (k % 30) == 0:
                    qL = q_curr[0:5]
                    qR = q_curr[7:12]
                    print(f"[k={k:05d}] q_left={np.array2string(qL, precision=3)}  q_right={np.array2string(qR, precision=3)}")
            except Exception:
                if (k % 300) == 0:
                    traceback.print_exc()

            # Camera display (bigger windows)
            head_image = tv_img_array.copy()
            if head_image is not None and head_image.size > 0 and np.any(head_image):
                if is_binocular:
                    h, w = head_image.shape[:2]
                    left_head  = head_image[:, :w // 2]
                    right_head = head_image[:, w // 2:]
                    cv2.imshow("Head Left",  cv2.resize(left_head,  DISPLAY_SCALE_HEAD))
                    cv2.imshow("Head Right", cv2.resize(right_head, DISPLAY_SCALE_HEAD))
                else:
                    h, w = head_image.shape[:2]
                    # double-ish
                    cv2.imshow("Head", cv2.resize(head_image, (max(2*w // 1, 1), max(2*h // 1, 1))))

            if has_wrist_cam and wrist_img_array is not None:
                wrist_image = wrist_img_array.copy()
                if wrist_image is not None and wrist_image.size > 0 and np.any(wrist_image):
                    h, w = wrist_image.shape[:2]
                    left_wrist  = wrist_image[:, :w // 2]
                    right_wrist = wrist_image[:,  w // 2:]
                    cv2.imshow("Wrist Left",  cv2.resize(left_wrist,  DISPLAY_SCALE_WRIST))
                    cv2.imshow("Wrist Right", cv2.resize(right_wrist, DISPLAY_SCALE_WRIST))

            # 30 Hz logging only when recording
            now = time.perf_counter()
            if recording and now >= next_log_t:
                # Frame: pick head-left (or single head), convert BGR->RGB, resize 640x480
                if head_image is not None and head_image.size > 0 and np.any(head_image):
                    if is_binocular:
                        h, w = head_image.shape[:2]
                        src = head_image[:, :w // 2]  # left eye
                    else:
                        src = head_image
                    frame_640x480 = cv2.resize(src, (640, 480))
                    frame_rgb = cv2.cvtColor(frame_640x480, cv2.COLOR_BGR2RGB)
                    frames_rgb.append(frame_rgb)

                # Obs
                if q_curr is not None:
                    obs_states.append(np.asarray(q_curr, dtype=np.float32))

                # schedule next tick (keeps cadence stable even if we miss one)
                next_log_t += (1.0 / LOG_HZ)

            # UI
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit command received.")
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                if head_image is not None and head_image.size > 0:
                    cv2.imwrite(f"head_{ts}.jpg", head_image)
                    print(f"Saved head_{ts}.jpg")

            k += 1

            # Loop pacing
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        # If recording when quitting, finalize the episode
        if recording and ep_id is not None:
            try:
                ep_dir = _episode_dir(ep_id)
                if len(frames_rgb) > 0:
                    frames_np = np.stack(frames_rgb, axis=0)
                    np.savez_compressed(os.path.join(ep_dir, "frames_rgb_640x480.npz"), frames_np)
                    print(f"[Episode] Saved {frames_np.shape} frames to {ep_dir}/frames_rgb_640x480.npz")
                if len(obs_states) > 0:
                    obs_np = np.stack(obs_states, axis=0).astype(np.float32)
                    np.savez_compressed(os.path.join(ep_dir, "obs_state.npz"), obs_np)
                    print(f"[Episode] Saved {obs_np.shape} obs to {ep_dir}/obs_state.npz")
            except Exception:
                traceback.print_exc()

        # Cleanup OpenCV and shared resources
        cv2.destroyAllWindows()
        try:
            from unitree_lerobot.eval_robot.utils.utils import cleanup_resources
            cleanup_resources(image_info)
        except Exception:
            pass
        for exo in (left_exo, right_exo):
            try:
                exo.disconnect()
            except Exception:
                pass

if __name__ == "__main__":
    main()
