"""Drive UR10e with a PS4 gamepad directly and live-print command→motion latency.

No lerobot env / processors / cameras / dataset. Just pygame + ur_rtde.

What the printed `latency` actually covers (the path):

  [stick deflected by user, OS/USB/HID buffers — UNMEASURED]
   │
   ▼
  pygame.event.pump() + js.get_axis()
   │
   ▼  ── t_cmd stamped when stick_mag crosses STICK_RISING_HIGH from below
  main loop writes target_pose under target_lock
   │
   ▼
  streaming thread (200 Hz) reads target_pose
   │
   ▼
  ctrl.servoL(pose, ...)  →  RTDE control socket  →  UR controller
                                                          │
                                                          ▼
                                                       servoL interpolation
                                                          │
                                                          ▼
                                                    motors physically move
                                                          │
                                                          ▼
                                              encoders update controller state
                                                          │
                                                          ▼
  rtde_recv ◄────────── RTDE receive socket ─────────────┘
   │
   ▼  ── t_motion stamped when |tcp_xyz(t) - tcp_xyz(armed)| crosses MOTION_DISP_M
  motion-watcher (500 Hz) emits (t_cmd, t_motion)

So `latency_ms = t_motion - t_cmd` = time from "stick rising-edge visible to
main thread" to "robot feedback confirms TCP has moved >= MOTION_DISP_M."
Includes streaming-thread pickup (≤ 5 ms), RTDE round-trip (≈ 4 ms), URScript
servoL interpolation, mechanical ramp-up, encoder + RTDE receive (≤ 2 ms).
Does NOT include the time from when the operator's thumb actually moves to
when pygame's event-queue is drained — that's invisible from Python.

Tap the stick (or hold it) from rest; the latency prints live. Ctrl+C
prints a summary.
"""

from __future__ import annotations

import csv
import datetime as _dt
import queue
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import pygame
import rtde_control
import rtde_receive


# -- user-tunable ---------------------------------------------------------------
UR10_IP            = "192.168.0.100"        # same as ur10_env_3cams.json
RTDE_FREQUENCY     = 500                    # UR10e native 500 Hz
STREAM_HZ          = 200                    # servoL streaming rate
SERVO_LOOKAHEAD    = 0.15                   # AGGRESSIVE (was 0.15) — less smoothing, snappier response, more ringing
SERVO_GAIN         = 100.0                  # AGGRESSIVE (was 100) — higher P gain, faster tracking, noisier on stick release

# Main-loop polling rate. 50 Hz catches taps as short as ~20 ms. Faster than
# this gives diminishing returns because pygame's event queue is also drained
# on a USB/HID cadence.
FPS                = 50

# Cartesian speed in m/s when stick is fully deflected. EE step per iteration
# = SPEED * (1 / FPS). At 50 Hz × 0.05 m/s that's 1 mm/step, max 5 cm/s —
# AGGRESSIVE (was 0.01). Keep your hand near the stick centre at first.
SPEED_M_PER_S      = 0.01

WORKSPACE_MIN      = (-0.5, -0.5, 0.05)     # m, base frame
WORKSPACE_MAX      = ( 0.5,  0.5, 0.70)
FIXED_RX           = 3.14159                # axis-angle, wrist pointing down
FIXED_RY           = 0.0
FIXED_RZ           = 0.0

STICK_DEADZONE     = 0.05
# Schmitt trigger for rising-edge arming. Lower thresholds → catches lighter
# taps; HIGH/LOW must straddle the deadzone so a "rest" stick never re-arms.
STICK_RISING_HIGH  = 0.15
STICK_RISING_LOW   = 0.06

# Motion-onset detection — TCP-displacement based (not velocity). The watcher
# captures `baseline_xyz = recv.getActualTCPPose()[:3]` the moment the stick
# rising-edge arms it, then trips when |xyz(t) - baseline_xyz| crosses this.
# 0.2 mm is ~20× the encoder noise floor (~10 µm on a stationary UR10e) and
# is crossed by even sub-100 ms taps, so the trigger fires reliably.
MOTION_DISP_M      = 0.0002

# Timeout after which the watcher gives up and reports "no-motion" with the
# max displacement observed. 0.5 s is generous for any real motion at the
# default speed.
MOTION_TIMEOUT_S   = 0.5

# Latency is only meaningful when the robot is at rest at t_cmd. If the user
# taps again while the previous motion is still settling (servoL lookahead
# is 150 ms; mechanical settle is a few hundred ms more), the watcher's
# baseline_xyz drifts past MOTION_DISP_M in tens of ms purely from residual
# motion — yielding bogus ~50 ms readings that have nothing to do with the
# new tap's propagation. Skip arming until TCP linear speed is below this.
# 1 mm/s is well above stationary-arm noise (~0.05 mm/s) and below typical
# residual speed after a tap (3-5 mm/s).
REST_TCP_SPEED     = 0.001                  # m/s

# CSV logging. Per-tick rows at FPS Hz to data.csv; sparse events to events.csv.
# Both share the same `t_rel = time.perf_counter() - t_start` clock, so events
# can be joined back to per-tick data on time when plotting.
CSV_DIR_BASE       = Path(__file__).parent / "latency_runs"

# PS4 controller via pygame on Linux:
#   axis 0 = left-stick X,  axis 1 = left-stick Y (up is negative)
#   axis 3 = right-stick X, axis 4 = right-stick Y (up is negative)
# The values below produce the same final stick→robot mapping as ur10_train_3cams.json
# (invert_delta_x=true, invert_delta_y=true, invert_delta_z=false). Note that
# gamepad_utils.py applies a base sign flip to Y and Z before its own invert flags,
# so the JSON's three flags do NOT translate one-to-one onto raw pygame axes — the
# settings here are the *net effective* signs.
AXIS_DX            = 0                      # left  stick X
AXIS_DY            = 1                      # left  stick Y
AXIS_DZ            = 4                      # right stick Y
INVERT_DX          = True                   # net: dx = -axis0  (push right → robot -x)
INVERT_DY          = False                  # net: dy = +axis1  (push up    → robot -y)
INVERT_DZ          = True                   # net: dz = -axis4  (push up    → robot +z)
# -------------------------------------------------------------------------------


def _deadzone(v: float) -> float:
    return 0.0 if abs(v) < STICK_DEADZONE else float(v)


def main() -> None:
    # -- gamepad ---------------------------------------------------------------
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick detected (is the PS4 controller plugged in?)")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Gamepad: {js.get_name()!r}  axes={js.get_numaxes()} buttons={js.get_numbuttons()}")

    # -- UR10e RTDE ------------------------------------------------------------
    print(f"Connecting to UR10e at {UR10_IP} ({RTDE_FREQUENCY} Hz) ...")
    ctrl = rtde_control.RTDEControlInterface(UR10_IP, float(RTDE_FREQUENCY))
    recv = rtde_receive.RTDEReceiveInterface(UR10_IP, float(RTDE_FREQUENCY))
    print("Connected.")

    # Initialise target = live TCP so the very first servoL is a no-op (no jump).
    initial_pose = list(recv.getActualTCPPose())
    target_pose  = list(initial_pose)
    target_xyz   = list(initial_pose[:3])
    target_lock  = threading.Lock()
    stop_event   = threading.Event()

    # -- latency machinery -----------------------------------------------------
    # `arm_request` carries (t_cmd, axes_tuple) from main → watcher. Using a
    # queue (not a shared list + Event) means brief overlaps between watcher
    # finishing an event and main arming the next one can't lose the second
    # arm.
    arm_request: "queue.Queue[tuple[float, tuple]]" = queue.Queue()
    # result tuple: (t_cmd, t_motion_or_None, value, axes, status)
    #   status="OK"      → value = trip_disp (m)
    #   status="TIMEOUT" → value = max_disp  (m)  [t_motion is None]
    #   status="SKIPPED" → value = tcp_speed (m/s) [t_motion is None]
    result_q: "queue.Queue[tuple[float, float | None, float, tuple, str]]" = queue.Queue()

    # -- streaming thread (200 Hz servoL) --------------------------------------
    def stream_loop() -> None:
        dt = 1.0 / STREAM_HZ
        while not stop_event.is_set():
            with target_lock:
                pose = list(target_pose)
            try:
                ctrl.servoL(pose, 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
            except Exception as e:
                print(f"servoL failed: {e!r}")
                stop_event.set()
                return

    # -- motion-watcher thread (~500 Hz) ---------------------------------------
    # Polls TCP pose. Captures a baseline xyz the instant a new arm-request
    # arrives, then trips on |xyz(t) - baseline| > MOTION_DISP_M.
    def motion_loop() -> None:
        active: tuple | None = None      # (t_cmd, axes, baseline_xyz)
        max_disp = 0.0
        while not stop_event.is_set():
            # Pick up a new arm request (without blocking)
            if active is None:
                try:
                    t_cmd, axes = arm_request.get_nowait()
                    baseline_xyz = np.asarray(recv.getActualTCPPose()[:3], dtype=np.float64)
                    active = (t_cmd, axes, baseline_xyz)
                    max_disp = 0.0
                except queue.Empty:
                    pass

            if active is not None:
                t_cmd, axes, baseline_xyz = active
                xyz = np.asarray(recv.getActualTCPPose()[:3], dtype=np.float64)
                disp = float(np.linalg.norm(xyz - baseline_xyz))
                if disp > max_disp:
                    max_disp = disp
                now = time.perf_counter()
                if disp > MOTION_DISP_M:
                    result_q.put((t_cmd, now, disp, axes, "OK"))
                    active = None
                elif now - t_cmd > MOTION_TIMEOUT_S:
                    result_q.put((t_cmd, None, max_disp, axes, "TIMEOUT"))
                    active = None

            time.sleep(0.002)

    streaming = threading.Thread(target=stream_loop, name="ur10-stream", daemon=True)
    watching  = threading.Thread(target=motion_loop, name="motion-watch", daemon=True)
    streaming.start()
    watching.start()

    # -- CSV setup -------------------------------------------------------------
    run_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = CSV_DIR_BASE / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = run_dir / "data.csv"
    events_path = run_dir / "events.csv"
    data_f = data_path.open("w", newline="")
    events_f = events_path.open("w", newline="")
    data_w = csv.writer(data_f)
    events_w = csv.writer(events_f)

    # data.csv columns: one row per main-loop tick at FPS Hz.
    data_w.writerow([
        "t_rel_s",
        "raw_x", "raw_y", "raw_z",                          # pygame axis values (pre-deadzone, pre-invert)
        "dx", "dy", "dz", "stick_mag",                      # net stick after deadzone + invert + magnitude
        "target_x", "target_y", "target_z",                 # commanded TCP target this tick (m, base frame)
        "tcp_x", "tcp_y", "tcp_z",                          # actual TCP position (m, base frame)
        "tcp_vx", "tcp_vy", "tcp_vz", "tcp_speed",          # actual TCP linear velocity + magnitude (m/s)
        "q0", "q1", "q2", "q3", "q4", "q5",                 # actual joint positions (rad)
        "qd0", "qd1", "qd2", "qd3", "qd4", "qd5",           # actual joint velocities (rad/s)
    ])
    # events.csv columns: one row per latency event.
    #   status: "OK" → arm + motion detected, latency_ms valid, trip_disp_mm valid
    #           "TIMEOUT" → arm but motion never crossed threshold; latency_ms blank,
    #                        trip_disp_mm holds the *max* displacement seen
    #           "SKIPPED" → rising edge while robot was still moving; t_motion + latency
    #                        blank, tcp_speed_at_arm_mm_s holds the speed that triggered the skip
    events_w.writerow([
        "status",
        "t_cmd_rel_s", "t_motion_rel_s",
        "latency_ms",
        "trip_disp_mm",
        "tcp_speed_at_arm_mm_s",
        "axis_dx", "axis_dy", "axis_dz",
    ])
    print(f"Logging to {run_dir}")

    # -- main loop -------------------------------------------------------------
    dt_main = 1.0 / FPS
    step_per_tick = SPEED_M_PER_S * dt_main   # m of target travel per main-loop tick at full deflection
    prev_above = False
    events: list[float] = []
    rows_since_flush = 0

    sx = -1.0 if INVERT_DX else 1.0
    sy = -1.0 if INVERT_DY else 1.0
    sz = -1.0 if INVERT_DZ else 1.0

    t_start = time.perf_counter()

    print(
        f"Running at {FPS} Hz; speed cap = {SPEED_M_PER_S*1000:.1f} mm/s; "
        f"motion threshold = {MOTION_DISP_M*1000:.2f} mm.  Tap or hold the sticks; Ctrl+C to stop."
    )
    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            pygame.event.pump()

            raw_x = float(js.get_axis(AXIS_DX))
            raw_y = float(js.get_axis(AXIS_DY))
            raw_z = float(js.get_axis(AXIS_DZ))

            dx = sx * _deadzone(raw_x)
            dy = sy * _deadzone(raw_y)
            dz = sz * _deadzone(raw_z)

            target_xyz[0] = float(min(max(target_xyz[0] + dx * step_per_tick, WORKSPACE_MIN[0]), WORKSPACE_MAX[0]))
            target_xyz[1] = float(min(max(target_xyz[1] + dy * step_per_tick, WORKSPACE_MIN[1]), WORKSPACE_MAX[1]))
            target_xyz[2] = float(min(max(target_xyz[2] + dz * step_per_tick, WORKSPACE_MIN[2]), WORKSPACE_MAX[2]))
            with target_lock:
                target_pose[:] = [target_xyz[0], target_xyz[1], target_xyz[2],
                                  FIXED_RX, FIXED_RY, FIXED_RZ]

            stick_mag = (dx * dx + dy * dy + dz * dz) ** 0.5

            # Telemetry for this tick — one snapshot each, reused for both the
            # rest-check and the CSV row so all values share a single timestamp.
            tcp_pose  = recv.getActualTCPPose()       # 6: x,y,z,rx,ry,rz
            tcp_speed_vec = recv.getActualTCPSpeed()  # 6: vx,vy,vz,wx,wy,wz
            q_pos = recv.getActualQ()                 # 6 joint positions
            q_vel = recv.getActualQd()                # 6 joint velocities
            tcp_lin_speed = float(np.linalg.norm(np.asarray(tcp_speed_vec[:3])))

            # Schmitt trigger: arm only on stick rest→active transition,
            # AND only when the robot is at rest.
            if stick_mag > STICK_RISING_HIGH:
                if not prev_above:
                    if tcp_lin_speed < REST_TCP_SPEED:
                        arm_request.put((t0, (dx, dy, dz)))
                    else:
                        # Push SKIPPED through result_q so it ends up in events.csv
                        # alongside successful events (uniform handling).
                        result_q.put((t0, None, tcp_lin_speed, (dx, dy, dz), "SKIPPED"))
                prev_above = True
            elif stick_mag < STICK_RISING_LOW:
                prev_above = False

            # Per-tick data row (50 Hz).
            t_rel = t0 - t_start
            data_w.writerow([
                f"{t_rel:.6f}",
                f"{raw_x:.6f}", f"{raw_y:.6f}", f"{raw_z:.6f}",
                f"{dx:.6f}", f"{dy:.6f}", f"{dz:.6f}", f"{stick_mag:.6f}",
                f"{target_xyz[0]:.6f}", f"{target_xyz[1]:.6f}", f"{target_xyz[2]:.6f}",
                f"{tcp_pose[0]:.6f}", f"{tcp_pose[1]:.6f}", f"{tcp_pose[2]:.6f}",
                f"{tcp_speed_vec[0]:.6f}", f"{tcp_speed_vec[1]:.6f}", f"{tcp_speed_vec[2]:.6f}",
                f"{tcp_lin_speed:.6f}",
                *[f"{v:.6f}" for v in q_pos],
                *[f"{v:.6f}" for v in q_vel],
            ])
            rows_since_flush += 1
            if rows_since_flush >= 100:
                data_f.flush()
                events_f.flush()
                rows_since_flush = 0

            # Drain results — print and log to events.csv.
            while True:
                try:
                    t_cmd, t_motion, value, axes, status = result_q.get_nowait()
                except queue.Empty:
                    break
                axes_str = f"({axes[0]:+.2f},{axes[1]:+.2f},{axes[2]:+.2f})" if axes else ""
                t_cmd_rel = t_cmd - t_start
                if status == "OK":
                    lat_ms = (t_motion - t_cmd) * 1000.0
                    events.append(lat_ms)
                    t_motion_rel = t_motion - t_start
                    print(f"latency: {lat_ms:7.1f} ms  stick={axes_str}  trip_disp={value*1000:.3f} mm")
                    events_w.writerow([
                        "OK",
                        f"{t_cmd_rel:.6f}", f"{t_motion_rel:.6f}",
                        f"{lat_ms:.3f}",
                        f"{value*1000:.4f}",
                        "",
                        f"{axes[0]:.4f}", f"{axes[1]:.4f}", f"{axes[2]:.4f}",
                    ])
                elif status == "TIMEOUT":
                    print(f"  no-motion within {MOTION_TIMEOUT_S*1000:.0f} ms "
                          f"(max disp={value*1000:.3f} mm) {axes_str}")
                    events_w.writerow([
                        "TIMEOUT",
                        f"{t_cmd_rel:.6f}", "",
                        "",
                        f"{value*1000:.4f}",
                        "",
                        f"{axes[0]:.4f}", f"{axes[1]:.4f}", f"{axes[2]:.4f}",
                    ])
                elif status == "SKIPPED":
                    print(f"  skipped: robot still settling (|v|={value*1000:.2f} mm/s) {axes_str}")
                    events_w.writerow([
                        "SKIPPED",
                        f"{t_cmd_rel:.6f}", "",
                        "",
                        "",
                        f"{value*1000:.4f}",
                        f"{axes[0]:.4f}", f"{axes[1]:.4f}", f"{axes[2]:.4f}",
                    ])

            elapsed = time.perf_counter() - t0
            if elapsed < dt_main:
                time.sleep(dt_main - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        streaming.join(timeout=1.0)
        watching.join(timeout=1.0)
        try:
            ctrl.servoStop(10.0)
        except Exception as e:
            print(f"servoStop failed: {e!r}")
        try:
            ctrl.stopScript()
        except Exception as e:
            print(f"stopScript failed: {e!r}")
        try:
            ctrl.disconnect()
        except Exception:
            pass
        try:
            recv.disconnect()
        except Exception:
            pass
        try:
            data_f.flush(); data_f.close()
            events_f.flush(); events_f.close()
            print(f"Saved: {data_path}  ({rows_since_flush} pending rows flushed)")
            print(f"Saved: {events_path}")
        except Exception as e:
            print(f"CSV close failed: {e!r}")
        pygame.quit()

        if events:
            print()
            print(f"Latency summary over {len(events)} events:")
            print(f"  mean  : {statistics.mean(events):7.1f} ms")
            print(f"  median: {statistics.median(events):7.1f} ms")
            print(f"  min   : {min(events):7.1f} ms")
            print(f"  max   : {max(events):7.1f} ms")
            if len(events) >= 2:
                print(f"  stdev : {statistics.stdev(events):7.1f} ms")
        else:
            print("No latency events recorded.")


if __name__ == "__main__":
    main()
