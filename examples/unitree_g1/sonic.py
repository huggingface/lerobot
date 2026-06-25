#!/usr/bin/env python
"""
SONIC planner with full mode control.

Keyboard controls:
    N / P      - next / previous motion set
    1-8        - select mode within current set
    WASD       - movement direction
    Q / E      - rotate facing left / right
    9 / 0      - decrease / increase speed
    - / =      - decrease / increase height
    R          - force replan
    M          - toggle SMPL motion playback <-> locomotion (needs --motion-file)
    Space      - emergency stop -> IDLE
    Esc        - quit

Gamepad controls (Unitree wireless controller):
    Left stick Y  - speed (forward = fast, back = stop)
    Left stick X  - movement direction (offset from facing)
    Right stick X - facing direction (incremental rotation)
    Right stick Y - height (up = tall 0.8m, down = low 0.1m)
    Buttons       - unused (mode selection is keyboard-only)

For teleop integration use --robot.controller=SonicWholeBodyController instead.
"""

import argparse
import contextlib
import faulthandler
import gc
import sys
import time

import numpy as np
from motion_loader import SmplMotion

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.controllers.sonic_pipeline import (
    CONTROL_DT,
    DEFAULT_ANGLES,
    LM,
    MOTION_SETS,
    RawKeyboard,
    compute_kp_kd,
    drain_keyboard,
)
from lerobot.robots.unitree_g1.controllers.sonic_whole_body import SonicRuntime
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1


def main():
    parser = argparse.ArgumentParser(description="SONIC planner with keyboard + gamepad control")
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Robot IP for real hardware (e.g. 192.168.123.164). Omit for simulation.",
    )
    parser.add_argument(
        "--log-csv",
        action="store_true",
        help="Write /tmp/sonic_pose_log.csv (disabled by default for teleop perf)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU ONNX Runtime (skip CUDA even if onnxruntime-gpu is installed)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Ignored for sim (stock UnitreeG1 uses hub MuJoCo defaults)"
    )
    parser.add_argument(
        "--gamepad",
        action="store_true",
        help="Read Unitree wireless gamepad in sim (default: keyboard-only in sim)",
    )
    parser.add_argument(
        "--keyboard-only", action="store_true", help="Ignore wireless gamepad (terminal keyboard only)"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Play an SMPL motion clip (.npz) via SONIC whole-body mode "
        "(encode_mode=2) instead of locomotion planning.",
    )
    parser.add_argument(
        "--no-loop", action="store_true", help="With --motion-file, play once instead of looping"
    )
    args = parser.parse_args()

    # Surface native crashes (onnxruntime / mujoco) with a real traceback, and
    # avoid losing buffered diagnostics if the process dies mid-loop.
    faulthandler.enable()
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)

    print("=" * 60)
    print("SONIC planner - full mode control")
    print("  N/P  cycle sets | 1-8 select mode | WASD move")
    print("  Q/E  rotate     | 9/0 speed       | -/= height")
    print("  R    replan     | Space IDLE       | Esc quit")
    if args.ip:
        print(f"  Robot IP: {args.ip}")
    else:
        print("  Mode: simulation")
    print("=" * 60 + "\n")

    cfg = UnitreeG1Config(controller=None)  # full-body SONIC; standalone loop owns publish
    if args.ip:
        cfg.is_simulation = False
        cfg.robot_ip = args.ip
    else:
        cfg.is_simulation = True
        if args.headless:
            print("[Note] --headless ignored: sim uses stock UnitreeG1 + hub env")
    robot = UnitreeG1(cfg)
    robot.connect()
    kp, kd = compute_kp_kd()
    robot.kp = kp.copy()
    robot.kd = kd.copy()

    runtime = SonicRuntime(force_cpu=args.cpu)
    controller = runtime.controller
    ms = runtime.ms

    motion = None
    if args.motion_file:
        motion = SmplMotion(args.motion_file, loop=not args.no_loop)
        controller.smpl_motion = motion  # lets 'M' key toggle playback
        controller.encode_mode = 2  # start in SONIC whole-body SMPL imitation
        dur = motion.num_frames / motion.fps
        print(f"\n[Motion] SMPL whole-body playback: {args.motion_file}")
        print(
            f"  frames={motion.num_frames} fps={motion.fps:.1f} "
            f"duration={dur:.1f}s loop={not args.no_loop} encode_mode=2"
        )
        print("  Press 'M' to toggle SMPL playback <-> locomotion at runtime.")

    runtime.controller.print_input_diagnostics()

    print(f"\nStarting: {MOTION_SETS[0][0]} (default mode: {LM(ms.mode).name})")
    [print(f"  {i + 1}: {m.name}") for i, m in enumerate(MOTION_SETS[0][1])]
    print(
        "\n[Ready] Click THIS terminal, then W/A/S/D to move. 1-6 change mode, 9/0 speed, Esc quit.\n",
        flush=True,
    )

    # Sim hub publishes wireless_remote bytes that can fight terminal WASD.
    base_joystick = not args.keyboard_only and (args.gamepad or args.ip is not None)

    with RawKeyboard() as kb:
        try:
            gc.disable()
            gc_timer = 0.0
            robot.reset(CONTROL_DT, DEFAULT_ANGLES)
            time.sleep(1.0)

            last_status = time.time() - 2.1
            loop_t, enc_t, dec_t, obs_t, act_t = [], [], [], [], []
            slow_n = blend_n = 0
            stall_src = ""
            did_blend = False
            t_start = time.time()

            log_path = "/tmp/sonic_pose_log.csv"
            jnames = [m.name for m in G1_29_JointIndex]
            log_ctx = open(log_path, "w") if args.log_csv else None  # noqa: SIM115
            if log_ctx:
                log_ctx.write(
                    "t,step,cursor,ts,blend,mode,"
                    + ",".join(f"q{i}" for i in range(29))
                    + ","
                    + ",".join(f"ref{i}" for i in range(29))
                    + ","
                    + ",".join(f"act{i}" for i in range(29))
                    + ",delta_max,action_norm,token_norm\n"
                )

            try:
                while not robot._shutdown_event.is_set():
                    t0 = time.time()
                    if drain_keyboard(kb, ms, controller):
                        break

                    obs = robot.get_observation()
                    t_obs = time.time()
                    obs_t.append(1000 * (t_obs - t0))
                    if not obs:
                        runtime.tick({}, use_joystick=False)
                        time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))
                        continue

                    # SMPL playback only while in whole-body mode; 'M' toggles it.
                    motion_active = motion is not None and controller.encode_mode == 2
                    if motion_active:
                        controller.smpl_joints_10frame_step1 = motion.step()
                        if motion.done:
                            print("\n[Motion] clip finished")
                            break

                    step_before = runtime.step
                    t_step = time.time()
                    action = runtime.tick(obs, use_joystick=base_joystick and not motion_active)
                    step_ms = 1000 * (time.time() - t_step)
                    do_enc = step_before % 5 == 0
                    (enc_t if do_enc else dec_t).append(step_ms)

                    t_act = time.time()
                    robot.send_action(action)
                    act_t.append(1000 * (time.time() - t_act))

                    if log_ctx and runtime.step % 5 == 0:
                        t_rel = time.time() - t_start
                        q_r = np.array([obs.get(f"{n}.q", 0) for n in jnames])
                        a_v = np.array([action.get(f"{n}.q", 0) for n in jnames])
                        cur, ts = controller.ref_cursor, controller.motion_timesteps
                        q_ref = (
                            controller.motion_joint_positions[min(cur, ts - 1)] if ts > 0 else np.zeros(29)
                        )
                        log_ctx.write(
                            f"{t_rel:.4f},{runtime.step},{cur},{ts},{int(did_blend)},{ms.mode},"
                            + ",".join(f"{v:.6f}" for v in q_r)
                            + ","
                            + ",".join(f"{v:.6f}" for v in q_ref)
                            + ","
                            + ",".join(f"{v:.6f}" for v in a_v)
                            + ","
                            + f"{np.max(np.abs(a_v - q_r)):.6f},"
                            f"{np.linalg.norm(a_v):.6f},"
                            f"{np.linalg.norm(controller.token):.6f}\n"
                        )
                        did_blend = False

                    now = time.time()
                    loop_ms = 1000 * (now - t0)
                    if loop_ms > 50:
                        stall_src = (
                            f"[STALL] {loop_ms:.0f}ms: "
                            f"obs={obs_t[-1]:.0f} step={step_ms:.0f} act={act_t[-1]:.0f}"
                        )
                    if loop_ms > CONTROL_DT * 1500:
                        slow_n += 1

                    if now - last_status > 2.0:

                        def _avg(lst):
                            return sum(lst) / len(lst) if lst else 0

                        hz = 1000 / _avg(loop_t) if _avg(loop_t) else 0
                        print(
                            f"\r  {ms.status_line()}  step={runtime.step} "
                            f"ref={controller.ref_cursor}/{controller.motion_timesteps} "
                            f"loop={_avg(loop_t):.1f}ms(max={max(loop_t, default=0):.1f}) hz={hz:.0f} "
                            f"enc={_avg(enc_t):.1f} dec={_avg(dec_t):.1f} obs={_avg(obs_t):.1f} "
                            f"slow={slow_n} blends={blend_n}",
                            end="",
                            flush=True,
                        )
                        if stall_src:
                            print(f"\n  {stall_src}")
                            stall_src = ""
                        last_status = now
                        loop_t, enc_t, dec_t, obs_t, act_t = [], [], [], [], []
                        slow_n = blend_n = 0

                    gc_timer += CONTROL_DT
                    if gc_timer >= 10.0:
                        gc.collect()
                        gc_timer = 0.0
                    loop_t.append(loop_ms)
                    time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))
            finally:
                if log_ctx:
                    log_ctx.close()

        except KeyboardInterrupt:
            pass
        finally:
            gc.enable()
            if args.log_csv:
                print(f"\n[Log] Saved to {log_path}")
            runtime.shutdown()
            print("\nStopping...")
            if robot.is_connected:
                robot.disconnect()
            print("Done.")


if __name__ == "__main__":
    main()
