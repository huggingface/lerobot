#!/usr/bin/env python
"""Prove the cart's own hardware works, with nothing else in the loop.

No network, no leader arms, no dataset, no policy. Just this machine, the
two buses and the three cameras. If this passes, every later failure is in
the parts this does not touch.

    uv run --project ../XLeRobot python config/cart_selftest.py
    ... --head          also sweep the head servos
    ... --base          also turn the base wheels      (WHEELS OFF THE GROUND)
    ... --arms          also nudge each arm joint      (ARMS CLEAR OF OBSTACLES)

Read-only by default: it connects, reads state and grabs camera frames.
The three movement tests are opt-in and each asks before it moves anything.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

import draccus
import numpy as np
import yaml

from lerobot_robot_xlerobot import XLerobot, XLerobotConfig

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def record(status: str, label: str, detail: str = "") -> None:
    results.append((status, label, detail))
    print(f"  {status}  {label}{('  ' + detail) if detail else ''}")


def load_config(path: Path) -> XLerobotConfig:
    """Accept either a top-level XLerobotConfig or a teleoperate-style file
    with the robot nested under `robot:`. cart.yaml is the latter."""
    raw = yaml.safe_load(path.read_text())
    inner = raw.get("robot", raw)
    if inner.get("type") not in (None, "xlerobot"):
        sys.exit(f"error: {path} describes a '{inner['type']}', not an xlerobot.")
    inner = {k: v for k, v in inner.items() if k != "type"}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(inner, f)
        tmp = f.name
    return draccus.parse(config_class=XLerobotConfig, args=["--config_path", tmp])


def ask(prompt: str) -> bool:
    try:
        return input(f"\n{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


# ---------------------------------------------------------------- state

def test_state(robot: XLerobot, n: int = 30) -> dict:
    print(f"\n--- state, {n} reads " + "-" * 40)
    samples, times = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        obs = robot.get_observation()
        times.append(time.perf_counter() - t0)
        samples.append(obs)

    hz = 1.0 / statistics.mean(times)
    detail = f"{hz:.1f} Hz, worst read {max(times) * 1000:.0f} ms"
    record(PASS if hz >= 25 else WARN, "observation rate", detail)
    if hz < 25:
        print("       Below 30 fps means recording cannot keep its contract.")
        print("       Usually a camera, not the servos - compare with --no-cameras.")

    last = samples[-1]
    print()
    for side in ("left", "right"):
        for j in ARM_JOINTS:
            k = f"{side}_arm_{j}.pos"
            print(f"      {k:<34}{last[k]:8.2f}")
    for k in ("head_motor_1.pos", "head_motor_2.pos"):
        print(f"      {k:<34}{last[k]:8.2f}")
    print()
    for k in ("x.vel", "y.vel", "theta.vel"):
        print(f"      {k:<34}{last[k]:8.3f}")

    # Constancy is NOT evidence of a fault here. connect() leaves torque on,
    # so a healthy servo holding position reports the same number every
    # read - that is what holding position means. The useful discriminator
    # is whether a joint MOVES when commanded, which is what --arms and
    # --head do. So this only reports, it does not judge.
    frozen = [
        k for k in last
        if k.endswith(".pos") and len({round(s[k], 4) for s in samples}) == 1
    ]
    n_pos = sum(1 for k in last if k.endswith(".pos"))
    if len(frozen) == n_pos:
        record(PASS, "all joints steady", "expected: torque is on and nothing is moving")
    elif frozen:
        record(PASS, f"{n_pos - len(frozen)}/{n_pos} joints show encoder jitter",
               "the rest are steady, which is normal under torque")
    else:
        record(PASS, "all joints show encoder jitter")

    return last


# -------------------------------------------------------------- cameras

def test_cameras(robot: XLerobot, outdir: Path) -> None:
    print(f"\n--- cameras " + "-" * 48)
    expected = dict(robot._cameras_ft)
    if not expected:
        record(FAIL, "no cameras in the config")
        return

    first = robot.get_camera_observation()
    time.sleep(0.4)
    second = robot.get_camera_observation()

    outdir.mkdir(parents=True, exist_ok=True)
    for name, want in expected.items():
        frame = second.get(name)
        if frame is None:
            record(FAIL, f"{name}: no frame")
            continue

        got = tuple(frame.shape)
        if got != tuple(want):
            record(FAIL, f"{name}: shape {got}, config says {tuple(want)}")
            continue

        mean = float(np.mean(frame))
        moved = float(np.mean(np.abs(frame.astype(np.int16) - first[name].astype(np.int16))))

        path = outdir / f"{name}.png"
        try:
            import cv2

            cv2.imwrite(str(path), frame)
            saved = f"-> {path}"
        except Exception as e:
            saved = f"(not saved: {e})"

        if mean < 2.0:
            # Lens cap first: it is the commonest cause by a distance, and
            # it looks exactly like a camera delivering nothing. The two are
            # told apart by the inter-frame delta - a capped camera is still
            # streaming, so its frames differ slightly from sensor noise,
            # while a starved one repeats the same empty buffer.
            why = "lens cap?" if moved > 0.05 else "no valid frame arriving"
            record(FAIL, f"{name}: black frame",
                   f"mean {mean:.1f}, delta {moved:.2f} - {why}  {saved}")
        elif moved < 0.25:
            record(WARN, f"{name}: frames identical 0.4s apart",
                   f"lens cap, or a stalled stream  {saved}")
        else:
            record(PASS, f"{name}: {got[1]}x{got[0]}",
                   f"mean {mean:.0f}, inter-frame delta {moved:.1f}  {saved}")

    print(f"\n      Open the PNGs in {outdir} and check they are the view you expect -")
    print("      a camera can be live, correctly sized and pointed at the wrong thing.")


# ----------------------------------------------------------------- head

def test_head(robot: XLerobot, base: dict, span: float = 12.0) -> None:
    print(f"\n--- head " + "-" * 51)
    print("      The wheels should not move; see the note under --arms.")
    if not ask(f"Sweep both head servos +/-{span:.0f} degrees?"):
        record(WARN, "head sweep skipped")
        return

    for motor in ("head_motor_1", "head_motor_2"):
        key = f"{motor}.pos"
        start = base[key]
        seen = []
        try:
            for target in (start + span, start - span, start):
                robot.send_action({key: target})
                time.sleep(0.6)
                seen.append(robot.get_observation()[key])
        finally:
            robot.send_action({key: start})
            time.sleep(0.4)

        travel = max(seen) - min(seen)
        if travel > span * 0.5:
            record(PASS, f"{motor} moved", f"{travel:.1f} deg over a {2 * span:.0f} deg command")
        else:
            record(FAIL, f"{motor} barely moved", f"{travel:.1f} deg - stalled, or blocked")


# ----------------------------------------------------------------- arms

def test_arms(robot: XLerobot, base: dict, span: float = 5.0) -> None:
    print(f"\n--- arms " + "-" * 51)
    print("      Each joint moves a few degrees and returns. Make sure both")
    print("      arms are clear of the table, each other and the cameras.")
    print()
    print("      The wheels should NOT move. send_action writes a goal velocity")
    print("      to all three on every call, and with no .vel keys that goal is")
    print("      zero - so a twitch here means the base calibration is wrong,")
    print("      not that the arms are misbehaving.")
    if not ask(f"Nudge all 12 arm joints by {span:.0f} degrees, one at a time?"):
        record(WARN, "arm nudge skipped")
        return

    for side in ("left", "right"):
        for j in ARM_JOINTS:
            key = f"{side}_arm_{j}.pos"
            start = base[key]
            seen = []
            try:
                for target in (start + span, start):
                    robot.send_action({key: target})
                    time.sleep(0.5)
                    seen.append(robot.get_observation()[key])
            finally:
                robot.send_action({key: start})
                time.sleep(0.3)

            travel = max(seen + [start]) - min(seen + [start])
            if travel > span * 0.4:
                record(PASS, f"{side}_arm_{j}", f"{travel:.1f} deg")
            else:
                record(FAIL, f"{side}_arm_{j} did not move", f"{travel:.1f} deg")


# ----------------------------------------------------------------- base

def test_base(robot: XLerobot, speed: float = 0.08, turn: float = 25.0) -> None:
    print(f"\n--- base " + "-" * 51)
    print("      The wheels will turn. PUT THE CART ON BLOCKS or hold it clear")
    print("      of the floor - it will drive away otherwise.")
    if not ask("Wheels are off the ground. Drive them?"):
        record(WARN, "base test skipped")
        return

    moves = [
        ("forward", {"x.vel": speed}, "x.vel", speed),
        ("backward", {"x.vel": -speed}, "x.vel", -speed),
        ("turn left", {"theta.vel": turn}, "theta.vel", turn),
        ("turn right", {"theta.vel": -turn}, "theta.vel", -turn),
    ]
    for label, action, key, want in moves:
        try:
            for _ in range(12):           # ~1.2s of commands, it is a velocity
                robot.send_action(action)
                time.sleep(0.1)
            got = statistics.median(robot.get_observation()[key] for _ in range(5))
        finally:
            robot.stop_base()
            time.sleep(0.5)

        same_sign = math.copysign(1, got) == math.copysign(1, want)
        if abs(got) < abs(want) * 0.2:
            record(FAIL, f"base {label}: no feedback", f"{key} read {got:.3f}, commanded {want:.3f}")
        elif not same_sign:
            record(FAIL, f"base {label}: wrong direction",
                   f"{key} read {got:.3f}, commanded {want:.3f} - a wheel is reversed")
        else:
            record(PASS, f"base {label}", f"{key} {got:.3f} vs {want:.3f} commanded")

    print("\n      Watch the wheels as well as the numbers. The body velocity is")
    print("      derived from all three, so two correct wheels can mask a third")
    print("      that is spinning the wrong way.")
    print()
    print("      Geometry is hardcoded in _body_to_wheel_raw: wheel_radius 0.05 m,")
    print("      base_radius 0.125 m. If x.vel reads consistently high or low by")
    print("      the same factor, measure yours - it is a scale error, not a fault.")


# ----------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_path", default="config/cart.yaml",
                    help="an XLerobotConfig, top-level or nested under robot:")
    ap.add_argument("--out", default="selftest-frames", help="where camera frames are written")
    ap.add_argument("--head", action="store_true")
    ap.add_argument("--arms", action="store_true")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--no-cameras", action="store_true", help="isolate the servos from a camera fault")
    args = ap.parse_args()

    cfg = load_config(Path(args.config_path))
    if args.no_cameras:
        cfg.cameras = {}

    print(f"config : {args.config_path}")
    print(f"id     : {cfg.id}")
    print(f"port1  : {cfg.port1}")
    print(f"port2  : {cfg.port2}")
    print(f"cameras: {', '.join(cfg.cameras) or '(none)'}")

    robot = XLerobot(cfg)
    print("\nconnecting...")
    robot.connect(calibrate=False)
    record(PASS, "connected", "both buses and every camera opened")

    if not robot.is_calibrated:
        record(FAIL, "NOT CALIBRATED", "joint readings below are raw and meaningless")
        print("\n       lerobot-calibrate --robot.type=xlerobot --robot.id=" + str(cfg.id) + " \\")
        print(f"         --robot.port1={cfg.port1} \\")
        print(f"         --robot.port2={cfg.port2}")
    else:
        record(PASS, "calibration loaded")

    try:
        base_state = test_state(robot)
        if cfg.cameras:
            test_cameras(robot, Path(args.out))
        if args.head:
            test_head(robot, base_state)
        if args.arms:
            test_arms(robot, base_state)
        if args.base:
            test_base(robot)
    finally:
        print("\ndisconnecting...")
        robot.disconnect()

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for status, label, detail in results:
        print(f"  {status}  {label}{('  ' + detail) if detail else ''}")

    failed = [r for r in results if r[0] == FAIL]
    warned = [r for r in results if r[0] == WARN]
    print(f"\n  {len(results) - len(failed) - len(warned)} passed, {len(warned)} warned, {len(failed)} failed")

    if not args.head or not args.arms or not args.base:
        skipped = [n for n, on in (("--head", args.head), ("--arms", args.arms), ("--base", args.base)) if not on]
        print(f"  Movement not exercised: {' '.join(skipped)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
