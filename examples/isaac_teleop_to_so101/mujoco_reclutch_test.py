#!/usr/bin/env python

"""Synthetic 20-cycle clutch/hold/re-clutch continuity test with real Placo IK."""

from __future__ import annotations

import argparse
import time

import numpy as np

from .common import HoldLatch
from .mujoco_sink import MOTOR_NAMES, MuJoCoSO101Sink
from .mujoco_xr_control import XRToSO101Controller

CONTROL_HZ = 60.0
PHYSICS_HZ = 500.0


def xr(pos: np.ndarray, squeeze: float, trigger: float = 0.25) -> dict:
    return {
        "grip_pos": pos,
        "grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "squeeze": squeeze,
        "trigger": trigger,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stability-seconds",
        type=float,
        default=0.0,
        help="additional simulated seconds of smooth 60 Hz Cartesian control",
    )
    args = parser.parse_args()

    sink = MuJoCoSO101Sink()
    controller = XRToSO101Controller(sink.get_observation())
    hold = HoldLatch(list(MOTOR_NAMES))
    grip = np.array([0.25, -0.10, 0.20])
    max_hold_drift_deg = 0.0
    max_loop_pause_s = 0.0
    loop_count = 0
    started = time.perf_counter()

    def frame(xr_action: dict) -> None:
        nonlocal max_loop_pause_s, loop_count
        frame_start = time.perf_counter()
        observation = sink.get_observation()
        action = hold.resolve(controller.compute(xr_action, observation), observation)
        sink.send_action(action)
        target_time = (loop_count + 1) / CONTROL_HZ
        step_count = max(1, round((target_time - sink.data.time) * PHYSICS_HZ))
        sink.step(step_count)
        loop_count += 1
        max_loop_pause_s = max(max_loop_pause_s, time.perf_counter() - frame_start)

    # Settle an initial engaged pose, then exercise small Cartesian moves between re-clutches.
    for _ in range(60):
        frame(xr(grip, 1.0))

    for cycle in range(20):
        delta = np.array([0.004, (-1.0) ** cycle * 0.003, 0.002])
        for alpha in np.linspace(0.0, 1.0, 8):
            frame(xr(grip + alpha * delta, 1.0, trigger=cycle / 19.0))

        # Release: the first idle frame latches, and subsequent frames must keep that target.
        frame(xr(grip + delta, 0.0))
        for _ in range(20):
            frame(xr(grip + delta + np.array([0.08, -0.05, 0.04]), 0.0))
        held_start = controller.joint_vector(sink.get_observation())
        for _ in range(20):
            frame(xr(grip + delta + np.array([0.08, -0.05, 0.04]), 0.0))
        held_end = controller.joint_vector(sink.get_observation())
        max_hold_drift_deg = max(max_hold_drift_deg, float(np.max(np.abs(held_end[:5] - held_start[:5]))))

        # Re-engage at an unrelated controller position without moving it on the edge frame.
        grip = grip + np.array([0.025, -0.018, 0.011])
        frame(xr(grip, 1.0, trigger=cycle / 19.0))

    stability_frames = max(0, round(args.stability_seconds * CONTROL_HZ))
    stability_origin = grip.copy()
    for index in range(stability_frames):
        phase = 2.0 * np.pi * index / max(round(10.0 * CONTROL_HZ), 1)
        position = stability_origin + np.array(
            [0.010 * np.sin(phase), 0.008 * np.sin(phase * 0.7), 0.006 * np.sin(phase * 1.3)]
        )
        frame(xr(position, 1.0, trigger=0.5 + 0.5 * np.sin(phase * 0.2)))
        if not np.all(np.isfinite(sink.qpos)) or not np.all(np.isfinite(sink.data.ctrl)):
            raise SystemExit(f"FAIL: non-finite simulation state at stability frame {index}")

    wall_elapsed = time.perf_counter() - started
    m = controller.metrics
    print(f"PASS synthetic reclutches={m.reclutches} control_frames={loop_count}")
    print(
        f"IK mean={m.ik_mean_ms:.3f}ms max={m.ik_max_s * 1e3:.3f}ms "
        f"max_loop_pause={max_loop_pause_s * 1e3:.3f}ms"
    )
    print(
        f"max_reclutch_ee_jump={m.max_reclutch_ee_jump_m:.6f}m "
        f"max_reclutch_joint_jump={m.max_reclutch_joint_jump_deg:.3f}deg"
    )
    print(
        f"max_hold_drift={max_hold_drift_deg:.3f}deg sim={sink.data.time:.3f}s "
        f"RTF={sink.data.time / wall_elapsed:.2f}x"
    )
    if stability_frames:
        print(
            f"PASS accelerated_stability={args.stability_seconds:.1f}s "
            f"frames={stability_frames} finite_state=true"
        )
    if m.reclutches != 21:
        raise SystemExit(f"FAIL: expected initial engage + 20 re-clutches, got {m.reclutches}")
    if m.max_reclutch_ee_jump_m > 1e-6:
        raise SystemExit("FAIL: re-clutch Cartesian target jumped")
    if m.max_reclutch_joint_jump_deg > 5.0:
        raise SystemExit("FAIL: re-clutch IK joint target jumped by more than 5 degrees")
    if max_hold_drift_deg > 1.0:
        raise SystemExit("FAIL: released clutch did not hold")


if __name__ == "__main__":
    main()
