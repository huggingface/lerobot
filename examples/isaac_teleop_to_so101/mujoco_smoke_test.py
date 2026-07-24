#!/usr/bin/env python

"""Headless fixed-target smoke test for the vendored SO-101 MuJoCo model."""

from __future__ import annotations

import argparse
import time

import numpy as np

from .mujoco_sink import MOTOR_NAMES, MuJoCoSO101Sink


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0, help="total simulated seconds")
    parser.add_argument("--control-hz", type=float, default=60.0)
    args = parser.parse_args()

    sink = MuJoCoSO101Sink()
    low, high = sink.ctrl_limits
    center = (low + high) / 2.0
    amplitude = np.minimum((high - low) * 0.18, np.deg2rad(20.0))
    target = center + amplitude * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    cycles = max(1, round(args.duration * args.control_hz))
    max_pause = 0.0
    wall_start = time.perf_counter()

    for cycle in range(cycles):
        loop_start = time.perf_counter()
        # Re-send one fixed six-axis target at the requested control rate, as the real loop does.
        sink.send_native_radians(target)
        # 500 Hz physics is not divisible by 60 Hz control. Alternate 8/9 steps by
        # following the absolute simulation deadline, avoiding long-run clock drift.
        step_count = max(1, round(((cycle + 1) / args.control_hz - sink.data.time) / sink.timestep))
        sink.step(step_count)
        max_pause = max(max_pause, time.perf_counter() - loop_start)

    wall_elapsed = time.perf_counter() - wall_start
    sim_elapsed = float(sink.data.time)
    control_rate = cycles / wall_elapsed
    print(f"PASS model: nq={sink.model.nq} nu={sink.model.nu} dt={sink.timestep:.4f}s")
    print(f"PASS actuators: {', '.join(MOTOR_NAMES)}")
    print(f"sim={sim_elapsed:.3f}s wall={wall_elapsed:.3f}s RTF={sim_elapsed / wall_elapsed:.2f}x")
    print(f"control throughput={control_rate:.1f}Hz max_loop_pause={max_pause * 1e3:.2f}ms")
    final_error = float(np.max(np.abs(sink.qpos - target)))
    print(f"target={np.round(target, 4).tolist()}")
    print(f"final_qpos={np.round(sink.qpos, 4).tolist()} max_final_error={final_error:.5f}rad")
    if not np.all(np.isfinite(sink.qpos)):
        raise SystemExit("FAIL: non-finite state")
    if sim_elapsed < args.duration * 0.98:
        raise SystemExit("FAIL: simulation did not advance for requested duration")
    if final_error > 0.03:
        raise SystemExit(f"FAIL: fixed target did not settle (max error {final_error:.5f}rad)")


if __name__ == "__main__":
    main()
