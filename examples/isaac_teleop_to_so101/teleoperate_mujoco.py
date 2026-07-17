#!/usr/bin/env python

"""Quest/CloudXR -> existing clutch + Placo IK -> MuJoCo SO-101."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import mujoco.viewer

from lerobot.utils.robot_utils import precise_sleep

from .common import CLOUDXR_ENV_FILE, RESET_ORIGIN_DEG, HoldLatch, _wait_for_xr_controller
from .isaac_teleop import XRController, XRControllerConfig
from .mujoco_sink import MOTOR_NAMES, MuJoCoSO101Sink
from .mujoco_xr_control import XRToSO101Controller


def print_metrics(
    controller: XRToSO101Controller,
    *,
    elapsed: float,
    sim_elapsed: float,
    loops: int,
    tracked_frames: int,
    max_loop_pause_s: float,
) -> None:
    metrics = controller.metrics
    print(
        f"metrics control={loops / elapsed:.1f}Hz XR={tracked_frames / elapsed:.1f}Hz "
        f"IK_mean={metrics.ik_mean_ms:.3f}ms IK_max={metrics.ik_max_s * 1e3:.3f}ms "
        f"RTF={sim_elapsed / elapsed:.3f} max_pause={max_loop_pause_s * 1e3:.2f}ms "
        f"reclutch={metrics.reclutches} EE_jump={metrics.max_reclutch_ee_jump_m:.6f}m "
        f"joint_jump={metrics.max_reclutch_joint_jump_deg:.3f}deg"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--control-hz", type=float, default=60.0)
    parser.add_argument("--clutch-threshold", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 runs until Ctrl-C")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--skip-cloudxr-autolaunch", action="store_true")
    args = parser.parse_args()

    sink = MuJoCoSO101Sink()
    # Start from the same comfortable pose as the real XR path, but settle it entirely in sim.
    sink.send_action(RESET_ORIGIN_DEG)
    sink.step(round(3.0 / sink.timestep))
    controller = XRToSO101Controller(sink.get_observation(), args.clutch_threshold)
    hold = HoldLatch(list(MOTOR_NAMES))
    xr_config = XRControllerConfig(
        hand_side=args.hand,
        clutch_threshold=args.clutch_threshold,
        auto_launch_cloudxr=not args.skip_cloudxr_autolaunch,
        cloudxr_env_file=CLOUDXR_ENV_FILE,
    )
    xr_device = XRController(xr_config)
    viewer_context = (
        nullcontext(None) if args.no_viewer else mujoco.viewer.launch_passive(sink.model, sink.data)
    )

    try:
        with viewer_context as viewer:
            xr_device.connect()
            _wait_for_xr_controller(xr_device)
            print("Quest controller live. Hold Grip/Squeeze to move; Trigger controls the jaw.")
            wall_start = time.perf_counter()
            sim_start = float(sink.data.time)
            next_report = 5.0
            loops = 0
            tracked_frames = 0
            max_loop_pause_s = 0.0

            while args.duration <= 0.0 or time.perf_counter() - wall_start < args.duration:
                loop_start = time.perf_counter()
                observation = sink.get_observation()
                xr_action = xr_device.get_action()
                tracked_frames += int(xr_device.is_tracking)
                action = hold.resolve(controller.compute(xr_action, observation), observation)
                sink.send_action(action)

                # Advance 500 Hz physics to the absolute wall-clock deadline. This naturally
                # alternates 8/9 physics steps under 60 Hz control without accumulating drift.
                elapsed = time.perf_counter() - wall_start
                desired_sim_time = sim_start + elapsed
                steps = max(1, round((desired_sim_time - sink.data.time) / sink.timestep))
                sink.step(steps)
                if viewer is not None:
                    viewer.sync()
                    if not viewer.is_running():
                        break
                loops += 1
                max_loop_pause_s = max(max_loop_pause_s, time.perf_counter() - loop_start)

                if elapsed >= next_report:
                    print_metrics(
                        controller,
                        elapsed=elapsed,
                        sim_elapsed=float(sink.data.time) - sim_start,
                        loops=loops,
                        tracked_frames=tracked_frames,
                        max_loop_pause_s=max_loop_pause_s,
                    )
                    next_report += 5.0
                precise_sleep(max(1.0 / args.control_hz - (time.perf_counter() - loop_start), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        xr_device.disconnect()


if __name__ == "__main__":
    main()
