# !/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teleoperate an SO-101 follower from an SO-101 *leader arm* via Isaac Teleop.

This is the joint-space sibling of ``teleoperate.py`` (which drives the follower from an
XR controller through clutch + IK). Here the input is a back-drivable SO-101 *leader* whose
six joint angles are streamed by NVIDIA Isaac Teleop's native ``so101_leader`` plugin over
the OpenXR tensor transport. Because the leader and follower share the SO-101 kinematics,
the control law is a direct 1:1 joint mirror -- there is no clutch, no IK, no URDF::

    so101_leader plugin ──(JointStateOutput over OpenXR)──▶ SO101LeaderArm.get_action()
                                                                  │  rad2deg + gripper->RANGE_0_100
                                                                  ▼
                                                          robot.send_action({joint}.pos)

This mirrors how ``lerobot-teleoperate`` drives a follower from the serial ``so101_leader``:
the leader's ``get_action()`` already returns follower-ready ``{joint}.pos``, sent straight
to the robot. :class:`SO101LeaderArm` does the unit conversion internally (see its module
docstring), so this script is a thin read->send loop.

Pieces that must be running:

* **CloudXR runtime** -- auto-launched by ``SO101LeaderArm.connect()`` (the shared
  ``IsaacTeleopTeleoperator`` base; first launch may prompt for the EULA and take ~30s).
  Opt out with ``--no-auto-launch-cloudxr`` if you run CloudXR externally.
* **so101_leader plugin** -- the C++ device that reads the physical leader's servos and
  pushes ``JointStateOutput``. Either start it yourself (``so101_leader_plugin <port>``) or
  pass ``--launch-plugin --plugin-bin <path> [--leader-port <port>]`` to have this script
  spawn it AFTER CloudXR is up (so it inherits the runtime env). With no ``--leader-port``
  the plugin runs its synthetic trajectory -- handy for a no-hardware dry run.

Startup safety: by default the follower is smoothly slewed from its current pose to the
leader's first reading over ``--align-duration`` seconds (``--no-align`` to skip), so the
arm does not snap when the 1:1 mirror begins. (``lerobot-teleoperate`` snaps; this is the
one deliberate divergence, matching the jump-free ethos of this example folder.) While the
leader is not streaming the follower is held at its measured pose.

Examples::

    # No hardware: synthetic leader trajectory, follower on /dev/ttyACM0.
    python teleoperate_leader.py --launch-plugin \
        --plugin-bin /path/to/IsaacTeleop/install/plugins/so101_leader/so101_leader_plugin

    # Real leader on /dev/ttyACM1, follower on /dev/ttyACM0.
    python teleoperate_leader.py --port /dev/ttyACM0 --launch-plugin \
        --plugin-bin /path/to/so101_leader_plugin --leader-port /dev/ttyACM1
"""

import argparse
import subprocess
import time
from pathlib import Path

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.isaac_teleop import SO101LeaderArm, SO101LeaderArmConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30

# CloudXR device-profile env file passed to the launcher (shared with teleoperate.py).
CLOUDXR_ENV_FILE = str(Path(__file__).parent / "default.env")

# Default duration [s] for the startup alignment slew (follower current -> leader first pose).
ALIGN_DURATION_S = 3.0

# How long to wait for the leader plugin to start streaming before aligning / looping.
LEADER_WARMUP_TIMEOUT_S = 20.0


def _wait_for_leader(teleop: SO101LeaderArm, timeout_s: float) -> dict[str, float]:
    """Poll the leader until it streams a live frame; return that frame's ``{joint}.pos``.

    Raises ``SystemExit`` if no live frame arrives within ``timeout_s`` (plugin not pushing,
    wrong ``--collection-id``, or CloudXR not up).
    """
    print(f"Waiting up to {timeout_s:.0f}s for the so101_leader plugin to stream…")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        action = teleop.get_action()
        if teleop.is_tracking:
            print("Leader is streaming.")
            return action
        time.sleep(1.0 / FPS)
    raise SystemExit(
        f"FAILED: leader did not stream within {timeout_s:.0f}s. Is the so101_leader plugin "
        "running and pushing (check --collection-id)? Is CloudXR up?"
    )


def align_follower_to_leader(
    teleop: SO101LeaderArm, robot, motor_names: list[str], duration_s: float
) -> None:
    """Slew the follower from its current pose to a continuously re-read live leader pose.

    Ramps ``alpha`` 0->1 from the follower's measured start toward the LIVE leader reading
    taken each step, so at ``alpha == 1`` the follower lands on the leader's *current* pose
    and the handoff to the 1:1 mirror loop is continuous even if the operator keeps moving
    the leader during the ramp (a fixed first-reading target would snap on handoff instead).
    """
    obs = robot.get_observation()
    start = {name: float(obs[f"{name}.pos"]) for name in motor_names}
    n_steps = max(1, int(duration_s * FPS))
    print(f"Aligning follower to leader over {duration_s:.1f}s ({n_steps} steps)…")
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        leader_now = teleop.get_action()  # re-read live each step so alpha=1 lands on the current pose
        target = {name: float(leader_now[f"{name}.pos"]) for name in motor_names}
        action = {f"{name}.pos": start[name] + alpha * (target[name] - start[name]) for name in motor_names}
        robot.send_action(action)
        precise_sleep(1.0 / FPS)
    print("Alignment complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port the SO-101 FOLLOWER arm is connected to (default: /dev/ttyACM0).",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="so101_follower_arm",
        help="Device id for the SO-101 follower arm (selects its calibration).",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default="so101_leader",
        help="Tensor collection id the leader plugin pushes on (must match the plugin).",
    )
    parser.add_argument(
        "--launch-plugin",
        action="store_true",
        help="Spawn the so101_leader plugin process automatically (AFTER CloudXR is up).",
    )
    parser.add_argument(
        "--plugin-bin",
        type=str,
        default=None,
        help="Path to the so101_leader_plugin binary (required with --launch-plugin). "
        "Built in the IsaacTeleop repo under install/plugins/so101_leader/.",
    )
    parser.add_argument(
        "--leader-port",
        type=str,
        default="",
        help="Serial port of the physical LEADER arm, passed to the launched plugin. "
        "Empty (default) -> the plugin runs its synthetic trajectory (no leader hardware).",
    )
    parser.add_argument(
        "--no-auto-launch-cloudxr",
        dest="auto_launch_cloudxr",
        action="store_false",
        help="Do not auto-launch CloudXR (assume it is already running externally).",
    )
    parser.add_argument(
        "--align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Slew the follower to the leader's first pose before mirroring (default: True). "
        "Pass --no-align to begin the 1:1 mirror immediately (the follower may snap).",
    )
    parser.add_argument(
        "--align-duration",
        type=float,
        default=ALIGN_DURATION_S,
        metavar="SEC",
        help=f"Duration in seconds for the startup alignment slew (default: {ALIGN_DURATION_S}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read-only: run the full leader read + conversion and log everything, but NEVER "
        "command the follower (no alignment slew, no send_action). Validates the joint mirror "
        "and gripper mapping safely; tolerates an absent follower.",
    )
    return parser.parse_args()


def _maybe_launch_plugin(args) -> subprocess.Popen | None:
    """Spawn the so101_leader plugin if requested (called AFTER teleop.connect())."""
    if not args.launch_plugin:
        return None
    if not args.plugin_bin:
        raise SystemExit("--launch-plugin requires --plugin-bin <path to so101_leader_plugin>")
    if not Path(args.plugin_bin).exists():
        raise SystemExit(
            f"plugin binary not found: {args.plugin_bin} (build it in the IsaacTeleop repo first)"
        )
    backend = f"leader on {args.leader_port}" if args.leader_port else "synthetic trajectory"
    print(f"launching plugin: {args.plugin_bin} ({backend})")
    # Positional args: [device_path] [collection_id]. Empty device_path -> synthetic backend.
    # Spawned after connect() so it inherits the CloudXR runtime env (XR_RUNTIME_JSON, ...).
    proc = subprocess.Popen([args.plugin_bin, args.leader_port, args.collection_id])
    time.sleep(1.5)  # let it create its OpenXR session and start pushing
    return proc


def main():
    args = parse_args()

    robot_config = SO100FollowerConfig(port=args.port, id=args.id, use_degrees=True)
    # SO100Follower is the shared SO-100/SO-101 follower class (registered under both ids).
    robot = SO100Follower(robot_config)
    motor_names = list(robot.bus.motors.keys())

    teleop_config = SO101LeaderArmConfig(
        collection_id=args.collection_id,
        auto_launch_cloudxr=args.auto_launch_cloudxr,
        cloudxr_env_file=CLOUDXR_ENV_FILE,
    )
    teleop = SO101LeaderArm(teleop_config)

    # Connect the follower (tolerating its absence only in --dry-run).
    robot_connected = False
    if args.dry_run:
        print("[DRY-RUN] Read-only mode: the follower will NOT be commanded.")
        try:
            robot.connect()
            robot_connected = True
        except Exception as e:  # noqa: BLE001  (bring-up convenience: any connect failure -> no-arm)
            print(f"[DRY-RUN] Follower not connected ({type(e).__name__}: {e}); continuing without it.")
    else:
        robot.connect()
        robot_connected = True

    # connect() auto-launches CloudXR (unless opted out); spawn the plugin AFTER so it
    # inherits the runtime env. The plugin is reaped in the finally block.
    teleop.connect()
    plugin_proc = None
    try:
        plugin_proc = _maybe_launch_plugin(args)

        if not teleop.is_connected:
            raise ValueError("Teleop is not connected!")
        if not args.dry_run and not robot_connected:
            raise ValueError("Follower is not connected!")

        # Block until the leader streams a live frame (clear error if it never does).
        _wait_for_leader(teleop, LEADER_WARMUP_TIMEOUT_S)

        if args.align and robot_connected and not args.dry_run:
            align_follower_to_leader(teleop, robot, motor_names, args.align_duration)

        print(
            "Starting DRY-RUN loop (follower will NOT move). Back-drive the leader and watch the readout."
            if args.dry_run
            else "Starting joint-mirror loop. Back-drive the leader to teleoperate the follower… (Ctrl-C to stop)"
        )
        _frame = 0
        while True:
            t0 = time.perf_counter()
            _frame += 1

            leader_action = teleop.get_action()

            # Hold the follower at its measured pose when the leader drops out (stale stream),
            # rather than commanding a held-last (possibly old) target.
            if teleop.is_tracking:
                action_to_send = leader_action
            elif robot_connected:
                obs = robot.get_observation()
                action_to_send = {f"{name}.pos": float(obs[f"{name}.pos"]) for name in motor_names}
            else:
                action_to_send = leader_action

            if not args.dry_run and robot_connected:
                robot.send_action(action_to_send)

            # Per-second heartbeat.
            if _frame % FPS == 0:
                pretty = "  ".join(f"{n}={action_to_send[f'{n}.pos']:+.2f}" for n in motor_names)
                tag = "" if teleop.is_tracking else "  [leader stale -> holding]"
                print(f"[t={_frame // FPS}s] {pretty}{tag}")

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        if plugin_proc is not None:
            plugin_proc.terminate()
            try:
                plugin_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                plugin_proc.kill()
        teleop.disconnect()
        if robot_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
