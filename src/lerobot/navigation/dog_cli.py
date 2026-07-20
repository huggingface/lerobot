#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""``dog-nav`` — interactive spatial-memory navigation REPL.

Behaviour:
  - **No prompt** (idle) → the base explores autonomously: value-map
    frontier selection, A* on the live occupancy map, obstacle-gated
    motion. The map grows/refreshes as it goes.
  - **Typed prompt** (e.g. ``find the couch``) → query the map; if a
    confident match exists, navigate to it; otherwise explore until it is
    found (or the budget is exhausted), then resume idle exploring.

A new prompt preempts the current goal. Ctrl-C latches an e-stop and
exits. ``--dry-run`` runs the whole loop against a synthetic scene with no
robot, camera, or models — the default until the live geometry pipeline
(LingBot-Map) is wired.

Run: ``python -m lerobot.navigation.dog_cli --dry-run`` and type object
names; empty line ⇒ one exploration step; ``quit`` ⇒ exit.
"""

from __future__ import annotations

import argparse
import logging
import select
import sys

from lerobot.navigation.agent import (
    AgentConfig,
    AgentResult,
    DeterministicAgent,
    HardcodedTaskParser,
)
from lerobot.navigation.skills import ExploreResult, SkillsConfig, SpatialSkills

LOG = logging.getLogger("dog-nav")


class DogController:
    """The behaviour loop over a :class:`SpatialSkills` toolset.

    Construct with a ready ``SpatialSkills`` (real robot or synthetic
    scene). :meth:`handle_prompt` runs a full locate/goto/explore task;
    :meth:`idle_tick` runs one autonomous exploration step. Both are
    plain calls, so the REPL and the tests share the same code.
    """

    def __init__(
        self,
        skills: SpatialSkills,
        agent: DeterministicAgent | None = None,
        parser: HardcodedTaskParser | None = None,
    ) -> None:
        self.skills = skills
        self.agent = agent or DeterministicAgent(skills)
        self.parser = parser or HardcodedTaskParser()

    def handle_prompt(self, text: str) -> AgentResult:
        """Query the map and navigate to the target (exploring if needed)."""
        LOG.info("prompt: %r", text)
        result = self.agent.execute_command(text, self.parser)
        for tr in result.target_results:
            if tr.reached:
                LOG.info("  reached %r at %s (conf %.3f)", tr.target, tr.final_xyz, tr.confidence)
            else:
                LOG.info("  did not reach %r: %s (conf %.3f)", tr.target, tr.reason, tr.confidence)
        return result

    def idle_tick(self) -> ExploreResult:
        """One autonomous exploration step: pick a frontier and drive to it."""
        ex = self.skills.explore(query=None)
        if ex.found_frontier and ex.target_xyz is not None:
            LOG.info("idle: exploring toward %s (value %.3f)", ex.target_xyz, ex.value)
            self.skills.goto(ex.target_xyz)
        else:
            LOG.debug("idle: no frontier to explore (%s)", ex.reason)
        return ex

    def stop(self) -> None:
        self.skills.base.stop()


def _build_dry_run() -> DogController:
    """Wire the controller against the synthetic kitchen scene."""
    from lerobot.navigation.base_controller import StubBaseController
    from lerobot.navigation.sim import kitchen_scene

    scene = kitchen_scene()
    base = StubBaseController()
    siglip = scene.feature_extractor()
    skills = SpatialSkills(
        scene.voxel_map,
        base,
        siglip,
        SkillsConfig(
            cell_size=0.2,
            obstacle_inflate_cells=0,
            goto_threshold=1.0,
            goto_max_steps=300,
            locate_threshold=0.5,
        ),
    )
    agent = DeterministicAgent(skills, AgentConfig(max_explore_iters=4))
    objs = ", ".join(o.name for o in scene.objects)
    LOG.info("dry-run kitchen scene ready — try one of: %s", objs)
    return DogController(skills, agent)


class LiveMapper:
    """One perceive→integrate step of live mapping on the robot.

    Each :meth:`tick` reads an observation (front camera + odometry),
    updates the base pose from odometry, runs the geometry model + feature
    extractor on the frame, and integrates the keyframe.

    Frame convention (important): the **odometry frame is the one world
    frame**. The geometry model supplies only relative camera-frame
    geometry (``local_points``/depth); those points are projected through
    the base's odometry pose, so the voxel map and the robot pose live in
    the same coordinates and ``goto`` drives to the right place. The
    model's own ``camera_poses`` (its internal monocular frame) are not
    used as the world frame. Constructed lazily — no SDK/model touched
    until the first tick.
    """

    def __init__(self, robot, base, geometry, siglip, voxel_map, pcfg=None) -> None:
        self.robot = robot
        self.base = base  # RobotBaseController (unwrapped) for feed_observation/pose
        self.safe = None  # optional SafeBaseController for the watchdog
        self.geometry = geometry
        self.siglip = siglip
        self.voxel_map = voxel_map
        self.pcfg = pcfg
        self._frame = 0

    def tick(self, t_sec: float) -> None:
        import numpy as np

        from lerobot.navigation.pipeline import (
            KeyframeContext,
            PipelineConfig,
            integrate_keyframe,
            local_points_to_world,
            upsample_features_to_view,
        )

        obs = self.robot.get_observation()
        self.base.feed_observation(obs)  # updates the odometry world pose
        pose = self.base.pose()  # camera-to-world in the odometry frame

        frame = obs.get("front")
        if frame is None:
            return
        views = np.asarray(frame)[None].astype(np.uint8)  # (1, H, W, 3)
        geo = self.geometry(views)
        h, w = frame.shape[:2]

        feat_map = None
        if self.siglip is not None:
            patches = self.siglip.encode_views(views)[0]  # (Hp, Wp, D)
            feat_map = upsample_features_to_view(patches, h, w)

        # World points come from the model's camera-frame geometry projected
        # through the odometry pose — NOT the model's own world frame.
        points_world = local_points_to_world(geo.local_points[0], pose)
        ctx = KeyframeContext(
            frame_idx=self._frame,
            t_sec=t_sec,
            rgb_uint8=views[0],
            points_world=points_world,
            local_points=geo.local_points[0],
            conf=geo.conf[0],
            pose=pose,
            feat_map=feat_map,
        )
        integrate_keyframe(self.voxel_map, ctx, self.pcfg or PipelineConfig())
        self._frame += 1
        if self.safe is not None:
            self.safe.feed_watchdog()


def _build_live(
    network_interface: str = "eth0",
    device: str = "cuda",
    camera_hfov_deg: float = 90.0,
    max_lin_speed: float = 0.4,
    max_yaw_rate: float = 0.8,
) -> tuple[DogController, LiveMapper]:
    """Wire the controller + live mapper against a real Unitree Go2.

    Nothing here touches the SDK or loads a model — construction is lazy;
    the DDS connection and model loads happen on first use.

    ``camera_hfov_deg`` sets the pinhole focal length used for free-space
    carving (``focal = W / (2·tan(HFOV/2))``). Calibrate it to the Go2
    front camera for correct carving; a wrong value only degrades dynamic
    removal, not the additive map. Speed caps are deliberately low for
    first bring-up.
    """
    import math

    from lerobot.navigation.base_controller import (
        RobotBaseController,
        RobotBaseControllerConfig,
        SafeBaseController,
    )
    from lerobot.navigation.features import SiglipFeatureExtractor
    from lerobot.navigation.geometry import LingBotMapRunner
    from lerobot.navigation.pipeline import PipelineConfig
    from lerobot.navigation.voxel_map import VoxelMap
    from lerobot.robots.unitree_go2 import UnitreeGo2, UnitreeGo2Config

    robot_cfg = UnitreeGo2Config(network_interface=network_interface)
    robot = UnitreeGo2(robot_cfg)
    inner = RobotBaseController(
        robot, RobotBaseControllerConfig(max_lin_speed=max_lin_speed, max_yaw_rate=max_yaw_rate)
    )
    safe = SafeBaseController(inner=inner, max_lin_speed=max_lin_speed, max_yaw_rate=max_yaw_rate)
    voxel_map = VoxelMap(voxel_size=0.05)
    siglip = SiglipFeatureExtractor(device=device)
    geometry = LingBotMapRunner(device=device)

    w = robot_cfg.front_camera_width
    focal_px = w / (2.0 * math.tan(math.radians(camera_hfov_deg) / 2.0))
    pcfg = PipelineConfig(focal_px=focal_px)

    skills = SpatialSkills(voxel_map, safe, siglip, SkillsConfig(cell_size=0.05))
    controller = DogController(skills, DeterministicAgent(skills, AgentConfig()))
    mapper = LiveMapper(robot, inner, geometry, siglip, voxel_map, pcfg=pcfg)
    mapper.safe = safe
    LOG.info(
        "live stack wired (iface=%s, device=%s, focal=%.1fpx, vmax=%.2f m/s) — connect the dog and run",
        network_interface,
        device,
        focal_px,
        max_lin_speed,
    )
    return controller, mapper


def _stdin_line_ready(timeout_s: float) -> bool:
    """True when a full line is available on stdin within ``timeout_s``.

    Uses ``select`` so idle ticks keep running while we wait for input.
    Falls back to blocking reads where ``select`` on stdin isn't supported
    (e.g. some Windows terminals).
    """
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        return bool(ready)
    except (OSError, ValueError):
        return True


def run_repl(controller: DogController, idle_period_s: float = 0.5) -> int:
    """Interactive loop: explore while idle, run a task on each typed line."""
    print("dog-nav ready. Type an object to find it, empty line to explore, 'quit' to exit.")
    try:
        while True:
            if _stdin_line_ready(idle_period_s):
                line = sys.stdin.readline()
                if not line:  # EOF
                    break
                text = line.strip()
                if text.lower() in {"quit", "exit"}:
                    break
                if text:
                    controller.handle_prompt(text)  # a new prompt preempts idle
                else:
                    controller.idle_tick()
            else:
                controller.idle_tick()
    except KeyboardInterrupt:
        LOG.warning("interrupted — stopping base")
    finally:
        controller.stop()
    return 0


def run_live_repl(
    controller: DogController,
    mapper: LiveMapper,
    idle_period_s: float = 0.2,
) -> int:
    """Live loop on the robot: map continuously, run tasks on typed lines.

    Each iteration integrates one keyframe (perceive → geometry → features →
    voxel map), then either handles a typed prompt or takes one exploration
    step. The DDS connection is opened here so ``--help`` stays model-free.
    """
    import time

    mapper.robot.connect()
    controller.skills.base.reset_watchdog()
    print("dog-nav (live). Type an object to find it, empty line to explore, 'quit' to exit.")
    t0 = time.monotonic()
    try:
        while True:
            mapper.tick(time.monotonic() - t0)
            if _stdin_line_ready(idle_period_s):
                line = sys.stdin.readline()
                if not line:
                    break
                text = line.strip()
                if text.lower() in {"quit", "exit"}:
                    break
                if text:
                    controller.handle_prompt(text)
                else:
                    controller.idle_tick()
            else:
                controller.idle_tick()
    except KeyboardInterrupt:
        LOG.warning("interrupted — stopping base")
    finally:
        controller.stop()
        mapper.robot.disconnect()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dog-nav", description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run against a synthetic scene (no robot/camera/models).",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Run on a real Unitree Go2 (DDS + LingBot-Map + SigLIP2 on the GPU host).",
    )
    ap.add_argument("--network-interface", default="eth0", help="Host interface wired to the dog.")
    ap.add_argument("--device", default="cuda", help="Torch device for the geometry/feature models.")
    ap.add_argument(
        "--camera-hfov-deg",
        type=float,
        default=90.0,
        help="Go2 front-camera horizontal FOV, for the carve focal length. Calibrate to your camera.",
    )
    ap.add_argument("--max-lin-speed", type=float, default=0.4, help="Body linear speed cap (m/s).")
    ap.add_argument("--max-yaw-rate", type=float, default=0.8, help="Yaw-rate cap (rad/s).")
    ap.add_argument("--command", default=None, help="Run a single command non-interactively, then exit.")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)-7s %(name)s: %(message)s"
    )

    if args.live:
        controller, mapper = _build_live(
            args.network_interface,
            args.device,
            camera_hfov_deg=args.camera_hfov_deg,
            max_lin_speed=args.max_lin_speed,
            max_yaw_rate=args.max_yaw_rate,
        )
        return run_live_repl(controller, mapper)

    if not args.dry_run:
        raise SystemExit("Choose a mode: --dry-run (synthetic scene) or --live (real Unitree Go2).")

    controller = _build_dry_run()
    if args.command is not None:
        result = controller.handle_prompt(args.command)
        controller.stop()
        return 0 if result.fully_successful else 1
    return run_repl(controller)


if __name__ == "__main__":
    sys.exit(main())
