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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dog-nav", description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run against a synthetic scene (no robot/camera/models). "
        "Currently the only supported mode until the live geometry pipeline lands.",
    )
    ap.add_argument("--command", default=None, help="Run a single command non-interactively, then exit.")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)-7s %(name)s: %(message)s"
    )

    if not args.dry_run:
        raise SystemExit(
            "Live mode needs the geometry pipeline (LingBot-Map + segment map), which is not "
            "wired yet. Run with --dry-run for now."
        )

    controller = _build_dry_run()
    if args.command is not None:
        result = controller.handle_prompt(args.command)
        controller.stop()
        return 0 if result.fully_successful else 1
    return run_repl(controller)


if __name__ == "__main__":
    sys.exit(main())
