# `lerobot.navigation` — spatial-memory navigation

Online spatio-semantic mapping (DynaMem-style), A* planning, obstacle
avoidance and open-vocabulary goto/explore for LeRobot mobile bases.
Ported from the dyna360 research stack; the physical robot layer lives in
`lerobot.robots` (e.g. [`unitree_go2`](../robots/unitree_go2)).

## Idea

Drive any LeRobot `Robot` on the standard REP-103 mobile-base contract —
body-velocity actions `x.vel`/`y.vel`/`theta.vel` and planar odometry
`x.pos`/`y.pos`/`theta.pos` — from a spatial memory that is built and
updated online from the robot's camera. With no prompt the base explores
autonomously; given a text prompt it queries the map and navigates to the
matching object, or explores to find it if it isn't there (or has moved).

## Architecture

The navigation layer talks to hardware only through LeRobot's own `Robot`
interface, so it is robot-agnostic and carries no SDK dependency.

```
BaseController (protocol)         world-frame move()/pose() seam
├── StubBaseController            kinematic integrator (sim, tests)
├── RobotBaseController           wraps any Robot; world<->body +
│                                 odometry<->world frame math
└── SafeBaseController            velocity clamp, occupancy gate,
                                  keyframe watchdog, e-stop latch
```

World frame is OpenCV (x right, y down, z forward); the base moves in the
XZ plane. `RobotBaseController.feed_observation(obs)` updates pose from
the observation the navigation loop already fetches (closed-loop
odometry), avoiding an extra camera read; absent odometry it integrates
open-loop so sim matches hardware.

## Status (branch `feat/unitree-go2`)

Implemented:
- `base_controller.py` — the controller seam above. SDK/torch-free;
  tests in `tests/navigation/test_base_controller.py`.

Planned (porting from dyna360, everything lands here — dyna360 is a
source to copy from, not a runtime dependency):
- `geometry.py` — LingBot-Map streaming reconstruction runner
  (`{points, local_points, conf, camera_poses}`), scale-anchored to
  odometry. (Pi3X is not carried over.)
- `voxel_map.py` — 5 cm `SegmentVoxelMap`, no point-cloud retention;
  one SigLIP2 feature per segment.
- `occupancy.py` — occupancy grid + A* + frontiers; `value_map.py`.
- `features.py` (SigLIP2), `segmenter.py` (SAM2) — run offboard.
- `agent.py`, `skills.py`, `dog_cli.py` — the interactive explore/query
  REPL (the deliverable).

## Target platform

Unitree Go2 EDU, no companion computer: the workstation (single RTX 5090)
talks DDS straight to the dog; geometry is monocular LingBot-Map from the
built-in front camera, scale-anchored to sport-mode odometry; the map is
5 cm voxels. See [`robots/unitree_go2`](../robots/unitree_go2).
