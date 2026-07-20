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
- `base_controller.py` — the controller seam (protocol, stub, safety
  wrapper, robot-backed controller + frame math).
- `voxel_map.py` — 5 cm sparse-hash `VoxelMap`: count-weighted geometry,
  free-space `carve` (dynamic updates), per-voxel feature + `query`. No
  point-cloud retention.
- `occupancy.py` — 3-class top-down grid + A* (no corner-cutting) +
  obstacle inflation + frontier extraction.
- `value_map.py` — DynaMem §3.4 recency (V_T) + similarity (V_S)
  exploration scoring.
- `features.py` — `SiglipFeatureExtractor` (lazy transformers) +
  `FeatureExtractor` protocol + `BasisVectorFeatureExtractor` stand-in.
- `geometry.py` — `GeometryRunner` protocol + `LingBotMapRunner` (lazy) +
  `FakeGeometryRunner`; `align_trajectory_to_odometry` (Umeyama) anchors
  the monocular scale to sport-mode odometry.
- `pipeline.py` — viz-free `integrate_keyframe` (carve → add) +
  feature upsampling.
- `skills.py` / `agent.py` — `SpatialSkills` (locate/goto/explore) +
  `DeterministicAgent` + regex parser.
- `sim.py` — self-contained synthetic scenes for model-free dry-runs.
- `dog_cli.py` — the `dog-nav` REPL (the deliverable).

Everything is model/hardware-free-testable (191 tests across the branch).
The one thing that needs the real dog + GPU models is `--live`.

## Running

```bash
# Synthetic scene, no robot/camera/models:
python -m lerobot.navigation.dog_cli --dry-run
python -m lerobot.navigation.dog_cli --dry-run --command "go to the couch"

# On a real Unitree Go2 (DDS + LingBot-Map + SigLIP2 on the GPU host):
python -m lerobot.navigation.dog_cli --live --network-interface enp2s0 --device cuda
```

Idle (no prompt) ⇒ autonomous exploration; a typed object name ⇒ navigate
to it, exploring to find it if it isn't mapped yet.

## Target platform

Unitree Go2 EDU, no companion computer: the workstation (single RTX 5090)
talks DDS straight to the dog; geometry is monocular LingBot-Map from the
built-in front camera, scale-anchored to sport-mode odometry; the map is
5 cm voxels. See [`robots/unitree_go2`](../robots/unitree_go2).

## Not yet ported (optional enhancement)

`SegmentVoxelMap` (object-centric per-segment features via SAM 2) is a
storage/precision optimization over the plain per-voxel features used
here; the locate/goto/explore stack is fully functional without it.
