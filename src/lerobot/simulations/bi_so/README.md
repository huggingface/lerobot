# Bimanual SO MuJoCo Simulation

This folder documents the isolated bimanual SO-arm simulation work added without modifying the main robot or teleoperator codepaths.

The goal of this work is:
- provide a `bi_so_follower_simulated` robot implementation that behaves like a bimanual SO follower at the API level
- keep compatibility with the existing `bi_so_leader` teleoperator so real leader arms can drive the simulation
- keep simulation-related additions grouped under new files and folders for easier continuation

## What Was Added

New robot package:
- `src/lerobot/robots/bi_so_follower_simulated/`

New simulation wrapper files:
- `src/lerobot/simulations/bi_so/`

New local MuJoCo XML assets:
- `src/lerobot/robots/bi_so_follower_simulated/mujoco/lerobot_pick_place_cube.xml`
- `src/lerobot/robots/bi_so_follower_simulated/mujoco/so_arm100.xml`
- `src/lerobot/robots/bi_so_follower_simulated/mujoco/so_arm100_right.xml`

New test file:
- `tests/robots/test_bi_so_follower_simulated.py`

## Current Architecture

The current control chain is:

`bi_so_leader` teleoperator
-> existing LeRobot teleoperation pipeline
-> `bi_so_follower_simulated` robot
-> external MuJoCo bridge
-> MuJoCo scene

This means the simulation does not read leader-arm hardware directly itself. It relies on the existing teleoperator stack:
- `lerobot.teleoperators.bi_so_leader.BiSOLeader`
- `lerobot.teleoperators.so_leader.SOLeader`

That is intentional. It keeps the simulation compatible with the same action keys used by the real bimanual follower.

## How The Teleoperation Mapping Works

`bi_so_leader` returns actions with keys like:
- `left_shoulder_pan.pos`
- `left_gripper.pos`
- `right_shoulder_pan.pos`
- `right_gripper.pos`

`bi_so_follower_simulated` accepts exactly those keys in `send_action()`.

Internally:
- the first five joints are treated as angle values compatible with the SO leader/follower convention
- the gripper is exposed in SO-style `0..100` units
- the gripper is converted internally to the MuJoCo actuator control range

So the intended real-hardware teleop path is already:
- leader arms connect through `bi_so_leader`
- their actions flow through the standard teleop loop
- the simulated robot receives those actions and writes them into the MuJoCo buses

## Files And Responsibilities

### `src/lerobot/robots/bi_so_follower_simulated/config_bi_so_follower_simulated.py`

Defines the robot config:
- `sim_root`
- `bridge_path`
- `xml_path`
- `bridge_factory_name`
- `robot_dofs`
- `render_size`
- `camera_names`
- `realtime`
- `slowmo`
- `launch_viewer`
- `max_relative_target`

### `src/lerobot/robots/bi_so_follower_simulated/bi_so_follower_simulated.py`

Implements the simulated robot wrapper.

Current features:
- exposes bimanual action keys matching the real bimanual SO follower style
- exposes observations using the same left/right prefixed motor naming
- supports optional rendered camera observations
- converts gripper between SO teleop units and MuJoCo actuator units
- supports `max_relative_target` clipping using the shared robot safety helper
- loads XML from the local package folder first
- loads the external bridge module dynamically

### `src/lerobot/robots/bi_so_follower_simulated/mujoco/`

Contains local XML scene definitions so the simulation has an internal default scene and does not depend on external XML files.

The scene currently includes:
- two simple SO-style arms
- a floor
- a movable cube body
- a target region
- front and top cameras
- end-effector sites (`ee_site`, `ee_site_r`)

These XMLs are simplified and self-contained. They do not depend on external mesh assets.

### `src/lerobot/simulations/bi_so/teleoperate_bi_so_follower_simulated.py`

This is a wrapper entrypoint.

Why it exists:
- the main CLI files were intentionally left untouched
- this wrapper imports the new simulated robot module first
- after that it calls the standard `lerobot_teleoperate` entrypoint

That allows the existing teleop system to discover the new robot through the registry/fallback machinery without changing the core CLI modules.

## XML Resolution Behavior

If `xml_path` is not passed, the robot currently searches in this order:

1. local package simulation folder:
   `src/lerobot/robots/bi_so_follower_simulated/mujoco/lerobot_pick_place_cube.xml`
2. any user-provided `sim_root`
3. repo-level `sim/`
4. sibling `AOSH/lerobot/sim/`
5. bridge folder fallback

This means the simulation now has a local default XML scene.

## Bridge Dependency

Important: the XML files are now local, but the MuJoCo bridge logic is still external unless you explicitly provide a local bridge file.

The current robot still expects:
- `task2_motors_bridge.py`
- `mujoco_task2.py`

The wrapper tries to locate those from:
- `bridge_path` if given
- `sim_root` if given
- `repo_root/sim`
- `repo_root/../AOSH/lerobot/sim`

So at the moment:
- XML is bundled locally
- bridge/backend Python is still expected from an external sim folder unless added later

## Recommended Local Run Approach

Using Conda is a reasonable choice for local testing.

Typical requirements are:
- Python environment with project dependencies installed
- `mujoco` installed
- access to the external bridge files unless they are later copied locally
- serial access to the real SO leader arms if using hardware teleoperation

Example command:

```shell
python -m lerobot.simulations.bi_so.teleoperate_bi_so_follower_simulated \
  --robot.type=bi_so_follower_simulated \
  --robot.sim_root=C:/Users/Ninja/AOSH/lerobot/sim \
  --robot.launch_viewer=true \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=COM5 \
  --teleop.right_arm_config.port=COM6 \
  --teleop.id=bimanual_leader \
  --fps=60
```

Optional rendered cameras:

```shell
--robot.camera_names="['front','top']" --robot.render_size="(480,640)"
```

## Project Status

What is already in place:
- new isolated simulated robot package
- left/right action compatibility with `bi_so_leader`
- local default MuJoCo XML files
- wrapper entrypoint for teleoperation
- gripper conversion logic
- optional offscreen camera exposure
- basic test coverage added for wrapper behavior

What is not yet fully localized:
- `task2_motors_bridge.py`
- `mujoco_task2.py`

What is not yet verified in this environment:
- end-to-end runtime test
- MuJoCo viewer launch
- real serial connection to the leader arms
- pytest execution

The last point is environmental: Python execution in this workspace was not usable during implementation, so runtime verification could not be completed here.

## Continuation Notes

If someone continues this work, the most useful next steps are:

1. Move or recreate the bridge/backend Python files into the local simulation folder so the package becomes fully self-contained.
2. Run a full Conda-based end-to-end test with:
   - MuJoCo installed
   - the new wrapper entrypoint
   - real `bi_so_leader` hardware
3. Verify that joint directions match the real teleoperator intuitively in the viewer.
4. Tune the local XML geometry and actuator gains if motion feels too rough or unrealistic.
5. Add more tests once a working Python runtime is available.

## Notes For Future Developers

- This work intentionally avoided modifying the existing main teleop files.
- The simulation is integrated through the existing teleoperator pipeline, not through direct controller reads inside the robot class.
- The action interface was kept aligned with `bi_so_follower` and `bi_so_leader` on purpose.
- The local XMLs are functional defaults, not final high-fidelity mechanical models.
