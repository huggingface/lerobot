# OpenArm — Episode Replay in Simulation

Replay a recorded bimanual-[OpenArm](https://openarm.dev) episode into an mp4 by driving the
official OpenArm MuJoCo model from a LeRobot dataset's recorded joint states.

By default the replay goes **through end-effector (Cartesian) space**: for every frame and
every arm it runs forward kinematics (recorded joints → EE pose) and then inverse kinematics
(EE pose → joints), and drives the simulator with the IK-recovered joints. This exercises the
exact `RobotKinematics` solver that `OpenArmFollower.make_kinematics()` builds (see the
[OpenArm docs](../../docs/source/openarm.mdx)), so the video is a visual sanity check of the
end-effector kinematics — not just of the raw recording. It also prints the FK→IK round-trip
error (mean joint error in degrees and mean EE-position error in mm). Pass `--joint-space` to
bypass kinematics and replay the raw recorded joints directly.

## Model provenance

Everything is pulled from Enactic's official, Apache-2.0 OpenArm repositories — nothing is
vendored into LeRobot:

| Asset                                                      | Source                                                                          | License    |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------- |
| MuJoCo MJCF (rendering)                                    | [`enactic/openarm_mujoco`](https://github.com/enactic/openarm_mujoco)           | Apache-2.0 |
| URDF / xacro (for `RobotKinematics`, the FK/IK round-trip) | [`enactic/openarm_description`](https://github.com/enactic/openarm_description) | Apache-2.0 |

End-effector kinematics use [`placo`](https://github.com/Rhoban/placo) under the hood (install
LeRobot with the `placo-dep` extra). The script auto-locates the URDF via `--urdf` →
`$OPENARM_URDF`; set the tip link with `--ee-frame` → `$OPENARM_EE_FRAME`.

Use the **v1** MuJoCo revision (`v1/openarm_bimanual.xml`). v2 is a different wrist hardware
revision (DM3507) and will look sign-flipped when replaying v1 recordings.

## Setup

```bash
# LeRobot in your env (see https://huggingface.co/docs/lerobot/installation)
# Plus the sim/replay deps:
pip install mujoco av pandas

# Get the OpenArm MuJoCo model (either works):
pip install openarm-mujoco                                   # installs models under <prefix>/share/openarm_mujoco/
# or
git clone https://github.com/enactic/openarm_mujoco.git      # then pass --mjcf .../v1/openarm_bimanual.xml
```

The script auto-locates the model in this order: `--mjcf` arg → `$OPENARM_MJCF` →
`<sys.prefix>/share/openarm_mujoco/v1/openarm_bimanual.xml`.

## Dataset layout

`observation.state` must be the 16-D bimanual vector (degrees):

```
right_joint_1..7, right_gripper, left_joint_1..7, left_gripper
```

Only the 14 arm joints affect the rendered pose; the two gripper scalars drive the fingers.

## Run

Headless rendering needs `MUJOCO_GL=egl`, and MuJoCo's GL libs on `LD_LIBRARY_PATH`
(in conda: `$CONDA_PREFIX/lib`).

```bash
# Replay episode 1 through end-effector kinematics (default)
LD_LIBRARY_PATH=$CONDA_PREFIX/lib MUJOCO_GL=egl \
python -m examples.openarm.render_episode \
    --dataset data/folding_src_meta \
    --episode 1 \
    --urdf /path/to/openarm.urdf \
    --ee-frame openarm_finger_tip_link \
    --out openarm_ep1.mp4

# Bypass kinematics and replay the raw recorded joints directly
LD_LIBRARY_PATH=$CONDA_PREFIX/lib MUJOCO_GL=egl \
python -m examples.openarm.render_episode \
    --dataset data/folding_src_meta --episode 1 --joint-space --out openarm_ep1_raw.mp4

# No dataset handy? Smoke-test with a synthetic wave (add --urdf to also exercise the kinematics):
LD_LIBRARY_PATH=$CONDA_PREFIX/lib MUJOCO_GL=egl \
python -m examples.openarm.render_episode --demo --joint-space --out openarm_demo.mp4
```

Useful flags: `--fps` (default 30), `--width` / `--height` (default 960×720), `--mjcf` to point
at an explicit model file, `--urdf` / `--ee-frame` for the kinematics, `--joint-space` to skip it.

## Troubleshooting

| Symptom                                      | Fix                                                                    |
| -------------------------------------------- | ---------------------------------------------------------------------- |
| `Could not find the OpenArm v1 MJCF`         | Pass `--mjcf`, set `$OPENARM_MJCF`, or install/clone `openarm_mujoco`. |
| `End-effector replay needs the OpenArm URDF` | Pass `--urdf` / set `$OPENARM_URDF`, or use `--joint-space`.           |
| Large FK→IK round-trip error reported        | Wrong `--ee-frame` link name, or the URDF doesn't match the recording. |
| `libEGL`/`GLEW` / blank window errors        | Ensure `MUJOCO_GL=egl` and `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.        |
| Wrists look mirrored / flipped               | You are on the v2 model; switch to **v1**.                             |
| `KeyError: 'observation.state'`              | Dataset isn't in the expected 16-D bimanual layout.                    |
