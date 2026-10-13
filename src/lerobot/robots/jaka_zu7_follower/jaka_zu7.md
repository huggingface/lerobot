# JAKA Zu 7 Follower

This driver exposes the six JAKA joints as a LeRobot follower. Its joint keys match the six arm joints in the customized SO-ARM101 leader in this repository: `shoulder_pan`, `heavy_joint`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, and `wrist_roll`. The leader's `gripper.pos` action is ignored; no JAKA gripper is configured.

The robot features use degrees to match the SO-ARM101 leader. The `jkrc` Python SDK returns joint positions in radians and accepts radians for `servo_j_extend`; the driver converts feedback and commands at the SDK boundary. This is the Python SDK interface, not JAKA's lower-level TCP JSON protocol, whose servo joint values use degrees. The driver filters each target with configured joint velocity and acceleration limits, then sends an absolute `servo_j_extend` target. The default `servo_step_num=2` spans two 8 ms controller cycles, which is close to the default 60 Hz LeRobot loop.

`joint_signs` can reverse individual axes using `-1`, and `joint_offsets_deg` can align the JAKA zero positions with the SO-ARM101 leader. Both settings contain six values in JAKA joint order. Defaults are all `1` and `0`; measure signs and offsets on the real setup before higher-speed use.

## Install the vendor SDK on Ubuntu

Use the vendor's Linux x86_64 Python SDK from JAKA SDK 2.1.14. Keep `jkrc.so` and `libjakaAPI.so` together in the SDK's `Linux/python3/x86_64-linux-gnu` directory. This repository does not redistribute those vendor binaries.

```bash
export JAKA_SDK_DIR="$HOME/jaka_sdk/Linux/python3/x86_64-linux-gnu"
export PYTHONPATH="$JAKA_SDK_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$JAKA_SDK_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import jkrc; print("JAKA SDK import OK")'
```

Run the import check with the same Python 3.12 environment used for LeRobot. The supplied binary is x86_64, but its CPython ABI compatibility with Python 3.12 must be confirmed on the target Ubuntu installation.

Install this LeRobot checkout and hardware dependencies in the same environment using the repository's normal setup instructions. Confirm the JAKA controller firmware is supported by the installed SDK; this driver has not been tested against a physical Zu 7 in this development environment.

## Connect and teleoperate

First place the robot in a clear workspace and put the SO-ARM101 leader and JAKA in corresponding safe poses. By default, the driver logs in and enables JAKA servo mode but does not power on or enable the robot. Prepare those states at the controller before connecting. `auto_power_on` and `auto_enable_robot` are opt-in config switches.

```bash
lerobot-teleoperate \
  --robot.type=jaka_zu7_follower \
  --robot.ip_address=192.168.1.10 \
  --robot.servo_step_num=2 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --fps=60
```

For the first motion test, use a low `--robot.max_joint_velocity_deg_s` and `--robot.max_joint_acceleration_deg_s2`, keep the emergency stop accessible, and verify each leader joint moves the intended JAKA axis and direction. The mapping assumes both arms' six joints are in the listed order; the sign and zero offset can be configured, but mechanical range matching is not inferred automatically. Increase the limits only after checking the mechanics, payload, workspace, and controller settings.

If the control loop frequency changes from 60 Hz, choose `servo_step_num` so `servo_step_num * 8 ms` is close to the action interval. The JAKA Python SDK and controller are responsible for their own network timing; this driver has only been unit-tested with a mocked SDK.
