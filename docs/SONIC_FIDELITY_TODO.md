# SONIC fidelity TODO — gaps vs gear_sonic / C++ deploy

Remaining differences between this lerobot SONIC port and the original gear_sonic +
`gear_sonic_deploy/g1_deploy_onnx_ref` reference. All three deployed encoder modes exist
(0 locomotion, 1 three-point, 2 full-body SMPL); these are faithfulness gaps in *how*
some pieces behave.

## 1. 3-point calibration & source (mode 1)
The original derives the 3 points from **full-body joints** and calibrates with
`ThreePointPose` against **live measured robot joints** (`reset_with_measured_q`), plus
neck→waist coupling. Our device path (head + controllers, fixed zero-q neutral) is an
approximation.
- Missing: measured-q recalibration + waist coupling.
- Note: the body-source `compute_3point` path (needs PICO Motion Trackers) is already faithful.

## 2. Locomotion stick/speed mapping (#6)
gear_sonic's `PlannerLoop` uses a yaw-accumulator + **mode-dependent speed curves**
(slow `0.1 + 0.5·mag`, run `1.5 + 3·mag`) with a facing-rotated movement vector. We reuse
the keyboard-parity `apply_joystick_axes` (no speed-by-mode curve).
- Also: PICO stick sign conventions untested (may need axis flips in `pico_publisher`).
- Self-contained and sim-testable.

## 3. Mode-2 root motion
`enable_smpl_root=False` by default; the original feeds the SMPL root orientation into the
anchor. Faithful only when enabled **and** the 30→50 Hz root trajectory is smoothed/
rate-matched (currently causes base-acceleration spikes → NaN QACC).

## 4. Start/stop handshake
C++ has `WAIT_FOR_CONTROL` (operator "start" gate) and a "stop" halt. We start policy
immediately after the startup ramp — no explicit arm/e-stop state machine.

## 5. Startup ramp target
C++ `InitControl` ramps to `default_angles`; we ramp to the policy's live commanded pose.
Deliberate (per request) — only a gap if exact C++ parity is wanted.

## 6. Hands / grippers
Original maps trigger/grip → Dex3 hand joints in modes 1 & 2 (simple gripping: thumb open,
trigger>0.5 closes middle finger, run through a gripper IK solver). Our 29-DOF path has no
hand control.
- Hardware here: **OpenArm grippers** (not Dex3) — could be routed through the same
  trigger→open/close path, but needs the robot interface to expose/command the gripper
  actuators (they're separate from the 29-DOF body).

## Not gaps (settled)
- `vr_5point_index` — vestigial constant in `policy_parameters.hpp`, not a deployed encoder
  mode. Nothing to implement.
- Action scaling, kp/kd, IsaacLab↔MuJoCo remaps, encoder cadence (encode ⅕ / decode 1×),
  per-mode replan intervals (0.1/0.2/1.0), 8-frame cross-fade blend — all faithful.
- Graceful damped shutdown — intentional safety addition, keep.

## Suggested priority
- High (change how the robot moves under teleop): #1, #2.
- Medium: #3, #4, #6 (grippers).
- Low / intentional: #5.
