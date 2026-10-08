# OpenArm Gripette Manual Policy Start

## Goal

Prepare the OpenArm at a manually chosen policy start pose without moving the arm between preparation and policy execution.

## Design

`run_grabette_policy.py` performs preparation after connecting the robot and before its existing final confirmation:

1. Disable torque on the seven OpenArm motors while leaving the Gripette connected.
2. Ask the operator to support and manually place the arm at the demonstrated policy start pose.
3. On ENTER, read the seven measured joint positions while torque remains disabled.
4. Re-enable arm torque and immediately command those measured positions so the arm holds its current pose.
5. Read the held pose again, display the seven joint positions and the corresponding `oak_l` Cartesian position, and use it as the initial FK/IK state.
6. Require the existing second ENTER before starting policy execution.

The script must not automatically move to a fixed home pose. The policy is camera-conditioned and uses relative Cartesian actions, so matching the demonstrated scene view matters more than a universal joint configuration.

## Safety and errors

- Prompts explicitly tell the operator to support the arm before torque is disabled.
- The arm torque transition is two-pass, matching the existing OpenArm hardware practice for fire-and-forget CAN commands.
- If preparation fails, normal cleanup disconnects the robot and disables arm and Gripette torque.
- `--dry_run` uses the same preparation path so it validates the actual chosen start pose.

## Testing

- Unit-test the torque-off, measured-position capture, torque-on, and hold-command sequence with a mocked robot.
- Keep the existing offline policy/IK check and plugin tests passing.
- Hardware verification remains operator-gated: manually position, confirm the reported joints and `oak_l` position, then run first with `--dry_run`.
