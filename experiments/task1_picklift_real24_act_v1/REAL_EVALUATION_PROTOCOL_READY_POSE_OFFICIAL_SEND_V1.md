# Task1 real evaluation: ready-pose and official-send protocol v1

Status: software implemented and tested without hardware. No real trial is
authorized by this document.

## What changed

1. Every trial moves directly to the same frozen Real-24 ready pose before the
   policy window:

   ```text
   [7.4285712242126465, -98.32967376708984,
    45.010990142822266, 92.21977996826172,
    1.8461538553237915, 19.765840530395508]
   ```

   The source is `task1_real24_ready_pose_reset_v1`, Real-24 episode 13 frame
   0, state SHA-256
   `ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56`.
   The current arm pose is recorded but is never reused as a return target.
   Post-trial return targets the same frozen ready pose.

2. ACT is reset only after the ready pose is observed. Tick 0 is then acquired,
   timed, and inferred. Evidence records the ready-pose requested state,
   observed state, delta, movement commands, and tick-0 state.

3. Formal deployment fixes `max_relative_target=None`. Policy output goes
   directly to `SO101Follower.send_action`; the real-evaluation runner applies
   no custom absolute calibration clamp and no custom relative step limiter.
   Every tick retains `raw_action`, identical `requested_action`, and the
   `sent_action` returned by the official robot API.

4. The loop uses LeRobot-style per-tick pacing:

   ```text
   run one tick
   sleep max(0.05 s - tick compute time, 0)
   ```

   It does not advance an absolute `next_tick` deadline. A slow first inference
   therefore cannot cause a burst of actions to catch up.

## Upstream behavior, not a runner transform

- `SO101Follower.send_action` applies relative clipping only when
  `max_relative_target` is not `None`; this profile fixes it to `None`.
- `FeetechMotorsBus._unnormalize` converts degree-mode body joints to raw motor
  units without calibration-range clamping.
- The upstream gripper `RANGE_0_100` conversion bounds the value to 0–100 before
  raw-unit conversion. This is LeRobot motor-bus behavior, not a custom runner
  clamp.
- `sent_action` is the normalized command returned by
  `SO101Follower.send_action`, before internal raw-register conversion.

## What did not change

- Fixed 100k checkpoint and model SHA-256
  `ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb`.
- Front-only canonical RGB `640x480`, `observation.state[6]`, and `action[6]`.
- 20 Hz nominal control over a 30-second wall-time window (600 ticks nominal;
  actual ticks are recorded and never backfilled).
- ACT `chunk_size=67`, `n_action_steps=67`, and the existing 12-cell order.
- Outcome definition and manual annotation categories.
- Historical evaluation evidence remains immutable.

## Versioned identities

- Profile:
  `real_evaluation_profile_ready_pose_official_send_v1.json`
- Profile SHA-256:
  `01e50c86adc1a03f2bb1675469502e6969e2cc6a4a51dc8db0b75e6049b5d4c5`
- Plan:
  `evaluation_plan_ready_pose_official_send_v1.json`
- Plan SHA-256:
  `a067615e37ae8b64b57d663e85c1b49be61ef881daa58c3d9159506caf9be048`

## Hardware gate

Software validation does not authorize a rollout. A future hardware run still
requires explicit user authorization and现场确认 for the first cell. Until
then, do not open the Follower serial port or camera, enable torque, or start
the evaluation command.
