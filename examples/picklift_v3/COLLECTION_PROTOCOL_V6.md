# PickLift collection protocol v6

`picklift_collection_v6_absolute_camera_sequence` is the future real
Leader-to-Follower collection contract. It does not rewrite
`picklift_collection_v5`, the frozen s03 dataset, or any historical manifest.
The task distribution remains `picklift_spawn_v5`: the same twelve 5×5 cm
cells, natural unmeasured 0–90° yaw, front-only aligned camera, and 20 FPS
training view.

## Supervision contract

- Both SO-101 devices use their established LeRobot degree calibration and
  joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
  `wrist_roll`, `gripper`.
- Leader values are sent as absolute Follower targets. The collector applies
  no relative rebase, custom clip, max-relative limit, threshold gate, ramp,
  or other hidden transform. `alignment_mode=direct_absolute` and
  `max_relative_target=null` are mandatory.
- For every control tick, the Follower state is read before sending the next
  target. `observation.state` records that float32[6] pre-action state.
- `action` records the float32[6] target returned by
  `SOFollower.send_action`, i.e. the target actually sent by the official
  robot implementation, not merely the Leader request.

## Ready and episode lifecycle

No precise numeric ready pose is required and no pose-difference threshold
blocks START. Before the first CONNECT/START, the operator brings both arms
into a similar ready area and checks the front view and gripper opening.

Immediately after END and result selection, live absolute Leader-to-Follower
control continues without dataset writes. The operator releases/resets the
cube, moves both arms back into a similar ready area, checks the view and
gripper, and then presses START NEXT. Successful motion is ended and saved
before any reset motion. FAILURE and DISCARD retry the same cell; only SUCCESS
advances the balanced spawn plan and enters the training dataset.

## Camera evidence

The immutable camera profile remains
`icspring_front_crop_1280x960_to_640x480_v1`: 30 FPS acquisition, canonical
640×480 RGB, sampled at 20 FPS, front only, with no wrist field.

Every attempt stores `picklift_camera_sequence_evidence_v1`:

- the ordered collector control-sample sequence used for each recorded image;
- monotonic timestamp offsets for those samples;
- SHA-256 of every canonical RGB frame;
- unique-content, repeated-content, and consecutive-duplicate frame counts;
- timestamp monotonicity and anomaly counts.

The collector sequence is not described as a hardware-camera sequence.
Exact RGB hashes provide deterministic repeated-image detection without
inventing unsupported camera metadata.
