# Task1 Real96 Leader–Follower collection

The active future collection identity is
`task1_picklift_real96_collection_v1`, implemented by
`picklift_continuous_batch_v4_real96_attempt_ledger` and
`picklift_collection_v7_real96_attempt_ledger`. It references research-control
commit `73908355df1add52cd04753216c13f8b1c0b400a` without presenting that commit
as the collector code commit.

The frozen research collection plan is copied byte-for-byte under
`contracts/`; its SHA-256 is
`f8d9ab2de3e7f6915dacafc2b8f70cc523373154b2df6b3abc8536fcfb623ef7`.
`real96_plan.py` reproduces all 96 plan items deterministically. Sessions 1 and
2 have independently transferred canonical compact JSON identities. Session 2
is 14,180 bytes, hashes to
`3cb86c9c176828405cc1cc838a119b7f4bd848a7d28f612d1624844342da0c37`,
and has sequence hash
`c81826685dc906e6bbf9d160e43fcb3986c146ecdc5ec020e73eb7c36ad05b98`.

## Operator lifecycle

Before CONNECT and every START:

1. Place the red cube at the exact displayed X/Y and nominal yaw (0° or 45°).
2. Make sure both Leader and Follower grippers start open.
3. Bring both arms to a similar ready area; no precise numeric pose or
   blocking difference threshold is required.
4. Check the canonical front image, remove hands/body from view, then START.

END stops dataset writes and live absolute Leader-to-Follower following
continues for reset. Reset motion is not part of the attempt. Each START
creates a new unique `attempt_id`. FAILURE or DISCARD retains that complete raw
Dataset v3 episode and its original outcome, links the next attempt to it, and
retries the same frozen `plan_item_id`. Only SUCCESS advances the plan.

The session raw dataset therefore contains every started attempt. The session
manifest's `accepted_dataset_episode_indices` is the deterministic accepted
success view: exactly one SUCCESS per frozen plan item. Failures are never
silently deleted, relabelled, or admitted to Real48/Real96 accepted subsets.

## Frozen data contract

- 20 FPS, 30 seconds maximum, 50 Hz control.
- `observation.state`: float32[6] Follower state read before action.
- `action`: float32[6] actual target returned by the official Follower send.
- joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
  wrist_roll, gripper.
- canonical front: uint8 RGB 640×480 using
  `icspring_front_crop_1280x960_to_640x480_v1`; no wrist camera.
- direct calibrated absolute Leader-to-Follower mapping, no relative rebase,
  custom clip, threshold gate, or max-relative limit.
- manual SUCCESS requires bilateral unsupported grasp, lift strictly greater
  than 5 cm, held continuously for 25 control steps (0.5 seconds at 50 Hz).

Leader–Follower provenance explicitly marks Quest-only `raw_human_target`,
`reachable_target`, and Grip-button state as
`not_applicable_not_fabricated`. The normal six-dimensional Follower gripper
state/action remains recorded.

## Offline preparation

Generate a new machine-local config without opening devices (use `--session 2`
and the corresponding `s02` paths for Session 2):

```bash
uv run python -m examples.picklift_v3.prepare_real96_session \
  --session 1 \
  --operator-id operator_01 \
  --device-config ~/.config/lerobot/picklift-practice.json \
  --dataset-root /home/ubuntu24/Teleop/artifacts/task1_picklift_real96_s01_raw_attempts_20260802 \
  --output ~/.config/lerobot/task1-picklift-real96-s01-20260802.json
```

The generated config deliberately leaves `powered_real_run_ack` empty. It is
filled only after the operator completes the powered safety check immediately
before collection.
