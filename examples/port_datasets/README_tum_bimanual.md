# Port a bimanual TUM-style dataset to LeRobot v3

`port_tum_bimanual.py` converts synchronized bimanual recordings into LeRobot
Dataset v3. One complete `session_*` directory becomes one episode.

The example is intended for a specific, documented source contract. It does
not infer arbitrary directory layouts or column orders.

## Source layout

```text
raw_root/
└── session_000001/
    ├── relative_transforms_left_to_right.txt
    ├── left_hand_device_a/
    │   ├── Calibration/rgb_intrinsic.json
    │   ├── Clamp_Data/clamp_data_tum.txt
    │   ├── IMU/imu.txt
    │   ├── Merged_Trajectory/merged_trajectory.txt
    │   └── RGB_Images/
    │       ├── timestamps.csv
    │       └── video.mp4
    └── right_hand_device_a/
        └── ...
```

Each session must contain exactly one `left_hand_*` directory and one
`right_hand_*` directory. Camera dimensions must match across both hands and
all converted sessions.

The input must be an extracted directory. ZIP files are not read directly.

## Numeric files

TUM poses use:

```text
timestamp tx ty tz qx qy qz qw
```

This format is used for both hand trajectories and the left-to-right relative
pose. The relative-pose timestamps are stream-local: the converter aligns the
first relative-pose timestamp to the first left-hand trajectory timestamp and
preserves all relative time differences.

Gripper files use:

```text
timestamp value
```

IMU files use:

```text
timestamp gyro_x gyro_y gyro_z accel_x accel_y accel_z
```

Two parenthesized covariance tuples may appear after the gyroscope and
accelerometer triples. They are ignored.

RGB timestamps support either:

```csv
timestamp
0.0
0.5
```

or:

```csv
frame_index,seq,header_stamp
0,10,0.0
1,11,0.5
```

In the indexed form, `frame_index` must start at zero and remain contiguous.
There must be one timestamp for each decoded source frame used by the
conversion.

## Convert locally

```bash
python examples/port_datasets/port_tum_bimanual.py \
  --raw-dir /path/to/raw_root \
  --repo-id namespace/dataset_name \
  --root /path/to/output \
  --fps 60 \
  --task "bimanual hand manipulation"
```

Existing output is rejected unless `--overwrite` is present. Input and output
paths may not overlap. Use `--skip-invalid-session` to record and skip an
invalid session while retaining valid sessions.

To publish after successful local finalization:

```bash
python examples/port_datasets/port_tum_bimanual.py \
  --raw-dir /path/to/raw_root \
  --repo-id namespace/dataset_name \
  --root /path/to/output \
  --push-to-hub
```

Authentication is handled by the normal Hugging Face Hub configuration. The
script never accepts or stores a token.

## Synchronization

The converter intersects the time coverage of both videos, both hand poses,
both gripper streams, both IMU streams, and the relative-pose stream. It does
not extrapolate.

- RGB uses the nearest source frame; an exact tie selects the earlier frame.
- Translation, gripper, gyroscope, and acceleration use linear interpolation.
- Quaternions use normalized, shortest-path SLERP.
- Output timestamps form a fixed-rate timeline at `--fps`.

## Output schema

- `observation.images.left_hand`: left RGB video.
- `observation.images.right_hand`: right RGB video.
- `observation.state`: `float32[16]`.
- `observation.imu`: `float32[12]`.
- `observation.relative_pose`: `float32[7]`.
- `action`: `float32[16]`.
- `source_timestamp`: `float64[1]`.

State and action order:

```text
left_tx, left_ty, left_tz,
left_qx, left_qy, left_qz, left_qw,
left_gripper,
right_tx, right_ty, right_tz,
right_qx, right_qy, right_qz, right_qw,
right_gripper
```

IMU order:

```text
left_gyro_x, left_gyro_y, left_gyro_z,
left_accel_x, left_accel_y, left_accel_z,
right_gyro_x, right_gyro_y, right_gyro_z,
right_accel_x, right_accel_y, right_accel_z
```

Actions follow the next-state convention:

```text
action[t] = observation.state[t + 1]
action[last] = observation.state[last]
```

Gripper values retain the source unit and range. The example does not
normalize or convert units.

## Data review before upload

Before using `--push-to-hub`, verify that you have the rights to publish every
recording and that frames, task text, paths, and metadata do not contain
personal, confidential, or organization-specific information. The converter
does not copy optional session metadata into the output.
