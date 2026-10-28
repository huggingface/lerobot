# OpenArm Gripette Policy Prediction Logging

## Goal

Persist every policy prediction and its derived robot targets during real and dry-run evaluation.

## Design

`run_grabette_policy.py` accepts `--log_dir`, defaulting to
`outputs/openarm_gripette_policy`. Each invocation creates one timestamped JSONL file.

The first record contains run metadata: policy, checkpoint, device, FPS, dry-run mode,
action horizon, and held initial joint positions. Each subsequent record represents one action
from a predicted chunk and contains:

- monotonic elapsed time, chunk index, and action index;
- raw named policy output (`x`, `y`, `z`, `r6d_0` through `r6d_5`, `strategy`, `closure`);
- seven IK joint targets in URDF degrees;
- decoded proximal and distal Gripette targets in radians;
- whether the target was sent to hardware.

Records are flushed after every write so an exception or Ctrl-C retains all completed actions.
The script prints the log path when it opens and closes the file.

## Testing

A focused test writes metadata and one action to a temporary directory, parses both JSONL
records, and verifies field names, numeric values, and dry-run execution state.
