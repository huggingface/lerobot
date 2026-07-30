# Task1 Eval-v2 systematic-offset audit v1

This experiment performs a hardware-free audit of the frozen Eval-v2 `1/12`
result. It reads the immutable canonical videos, step evidence, accepted Real
camera geometry audit, and Real24 Dataset v3 without running a policy or
writing the Dataset.

Run:

```bash
.venv/bin/python \
  experiments/task1_picklift_evalv2_systematic_offset_audit_v1/systematic_offset_audit.py
```

The runner refuses to overwrite the immutable evidence root and verifies every
frozen input hash before analysis. Validate the completed output independently:

```bash
.venv/bin/python \
  experiments/task1_picklift_evalv2_systematic_offset_audit_v1/validate_systematic_offset_audit.py
```

Large videos, decoded frames, overlays, and evidence remain under:

`/home/ubuntu24/Teleop/artifacts/analysis/task1_picklift_evalv2_systematic_offset_audit_v1`

The tracked `systematic_offset_audit_result_v1.json` is the small result index.
