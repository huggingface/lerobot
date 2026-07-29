# Task1 Real24 + Quest-Sim24 ACT v1

This experiment implements research plan
`task1-picklift-real24-questsim24-act-v1` at research commit `63874ee`.

- Derived Dataset v3: 24 accepted Real episodes followed by 24 accepted
  Quest-to-Remote MuJoCo episodes, each exactly once.
- Sampling: standard frame sampling over all 7,966 frames; no domain weights,
  duplication, upsampling, or downsampling.
- Metadata-only normalization: canonical SO-101 `.pos` joint names, generic
  `so101` robot type, and one canonical PickLift task string. Original metadata,
  episode identity, subset hashes, and tree hashes remain in the external
  provenance map.
- Video: source AV1 and H.264 files remain separate and are not re-encoded.
  High-resolution Quest sidecars are excluded.
- Normalization: numeric statistics are recomputed from the derived parquet
  files; visual statistics are computed over every decoded 640x480 training
  pixel. `use_imagenet_stats=false`.
- Policy: ACT ResNet18/ImageNet initialization, front-only, state-to-action,
  67-step chunks at 20 Hz, fixed 500-step smoke followed by one 100,000-step
  run from scratch.

All dataset, checkpoint, log, and validation evidence stays under
`/home/ubuntu24/Teleop/artifacts`. No hardware or simulation rollout is part of
this experiment.
