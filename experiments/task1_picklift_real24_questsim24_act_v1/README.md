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
- `combined48_v1` was retained as a pre-training failed candidate because
  evidence-only strings were accidentally embedded in its visual stats entry.
  `combined48_v2` moves those strings to the external manifest and is the only
  dataset identity used by smoke and full training.
- Policy: ACT ResNet18/ImageNet initialization, front-only, state-to-action,
  67-step chunks at 20 Hz, fixed 500-step smoke followed by one 100,000-step
  run from scratch.

All dataset, checkpoint, log, and validation evidence stays under
`/home/ubuntu24/Teleop/artifacts`. No hardware or simulation rollout is part of
this experiment.

## Completed result

- Combined48 v2 tree:
  `cf70d2195325779187a8433992754f739ff8c99541182e5b32da90cd68ac2086`
- Fixed step-100000 model:
  `e054e682057f09a4653af00a4580da173d3d1658ef5c34244bdbf3ca1a125de5`
- Final logged metrics: loss `0.051`, L1 `0.050`, KLD `0.000`.
- Offline reload and inference: pass, output shape `[1, 6]`, all finite.

See `combined48_result.json`, `run_result.json`, and
`offline_validation_result.json` for immutable artifact paths and hashes.
