# Hand annotation worker

This image runs EgoHand/SOMA with DROID-W + MoGe on HF Jobs. The controller,
camera and hand workers use isolated environments because their PyTorch/CUDA
versions differ. The image installs precompiled camera wheels; it needs no CUDA
compiler. Model assets are embedded during the image build using public upstream
downloads and the frozen manifest, so Jobs need no model-repository login.

## Build context

Use this directory with the operator's immutable runtime artifacts:

- `camera-adapter-source.tar.gz`: calibrated camera adapter, license notices and
  `runtime-source.json` with per-file checksums.
- `camera-native-source.tar.gz`: DROID-W/MoGe source and TensorRT wrappers.
- `camera-startup-source.tar.gz` and `camera-startup-sources.json`: supplemental
  Python imports and their notices, checked before installation.
- `camera-wheels/build.json` and `camera-wheels/wheels/*.whl`: CPU-built LieTorch
  and DROID operators for Python 3.10, with SM80/SM90/SM120 support.
- `image-version.json`: source, wheel and manifest checksums.
- `model-assets/`: the frozen `assets.json`, notices, camera support files and
  model files small enough to keep in the Space repository. Large upstream
  downloads are pinned in `model-sources.json`.
- `egohand-checkpoint-template.zip`: accepted checkpoint container metadata and
  tensor hashes, without tensor values. The build streams the 817 inference
  tensor records from the public checkpoint with bounded memory, without
  unpickling it, and requires a byte-identical inference-only checkpoint.

```bash
docker build -t annotation-hands docker/annotation_hands
```

The build checks wheel hashes, complete offline imports and the cuDSS 0.8.0.10
ABI as the Job user. Compatibility links preserve the frozen camera wrapper's
checkpoint and TensorRT paths. Both camera compilation and inference require
`NVIDIA_TF32_OVERRIDE=0`.

## Deployment

Pin the built image's HF Space revision and `image-version.json` checksum in
Studio's `hand-worker-image.json`. Also pin the model release revision and
`assets.json` checksum. Publish this build context in the public worker Space and
set the asset profile's `embedded` flag. Jobs use the preinstalled, checksummed
assets and native inference runs offline. The user's HF login is needed
for Jobs and private datasets. Preserve upstream terms
and notices when preparing the runtime and model releases.

The CPU health endpoint reports build readiness. Before enabling a new image,
validate a GPU Job that compiles its camera engines and writes native SOMA
actions. TensorRT plans are built on the selected GPU rather than copied from
another machine.
