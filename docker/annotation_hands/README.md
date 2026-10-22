# EgoHand/SOMA HF Jobs image

This is the CPU build recipe for the experimental `Hand actions` runtime. It
keeps DROID-W + MoGe and EgoHand in their validated separate Python/PyTorch
versions. Model weights are fetched at Job startup, never embedded in an image.

The operator build context is maintained in the private HF Docker Space
`finerobotics/egohand-soma-worker-v4`. It includes:

- This Dockerfile and the pinned environment requirements.
- Studio's `worker-requirements.txt` and the CPU-only health endpoint.
- `camera-native-source.tar.gz`: the frozen DROID-W + MoGe source, headers and
  TensorRT hooks, without machine-specific binaries, weights or recordings.
- `camera-adapter-source.tar.gz`: the pinned calibrated camera adapter, its
  upstream license, and `runtime-source.json` with per-file checksums.
- `image-version.json`: source/archive hashes and native revision identifiers.

The private source context is separate from the public LeRobot distribution.
Building preserves all applicable upstream notices. GPU extensions compile from
source; RTX 5090 binaries and TensorRT plans must not be reused on other GPUs.
Studio checks both the image Space commit and the image-version checksum.

The private model release `finerobotics/egohand-soma-runtime` contains the verified
inference-only checkpoint, FP32 detector ONNX, native SOMA/MHR assets, MoGe/DROID
weights, exact camera ONNX graphs, local initialization caches and model notices.
`assets.json` pins every file's SHA-256 and byte count. It contains no MANO,
recordings, benchmark predictions, reference cameras or evaluation targets.

After a successful CPU build, provision Studio's `hand-worker-image.json` with the
image revision and image-version hash, plus the immutable model release and asset
manifest hash. Its backend paths must match the Dockerfile. Mark `build_verified`
only after the image passes its CPU import checks and serves the health receipt.

On the user's GPU Job, `hand_setup.py` compiles the frozen camera graphs using the
validated TensorRT 10.13.3.9 settings. The detector builds its own FP32 provider
cache. This is deployment compilation, not automatic benchmarking or speed tuning.
No GPU acceptance or performance claim follows from a successful CPU image build.
