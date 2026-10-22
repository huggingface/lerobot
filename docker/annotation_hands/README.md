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

The free builder has a 30-minute build limit; a longer startup timeout does not
extend it. Build the two camera wheels in a CPU-only HF Job using the pinned
devel image, Python 3.10.20, PyTorch 2.8.0+cu128 and setuptools 79.0.1. The private
operator build records source hashes, import checks, ELF architecture listings
and wheel hashes. Copy its `build.json` and `wheels/` to `camera-wheels/` in this
image context and pin their hashes plus the source build identity in
`image-version.json`. `verify_camera_wheels.py` rejects missing operators,
changed sources, architectures or unverified bytes before installation. No
weights or GPU-specific TensorRT plans enter these wheels.

Compile only SM80 (A100), SM90 (H200) and SM120 + PTX (RTX PRO 6000), with two
bounded compiler jobs. The build helper
filters out `lietorch_extras`, which is referenced only by vendor examples;
DROID-W uses `lietorch_backends` and its own `droid_backends`. Vendor inference
sources and compiler optimization flags stay unchanged. This changes build
packaging, not model precision or runtime sampling.

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
