# PickLift aligned front camera profile

The immutable real-camera profile is
`icspring_front_crop_1280x960_to_640x480_v1`.

## Executable physical-camera pipeline

1. Acquire the icSpring front camera as 1920×1080 RGB, MJPEG, requested
   30 FPS.
2. Preserve the camera's raw lens distortion.
3. Center-crop the exact rectangle:
   `x=320, y=60, width=1280, height=960`.
4. Resize that 4:3 crop to 640×480 RGB with OpenCV `INTER_AREA`.
5. Sample synchronized Dataset v3 frames at exactly 20 FPS.

There is no software crop after the 640×480 output, no aspect-ratio stretch,
no rotation, and no upsampling. The source resolution, crop rectangle,
resize algorithm, output geometry, and reference FOV are copied into every
dataset/session/episode provenance manifest.

## FOV evidence

- The accepted MuJoCo alignment reference uses a vertical FOV of **47°** with
  the same 4:3 1280×960 master / 640×480 output geometry.
- A prior grid-based estimate of the physical cropped image was approximately
  **63° horizontal × 49° vertical**. This is an approximate measurement, not
  calibrated intrinsics.

The physical implementation is therefore defined by the native mode and exact
crop/resize pipeline above, not by attempting to set a `fovy` value on the USB
camera.

Native 1920×1080 frames are the bounded evidence source but are not currently
written as a sidecar; manifests explicitly record `raw_evidence=not_recorded`.
