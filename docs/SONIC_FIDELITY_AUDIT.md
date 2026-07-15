# SONIC fidelity audit — Python/ONNX port vs C++ deploy reference

Compares the lerobot Python/ONNX SONIC port against the original C++ deploy stack
(`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` and headers).

**Verdict:** the port is algorithmically faithful in the core control math (the parts
that determine stability and pose tracking). The gaps are concentrated in (a) update
cadence, (b) the safety layer, and (c) two modes/paths present in C++ but never wired up.
No silent math bug was found; divergences are deliberate or missing-feature.

## Genuinely faithful (verified, no action needed)

- **Action production** — both do `q_target = DEFAULT_ANGLES + policy_out[isaaclab→mujoco] * ACTION_SCALE`
  (residual-to-default, not residual-to-previous). C++ `g1_deploy_onnx_ref.cpp:3119-3127`,
  Python `sonic_pipeline.py:708-724`.
- **Joint-order remap** — `ISAACLAB_TO_MUJOCO` / `MUJOCO_TO_ISAACLAB` identical arrays.
- **Gains** — same `Kp = armature·ω²`, `Kd = 2ζ·armature·ω`, ω = 10·2π, ζ = 2, and the same
  2× set `{4,5,10,11,13,14}` (ankles + waist roll/pitch).
- **History normalization** — robot `q` subtracts defaults, velocities raw,
  gravity = `quat_rotate(conj(base), [0,0,-1])`, oldest→newest ordering.
- **6D anchor rotation** — verified element-by-element: C++ takes the first two *columns*
  of the rotation matrix flattened row-wise (`.cpp:677-683`); Python `quat_to_6d`
  (`sonic_pipeline.py:227-240`) produces the identical 6 values. Only the Python docstring
  wording ("rows") is misleading — the math is correct.
- **Planner** — replan intervals (RUN 0.1 / CRAWL 0.2 / boxing 1.0 / default 1.0 s),
  8-frame slerp crossfade blend, 30→50 Hz linear+slerp resample, `MOTION_LOOK_AHEAD = 2`,
  4-frame 30 Hz context. All match.
- **SLERP / heading / FK conventions** — wxyz, shortest-path slerp with 0.9995 linear
  fallback, `calc_heading` yaw extraction.

## Divergences that reduce fidelity (ranked)

### 1. Encoder cadence: 10 Hz vs 50 Hz (biggest)
C++ runs the encoder every control tick — `GatherTokenState → Encode()` is unconditional
inside the 50 Hz `Control()` loop (`.cpp:1644-1684`, no N-step gate). The port recomputes
the token only every 5 ticks (`ENCODER_UPDATE_EVERY = 5`, `sonic_pipeline.py:150`), so the
latent is up to 80 ms stale and the decoder consumes a held token for 4 of every 5 ticks.
Likely a perf shortcut. For full faithfulness set `ENCODER_UPDATE_EVERY = 1` (cheap on GPU).

### 2. SMPL root anchor disabled by default
C++ always feeds the reference root orientation into the anchor/heading. The port sets
`smpl_root_quat = None` unless `enable_smpl_root=True` (`sonic_whole_body.py:366`), because
the raw 30 Hz root caused QACC spikes. Faithful fix: slerp-resample the root 30→50 Hz like
the joints, then re-enable by default. Until then, mode-2 heading steering isn't faithful.
See `SONIC_REPLAY_DEBUGGING.md`.

### 3. VR 3-point teleop (`encode_mode = 1`) — now wired (was inert)
Originally the encoder layout for mode 1 existed but nothing set `encode_mode=1` or
filled `vr_3point_local_target` / `vr_3point_local_orn_target`. **Now implemented
end-to-end:**
- Producer `pico_publisher.py` computes the 3 keypoints via `smpl_fk.compute_3point`
  (ported from gear_sonic `_process_3pt_pose`) and adds `vr3_pos` (9) / `vr3_orn` (12)
  to the `rt/smpl` message.
- `SmplStream` parses them (`has_vr3`); `PicoHeadset(mode="vr3")` emits `vr3_pos.*` /
  `vr3_orn.*` action keys.
- `SonicWholeBodyController` extracts them, switches to `encode_mode=1`, fills the
  controller targets, and drives locomotion from the joystick/keyboard planner
  (`use_joystick=True`); `PlannerController.build_encoder_obs` gained a mode-1 branch
  (lower body per-frame step 5 + VR targets + anchor).

Still not ported: `vr_3point_compliance`, and the operator calibration
(`ThreePointPose.apply_calibration`) / physical wrist offsets — the raw tracked joint
poses are used, so hand-tracking may need calibration tuning.

### 4. Safety layer largely absent
- **Joint-velocity kill switch** at >35 rad/s (`.cpp:2829-2832`) — missing.
- **E-stop damping**: C++ e-stop commands `kp=0, kd=8` (active damping, `.cpp:2708-2714`).
  The port's Space/e-stop just sets `playing=False` + `LM.IDLE` and stops the cursor — it
  does not switch to a damped hold. Less safe.
- **Motor-temperature** monitor (90 °C / 85 °C hysteresis) — missing.
- **Stale-/late-state watchdogs** (500 ms fail, 50 ms warn, 200 ms token timeout) — not in
  the SONIC layer (partly covered by `UnitreeG1`, not equivalently).
- **Per-tick delta clamp** — C++ has none, so the reverted `MAX_DELTA_PER_STEP` was correctly
  removed; that part is faithful.

### 5. Idle readaptation blend missing
At planner IDLE at a motion end, C++ runs a double-threshold blend (0.98/0.02 toward
robot-current or original target; thresholds 0.10/0.05/0.045 rad; `.cpp:3303-3361`). The
port just holds. Minor; only matters at motion-end idle.

### 6. Input/streaming paths not ported
- **ZMQ Protocol V1 joint streaming** (`encode_mode=0` from a live joint stream via
  `StreamedMotionMerger`) — not implemented; the port's streaming path is SMPL-only.
- **External token injection** (tokens over ZMQ/ROS2 bypassing the encoder) — not supported;
  the port always encodes locally. Fine for standalone.
- **Gamepad** — C++ has a full gamepad map (EMA smooth 0.3, deadzone 0.05); the port has
  joystick byte-parsing + keyboard, no gamepad manager. Functionally close.

### 7. Minor input-feel differences
Delta-heading is continuous (±0.02 rad/tick) in the port vs discrete steps (±π/6, ±π/12) in
C++; speed/height increments differ slightly. Behavioral feel only, not correctness.

## Recommendation (priority order)

1. `ENCODER_UPDATE_EVERY = 1` (or a param defaulting to 1) — closes the biggest gap for
   near-zero cost on GPU.
2. Rate-match the SMPL root 30→50 Hz (slerp) and re-enable `enable_smpl_root` by default.
3. Add the safety envelope: joint-velocity kill (35 rad/s) and a proper damped e-stop
   (`kp=0, kd≈8`). Real hardware-safety consequences.
4. Wire `encode_mode=1` from the pico headset (3-point targets), or document as out of scope.
5. Fix the `quat_to_6d` docstring wording ("rows" → "first two columns flattened row-wise").

Items 4–7 are feature-completeness; 1–3 are what to do for faithful behavior on the robot.
