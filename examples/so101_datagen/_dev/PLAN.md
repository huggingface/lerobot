# SO-101 datagen v2: side grasps, retries, denser boxes, wider cameras, motion noise

## Context

The `examples/so101_datagen` generator (branch `feat/so101-datagen`, worktree `lerobot_so101_datagen`, head `48384bbe1`) produced the 4k dataset that `g05_so101_sim4k` was trained on. Its limits, after reviewing MimicGen, MolmoBot and GraspVLA:

- **Grasps.** There is only one grasp family: from above, tilted 0–55° outwards (`motion.grasp_for`, `MotionConfig.tilt/steep_tilt`). There's no horizontal side grasp.
- **No recovery data.** A failed grasp discards the episode (`episode.simulate` raises on the `lifted` check, and `worker.py` draws a new seed). The policy never sees a miss followed by a retry, which is MolmoBot's biggest robustness lever.
- **Too few bbox targets for G0.5's bbox CoT.**
  - G0.5's bbox target is `emitted_at(t, style=vqa, role=assistant, camera=front)` (`lerobot_g05/.../configuration_g05.py:47`). Only frames carrying a front VQA _detections_ answer train it.
  - Today VQA runs at 1 Hz, alternating cameras, and 40% of questions are bbox. That gives about 0.2 Hz of front boxes, or 1.4% of frames.
  - GraspVLA's ablation credits about +13 points to the box step alone.
- **Cameras.** Front-camera views are conservative: azimuth ±35°, elevation 35–80°, roll ±3°, fovy 50–70°.
- **Command noise.** It is time-driven only (OU drift, jerks). There is no noise that scales with how fast the hand moves (MolmoBot's motion-proportional noise).

**Table height: not needed as such.** The arm is mounted on the table: base and tabletop are both at z=0 (`scene.py:216-220`). Raising the table therefore moves the robot and every object together. That is the same as moving the camera relative to the workspace, which we already randomize (distance, elevation, look-at), plus a cosmetic floor/horizon shift.

GraspVLA randomizes table height because its Franka stands on its own pedestal, so the table moves _relative to the robot_. The SO-101 equivalent is the arm on a riser above the objects' surface. Real SO-101 setups clamp to the table, so this is left out; it could become a small `mount_height` knob later.

## 0. Carry height, the real-robot "doesn't go up enough" (`motion.py`, `config.py`, `processors.py`)

**What was measured.** A physics-only run of seeds 0–399 with the current code (196 place episodes, heights of the object's bottom):

|                                              | place_in, median (10%) | place_near, median (10%) |
| -------------------------------------------- | ---------------------- | ------------------------ |
| Height at the `lifted` mark                  | 8.0 cm (5.5)           | 7.4 cm (5.2)             |
| Highest point in the carry                   | 8.6 cm (6.3)           | 7.9 cm (5.8)             |
| Peak position in the carry                   | first 10%              | first 10%                |
| Clearance over the box rim                   | 5.5 cm (2.7, min 1.2)  |                          |
| Sim sag (commanded − executed TCP z at lift) | ≤ 1 mm                 | ≤ 1 mm                   |

**Causes, three of them:**

1. **The data carries low too often.**
   - The place approach height (5–11 cm) is measured from the release point inside the box, not from the rim. So a quarter of `place_in` episodes clear the rim by under 4 cm.
   - The carry also peaks right after the lift and then slopes down toward the box. There is no "carry high" phase.
2. **No sag in sim.** The simulated servos hold position perfectly. A real STS3215 sags under gravity (shoulder_lift and elbow with the arm extended). A policy that learned action ≈ state therefore ends lower on the real arm, which costs centimetres at 25 cm reach.
3. **Undertrained.** Step 4000 × batch 24 is 96k samples, about 9% of one epoch of the 1.02M frames. Resuming toward the planned 20k steps (public repo) is the cheapest test of this cause.

**Fixes in the generator:**

- **Place hover height measured from the obstacle.** It becomes `max(sampled, rim_top − object_bottom_at_release + margin)`, with margin 4–8 cm. `place_near` uses the tallest neighbour along the path.
- **Higher lifts.** `lift_height` changes from (0.06, 0.13) to (0.08, 0.15), and the transit's via point stays at least at the lift height (`via_height` raised relative to the higher of the two endpoints).
- **Sag.** Measure the action − state gap on shoulder_lift and elbow during carries in 2–3 real SO-101 Hub datasets, and compare it with ours.
  - If the real gap is a few degrees, add a `GravitySag` actuation step: executed = command − gravity torque / stiffness, with the torque from `qfrc_bias` and stiffness sampled per episode.
  - Alternatively, widen `DynamicsConfig.kp_scale` downwards until the sim gap matches.
  - Either way it lands in the state (robot side), so the policy learns to command above where it wants to be.

## 1. Two grasp styles: top and side (`motion.py`, `config.py`)

**Sampling.** Each pick samples a style from `MotionConfig.grasp_styles = {"top": 0.65, "side": 0.35}`, falling back to the other style when the sampled one has no reachable or collision-free solution.

**`top`**

- The existing `grasp_for` path, with tilt sampled in `tilt=(0, 25)`.
- The `steep_tilt` sampling mode is removed. `candidate_tilts` still falls back to steeper tilts when a near-vertical one is out of reach.

**`side`: a new `side_grasp_for(arm, obj, ...)`**

- **The wrist roll is what defines the style.** The jaws are rolled 90° so they close horizontally, across the reach (tangential). That way they close around the object, instead of the fixed jaw pushing it away along the reach. The roll's sign is picked nearest the current roll, as `grasp_for` does.
- **Approach tilt** is anywhere from 60° to 90° from vertical, radial (`tilted_approach(center, tilt)`), sampled per pick. A 5-DoF arm keeps its approach in the pan plane, so side grasps always come from the robot's side of the object.
- **The same rule fixes steep top grasps.** Any grasp tilted past about 45° gets tangential jaws as well. Today `grasp_for` keeps the object's narrow axis there, which can leave the jaws along the reach.
- **Width** is the object's extent across the tangential axis:
  - Boxes and meshes qualify only when one footprint axis is within ±20° of radial; otherwise the jaws would land on corners.
  - Cylinders always qualify.
  - The width must fit the jaw gap (`arm.opening_for_width`).
- **Grip height** is sampled between `max(side_min_height, 0.4·h)` and `top − 1 cm`. Objects too short for the gripper housing to clear the table are not side-graspable. `side_min_height` is measured from the model during implementation, expected around 2.5–3 cm.
- **Collision check.** A new `Arm.collides(q, ignore)` helper checks arm-vs-scene contacts on the arm's scratch `MjData`, in the same way as `episode._free`. It is applied at the grasp, pregrasp and release poses. A pose touching the table or another object rejects that candidate, which replaces guessing gripper geometry.
- **Motion:**
  - pregrasp at the grip height, backed off 5–9 cm along the approach toward the base, entered via `transit`;
  - a horizontal `linear_move` slide-in, close, then a vertical lift keeping orientation (the existing `_highest(..., grasp)` path).

**`place` for side-held objects**

- The release pose is approached **vertically from above**, keeping the grasp orientation, instead of along the horizontal approach, which would sweep the fingers through container walls.
- The release pose gets the same `Arm.collides` check (container walls).
- `Grasp` gains a `style` field so `place`/`reach` know which approach to use.
- `candidate_tilts` gets a side grid (60–90°) used when `style == "side"`.

**Feasibility first.** A physics-only sweep over radius 0.17–0.30 m and object heights 3–8 cm. If side grasps turn out unreachable near the inner radius, side picks will draw a reachable object or fall back to top.

## 2. Retry and recovery episodes (`episode.py`, `motion.py`, `tasks/`, `worker.py`)

**Interleaved planning and physics.** Planning and physics now alternate, because a retry must react to where the object actually is.

- **Data flow.** `prepare()` keeps sampling the scene and start pose but no longer scripts the motion.
  - A new `Rollout` class (in `episode.py`) owns `MjData`, the command and actuation pipelines, and the recorded `qpos`/`commands`.
  - `Rollout.run_until(frame)` executes the planner's pending commands. It takes over the body of today's `simulate` loop.
  - `Rollout.object_now(obj) -> ObjectParams | None` returns the object's current pose (xy, yaw from `xquat`). It returns None when the object has toppled or left the reachable annulus.
- **Planner.** The planner gets an optional `rollout`. At a checkpoint it flushes its pending frames into the rollout and sets `_blendable = 0`, so a later move never blends into already-executed frames. The cost is a short settle after a lift, which an operator does anyway.
- **Pass 2 (rendering) is unchanged.** It still replays recorded `qpos`, so only successes are rendered.

**`Planner.pick` becomes a retry loop**, with `RecoveryConfig.max_attempts = 3`. MolmoBot also discards after more than 3 retries.

1. Plan one grasp attempt. With probability `RecoveryConfig.miss_prob = 0.2`, deliberately spoil it in one of these ways:
   - `high`: close 2–4 cm above the grip height;
   - `short`: close 1.5–3 cm short along the approach;
   - `wide`: offset 1.5–2.5 cm along the jaw axis;
   - `loose`: squeeze less, so the object slips on lift.
2. Execute up to the `lifted` mark and check `is_lifted`.
3. On failure:
   - open the gripper;
   - back off up or away (`linear_move` along `−approach`);
   - optionally dwell, like an operator noticing the miss;
   - read the object's new pose with `Rollout.object_now`, then plan the next attempt from the arm's current command.
   - If the object toppled, raise `EpisodeError("toppled")`.
4. Natural failures go through the same path, so episodes that used to be discarded become recovery episodes.

**Drops during the carry.** `place` executes up to its `release` mark and checks the object is still held. A drop raises `Dropped`. A new shared `Task.pick_and_place` helper, used by `place_in` and `place_near`, catches it and re-picks from the table, within the same attempt budget.

**Subtasks and memory:**

- A retry opens a new `pick up the X` subtask, so `persistent_rows` emits a fresh subtask row at the retry.
- `Planner.subtask(label, memory=None)` records a memory text per subtask, e.g. "My grasp of the X missed; trying again." `Plan.memories` replaces `TaskInstance.memories` in `annotations.persistent_rows`.

**Bookkeeping:**

- The sidecar `episodes.jsonl` gets `attempts`, `miss_modes` and `retry_frames` per episode, for later upsampling (MolmoBot upsamples retry segments 3×).
- `generation_report.json` counts retries.
- `max_episode_seconds` goes from 30 to 40.

## 3. Denser front boxes and a grasp-point keypoint (`annotations.py`, `worker.py`, `config.py`)

**Front-box stream**

- `AnnotationConfig.front_bbox_hz = 3.0` sets a dedicated stream of front-camera bbox pairs. The same seg render costs about 3 ms each.
- The answer lists the task's objects (the object, plus the target container or object): that's what G0.5's "predict bbox" CoT should output. With probability 0.25 it lists every visible object instead.
- Questions are phrased like "Detect the {obj} and the {target}."
- **One front answer per frame.** `emitted_at` resolves a single front answer per frame, so the general VQA stream (kept at 1 Hz) takes its front-camera frames from those not used by the bbox stream.

**Grasp-point keypoint (GraspVLA-style)**

- A new `grasp` question kind in the general VQA stream, on front and wrist: "Where will you grasp the X?" → `{"label": "grasp point of the X", "point_format": "xy", "point": [u, v]}`.
- **Where the point comes from:** the planned grasp center (between the jaws at grip height), stored in the object's frame when the attempt is planned. On each annotated frame it is transformed by the object's pose in that frame's `qpos` and projected with the camera's `cam_xpos`, `cam_xmat` and fovy.
- It is skipped when off-image or when the object isn't visible in the seg render. After a retry it follows the new attempt's grasp.
- **Usage caveat:** today's G0.5 recipe only consumes front detections. These keypoint rows serve other consumers, or a future grasp-point CoT field in G0.5, which would need G0.5 changes.

**Expected density:** about 10% of frames carry a front box, against 1.4% today.

## 4. More aggressive front cameras (`config.py`, `randomization.py`, `worker.py`)

**Two modes.** `FrontCameraConfig` gets a second sampling mode, `wide`, with `wide_prob = 0.4` and fields:

| Field           | Current    | `wide`                                   |
| --------------- | ---------- | ---------------------------------------- |
| `azimuth`       | ±35°       | ±150° (side and over-the-shoulder views) |
| `elevation`     | 35–80°     | 10–88° (grazing to near top-down)        |
| `distance`      | 0.3–0.55 m | 0.25–0.8 m                               |
| `roll`          | ±3°        | ±20°                                     |
| `fovy`          | 50–70°     | 40–95°                                   |
| `lookat_jitter` |            | 0.10                                     |

The current ranges stay as the `typical` mode.

**Frustum check.** `_in_view` keeps requiring every object whole in frame. `CAMERA_TRIES` goes up for the wide mode.

**Occlusion check (MolmoBot).** Before rendering an episode, one seg render of frame 0 must show the task's objects with at least `min_visible_pixels`; otherwise `EpisodeError("occluded")`. This matters once views come from behind the base.

**Wrist camera** stays a mounting tolerance.

## 5. Motion-proportional command noise (`processors.py`, `config.py`)

A new teleop-side step, `MotionNoise(RobotActionProcessorStep)`, sits in `make_command_pipeline` next to `CommandDrift` and `CommandJerk`.

**Per frame:**

1. Compute the TCP displacement `Δx` between the previous and current planned command, by FK.
2. Sample a Cartesian perturbation `ε ~ N(0, (k·|Δx|)²)` on position. The rotation part uses `0.1·σ_pos` rad, as in MolmoBot.
3. Map it to the 5 arm joints by damped least squares through `mj_jacSite` (`dq = Jᵀ(JJᵀ+λI)⁻¹ε`).
4. Accumulate it into an offset that decays with `tau`, and add the offset to the command. The gripper isn't noised.

**Effect.** Noise appears while the hand moves fast and fades near grasps and releases. MolmoBot's white per-step version would only jitter our open-loop commands, hence the accumulating offset.

**Defaults:** `NoiseConfig.motion_noise_k = (0.2, 0.5)` per episode and `motion_noise_tau = 0.5` s. That's roughly 2–5 mm of deviation at peak speed; it gets tuned on plots.

**Kinematics source.** The step owns an arm-only model (`Arm(so101.arm_spec().compile())`). Arm kinematics don't depend on the scene, so it needs nothing from the episode. It is seeded through `start_episode(rng)` like the other steps.

## Files

- `motion.py`:
  - grasp styles, `side_grasp_for`;
  - the retry loop in `pick`;
  - the vertical place for side grasps;
  - planner flush to the rollout;
  - `subtask(label, memory)`.
- `episode.py`: `Rollout` (absorbs `simulate`); `prepare` without scripting; the occlusion check hook.
- `so101.py`: `Arm.collides(q, ignore)`.
- `tasks/__init__.py`, `place_in.py`, `place_near.py`, `pick_lift.py`: `pick_and_place` with drop recovery; memories passed via `planner.subtask`.
- `annotations.py`: the front bbox stream, the `grasp` keypoint question, `persistent_rows` from `Plan.memories`.
- `render.py`: camera projection helper (`project(camera, points)`).
- `randomization.py`, `config.py`: camera modes, `RecoveryConfig`, grasp styles, noise and annotation knobs.
- `processors.py`: `MotionNoise`.
- `worker.py`: rollout wiring, sidecar and report fields, frame-0 occlusion check.
- `tests/test_so101_datagen.py`: extend the e2e test (validator still passes, front box density is at least 2/s); add one recovery test (`miss_prob=1` → `attempts ≥ 2` and success).

Run and debug scripts go in `robots/runs/so101_datagen/`, not in the PR.

## Visual sign-off (you validate each feature before the next builds on it)

- **Where.** Everything goes in `robots/runs/so101_datagen/review/<feature>/`: mp4s you can open in VS Code, plus PNG contact sheets. Each is rendered from real generated episodes, with a caption per tile (seed, task, style, attempt).
- **Order.** I stop and ask for your OK at each checkpoint in this order: 0, then 1, then 2, then 3–5 together.

| #   | Feature               | What you get                                                                                                                                                                                                   |
| --- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0   | Carry height          | Side-by-side clips, before and after, of the same seeds. A plot of object-bottom height along the carry with the box rim drawn, for 50 episodes. The real-vs-sim action − state gap plot.                      |
| 1   | Grasp styles          | 12 clips (6 top, 6 side) in front and wrist views, slowed to 0.5×, with the planned grasp point drawn. A sheet of side-grasp start/grasp/lift frames over different objects.                                   |
| 2   | Retries               | 8 clips, two per miss mode (`high`, `short`, `wide`, `loose`) plus natural misses, with the current subtask and memory text burned in. The retry frames are shaded on a timeline under each clip.              |
| 3   | Boxes and grasp point | 6 clips of the front camera with every front bbox answer and grasp keypoint drawn on its frame, so you can see the density (3 Hz) and how well they fit. Plus `lerobot-dataset-viz` on the 20-episode dataset. |
| 4   | Wide cameras          | A sheet of 48 first frames, `typical` vs `wide` (24 each), plus 4 episodes rejected by the occlusion check, to show what it filters.                                                                           |
| 5   | Motion noise          | Planned vs executed command per joint for 3 episodes, with TCP speed overlaid. One clip with k at its max, so the wobble is visible.                                                                           |

## Verification

1. **Physics-only sweeps** (no rendering, 200 seeds per variant):
   - success rate per grasp style and per task;
   - side-grasp reachability vs radius and height;
   - recovery success with `miss_prob=1`, per miss mode.

   Target: overall success no worse than today and recovery at least 70% (the report prints both).

2. **Visual check:** render about 12 episodes (side grasps, misses with retries, wide cameras) into a contact sheet and mp4 in `runs/so101_datagen/`, then look at them before going further.
3. **Annotations:**
   - Generate a 20-episode dataset. Check `StagingValidator` passes and that `render_sample` with G0.5's recipe resolves front bbox targets with no ambiguous-resolver errors.
   - Count frames with a front detections answer (target about 10%).
   - Overlay the boxes and the projected grasp point on decoded frames.
4. **Noise:** plot planned vs executed commands for one episode, to confirm the deviation scales with speed and stays small at the grasp.
5. **Tests and throughput:**
   - `uv run pytest tests/test_so101_datagen.py`, scoped (never the full suite).
   - Throughput on 6–8 workers, never 14, compared against today's frames/s.
6. Commit only when asked: one-line messages, no trailers.

## Housekeeping (pending from the last session)

Write the MimicGen / MolmoBot / GraspVLA technical analysis to `robots/runs/so101_datagen/SOTA_ANALYSIS.md`, kept local and not committed. Add a memory pointer (`memory/sota_synthetic_datagen.md` plus a line in `MEMORY.md`).
