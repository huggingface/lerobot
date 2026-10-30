# Synthetic manipulation data: MimicGen, MolmoBot, GraspVLA (analysis 2026-10-29)

These notes were taken for `examples/so101_datagen`, our scripted MuJoCo SO-101 generator.

**About the numbers:**

- Some come from summaries of appendices, not verbatim text: MimicGen's noise ablation, GraspVLA's friction value and CoT token counts, and MolmoBot's lighting and dynamics ranges.
- MolmoBot's π0.5 baseline appears as both 31.3% and 39.2% in the paper itself.

## MimicGen (NVlabs, CoRL 2023, `NVlabs/mimicgen`)

**Source data**

- About 10 human demos per task (robomimic HDF5).
- Each timestep stores `datagen_info`:
  - `eef_pose` (4x4);
  - `object_poses` (a dict of 4x4);
  - `subtask_term_signals` (0/1);
  - `target_pose` (from `action_to_target_pose`);
  - `gripper_action`.

**Subtasks**

- A subtask ends at the first 0→1 edge of an automatic signal (contact, lift, a partial success check).
- Per subtask, the user sets:
  - one `object_ref`, the term signal and a term offset range;
  - the selection strategy and the action noise;
  - `num_interpolation_steps` (5) and `num_fixed_steps` (0).
- The last subtask has offset (0, 0).

**Transform:** `C'_t = O'_0 · O_0^-1 · C_t`. In code, `src_eef_rel_obj = inv(src_obj) @ src_eef`, then `obj_pose @ rel`. Gripper actions are copied verbatim and never noised.

**Controller:** robosuite OSC_POSE delta at 20 Hz.

- Position: `clip((target_pos − curr_pos)/max_dpos)`.
- Rotation: `clip(axisangle(target_rot·curr_rotᵀ)/max_drot)`.

**Selecting the source demo**

- `random` is the default.
- `nearest_neighbor_object`: `nn_k=3`, cost `‖Δp‖ + arccos((tr(R_src R_objᵀ)−1)/2)`.
- `nearest_neighbor_robot_distance`.
- `select_src_per_subtask` decides whether the demo is re-picked per subtask.

**Joining segments:** linear position and axis-angle rotation interpolation. 5 steps in sim; 25 steps plus 25 fixed steps on the real robot.

**Noise:** `σ·randn(6)` on the normalized pose action, σ=0.05 in sim and 0.02 on the real robot.

**Filtering:** success is the OR of `is_success` over the executed steps. Data generation rate (DGR) = successes / attempts.

**Results**

- 50k demos from about 200 human demos, over 18 tasks (1000 per variant).
- DGR:

  | Task        | DGR   |
  | ----------- | ----- |
  | Stack       | 94.3% |
  | Square      | 73.7% |
  | Threading   | 51%   |
  | Mug Cleanup | 29.5% |
  | Gear D1     | 8.2%  |

- Image BC-RNN at 84×84, front plus wrist: Square D0 goes from 11.3% to 90.7%.
- Dataset size: big gains from 200 to 1000 demos, diminishing to 5000.
- More interpolation hurts: Stack D1 drops from 99.3% to 68.7%.
- Noise ablation: removing noise raises DGR (51% → 84.5%) but drops policy success (98% → 59.3%), so noise matters.
- Works across robots.
- Real robot: Stack 36%, Coffee 14%, against 0% from source demos alone.

**Limits:**

- quasi-static rigid objects and a known subtask order;
- one reference object per subtask;
- no collision avoidance and no reachability checks.

**SkillMimicGen** (2410.18907)

- Segments demos into skills and plans the transit moves (TRAC-IK, RRT-Connect), with 5 cm approach and retreat moves around each skill.
- Adds a Hybrid Skill Policy.
- Generation rate averages 75% vs 41% for MimicGen.
- Initiation augmentation: translation U[−8, 8] cm and rotation U[0, 80°], plus a recovery segment.

**For us**

- The 5-DoF SO-101 can't follow full SE(3). Use planar transforms instead: `p' = p_O0' + R_z(Δψ)(p − p_O0)`, `yaw' = yaw + Δψ`, then IK, rejecting failures.
- Its value over our scripted expert is reusing human motion shapes, recorded with the sim teleop of #5109 and a leader arm.

## MolmoBot / MolmoSpaces (AI2, arXiv 2603.16861v2)

**Licences:** code Apache-2.0; data ODC-BY 1.0 and CC BY 4.0. Some Objaverse assets are non-commercial; filter them with `MLSPACES_LICENSE_POLICY=commercial_safe`.

**Scenes:** prebuilt MuJoCo houses, more than 200k in total, of which the dataset uses 94.2k.

| Source             | Houses     |
| ------------------ | ---------- |
| iTHOR              | 120        |
| ProcTHOR-10k       | about 120k |
| ProcTHOR-Objaverse | about 110k |
| Holodeck           | about 110k |

**Assets**

- iTHOR (about 2k) plus Objaverse (about 129k converted).
- Filters:
  - receptacle footprint under 50 cm and height at most 15 cm;
  - the pickup diagonal smaller than the receptacle;
  - watertight colliders and semantic relevance.
- Result: 11.4k pickup and 9.4k receptacle assets.

**Tasks:** 8 task-robot configs over 6 task definitions. Success criteria:

- Pick: lifted at least 1 cm and unsupported.
- Pick-and-place: at least 50% of the object's weight on the receptacle.
- Next-To: an XY gap of 0–5 cm.
- Open: at least 67% of the joint range.

**Robots:**

- Franka FR3 with a Robotiq 2F-85, DROID configuration, 15 Hz, absolute joint targets (better than deltas on the real robot).
- RB-Y1: delta joints plus base, policy at 100 ms.

**Expert**

- Franka uses IK interpolation; RB-Y1 uses cuRobo trajectory optimization.
- **Grasps:** a precomputed 6-DoF grasp library per object, ranked by a weighted cost (TCP proximity, rotation similarity, vertical alignment, distance to the centre of mass).
  - Collision-checked with phantom grippers, 128 at a time.
  - Then batch IK, 256 at a time.
- **Phases:** Pregrasp, Grasp, Lift, Preplace, Place, Postplace, Stow.
- **Retries:** on a detected failure the expert resets to the first phase. Episodes with more than 3 retries are discarded.
- **Initial-joint noise** is graduated per joint (Franka radii from 0.025 to 0.175 rad), capped at 10 cm of TCP displacement.
- **Action noise** is proportional to motion:
  - TCP σ = 0.1·|Δx|, clipped at ±2 cm;
  - rotation σ = 0.1·σ_pos, clipped at ±0.1 rad;
  - projected to joints by least squares on the Jacobian, and off at evaluation.
- Gripper: close 0.5 s, open 0.25 s.
- Recorded privileged signals: object poses, visibility, intrinsics and extrinsics, image-frame points, phase, retry count.

**Domain randomization**

- Lights: count, position, intensity, colour, shadows.
- Textures: procedural and AI2-THOR.
- Dynamics: friction, mass, damping.
- **Wrist camera:** 52° FOV, position ±1.5/0.5/2 cm, roll ±8°, pitch and yaw ±4°.
- **Exocentric cameras:**
  - ZED-like: 64–72° FOV at 0.2–0.8 m;
  - GoPro-like: 137–140° FOV at 0.2–0.5 m;
  - height 0.05–0.6 m, 360° azimuth, look-at ±10 cm;
  - resampled up to 20 times when the objects aren't visible.
- Images are 624×352. Rendering is classic MuJoCo; Filament is used only for evaluation.
- Policy-side image augmentation: ColorJitter, GaussianBlur, Posterize, Sharpness, Grayscale.

**Scale:** 1.7M episodes, 295M frames, 5,704 h. About 660 successful episodes per A100-hour, about 6,500 GPU-hours in total.

**Policy**

- Molmo2-4B with a DiT flow-matching head, up to 3 frames per view.
- Chunk of 16, executing 8. Batch 1024, LR 1e-5.
- Upsampling: retry-grasp segments ×3, successful picks ×2.

**Results:** real DROID pick-and-place 79.2%, against 31.3% (or 39.2%) for π0.5-DROID.

**Ablations:**

- The number of demos is the main driver.
- The number of environments matters little.
- Object diversity helps in sim, not on the real robot.

**Language:** templated instructions with CLIP-chosen referring expressions (softmax τ=0.02, dropped below margin 0.03), plus point annotations. No boxes.

**For us:**

- a retry/recovery phase loop (reset to pregrasp, at most 3 retries, upsample the retry segments);
- motion-proportional noise through the Jacobian;
- graduated initial-joint noise;
- a camera visibility rejection and wider exocentric camera ranges;
- a grasp library with collision and IK filtering;
- absolute joint targets (already ours);
- a commercial-safe asset filter.

## GraspVLA / SynGrasp-1B (PKU, arXiv 2505.03233v3)

**Licences:** paper CC BY-NC-SA 4.0; code badge BY-NC 4.0. The dataset `vegebirrd/SynGrasp-1B` (4.91 TB) has no licence or card. The generator code isn't released.

**Assets:** Objaverse LVIS, 240 categories and 10,680 instances.

- Weapons filtered out.
- Per-category size limits, 2–35 cm.
- ACVD mesh simplification.

**Scenes:** clutter in a 0.4×0.5 m area, objects dropped in random poses. Table height −0.1 to 0.2 m relative to the robot's own pedestal.

**Expert**

- Grasps: BoDex antipodal / force-closure grasps (friction 0.15).
- cuRobo plans one collision-free trajectory through grasp and lift, preferring smoothness over planning success.
- The start pose is randomized.
- Physics is validated in MuJoCo; images are ray-traced in Isaac Sim.

**Domain randomization**

- Point, directional and dome lights.
- About 1,000 table textures and 1,200 floor/wall textures.
- Two cameras, each jittered in a 15 cm ball and ±5° per axis:
  - front at (1.35, 0, 0.54), looking at (0.2, 0, 0);
  - side at (0.5, 0.69, 0.5), looking at (0.5, 0, 0.1).
- Robot material randomized; 30+ distractors.

**Scale:** 10M trajectories × about 100 frames at 10 Hz, about 1B frames. 160 RTX 4090s for 10 days (about $5K), about 7 frames/s per GPU. Asynchronous EnvLogger writes, one UUID folder per process.

**Model**

- InternLM2-1.8B with frozen DINOv2 and SigLIP encoders, plus a π0-style flow-matching action expert.
- **PAG chain of thought:** a 2D bbox (8 tokens), then a 3D grasp pose in the base frame (6 tokens), then the action, conditioned on the KV cache.
- Co-trained with GRIT internet grounding data, on the bbox loss only.
- Losses: `L_S2 = −Σ log P(bbox) − 1_syn Σ log P(grasp | bbox)`; `L_S1` is the flow loss.
- Action chunk of 4 end-effector deltas. Batch 384, LR 1.6e-4, about 200 ms latency.

**Results**

- About 93% real zero-shot across five test sets.
- PAG ablation:

  | Variant    | Success | SPL |
  | ---------- | ------- | --- |
  | No CoT     | 66.6%   | 39  |
  | + 2D box   | 80.0%   |     |
  | + 3D grasp | 93.3%   | 90  |

- Transparent objects: 86.6%, against 10% for AnyGrasp.
- LIBERO zero-shot 82–94%.
- Multi-view 93% vs single view about 60%.
- Adapts to new hardware with 5k trajectories.
- Failure modes: ambiguous language 31%, misidentification 27%, slipping 21%.

**For us:**

- dense front-box CoT targets (G0.5 already has the bbox CoT);
- a grasp-point/pose target (a keypoint row now; a G0.5 CoT field later);
- co-training on internet grounding data (training side);
- category diversity over instance count;
- a precomputed grasp library;
- more textures.

## Decisions taken (2026-10-29), see the plan in `~/.claude/plans/pasted-content-id-f641-i-d-like-linear-creek.md`

- Top grasps and side grasps. A side grasp means wrist roll at 90°, so the jaws close tangentially, with a 60–90° tilt.
- In-episode retries with deliberate misses.
- A 3 Hz front-box stream, plus a grasp-point keypoint.
- A `wide` front-camera mode with an occlusion check.
- Motion-proportional Jacobian noise.
- Higher carry over obstacles; check real-arm gravity sag.
- Table height left out: the arm is on the table, so it's equivalent to moving the camera.
