# integration plan: sim HIL-SERL (UR10e+2F85, 3-obj nesting)

branch: feature/reward-models-port
sim: ../simulator_for_IL_RL (AssemblingEnv, UR10e+2F85, 3 objs nesting, NO reward, fast-mode OK)
teleop: ../RC10_control (PS4Joystick, pygame, DualSense-compat)
target: 3 modes (manual / binary / sarm) runnable in sim via lerobot fork

user constraint: NO changes to simulator_for_IL_RL. RC10 mods OK.

## what sim gives / lacks

gives:
- `AssemblingEnv(xml_path, sim_timestep, control_hz, mode='realtime'|'fast', use_task_space, render_mode)` — raw gym.Env, NOT registered
- nested obs dict: `state.{joint_pos,joint_vel,ee_pos,ee_quat,ee_lin_vel,ee_ang_vel}` (7+7+3+4+3+3 = 27 dims) + `objects.{bottom,mid,cap}.{pos,quat,lin_vel,ang_vel}` + `images.cam_{front,side,gripper,state}` 240x320x3 uint8
- action: task-space 8D [xyz, quat_wxyz, gripper ∈ [0,1]] or joint 7D
- reward=0.0 always. terminated=False always. truncated at max_episode_steps.
- fast mode: no sleep throttle, ~91 steps/s for 20Hz control (~4.5× real-time on this box)

lacks (for HIL-SERL):
- gym.register id
- flat obs compatible w/ `observation.images.*` + `observation.state`
- reward signal (port already has height_gripper / cnn / sarm dispatch — plug in)
- teleop wiring (RC10_control is hand-driven; needs Teleoperator adapter)

## integration surface

### A. gym registration
target: `src/lerobot/envs/sim_assembling.py` (NEW)
- entry fn `make_assembling_env(**kw) -> gym.Env` that instantiates AssemblingEnv + wraps w/ adapter
- `gymnasium.register(id="sim_assembling/AssembleBase-v0", entry_point=make_assembling_env)` at import time
- kwargs: xml_path='scene.xml', control_hz=20.0, mode='fast', use_task_space=True, render_mode='rgb_array', image_obs=True, resize_to=(128,128)

### B. obs/action adapter
gym wrapper `AssemblingHILAdapter(gym.Wrapper)`:
- obs flatten:
  - `observation.state` = concat([joint_pos(7), ee_pos(3), ee_quat(4), gripper_q(1)]) → 15D float32 (use joint_pos idx for gripper)
  - `observation.images.front` = cam_front resized 128x128 (H,W,3) uint8
  - `observation.images.wrist` = cam_gripper resized 128x128 (H,W,3) uint8
  - drop side/state cams (fork cfg uses 2 cams max, keep extensible)
- action adapter: accept 3D delta XYZ (gym-hil convention) + discrete gripper (num_discrete_actions=3 like HF). internally:
  - maintain ee ref pose (initialized at reset)
  - pos += delta * step_size (default 0.025m per-axis clip)
  - keep quat fixed (rot held constant; matches SAC default 3D action shape)
  - gripper map: 0=noop, 1=open, 2=close → gripper ∈ [0,1]
  - build 8D task-space action → pass through to AssemblingEnv
- reset: obs reset + reset internal ee_ref pose

### C. env factory branch in gym_manipulator.py
extend `make_robot_env` in fork:
- new branch `cfg.name == "sim_assembling"` → `gym.make("sim_assembling/AssembleBase-v0", **kwargs_from_cfg)`. teleop_device = None (handled later via separate teleop config block).
- extend `make_processors` sim_assembling branch mirror of gym_hil:
  - `GymHILAdapterProcessorStep` already handles flat-obs rename — but our wrapper already emits lerobot keys. Option 1: use VanillaObservationProcessorStep directly. Option 2: add a passthrough adapter.
  - add reward_step hook (`_maybe_build_reward_step`) — already generic.

### D. DualSense teleop → lerobot Teleoperator
target: `src/lerobot/teleoperators/dualsense_sim/` (NEW, don't touch existing `ps4_joystick`)
- class `DualSenseSimTeleop(Teleoperator)`:
  - wraps `rc10_api.ps4_joystick.PS4Joystick`
  - `action_features`: {"delta.x", "delta.y", "delta.z", "gripper"} or 3D action + gripper discrete
  - implements `get_action() -> dict` returning sticks via `get_normalized_deltas()` scaled to action step size
  - implements `get_teleop_events() -> dict` using `get_button_states()` → `TeleopEvents.{SUCCESS, TERMINATE_EPISODE, RERECORD_EPISODE}` + `is_intervention`
- register via `@TeleoperatorConfig.register_subclass("dualsense_sim")`
- env var `SDL_VIDEODRIVER=dummy` to survive headless (tested already)
- headless-mode fallback: if no controller found, detect env var `LEROBOT_SIM_TELEOP=keyboard` → use existing fork keyboard teleop OR emit zero action + no events (for CI)

RC10 mods (user-permitted):
- add `is_available()` classmethod that returns False if no joystick found (raises today)
- small wrapper in RC10_control to make PS4Joystick constructor non-fatal — OR do it via a thin subclass in lerobot side. Prefer subclass to keep RC10 clean.

### E. faster-than-real-time support
- sim already has `mode='fast'`. Expose `fps_throttle: bool = True` in HILSerlProcessorConfig extension.
  - when `fps_throttle=False`, AssemblingEnv instantiated w/ `mode='fast'` — no per-step sleep → actor drives env as fast as GPU+CPU allow
  - when True (default), use `mode='realtime'` at env's own `control_hz`
- actor's `precise_sleep(1/fps)` must be skipped when fast. Patch actor.py or expose cfg (add `actor.realtime=False`).

### F. 3 reward-model JSON env configs
new files:
- `src/lerobot/rl/sim_manual_env.json` — `reward_model.type=manual`, demo-record + actor uses teleop SUCCESS btn
- `src/lerobot/rl/sim_cnn_env.json` — `reward_model.type=cnn`, pretrained_path pointer + device
- `src/lerobot/rl/sim_sarm_env.json` — `reward_model.type=sarm`, task="assemble_nesting", reward_mode=delta, stats_dataset_repo_id

training config mirror `train_hil_serl_env.json` but w/ `env.name="sim_assembling"` + `env.task="AssembleBase-v0"` + env features (state shape=[15], 2 images 128x128, action shape=[3]).

### G. pyproject adjustments
fork's pyproject.toml: add optional extra:
```
sim = ["simulator-for-il-rl @ file:///${PROJECT_ROOT}/../simulator_for_IL_RL", "rc10-api @ file:///${PROJECT_ROOT}/../RC10_control"]
```
or just document manual install (`uv pip install -e ../simulator_for_IL_RL ../RC10_control`) — user's monorepo-ish layout suggests editable-deps w/o hard pin is fine.

## flow: 3 scenarios

### manual
1. record demos w/ sim_manual_env.json (control_mode=dualsense_sim). press Triangle for success.
2. launch actor+learner w/ same env cfg (reward comes from teleop SUCCESS via InterventionActionProcessorStep).

### binary classifier
1. record success + failure demos separately
2. (optional) crop/resize already @128x128 in adapter
3. split_dataset for classifier train/val
4. lerobot-train-reward-classifier → checkpoint
5. set `reward_model.pretrained_path` in sim_cnn_env.json
6. actor+learner

### SARM
1. success + failure demos
2. prepare_sarm_data → merged+annotated+split
3. lerobot-train (policy=sarm, annotation_mode=dense_only)
4. relabel offline buffer (lerobot-relabel-sarm --reward-mode delta)
5. set `reward_model.pretrained_path` + reward_mode=delta + stats_dataset_repo_id
6. actor+learner

## risks / gotchas

- nested obs flatten: AssemblingEnv returns obs dict keyed `state`/`objects`/`images`. wrapper must emit `observation.images.front`, `observation.images.wrist`, `observation.state` (flat). do NOT rely on GymHILAdapterProcessorStep which is tuned for gym_hil's `pixels`/`agent_pos`.
- action convention: fork's gym_hil branch expects 3D delta (task-space). our env takes 8D task-space absolute [xyz, quat, gripper]. wrapper converts delta -> accumulated ref pose -> abs task-space.
- IK solver: sim's PinKinematics handles IK inside env.step. adapter must NOT duplicate; only maintain a persistent ee ref pose between steps.
- initial pose: reset returns ee_pos in world frame; wrapper's ref pose must be captured from obs at reset, not hardcoded.
- quaternion fixed during delta control: task is vertical pick — pose-change not needed. document this constraint.
- mujoco rendering headless → `MUJOCO_GL=egl` required. add to scripts/env or detect.
- pygame headless: `SDL_VIDEODRIVER=dummy` for CI tests, skip teleop init entirely if no controller.
- image dim mismatch: env emits 240x320 → must resize 128x128 at wrapper OR via `ImageCropResizeProcessorStep`. prefer wrapper for speed.

## deliverables

code (src/):
- src/lerobot/envs/sim_assembling.py (gym registration + factory + adapter wrapper)
- src/lerobot/teleoperators/dualsense_sim/{__init__.py, config_dualsense_sim.py, teleop_dualsense_sim.py} (teleop adapter)
- src/lerobot/rl/gym_manipulator.py (add sim_assembling branch to make_robot_env + make_processors)
- src/lerobot/rl/sim_{manual,cnn,sarm}_env.json (record/actor/learner env cfgs)
- src/lerobot/rl/sim_{manual,cnn,sarm}_train.json (training cfgs)

tests (tests/, not committed):
- tests/envs/test_sim_assembling.py (gym registry, obs/action reshape)
- tests/teleoperators/test_dualsense_sim_headless.py (headless fake; no controller required)
- tests/rl/test_sim_reward_wiring.py (make_processors dispatch on sim env)
- tests/rl/test_sim_three_scenarios_smoke.py (short rollout w/ each reward_model type)
- tests/fixtures/sim_synth_dataset.py (generate small success + failure demos for classifier/SARM training smoke)

docs (not committed):
- docs/port/2026-04-22-sim-hilserl-commands.md (end-to-end cmd list — extend prior commands.md for sim specifically)

## non-goals

- modifying simulator_for_IL_RL
- porting side/state cameras
- running real robot (hw is out of scope)
- hydra configs (fork uses JSON+draccus)
- achieving production-quality RL convergence (smoke only)
