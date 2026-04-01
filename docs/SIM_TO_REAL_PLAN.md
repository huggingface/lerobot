# HIL-SERL Sim-to-Real Plan: SO-ARM 101 Pick-and-Place

## Context

Phase 1-2 (simulation) are complete: environment verified, configs created, demo dataset loaded (`aractingi/franka_sim_pick_lift_5`), actor-learner training tested on Apple Silicon. Simulation training is too slow on CPU/MPS (~158 hours for 100k steps) — practical training requires a GPU machine.

**Goal:** Deploy HIL-SERL on a real SO-ARM 101 robot for a pick-cube task, achieving ~70%+ grasp success rate within 1-3 hours of active training.

**Current state:**
- Fork: `LIJianxuanLeo/lerobot` on GitHub
- Local clone: `/Users/studyandwork/project/lerobot`
- Conda env `hilserl`: Python 3.12, PyTorch 2.10, lerobot 0.5.1
- Sim configs: `configs/sim/{env_config,train_config,record_config}.json`
- Guide: `docs/HIL-SERL_REPRODUCTION_GUIDE.md`

**Important:** HIL-SERL does **NOT** do direct sim-to-real policy transfer. The sim phase validates the training pipeline works. Real training starts from scratch on the real robot.

---

## Phase 3: Real SO-ARM 101 Deployment

### Step 1: Hardware Setup & Assembly

**Required hardware:**

| Item | Quantity | Notes |
|------|----------|-------|
| SO-ARM 101 kit | 2 | Leader + Follower, each with 6x STS3215 servos |
| USB cables | 2 | Leader + Follower → Mac |
| USB cameras | 2 | Front view + wrist-mounted |
| Small cube | 1 | ~4cm for grasping task |
| Stable desk/surface | 1 | Rigid mounting surface |
| Desk lamp | 1 | Consistent lighting (critical for reward classifier) |
| Power supply | 1 | 5V, each arm ~2A |

**Physical setup:**
1. Assemble both arms following SO-101 guide: `docs/source/so101.mdx`
2. Mount follower and leader ~50cm apart on stable surface
3. Mount front camera ~40cm away, angled down at workspace
4. Mount wrist camera on follower's wrist (if using wrist view)
5. Place cube in center of follower's reachable workspace
6. Set up consistent desk lamp lighting (critical for reward classifier)

---

### Step 2: Software Setup & Port Discovery

```bash
conda activate hilserl

# Find USB ports for both arms
ls /dev/tty.usbmodem*
# Example output: /dev/tty.usbmodem58760432981  /dev/tty.usbmodem58760434471

# Or use lerobot-find-port for easier identification
lerobot-find-port
```

**Motor setup (only needed once per new arm):**
```bash
lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX
```

---

### Step 3: Calibrate Both Arms

```bash
# Calibrate follower
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemFOLLOWER \
  --robot.id=my_follower

# Calibrate leader
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemLEADER \
  --teleop.id=my_leader
```

**Process:** Disable torque → move to middle → sweep through full range → save calibration to `~/.cache/huggingface/lerobot/calibration/`

---

### Step 4: Verify Teleoperation

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemFOLLOWER \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemLEADER \
  --teleop.id=my_leader \
  --display-cameras 1
```

**Verify:** Leader arm movements are mirrored by follower. Camera feeds display correctly.

---

### Step 5: Find Joint Limits & EE Workspace Bounds

```bash
lerobot-find-joint-limits \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemFOLLOWER \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemLEADER \
  --teleop.id=my_leader \
  --urdf_path=SO101/so101_new_calib.urdf \
  --target_frame_name=gripper_frame_link \
  --teleop_time_s=30
```

**During this step:**
- Move the leader arm through the full desired workspace
- Focus on the area where the cube will be placed
- The tool outputs `min_ee` and `max_ee` bounds — **save these values**

**Known issue:** SO-101 EE bounds may be too small ([#1387](https://github.com/huggingface/lerobot/issues/1387)). Manually widen the bounds by ~20% in each direction.

**Output needed:**
```json
"end_effector_bounds": {
    "min": [x_min, y_min, z_min],
    "max": [x_max, y_max, z_max]
}
```

Also record a safe reset joint position (arm raised above table):
```json
"fixed_reset_joint_positions": [0.0, -30.0, 90.0, -60.0, 0.0, 0.0]
```
(These are degrees; adjust by moving arm to desired reset pose and reading joint positions)

---

### Step 6: Download URDF for Inverse Kinematics

```bash
mkdir -p SO101
curl -L -o SO101/so101_new_calib.urdf \
  https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf
```

---

### Step 7: Create Real Robot Environment Config

Create `configs/real/env_config.json` — see the template in the configs directory.

**Key parameters to customize:**
- `robot.port` — your follower's USB port
- `teleop.port` — your leader's USB port
- `cameras` — camera indices (verify with OpenCV)
- `end_effector_bounds` — values from Step 5
- `fixed_reset_joint_positions` — safe pose from Step 5

**Config structure overview:**

| Field | Value | Notes |
|-------|-------|-------|
| `observation.state` shape | `[9]` | 6 joint positions + 3 EE position |
| `action` shape | `[4]` | [delta_x, delta_y, delta_z, gripper] |
| `control_mode` | `"leader"` | Uses leader arm for teleoperation |
| `fps` | `10` | Control frequency |
| `image size` | `[128, 128]` | After crop + resize |

---

### Step 8: Record Demonstrations (20-30 episodes)

```bash
python -m lerobot.rl.gym_manipulator --config_path configs/real/env_config.json
```

**Recording protocol:**
1. Robot resets to safe position automatically
2. Use leader arm to teleoperate follower to pick up the cube
3. Complete each pick in 5-10 seconds
4. Press `s` to mark success, `Esc` to discard failed attempt
5. Press `Space` to toggle leader control on/off
6. Record **25 successful episodes** minimum

**Tips for good demos:**
- Approach from above, lower slowly
- Close gripper firmly around cube center
- Lift straight up after grasping
- Keep movements smooth and consistent
- Maintain consistent cube placement between episodes

---

### Step 9: Process Dataset — ROI Crop

```bash
python -m lerobot.rl.crop_dataset_roi --repo-id LIJianxuanLeo/real_pick_cube
```

**Interactive tool:**
1. For each camera, draw a rectangle around the workspace area
2. **Exclude** background clutter (walls, other objects, cables)
3. **Include** the cube, gripper, and immediate workspace
4. Press `c` to confirm, `r` to reset
5. Output: crop parameters `(top, left, height, width)` per camera

Update `configs/real/env_config.json` with the crop parameters:
```json
"crop_params_dict": {
    "observation.images.front": [100, 150, 280, 340],
    "observation.images.wrist": [80, 120, 320, 400]
}
```
(Replace with actual values from the tool)

---

### Step 10: Train Reward Classifier

**Purpose:** Vision-based binary classifier that predicts "is the cube successfully grasped and lifted?" — used to auto-terminate episodes and provide reward signal during RL training.

**Reference:** `examples/tutorial/rl/reward_classifier_example.py`

**Steps:**
1. Record additional episodes with `terminate_on_success: false` to capture full trajectories including both success and failure frames
2. Label frames: success frames (cube lifted) vs failure frames (cube on table)
3. Train classifier:
   ```bash
   python -m lerobot.rl.train_reward_classifier \
     --repo-id LIJianxuanLeo/real_pick_cube \
     --output-dir outputs/reward_classifier
   ```
4. Test classifier accuracy on held-out data — should be >95% for reliable training

**Alternative:** Skip reward classifier initially and use manual `s`/`Esc` annotations during training (slower but simpler).

---

### Step 11: Create Real Robot Training Config

Create `configs/real/train_config.json` — see the template in the configs directory.

**Key SAC hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `discount` | 0.97 | Reward discount factor |
| `temperature_init` | 0.01 | Entropy temperature (increase for more exploration) |
| `critic_lr` / `actor_lr` | 3e-4 | Learning rates |
| `utd_ratio` | 2 | Update-to-data ratio |
| `online_step_before_learning` | 100 | Warm-up steps before training starts |
| `online_buffer_capacity` | 100000 | Online replay buffer size |
| `offline_buffer_capacity` | 100000 | Offline (demo) buffer size |

**Actor-Learner config:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learner_host` | `127.0.0.1` | Learner IP (change for remote GPU) |
| `learner_port` | `50051` | gRPC port |
| `policy_parameters_push_frequency` | 4 | Seconds between policy syncs |

---

### Step 12: Run Real-World RL Training (Actor-Learner)

#### Option A: Both on Mac (slow but simple)
```bash
# Terminal 1: Learner
python -m lerobot.rl.learner --config_path configs/real/train_config.json

# Terminal 2: Actor (connected to robot)
python -m lerobot.rl.actor --config_path configs/real/train_config.json
```

#### Option B: Learner on GPU server, Actor on Mac (recommended)
```bash
# On GPU server (e.g., cloud GPU with CUDA):
# 1. Clone repo, install lerobot
# 2. Change config: "device": "cuda", learner_host: "0.0.0.0"
python -m lerobot.rl.learner --config_path configs/real/train_config.json

# On Mac (connected to robot):
# Change config: learner_host: "<GPU_SERVER_IP>"
python -m lerobot.rl.actor --config_path configs/real/train_config.json
```

**Network requirements for Option B:**
- Both machines on same network (or port forwarding for port 50051)
- Learner needs GPU (NVIDIA recommended)
- Actor needs USB access to robot + cameras

---

### Step 13: Human Intervention Strategy

| Phase | Episodes | What to Do |
|-------|----------|-----------|
| Exploration | 1-5 | Let policy explore freely, **do NOT intervene** |
| Light guidance | 5-20 | Provide **short corrective nudges** via leader arm |
| Hands-off | 20+ | Reduce interventions, let policy handle most tasks |

**Intervention controls:**
- `Space` — toggle leader control (take over / release)
- `s` — mark episode as success
- `Esc` — mark episode as failure / discard

**Critical rules:**
- **Short nudges only** — intervene for 1-2 seconds, then release
- **Never complete the whole task** — just correct the trajectory direction
- Extended takeovers cause Q-function overestimation and hurt learning
- Aim for intervention rate to decrease over time

---

### Step 14: Monitor & Tune

**Key metrics to watch:**
- Episode reward (should increase over time)
- Intervention rate (should decrease)
- Policy frequency Hz (should be ~10 Hz)
- Success rate over last 10 episodes

**Hyperparameter tuning:**

| Parameter | Default | When to Change |
|-----------|---------|---------------|
| `temperature_init` | 0.01 | → 0.1 if robot barely moves (too little exploration) |
| `policy_parameters_push_frequency` | 4s | → 1-2s for faster policy updates to actor |
| `discount` | 0.97 | Keep as-is |
| `utd_ratio` | 2 | → 1 if learner is bottleneck |
| `critic_lr` / `actor_lr` | 3e-4 | → 1e-4 if training unstable |
| `gripper_penalty` | -0.02 | → -0.05 if gripper toggles too much |

---

## Sim vs Real: Key Differences

| Aspect | Simulation | Real Robot |
|--------|-----------|------------|
| Robot | Franka Panda + Robotiq 2F-85 | SO-ARM 101 (6-DOF + gripper) |
| Control | Keyboard (3D delta) | Leader arm (4D EE delta + IK) |
| Observation | MuJoCo renders + 18D state | USB cameras + 6 joints + 3 EE |
| Action space | 3D (dx, dy, dz) | 4D (dx, dy, dz, gripper) |
| Reward | Physics-based (cube height) | Learned classifier (vision) |
| Reset | Automatic (MuJoCo reset) | Automatic (servo trajectory) |
| Training speed | ~158h for 100k steps (MPS) | Real-time (~10 Hz) |

**The simulation is valuable for:**
1. Verifying the actor-learner pipeline works end-to-end
2. Testing config format and parameters
3. Building familiarity with the training loop
4. Tuning hyperparameters before expensive real-robot time

---

## Expected Results

| Metric | Target |
|--------|--------|
| Demo recording time | ~30 minutes for 25 episodes |
| Training time | 1-3 hours of active monitoring |
| Grasp success rate | ~70%+ from scratch |
| Intervention rate at end | <10% |

---

## Known Issues & Workarounds

| Issue | Workaround |
|-------|-----------|
| SO-101 EE bounds too small ([#1387](https://github.com/huggingface/lerobot/issues/1387)) | Manually widen bounds by ~20% after find-joint-limits |
| Keyboard teleop broken (0.4.1+) | Use leader arm (`control_mode: "leader"`) |
| Calibration motor errors ([#1694](https://github.com/huggingface/lerobot/issues/1694)) | Check LEDs, replace bad motors, delete calibration cache |
| `transformers` 5.x breaks ResNet10 | Pin `transformers>=4.45,<5.0` |
| Training slow on Apple Silicon | Use cloud GPU for learner (actor stays on Mac) |
| Reward classifier lighting sensitive | Consistent desk lamp, tight ROI crop |
| IK solver convergence failures | Widen EE bounds, reduce step sizes |
| Robot jerky movements | Reduce `end_effector_step_sizes`, increase `reset_time_s` |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    LEARNER (GPU Server)                  │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ Online      │    │ SAC Policy   │    │ Offline   │  │
│  │ Replay      │───▶│ (Actor +     │◀───│ Replay    │  │
│  │ Buffer      │    │  Critic +    │    │ Buffer    │  │
│  │ (all data)  │    │  Temperature)│    │ (demos +  │  │
│  └─────────────┘    └──────┬───────┘    │ interv.)  │  │
│                            │            └───────────┘  │
│                     gRPC (policy params)               │
│                            │                            │
└────────────────────────────┼────────────────────────────┘
                             │
                      ┌──────┴──────┐
                      │   Network   │
                      └──────┬──────┘
                             │
┌────────────────────────────┼────────────────────────────┐
│                    ACTOR (Mac + Robot)                   │
│                            │                            │
│  ┌─────────────┐    ┌──────┴───────┐    ┌───────────┐  │
│  │ SO-ARM 101  │◀───│ Policy       │◀───│ Reward    │  │
│  │ Follower    │    │ Inference    │    │ Classifier│  │
│  │ (6-DOF)     │    │ + IK Solver  │    │ (Vision)  │  │
│  └──────┬──────┘    └──────┬───────┘    └───────────┘  │
│         │                  │                            │
│  ┌──────┴──────┐    ┌──────┴───────┐                   │
│  │ USB Cameras │    │ SO-ARM 101   │                   │
│  │ (front +    │    │ Leader       │                   │
│  │  wrist)     │    │ (human       │                   │
│  └─────────────┘    │  intervention)│                   │
│                     └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

**Data flow:**
1. Actor runs policy on robot, collects transitions (obs, action, reward, next_obs)
2. Transitions sent to learner via gRPC
3. Learner trains SAC on combined online + offline data
4. Updated policy parameters pushed back to actor every N seconds
5. Human can intervene via leader arm — intervention transitions go to offline buffer

---

## Action Processing Pipeline (Real Robot)

```
Policy Output: [delta_x, delta_y, delta_z, gripper]  (4D tensor)
        ↓
MapTensorToDeltaActionDictStep
        ↓
{delta_x, delta_y, delta_z, gripper}
        ↓
InterventionActionProcessorStep  ← Human override via leader arm
        ↓
EEReferenceAndDelta  ← FK(current_joints) + delta → target_EE
        ↓
EEBoundsAndSafety  ← Clip to workspace, prevent large jumps
        ↓
GripperVelocityToJoint  ← Integrate gripper velocity
        ↓
InverseKinematicsRLStep  ← IK(target_EE) → joint_targets
        ↓
Robot Hardware: FeetechMotorsBus → STS3215 servos
```

---

## Verification Checklist

- [ ] Both arms assembled, USB connected, ports identified
- [ ] Both arms calibrated (`~/.cache/huggingface/lerobot/calibration/`)
- [ ] Leader-follower teleoperation works smoothly
- [ ] Camera feeds display correctly
- [ ] Joint limits and EE bounds recorded
- [ ] URDF downloaded
- [ ] 25+ successful demo episodes recorded
- [ ] ROI crop parameters determined
- [ ] Reward classifier trained (>95% accuracy)
- [ ] Actor-learner training loop running
- [ ] Success rate increasing over episodes
- [ ] Intervention rate decreasing over episodes

---

## Key References

- HIL-SERL Paper: https://hil-serl.github.io/
- LeRobot HIL-SERL Docs: https://huggingface.co/docs/lerobot/en/hilserl
- LeRobot Sim Training: https://huggingface.co/docs/lerobot/en/hilserl_sim
- SO-101 Real Grasping Blog: https://ggando.com/blog/so101-hil-serl/
- Community SO-101 Fork: https://github.com/ubi-coro/lerobot-hil-serl
- Original HIL-SERL Repo: https://github.com/rail-berkeley/hil-serl

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/lerobot/rl/actor.py` | Actor process — runs policy on robot |
| `src/lerobot/rl/learner.py` | Learner process — trains SAC policy |
| `src/lerobot/rl/gym_manipulator.py` | `make_robot_env()`, `RobotEnv`, `make_processors()` |
| `src/lerobot/envs/configs.py` | All config dataclasses |
| `src/lerobot/rl/crop_dataset_roi.py` | Interactive ROI cropping tool |
| `src/lerobot/robots/so_follower/robot_kinematic_processor.py` | EE control, IK, safety bounds |
| `src/lerobot/processor/hil_processor.py` | Intervention handling |
| `examples/tutorial/rl/hilserl_example.py` | Complete actor-learner example |
| `examples/tutorial/rl/reward_classifier_example.py` | Reward classifier training |
