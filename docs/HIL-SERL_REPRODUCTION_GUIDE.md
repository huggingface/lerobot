# HIL-SERL Reproduction Guide: Simulation to Real SO-ARM 101

> Full guide for reproducing Human-in-the-Loop Sample-Efficient Robotic RL on this Mac (Apple Silicon), then deploying to a real SO-ARM 101 robot.

---

## Phase 1: Environment Setup (COMPLETED)

### 1.1 Fork & Clone

```bash
# Fork huggingface/lerobot to your GitHub account
gh repo fork huggingface/lerobot --clone=false

# Local clone at /Users/studyandwork/project/lerobot
cd /Users/studyandwork/project/lerobot
git remote rename origin upstream
git remote add origin https://github.com/LIJianxuanLeo/lerobot.git
```

### 1.2 Conda Environment

```bash
# Already created: hilserl env with Python 3.12
conda activate hilserl

# Install lerobot with HIL-SERL extras
cd /Users/studyandwork/project/lerobot
pip install -e ".[hilserl]"

# Install FFmpeg for video encoding
conda install -c conda-forge ffmpeg

# Downgrade transformers for ResNet10 compatibility
pip install "transformers>=4.45,<5.0"
```

### 1.3 Verified Packages

| Package | Version | Purpose |
|---------|---------|---------|
| lerobot | 0.5.1 | Core framework |
| torch | 2.10.0 | Deep learning |
| jax | 0.9.2 | JAX backend |
| mujoco | 3.6.0 | Physics simulation |
| gym-hil | 0.1.13 | Gymnasium HIL environments |
| grpcio | 1.73.1 | Actor-learner communication |
| ffmpeg | 8.0.1 | Video encoding |

---

## Phase 2: Simulation Training (COMPLETED)

### 2.1 Configuration Files

Location: `configs/sim/`

**env_config.json** - Environment config for recording/replay:
```json
{
    "env": {
        "name": "gym_hil",
        "task": "PandaPickCubeKeyboard-v0",
        "fps": 10,
        "robot": null,
        "teleop": null,
        "processor": {
            "control_mode": "keyboard",
            "gripper": {
                "use_gripper": true,
                "gripper_penalty": -0.02
            },
            "reset": {
                "fixed_reset_joint_positions": [0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785],
                "reset_time_s": 2.0,
                "control_time_s": 15.0,
                "terminate_on_success": true
            }
        }
    },
    "dataset": {
        "repo_id": "LIJianxuanLeo/sim_pick_cube",
        "root": null,
        "task": "PandaPickCubeKeyboard-v0",
        "num_episodes_to_record": 10,
        "replay_episode": 0,
        "push_to_hub": false
    },
    "mode": null,
    "device": "cpu"
}
```

**train_config.json** - Training config (see `configs/sim/train_config.json` for full version):
- Policy: SAC with ResNet10 vision encoder
- Device: `mps` (Apple Silicon GPU)
- Storage device: `cpu` (replay buffer on CPU to avoid MPS memory limits)
- Dataset: `aractingi/franka_sim_pick_lift_5` (official 30-episode demos)
- W&B: disabled (set `enable: true` if you have W&B configured)

### 2.2 macOS-Specific Notes

- **Must use `mjpython` instead of `python`** for any script that opens a MuJoCo viewer window
  ```bash
  # Location: /Users/studyandwork/miniconda3/envs/hilserl/bin/mjpython
  mjpython scripts/auto_record_demos.py
  ```
- **Accessibility permissions**: Terminal app needs Accessibility access for keyboard control
  - System Settings -> Privacy & Security -> Accessibility -> Add Terminal/iTerm2
- **MPS GPU**: Apple Silicon MPS works for policy forward/backward pass, but replay buffer must stay on CPU (18+ GiB too large for MPS)

### 2.3 Available Simulation Environments

| Task ID | Control | Viewer |
|---------|---------|--------|
| `PandaPickCubeBase-v0` | None (API only) | No |
| `PandaPickCubeKeyboard-v0` | Keyboard | Yes |
| `PandaPickCubeGamepad-v0` | Gamepad | Yes |

**Keyboard controls (PandaPickCubeKeyboard-v0):**
- Arrow keys: Move X-Y
- Shift / Shift_R: Move Z
- Right Ctrl / Left Ctrl: Open/close gripper
- Enter: End episode SUCCESS
- Backspace: End episode FAILURE
- Space: Toggle intervention
- ESC: Exit

### 2.4 Demo Dataset

**Official dataset used**: `aractingi/franka_sim_pick_lift_5`
- 30 episodes, 558 frames, v3.0 format, 10 FPS
- 100% success rate (all episodes end with reward=1.0)
- Features: front camera (128x128), wrist camera (128x128), state (18D), action (4D)
- Actions are 4D: `[dx, dy, dz, gripper]` - EE deltas + gripper command
- Average episode length: 18.6 frames (~1.9 seconds)

### 2.5 Running Simulation Training

```bash
conda activate hilserl
cd /Users/studyandwork/project/lerobot

# Terminal 1: Start learner (trains SAC policy)
python -m lerobot.rl.learner --config_path configs/sim/train_config.json

# Terminal 2: Start actor (runs policy in MuJoCo, collects data)
mjpython -m lerobot.rl.actor --config_path configs/sim/train_config.json
```

**Prevent macOS sleep during training:**
```bash
caffeinate -dims -w <LEARNER_PID> &
```

**Monitor training:**
```bash
tail -f /tmp/learner.log
tail -f /tmp/actor.log
```

### 2.6 Simulation Training Performance (Apple Silicon)

| Metric | Value |
|--------|-------|
| Actor speed | ~455 steps/min |
| Learner optimization rate | ~0.17 Hz (~10 opt/min) |
| Time to 100k learner steps | ~158 hours (CPU-bound) |
| Time to 20k learner steps (minimum useful) | ~32 hours |

**Recommendation**: Use a cloud GPU (NVIDIA) for the learner to get 10-50x speedup. The actor can remain on the Mac.

### 2.7 Output & Checkpoints

- Output directory: `outputs/train/<date>/<time>_franka_sim_sac/`
- Checkpoints saved every 20,000 steps
- Logs: `outputs/train/.../logs/`

---

## Phase 3: Real SO-ARM 101 Deployment

### 3.1 Hardware Requirements

| Component | Quantity | Notes |
|-----------|----------|-------|
| SO-ARM 101 Follower | 1 | Robot being trained (6-DOF + gripper) |
| SO-ARM 101 Leader | 1 | For human teleoperation interventions |
| USB-C cables | 2 | One per arm |
| Power supply (5V or 12V) | 2 | 5V for standard, 12V for Pro follower |
| USB camera (gripper-mounted) | 1 | For wrist observations |
| USB camera (external/overhead) | 1 | For front/side observations |
| Desk lamp | 1 | **Critical** for consistent lighting |
| Stable table + clamps | 1 | Prevent arm movement |
| Small cube/object | 1 | Task object |

**Important**: USB does NOT power the arms. Both USB cable AND power supply must be connected simultaneously.

### 3.2 Workspace Setup

1. **Mount arms** on stable surface, ~50cm apart to prevent collisions
2. **Clamp both arms** securely to the table
3. **Mount cameras**:
   - Gripper camera: attach to follower's wrist/gripper
   - External camera: overhead or angled front view of workspace
4. **Desk lamp**: Position as dominant light source over workspace
   - Consistent lighting is **critical** for the reward classifier
   - Shadows and ambient light changes confuse the vision-based reward
5. **Task object**: Place a small cube in the workspace center

### 3.3 Step 1: Find USB Ports

```bash
# List connected USB devices
ls /dev/tty.usbmodem*

# Example output:
# /dev/tty.usbmodem58760431541  (follower)
# /dev/tty.usbmodem58760431551  (leader)
```

Note which port corresponds to which arm.

### 3.4 Step 2: Calibrate Arms

```bash
conda activate hilserl
cd /Users/studyandwork/project/lerobot

# Calibrate follower arm
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX \
  --robot.id=my_follower

# Calibrate leader arm
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemYYYY \
  --teleop.id=my_leader
```

**Calibration process:**
1. Move robot to middle position for all joints
2. Press Enter
3. Move each joint through its full range of motion
4. Settings saved to `~/.cache/huggingface/lerobot/calibration/`

**If calibration fails** (motor check errors):
- Check motor LED status: blinking = failed motor, steady ON = healthy
- Delete cached calibration and retry: `rm -rf ~/.cache/huggingface/lerobot/calibration/`
- Replace any malfunctioning motors

### 3.5 Step 3: Find Joint Limits & Workspace Bounds

```bash
lerobot-find-joint-limits \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodemXXXX \
  --robot.id=my_follower \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodemYYYY \
  --teleop.id=my_leader
```

This outputs min/max end-effector positions and joint angles. Example:
```
Max ee position [0.2417 0.2012 0.1027]
Min ee position [0.1663 -0.0823 0.0336]
```

**Save these values** — they go into the training config's `end_effector_bounds`.

> **Known Issue**: SO-101 EE bounds can be very small, limiting human intervention range. Widen the bounds manually if needed.

### 3.6 Step 4: Record Demonstrations (20-30 episodes)

Create a real robot env config (e.g., `configs/real/env_config.json`):

```json
{
    "env": {
        "name": "real_robot",
        "task": "pick_cube",
        "fps": 10,
        "robot": {
            "type": "so100_follower",
            "port": "/dev/tty.usbmodemXXXX",
            "id": "my_follower",
            "cameras": {
                "front": {"type": "opencv", "index": 0, "width": 640, "height": 480},
                "wrist": {"type": "opencv", "index": 1, "width": 640, "height": 480}
            }
        },
        "teleop": {
            "type": "so100_leader",
            "port": "/dev/tty.usbmodemYYYY",
            "id": "my_leader"
        },
        "processor": {
            "control_mode": "leader",
            "gripper": {
                "use_gripper": true,
                "gripper_penalty": -0.02
            },
            "image_preprocessing": {
                "crop_params_dict": {},
                "resize_size": [128, 128]
            },
            "reset": {
                "control_time_s": 15.0,
                "terminate_on_success": true
            }
        }
    },
    "dataset": {
        "repo_id": "LIJianxuanLeo/real_pick_cube",
        "task": "pick_cube",
        "num_episodes_to_record": 20,
        "push_to_hub": false
    },
    "mode": "record",
    "device": "cpu"
}
```

```bash
python -m lerobot.rl.gym_manipulator --config_path configs/real/env_config.json
```

**During recording:**
- Use leader arm to teleoperate follower
- Complete task in 5-10 seconds
- Press `s` for success, `Esc` for failure
- Press `Space` to toggle leader control
- Record 20-30 successful episodes

### 3.7 Step 5: Process Dataset (ROI Crop)

```bash
python -m lerobot.rl.crop_dataset_roi --repo-id LIJianxuanLeo/real_pick_cube
```

This opens an interactive tool:
1. Select region of interest around workspace for each camera
2. Exclude background clutter (walls, other objects)
3. Output: crop parameters dictionary

Update your config's `crop_params_dict` with the output:
```json
"crop_params_dict": {
    "observation.images.front": [180, 250, 120, 150],
    "observation.images.wrist": [180, 207, 180, 200]
}
```

### 3.8 Step 6: Train Reward Classifier (Optional but Recommended)

1. **Collect labeled data** with `terminate_on_success: false` to capture both success and failure frames
2. **Train classifier**:
   ```bash
   lerobot-train --config_path configs/real/reward_classifier_config.json
   ```
3. **Deploy** by setting `pretrained_path` in the processor config:
   ```json
   "reward_classifier": {
       "pretrained_path": "outputs/reward_classifier/checkpoint/",
       "success_threshold": 0.5,
       "success_reward": 1.0
   }
   ```

### 3.9 Step 7: Real-World RL Training (Actor-Learner)

Create training config `configs/real/train_config.json` with:
- Same structure as sim train config
- Real robot and teleop settings
- Learned reward classifier path
- EE bounds from Step 3
- Crop params from Step 5

```bash
# Terminal 1: Learner (ideally on a GPU machine)
python -m lerobot.rl.learner --config_path configs/real/train_config.json

# Terminal 2: Actor (on the Mac connected to the robot)
python -m lerobot.rl.actor --config_path configs/real/train_config.json
```

### 3.10 Human Intervention Strategy

| Phase | What to do |
|-------|-----------|
| First 5 episodes | Let the policy explore freely, don't intervene |
| Episodes 5-20 | Provide **short, corrective** nudges via leader arm |
| Episodes 20+ | Reduce interventions, let policy handle most of it |

**Key rules:**
- Press `Space` to take over, `Space` again to release
- **Do NOT intervene all the way to success** — just nudge the robot in the right direction, then release
- Extended takeovers cause Q-function overestimation and hurt learning
- Press `s` for success, `Esc` for failure annotation

### 3.11 Hyperparameter Tuning

| Parameter | Default | Recommendation |
|-----------|---------|---------------|
| `temperature_init` | 0.01 | Increase to 0.1 if too little exploration |
| `policy_parameters_push_frequency` | 4s | Reduce to 1-2s for faster policy updates |
| `discount` | 0.97 | Keep as-is |
| `utd_ratio` | 2 | Reduce to 1 if learner is too slow |
| `online_step_before_learning` | 100 | Keep as-is |

### 3.12 Expected Results

| Metric | Target |
|--------|--------|
| Training time | 1-3 hours of active monitoring |
| Grasp success rate | ~70%+ from scratch |
| Intervention rate | Should decrease over time |

---

## Known Issues & Workarounds

| Issue | Workaround |
|-------|-----------|
| SO-101 EE bounds too small ([#1387](https://github.com/huggingface/lerobot/issues/1387)) | Manually widen bounds after `find-joint-limits` |
| Keyboard teleop broken in LeRobot 0.4.1+ | Use leader arm (not keyboard) for real robot |
| Calibration fails with motor check errors ([#1694](https://github.com/huggingface/lerobot/issues/1694)) | Check LEDs, replace bad motors, delete calibration cache |
| `transformers` 5.x breaks ResNet10 loading | Pin `transformers>=4.45,<5.0` |
| MPS replay buffer too large (18+ GiB) | Set `storage_device: "cpu"` |
| `gripper_penalty_in_reward` not valid in current version | Remove from config |
| `use_amp`, `online_env_seed`, `repo_id` (in policy) not valid | Remove from config |
| MuJoCo viewer requires `mjpython` on macOS | Use `mjpython` instead of `python` |
| Training slow on Apple Silicon | Use cloud GPU for learner process |

---

## Key References

- HIL-SERL Paper: https://hil-serl.github.io/
- LeRobot HIL-SERL Docs: https://huggingface.co/docs/lerobot/en/hilserl
- LeRobot Sim Training: https://huggingface.co/docs/lerobot/en/hilserl_sim
- SO-101 Real Grasping Blog: https://ggando.com/blog/so101-hil-serl/
- Community SO-101 Fork: https://github.com/ubi-coro/lerobot-hil-serl
- Original HIL-SERL Repo: https://github.com/rail-berkeley/hil-serl

---

## File Structure

```
lerobot/
├── configs/
│   ├── sim/
│   │   ├── env_config.json          # Sim environment config
│   │   ├── record_config.json       # Sim recording config
│   │   └── train_config.json        # Sim training config (SAC + MPS)
│   └── real/                        # Phase 3: Real robot configs
│       ├── env_config.json          # Real robot env + recording config
│       └── train_config.json        # Real robot SAC training config
├── SO101/
│   └── so101_new_calib.urdf         # SO-101 URDF for inverse kinematics
├── scripts/
│   └── auto_record_demos.py         # Automated demo recording script
├── outputs/
│   └── train/                       # Training checkpoints & logs
└── docs/
    ├── HIL-SERL_REPRODUCTION_GUIDE.md  # This file
    └── SIM_TO_REAL_PLAN.md             # Detailed sim-to-real deployment plan
```
