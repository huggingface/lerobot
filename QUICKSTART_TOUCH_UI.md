# Quick Start: Bimanual SO-101 Touch UI

This guide will get your bimanual SO-101 setup running with the touch UI in minutes!

## What You Built

A complete graphical touch interface for your 4 SO-101 arms:
- **2 Follower arms** (execute actions)
- **2 Leader arms** (teleoperation input)
- **7-inch landscape touchscreen** (1024x600)
- **Bimanual teleoperation** (leaders control followers)
- **Dataset recording** for imitation learning

## Hardware Setup

1. **Connect the Arms**:
   ```
   Left Follower  → USB Port → /dev/ttyUSB0
   Right Follower → USB Port → /dev/ttyUSB1
   Left Leader    → USB Port → /dev/ttyUSB2
   Right Leader   → USB Port → /dev/ttyUSB3
   ```

2. **Check USB Connections**:
   ```bash
   ls -l /dev/ttyUSB*
   ```

3. **Fix Permissions** (if needed):
   ```bash
   sudo usermod -a -G dialout $USER
   # Log out and back in
   ```

## Installation

1. **Install LeRobot with Feetech Support**:
   ```bash
   cd /home/user/lerobot
   pip install -e ".[feetech]"
   ```

   This installs:
   - LeRobot core
   - Feetech motor SDK
   - pygame (for the UI)
   - All dependencies

## Launch the Touch UI

### Quick Launch

```bash
lerobot-touch-ui \
    --follower-left-port=/dev/ttyUSB0 \
    --follower-right-port=/dev/ttyUSB1 \
    --leader-left-port=/dev/ttyUSB2 \
    --leader-right-port=/dev/ttyUSB3
```

### Or use Python module directly

```bash
python -m lerobot.scripts.lerobot_touch_ui \
    --follower-left-port=/dev/ttyUSB0 \
    --follower-right-port=/dev/ttyUSB1 \
    --leader-left-port=/dev/ttyUSB2 \
    --leader-right-port=/dev/ttyUSB3
```

## First-Time Workflow

### Step 1: Motor Setup (MUST DO FIRST!)

Each SO-101 arm has 6 motors that need unique IDs:

1. Tap **"Motor Setup"** from main menu
2. Tap **"Setup Left Follower"**
3. Follow the console prompts:
   - Connect ONLY motor 1 (shoulder_pan)
   - Press ENTER
   - Connect ONLY motor 2 (shoulder_lift)
   - Press ENTER
   - ... repeat for all 6 motors
4. Repeat for other arms

**Motor IDs per arm**:
- Motor 1: shoulder_pan
- Motor 2: shoulder_lift
- Motor 3: elbow_flex
- Motor 4: wrist_flex
- Motor 5: wrist_roll
- Motor 6: gripper

### Step 2: Calibration

Set the range of motion for each joint:

1. Tap **"Calibration"** from main menu
2. Tap **"Calibrate Followers"**
3. Follow prompts:
   - Move arm to center position
   - Press ENTER
   - Move all joints through their full range
   - Press ENTER when done
4. Repeat for **"Calibrate Leaders"**

Calibration files are saved to: `~/.cache/lerobot/calibration/`

### Step 3: Connect Arms

1. Tap **"System Status"** from main menu
2. Tap **"Connect Followers"**
3. Tap **"Connect Leaders"**
4. Verify all show "Connected"

### Step 4: Teleoperation

Test bimanual control:

1. Tap **"Teleoperation"** from main menu
2. Tap **"START"**
3. Move the leader arms - followers will mirror!
4. Tap **"STOP"** to end

### Step 5: Record Datasets

Collect demonstration data:

1. Ensure teleoperation is working
2. Tap **"Record Dataset"** from main menu
3. Tap **"Start Recording"**
4. Perform your task demonstration
5. Data is saved automatically

## Touch UI Screens

### Main Menu
- **Motor Setup** - Configure motor IDs
- **Calibration** - Set joint ranges
- **Teleoperation** - Bimanual control
- **Record Dataset** - Data collection
- **System Status** - Connection info
- **Exit** - Close app

### All screens have:
- **Large touch buttons** (80px height)
- **Back button** (bottom left)
- **Clear status messages**
- **Error display**

## Using the API Directly

The touch UI uses the new bimanual SO-101 classes:

### Python Example

```python
from lerobot.robots.bi_so101_follower import BiSO101Follower, BiSO101FollowerConfig
from lerobot.teleoperators.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig

# Create follower
follower_config = BiSO101FollowerConfig(
    left_arm_port="/dev/ttyUSB0",
    right_arm_port="/dev/ttyUSB1",
    id="bimanual_follower"
)
follower = BiSO101Follower(follower_config)
follower.connect()

# Create leader
leader_config = BiSO101LeaderConfig(
    left_arm_port="/dev/ttyUSB2",
    right_arm_port="/dev/ttyUSB3",
    id="bimanual_leader"
)
leader = BiSO101Leader(leader_config)
leader.connect()

# Teleoperation loop
while True:
    action = leader.get_action()
    follower.send_action(action)
```

## Command-Line Examples

### Teleoperate (without UI)

```bash
lerobot-teleoperate \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/ttyUSB0 \
  --robot.right_arm_port=/dev/ttyUSB1 \
  --robot.id=bimanual_follower \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=/dev/ttyUSB2 \
  --teleop.right_arm_port=/dev/ttyUSB3 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

### Record Dataset (without UI)

```bash
lerobot-record \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/ttyUSB0 \
  --robot.right_arm_port=/dev/ttyUSB1 \
  --robot.id=bimanual_follower \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=/dev/ttyUSB2 \
  --teleop.right_arm_port=/dev/ttyUSB3 \
  --teleop.id=bimanual_leader \
  --dataset.repo_id=your-username/bimanual-so101-dataset \
  --dataset.num_episodes=50 \
  --dataset.single_task="Bimanual manipulation task"
```

### Calibrate Individual Arm

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=left_follower
```

## Auto-Start on Boot

To launch the touch UI automatically on boot:

1. Create a desktop file:
   ```bash
   nano ~/.config/autostart/lerobot-touch-ui.desktop
   ```

2. Add:
   ```ini
   [Desktop Entry]
   Type=Application
   Name=LeRobot Touch UI
   Exec=/usr/bin/python3 -m lerobot.scripts.lerobot_touch_ui --follower-left-port=/dev/ttyUSB0 --follower-right-port=/dev/ttyUSB1 --leader-left-port=/dev/ttyUSB2 --leader-right-port=/dev/ttyUSB3
   Terminal=false
   ```

3. Make executable:
   ```bash
   chmod +x ~/.config/autostart/lerobot-touch-ui.desktop
   ```

## Troubleshooting

### "Permission denied" on USB ports
```bash
sudo usermod -a -G dialout $USER
# Log out and back in
```

### Touch not working
```bash
# Calibrate touchscreen
sudo apt install xinput-calibrator
xinput_calibrator
```

### Display too small/large
Edit `SCREEN_WIDTH` and `SCREEN_HEIGHT` in:
`/home/user/lerobot/src/lerobot/touch_ui/touch_ui_app.py`

### Motors not responding
1. Check USB connections
2. Verify motor IDs are set correctly
3. Check calibration files exist
4. Try reconnecting in System Status

### "Module not found" errors
```bash
pip install -e ".[feetech]"
```

## What's New

This implementation adds:

1. **Bimanual SO-101 Classes**:
   - `BiSO101Follower` - Coordinated follower arms
   - `BiSO101Leader` - Coordinated leader arms
   - Full LeRobot API compatibility

2. **Touch UI App**:
   - Modern pygame-based interface
   - 7-inch landscape optimization
   - Touch-friendly large buttons
   - Multi-screen navigation

3. **Integration**:
   - Registered in `utils.py`
   - Added to `pyproject.toml`
   - Console script entry point
   - Example configurations

## File Locations

- **Bimanual SO-101 Follower**: `/home/user/lerobot/src/lerobot/robots/bi_so101_follower/`
- **Bimanual SO-101 Leader**: `/home/user/lerobot/src/lerobot/teleoperators/bi_so101_leader/`
- **Touch UI**: `/home/user/lerobot/src/lerobot/touch_ui/`
- **Launcher Script**: `/home/user/lerobot/src/lerobot/scripts/lerobot_touch_ui.py`
- **Documentation**: `/home/user/lerobot/src/lerobot/touch_ui/README.md`

## Next Steps

1. **Test teleoperation** - Verify leaders control followers
2. **Record a dataset** - Collect some demonstrations
3. **Train a policy** - Use recorded data for imitation learning
4. **Customize UI** - Modify colors, buttons, layouts as needed

## Support

- Full README: `/home/user/lerobot/src/lerobot/touch_ui/README.md`
- LeRobot Docs: https://huggingface.co/docs/lerobot
- SO-101 Guide: https://huggingface.co/docs/lerobot/so101

Enjoy your bimanual robot! 🤖🤖
