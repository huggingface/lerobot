# Bimanual SO-101 Touch UI

A modern, touch-friendly graphical interface for controlling bimanual SO-101 robot arms on a 7-inch landscape touchscreen.

## Features

- **Motor Setup**: Configure motor IDs for all 4 arms (2 followers + 2 leaders)
- **Calibration**: Calibrate range of motion for all joints with visual feedback
- **Teleoperation**: Real-time bimanual control (leaders control followers)
- **Dataset Recording**: Record demonstration datasets for imitation learning
- **Status Monitoring**: View connection status and error messages
- **Touch-Optimized**: Large buttons designed for 7-inch (1024x600) touchscreen

## Hardware Requirements

- Mini PC with Ubuntu Desktop
- 7-inch touchscreen (landscape orientation, 1024x600 or 800x480)
- 4x SO-101 arms:
  - 2 Follower arms
  - 2 Leader arms
- 4 USB serial ports for arm connections

## Installation

1. Install LeRobot with Feetech support:

```bash
pip install -e ".[feetech]"
```

2. Install pygame for the UI:

```bash
pip install pygame
```

## USB Port Configuration

Connect your arms to the mini PC via USB. Note the device paths (usually `/dev/ttyUSB*`):

```bash
# List connected USB devices
ls -l /dev/ttyUSB*
```

Example configuration:
- Left Follower: `/dev/ttyUSB0`
- Right Follower: `/dev/ttyUSB1`
- Left Leader: `/dev/ttyUSB2`
- Right Leader: `/dev/ttyUSB3`

## Usage

### Launch the Touch UI

```bash
python -m lerobot.scripts.lerobot_touch_ui \
    --follower-left-port=/dev/ttyUSB0 \
    --follower-right-port=/dev/ttyUSB1 \
    --leader-left-port=/dev/ttyUSB2 \
    --leader-right-port=/dev/ttyUSB3
```

Or if installed via setuptools:

```bash
lerobot-touch-ui \
    --follower-left-port=/dev/ttyUSB0 \
    --follower-right-port=/dev/ttyUSB1 \
    --leader-left-port=/dev/ttyUSB2 \
    --leader-right-port=/dev/ttyUSB3
```

### First-Time Setup Workflow

1. **Motor Setup** (do this first!):
   - Tap "Motor Setup" from main menu
   - Follow prompts to configure motor IDs
   - Do this for each arm individually

2. **Calibration**:
   - Tap "Calibration" from main menu
   - Choose to calibrate followers or leaders
   - Follow on-screen instructions to move arms through full range
   - Calibration data is saved automatically

3. **Connect Arms**:
   - Tap "System Status" from main menu
   - Tap "Connect Followers" and "Connect Leaders"
   - Verify all connections show "Connected"

4. **Teleoperation**:
   - Tap "Teleoperation" from main menu
   - Tap "START" to begin
   - Move leader arms - followers will mirror movements
   - Tap "STOP" to end session

5. **Dataset Recording**:
   - Complete teleoperation setup first
   - Tap "Record Dataset" from main menu
   - Tap "Start Recording" to begin an episode
   - Perform demonstration
   - Episodes are automatically saved

## Screen Reference

### Main Menu
- **Motor Setup**: Configure motor IDs
- **Calibration**: Set joint ranges
- **Teleoperation**: Manual bimanual control
- **Record Dataset**: Capture demonstration data
- **System Status**: View connection status
- **Exit**: Close application

### Motor Setup Screen
- Setup motors for each arm individually
- Follow console prompts (will appear on terminal)
- Motors must be connected one at a time

### Calibration Screen
- Calibrate followers or leaders separately
- Follow prompts to move through full range
- Calibration saves to `~/.cache/lerobot/calibration/`

### Teleoperation Screen
- Real-time control mode
- Left leader controls left follower
- Right leader controls right follower
- 30 FPS control loop

### Recording Screen
- Combines teleoperation + data recording
- Episode counter tracks recorded demonstrations
- Data saved in LeRobot dataset format

### Status Screen
- Connection status for all 4 arms
- Error messages displayed
- Connect/reconnect buttons

## API Integration

The touch UI uses the LeRobot API:

### Bimanual SO-101 Follower
```python
from lerobot.robots.bi_so101_follower import BiSO101Follower, BiSO101FollowerConfig

config = BiSO101FollowerConfig(
    left_arm_port="/dev/ttyUSB0",
    right_arm_port="/dev/ttyUSB1",
    id="bimanual_follower"
)
follower = BiSO101Follower(config)
follower.connect()
```

### Bimanual SO-101 Leader
```python
from lerobot.teleoperators.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig

config = BiSO101LeaderConfig(
    left_arm_port="/dev/ttyUSB2",
    right_arm_port="/dev/ttyUSB3",
    id="bimanual_leader"
)
leader = BiSO101Leader(config)
leader.connect()
```

### Teleoperation Loop
```python
# Get action from leaders
action = leader.get_action()

# Send to followers
follower.send_action(action)
```

## Touchscreen Calibration

If your touchscreen needs calibration:

```bash
# Install xinput_calibrator
sudo apt install xinput-calibrator

# Run calibration
xinput_calibrator
```

Follow on-screen instructions and save the calibration.

## Troubleshooting

### USB Permissions
If you get permission errors:

```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Log out and back in for changes to take effect
```

### Display Issues
For fullscreen mode on boot, edit `/etc/xdg/lxsession/LXDE-pi/autostart`:

```bash
@python3 /path/to/lerobot/src/lerobot/scripts/lerobot_touch_ui.py --args
```

### Pygame on Headless Systems
If running without X11:

```bash
# Use framebuffer
SDL_VIDEODRIVER=fbcon python -m lerobot.scripts.lerobot_touch_ui ...
```

## Development

The touch UI is modular and extensible:

- `touch_ui_app.py`: Main application and screen management
- Screen classes inherit from base UI components
- Easy to add new screens or modify button layouts
- All LeRobot API calls are centralized

### Adding Custom Screens

1. Define new `Screen` enum value
2. Create `draw_<screen>_screen()` method
3. Add button handler in event loop
4. Update main menu to include new screen

## Performance Notes

- UI runs at 30 FPS
- Teleoperation loop runs in same thread (non-blocking)
- For higher control rates, run teleoperation in separate thread
- Touch events processed every frame

## Credits

Built on top of:
- LeRobot by HuggingFace
- SO-101 arms by TheRobotStudio & HuggingFace
- pygame for UI rendering
