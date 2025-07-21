# Inference Logging for LeRobot

This guide shows you how to use the comprehensive inference logging system to capture and analyze robot state, policy outputs, and trajectories during inference.

## Quick Start

### 1. Check Robot State (Debug)

First, check if your robot is connected and working properly:

```bash
python debug_robot_state.py
```

This will show you:

- Current servo positions
- Raw motor data
- Motor temperatures, voltages, currents
- Robot connection status

### 2. Record with Inference Logging

Enable logging during recording by adding the `--log=true` flag:

```bash
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760434091 \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --robot.id=my_robot \
    --dataset.repo_id=adungus/test_logging \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=30 \
    --dataset.single_task="Pick up object" \
    --policy.path=your_policy_path \
    --log=true
```

**Important Notes:**

- Logging only works when using a policy (not teleoperation)
- Use lower camera resolution (640x480) to avoid timeout issues
- The `--log=true` flag enables comprehensive logging

### 3. Analyze Logs

After recording, analyze the generated CSV files:

```bash
python analyze_inference_logs.py inference_logs/your_dataset_name/timestamp/
```

This generates:

- `servo_positions.png` - Servo trajectories over time
- `policy_outputs.png` - Policy actions and inference timing
- `servo_correlation.png` - Correlation matrix between servos
- `inference_summary.txt` - Statistical summary

## What Gets Logged

### Robot State (`*_robot_state.csv`)

- Timestamp and step number
- All servo positions (normalized and raw)
- Motor temperatures, voltages, currents (if available)
- Robot connection status
- Camera information

### Policy Inference (`*_policy_inference.csv`)

- Timestamp and step number
- Inference timing (milliseconds)
- Policy input observations (non-image data)
- Policy action outputs
- Raw policy output statistics (mean, std, min, max)
- Task description

### Console Output

During inference, you'll see real-time output like:

```
📊 INFERENCE STEP 1 @ 2.35s
============================================================
🔧 SERVO POSITIONS:
   shoulder_pan.pos:    45.23
   shoulder_lift.pos:  -12.67
   elbow_flex.pos:      78.45
   wrist_flex.pos:     -23.12
   wrist_roll.pos:       5.67
   gripper.pos:         50.00

🎯 POLICY OUTPUT:
   shoulder_pan.pos:    45.78
   shoulder_lift.pos:  -13.02
   elbow_flex.pos:      78.92
   wrist_flex.pos:     -23.45
   wrist_roll.pos:       5.23
   gripper.pos:         52.34

⏱️  TIMING: Inference took 15.3ms
📋 TASK: Pick up object
============================================================
```

## File Structure

```
inference_logs/
└── your_dataset_name/
    └── 20240121_143022/  # timestamp
        ├── so100_follow_robot_state.csv
        ├── so100_follow_policy_inference.csv
        └── (trajectory files if end-effector control used)
```

## Troubleshooting

### Camera Timeout Issues

If you get camera timeout errors:

1. Use lower resolution: 640x480 instead of 1920x1080
2. Reduce FPS: try 15 or 10 instead of 30
3. Try different USB ports
4. For debugging, use no cameras: `--robot.cameras="{}"`

### No Logging Output

Make sure:

- You're using a policy (not just teleoperation)
- The `--log=true` flag is set
- The policy path is valid and loads correctly

### Performance Issues

- Logging adds minimal overhead (~1-2ms per inference)
- CSV files are flushed after each write for safety
- Large episodes may generate large CSV files

## Example Analysis Workflows

### Basic Performance Analysis

```bash
# Record with logging
python -m lerobot.record --policy.path=my_policy --log=true ...

# Analyze timing
python analyze_inference_logs.py inference_logs/my_dataset/timestamp/
grep "Mean inference time" inference_summary.txt
```

### Servo Trajectory Analysis

```bash
# View servo correlations
python analyze_inference_logs.py inference_logs/my_dataset/timestamp/
# Check servo_correlation.png for coupled movements
```

### Custom Analysis

```python
import pandas as pd

# Load your data
robot_df = pd.read_csv('inference_logs/.../so100_follow_robot_state.csv')
policy_df = pd.read_csv('inference_logs/.../so100_follow_policy_inference.csv')

# Custom analysis
print(f"Average inference time: {policy_df['inference_time_ms'].mean():.1f}ms")
print(f"Servo range of motion: {robot_df['shoulder_pan.pos'].max() - robot_df['shoulder_pan.pos'].min():.1f}")
```

## Integration with Other Tools

The CSV format makes it easy to:

- Import into Excel/Google Sheets
- Use with pandas/numpy for analysis
- Visualize with matplotlib/plotly
- Feed into ML pipelines for analysis

## Tips for Best Results

1. **Start Simple**: Test with short episodes first
2. **Monitor Performance**: Check inference timing doesn't exceed your control loop frequency
3. **Regular Analysis**: Analyze logs after each session to catch issues early
4. **Compare Policies**: Use logging to compare different policy performances
5. **Debug Issues**: Use the debug script first to verify robot connectivity
