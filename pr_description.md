# Async Inference Third-Party Hardware Support

## Summary

This PR enables third-party robots to work with LeRobot's async inference system by implementing automatic device discovery and early registration.

## Problem

Third-party robots couldn't be used with `lerobot.async_inference.robot_client` because:
- The argument parser couldn't see dynamically registered robot types
- Device discovery didn't handle editable package installations

## Solution

**Three key improvements:**

#### 1. Early Device Registration (`robot_client.py`)
```python
# Register third-party devices BEFORE importing configs
register_third_party_devices()
from .configs import RobotClientConfig  # Now sees all registered robots
```

#### 2. Dynamic Robot Discovery (`constants.py`)
```python
def get_supported_robots():
    """Auto-discover supported robots from registered RobotConfig subclasses."""
    return list(RobotConfig.get_known_choices().keys())
```

#### 3. Enhanced Device Discovery (`import_utils.py`)
- Automatic detection of regular pip-installed packages
- Automatic detection and extraction of editable package finder objects
- Graceful import of robot config modules for registration

## Files Changed

- `src/lerobot/async_inference/constants.py` (9 lines)
- `src/lerobot/async_inference/robot_client.py` (8 lines)
- `src/lerobot/utils/import_utils.py` (enhanced)

## Impact

**For Third-Party Robot Developers:**
- ✅ Works with both `pip install` and `pip install -e`
- ✅ No core LeRobot modifications needed
- ✅ Standard CLI commands work immediately
- ✅ Async inference support out-of-the-box

**For LeRobot Maintainers:**
- ✅ Automatic robot type discovery
- ✅ No hardcoded robot lists to maintain
- ✅ Backward compatible

## Testing

The implementation enables automatic discovery of third-party robots:

```bash
# Any third-party robot package installed via:
pip install lerobot-robot-custom          # ✅ Regular install
pip install -e lerobot-robot-custom       # ✅ Editable install

# Will be automatically discovered and available:
python -m lerobot.async_inference.robot_client --robot.type=custom  # ✅
lerobot-record --robot.type=custom                                 # ✅
lerobot-teleoperate --robot.type=custom                            # ✅
```
