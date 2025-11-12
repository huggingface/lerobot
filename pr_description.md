# Async Inference Third-Party Hardware Support

## Summary

This PR enables third-party robots to work with LeRobot's async inference system by fixing registration timing and dynamic robot discovery.

## Problem

Third-party robots couldn't be used with `lerobot.async_inference.robot_client` because the argument parser couldn't see dynamically registered robot types.

## Solution

**Two minimal changes:**

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

## Files Changed

- `src/lerobot/async_inference/constants.py` (9 lines added)
- `src/lerobot/async_inference/robot_client.py` (8 lines added)
- `src/lerobot/utils/import_utils.py` (enhanced with editable package support)

## Impact

**For Third-Party Robot Developers:**
- ✅ Standard pip installs work immediately
- ✅ No core LeRobot modifications needed
- ✅ Async inference support out-of-the-box

**For LeRobot Maintainers:**
- ✅ Automatic robot type discovery
- ✅ No hardcoded robot lists
- ✅ Backward compatible

## Testing

Verified with Piper robot:
```bash
python -m lerobot.async_inference.robot_client --robot.type=piper
# Now accepts piper as valid robot type ✅

python -m lerobot.async_inference.robot_client \
    --robot.type=piper \
    --server_address=127.0.0.1:8080 \
    # ... full command now works ✅
```

## Future

Editable package support and integration tests can be added in follow-up PRs.
