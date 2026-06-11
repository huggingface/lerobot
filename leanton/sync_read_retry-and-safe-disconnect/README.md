# sync_read_retry-and-safe-disconnect

**Target:** `src/lerobot/motors/motors_bus.py`
**Status:** `active`
**GitHub:** Not filed

## What

1. `sync_read` default `num_retry` changed from 0 → 2 (3 total attempts instead of 1)
2. `disconnect()` wraps `disable_torque()` in try/except `ConnectionError` so the serial port always closes even if a motor is unresponsive

## Why

SO-101 Feetech servos on the shared RS-485 bus occasionally brown out near mechanical limits. A single missed status packet kills the entire teleop session. The extra retries tolerate transient dropouts. The disconnect guard prevents the arm from staying powered on after a crash.

## Validate

**User:** Run a teleop session and move the arm to its full extension (near mechanical limits). The session should not crash with "Failed to sync read" / "There is no status packet!" errors. If it does crash, the arm should power off (torque disabled) rather than staying locked.

**Agent:**
```bash
grep -q "num_retry: int = 2" ~/lerobot/src/lerobot/motors/motors_bus.py && echo "Retry count ✅" || echo "MISSING"
grep -q "except ConnectionError:" ~/lerobot/src/lerobot/motors/motors_bus.py && echo "Safe disconnect ✅" || echo "MISSING"
```
