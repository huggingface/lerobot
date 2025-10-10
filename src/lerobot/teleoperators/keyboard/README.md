# Keyboard Teleoperators

This directory contains keyboard-based teleoperators for various robots.

## Issues Fixed

### Keyboard Listener Conflicts (SO-101 MuJoCo)

**Original Problem:**
- When running `lerobot_record.py` with `so101_mujoco` robot and `so101_keyboard` teleop, two separate `pynput` keyboard listeners were initialized:
  1. One in `teleop_so101_keyboard.py` (for WASD, Q/E, etc.)
  2. One in `lerobot_record.py` via `init_keyboard_listener()` (for arrow keys, ESC)
- These conflicted with each other, preventing proper keyboard input handling

**Solution: Shared KeyboardEventManager**

Created a singleton `KeyboardEventManager` (`lerobot/utils/keyboard_event_manager.py`) that:
- Maintains a single `pynput` listener for the entire application
- Allows multiple components to register handlers for different keys
- Distributes keyboard events to all registered handlers
- Is thread-safe (uses `RLock` for concurrent registration/access)

**Key Design Decisions:**
- **Singleton pattern**: All components get the same manager instance via `get_keyboard_manager()`
- **Order-independent**: Works whether recording script or teleop connects first
- **Dynamic registration**: Handlers can be registered before or after the listener starts
- **Single source of truth**: Only the manager checks `pynput` availability
- **Graceful disconnection**: Components disconnect without stopping the shared manager

**Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│        KeyboardEventManager (Singleton)                 │
│  - Created by whoever calls get_keyboard_manager()     │
│  - Started by first component that needs it            │
│  - Shared by all: record script, teleop, etc.          │
│  - Single pynput.keyboard.Listener                     │
└─────────────────────────────────────────────────────────┘
                      ▲          ▲
                      │          │
            ┌─────────┘          └─────────┐
            │                                │
  lerobot_record.py                 teleop_so101_keyboard
  (arrow keys, ESC)                 (WASD, Q/E, shift, ctrl)
  - exit_early                      - Robot control keys
  - rerecord_episode                - Via event_queue
  - stop_recording                  - Shared manager lifecycle
```

**Usage Flow:**

1. **Scenario: Recording with keyboard teleop**
   - `lerobot_record.py` calls `init_keyboard_listener()` → creates and starts manager
   - Registers handlers for arrow keys and ESC
   - `SO101KeyboardTeleop.connect()` → gets same manager, registers WASD/Q/E/etc.
   - Single listener routes events to both sets of handlers
   - On exit: `lerobot_record.py` stops the manager

2. **Scenario: Standalone keyboard teleop (no recording)**
   - `SO101KeyboardTeleop.connect()` → gets manager, starts it
   - Registers handlers for control keys
   - Single listener routes events to teleop
   - On exit: teleop disconnects (doesn't stop shared manager)

**Files Modified:**
- `lerobot/utils/keyboard_event_manager.py` (new)
- `lerobot/utils/control_utils.py` (`init_keyboard_listener` now uses shared manager)
- `lerobot/teleoperators/keyboard/teleop_so101_keyboard.py` (uses shared manager, removed duplicate pynput check)
- `lerobot/scripts/lerobot_record.py` (cleanup call updated for manager)

