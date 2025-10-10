#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared keyboard event manager to avoid conflicts when multiple components need keyboard input.

This singleton-style manager ensures only one keyboard listener is active and routes
events to multiple registered handlers.

## Usage Flow

The manager uses a singleton pattern, so all components get the same instance.

**Scenario 1: Recording with keyboard teleop**
1. `lerobot_record.py` calls `init_keyboard_listener()` → creates and starts the manager
2. Registers handlers for arrow keys and ESC (recording control)
3. Keyboard teleop connects → gets same manager, registers WASD, Q/E, etc.
4. Single listener captures all keys, routes to appropriate handlers
5. On exit: `lerobot_record.py` stops the manager

**Scenario 2: Standalone keyboard teleop (no recording)**
1. Keyboard teleop connects first → gets manager, starts it
2. Registers handlers for WASD, Q/E, shift, ctrl, etc.
3. Single listener captures keys, routes to teleop handlers
4. On exit: teleop disconnects (doesn't stop the shared manager)

**Key benefits**:
- ✅ No conflicts: only one pynput listener ever exists
- ✅ Order-independent: works whether record or teleop connects first
- ✅ Flexible: works with non-keyboard teleoperators (gamepad, etc.) too
- ✅ Simple: keyboard teleops always use the shared manager, no fallback needed
"""

import logging
import os
import sys
from threading import RLock
from typing import Callable

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class KeyboardEventManager:
    """
    Singleton-style keyboard event manager that handles multiple consumers.
    
    This manager creates a single keyboard listener and distributes events to
    registered handlers, avoiding conflicts when both teleop and recording scripts
    need keyboard input.
    
    Example usage:
        ```python
        # In lerobot_record.py (or any script)
        from lerobot.utils.keyboard_event_manager import get_keyboard_manager
        from pynput import keyboard
        
        manager = get_keyboard_manager()
        
        # Register recording control handlers
        def on_right_arrow():
            events["exit_early"] = True
        manager.register_key_press_handler(keyboard.Key.right, on_right_arrow)
        
        # Start the manager (safe to call multiple times)
        manager.start()
        
        # In keyboard teleop
        # Always uses the same singleton manager
        manager = get_keyboard_manager()
        manager.register_char_press_handler('w', lambda: move_forward())
        manager.start()  # Safe to call even if already started
        ```
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.listener = None
        self._key_press_handlers = {}  # key -> list of callbacks
        self._key_release_handlers = {}  # key -> list of callbacks
        self._char_press_handlers = {}  # char -> list of callbacks
        self._char_release_handlers = {}  # char -> list of callbacks
        self._is_started = False
        self._lock = RLock()  # Thread safety for handler registration/access
        
    def register_key_press_handler(self, key, callback: Callable[[], None]):
        """
        Register a handler for a special key press event.
        
        Can be called before or after starting the listener (thread-safe).
        
        Args:
            key: A pynput.keyboard.Key value (e.g., keyboard.Key.right, keyboard.Key.esc)
            callback: Function to call when the key is pressed (takes no arguments)
        """
        with self._lock:
            if key not in self._key_press_handlers:
                self._key_press_handlers[key] = []
            self._key_press_handlers[key].append(callback)
        logging.debug(f"Registered press handler for key: {key}")
    
    def register_key_release_handler(self, key, callback: Callable[[], None]):
        """
        Register a handler for a special key release event.
        
        Can be called before or after starting the listener (thread-safe).
        
        Args:
            key: A pynput.keyboard.Key value (e.g., keyboard.Key.right, keyboard.Key.esc)
            callback: Function to call when the key is released (takes no arguments)
        """
        with self._lock:
            if key not in self._key_release_handlers:
                self._key_release_handlers[key] = []
            self._key_release_handlers[key].append(callback)
        logging.debug(f"Registered release handler for key: {key}")
    
    def register_char_press_handler(self, char: str, callback: Callable[[], None]):
        """
        Register a handler for a character key press event.
        
        Can be called before or after starting the listener (thread-safe).
        
        Args:
            char: A single character string (e.g., 'w', 'a', 's', 'd')
            callback: Function to call when the character is pressed (takes no arguments)
        """
        with self._lock:
            if char not in self._char_press_handlers:
                self._char_press_handlers[char] = []
            self._char_press_handlers[char].append(callback)
        logging.debug(f"Registered press handler for char: {char}")
    
    def register_char_release_handler(self, char: str, callback: Callable[[], None]):
        """
        Register a handler for a character key release event.
        
        Can be called before or after starting the listener (thread-safe).
        
        Args:
            char: A single character string (e.g., 'w', 'a', 's', 'd')
            callback: Function to call when the character is released (takes no arguments)
        """
        with self._lock:
            if char not in self._char_release_handlers:
                self._char_release_handlers[char] = []
            self._char_release_handlers[char].append(callback)
        logging.debug(f"Registered release handler for char: {char}")
    
    def _on_press(self, key):
        """Internal press handler that distributes to registered callbacks."""
        # Handle special keys
        with self._lock:
            callbacks = list(self._key_press_handlers.get(key, []))
            if hasattr(key, "char"):
                callbacks.extend(self._char_press_handlers.get(key.char, []))
        
        # Call callbacks outside the lock to avoid deadlock
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Error in key press handler for {key}: {e}")
    
    def _on_release(self, key):
        """Internal release handler that distributes to registered callbacks."""
        # Handle special keys
        with self._lock:
            callbacks = list(self._key_release_handlers.get(key, []))
            if hasattr(key, "char"):
                callbacks.extend(self._char_release_handlers.get(key.char, []))
        
        # Call callbacks outside the lock to avoid deadlock
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Error in key release handler for {key}: {e}")
    
    def start(self):
        """Start the keyboard listener if not already started."""
        if self._is_started:
            logging.debug("Keyboard event manager already started")
            return
        
        if not PYNPUT_AVAILABLE:
            logging.warning("pynput not available - keyboard event manager cannot start")
            return
        
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()
        self._is_started = True
        logging.info("Keyboard event manager started")
    
    def stop(self):
        """Stop the keyboard listener."""
        if self.listener is not None:
            self.listener.stop()
            self._is_started = False
            logging.info("Keyboard event manager stopped")
    
    @property
    def is_active(self) -> bool:
        """Check if the manager has been started."""
        return self._is_started
    
    def clear_handlers(self):
        """Clear all registered handlers."""
        self._key_press_handlers.clear()
        self._key_release_handlers.clear()
        self._char_press_handlers.clear()
        self._char_release_handlers.clear()
        logging.debug("Cleared all keyboard handlers")


def get_keyboard_manager() -> KeyboardEventManager | None:
    """
    Get the global keyboard event manager instance.
    
    Returns:
        KeyboardEventManager instance if pynput is available, None otherwise.
    """
    if not PYNPUT_AVAILABLE:
        return None
    return KeyboardEventManager()

