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

from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from lerobot.utils.import_utils import make_device_from_device_class

from .config import TeleoperatorConfig

if TYPE_CHECKING:
    from .teleoperator import Teleoperator


class TeleopEvents(Enum):
    """Shared constants for teleoperator events across teleoperators."""

    SUCCESS = "success"
    FAILURE = "failure"
    RERECORD_EPISODE = "rerecord_episode"
    IS_INTERVENTION = "is_intervention"
    TERMINATE_EPISODE = "terminate_episode"


class KeyboardTeleopEvents:
    """Keyboard-based episode control for teleoperators without hardware buttons.

    Key bindings match the HIL-SERL docs for leader arm usage:
    - Space: toggle intervention (take over / release control)
    - s: mark episode as success
    - Escape: mark episode as failure
    - r: rerecord episode
    """

    def __init__(self):
        import logging
        import os
        import sys
        from queue import Queue

        self._logger = logging.getLogger(__name__)
        self._listener = None
        self._event_queue: Queue[str] = Queue()
        self._intervention_active = False
        self._success_active = False

        self._pynput_keyboard = None
        try:
            if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
                raise ImportError("No DISPLAY set — keyboard listener unavailable.")
            from pynput import keyboard as pynput_keyboard

            self._pynput_keyboard = pynput_keyboard
        except ImportError:
            pass

    def start(self) -> None:
        """Start the keyboard listener. Called automatically on first get_events()."""
        if self._listener is not None:
            return

        if self._pynput_keyboard is None:
            raise ImportError(
                "pynput is required for keyboard-based teleop events but is unavailable. "
                "On headless systems (no DISPLAY), keyboard input is not supported."
            )

        pynput_kb = self._pynput_keyboard

        def on_press(key):
            try:
                if key == pynput_kb.Key.space:
                    self._intervention_active = not self._intervention_active
                elif key == pynput_kb.Key.esc:
                    self._event_queue.put("failure")
                elif hasattr(key, "char"):
                    if key.char == "s":
                        self._success_active = True
                    elif key.char == "r":
                        self._event_queue.put("rerecord")
            except AttributeError:
                pass

        def on_release(key):
            try:
                if hasattr(key, "char") and key.char == "s":
                    self._success_active = False
            except AttributeError:
                pass

        self._listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        self._logger.info(
            "Keyboard listener started for teleop events "
            "(Space=intervention, s=success, Esc=failure, r=rerecord)"
        )

    def stop(self) -> None:
        """Stop the keyboard listener if running."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._success_active = False

    def get_events(self) -> dict[str, Any]:
        """Get pending teleop events, starting the listener on first call.

        Returns:
            Dictionary containing episode control events.
        """
        from queue import Empty

        if self._listener is None:
            self.start()

        terminate_episode = False
        success = self._success_active
        rerecord_episode = False

        while True:
            try:
                event = self._event_queue.get_nowait()
            except Empty:
                break
            if event == "failure":
                terminate_episode = True
                self._success_active = False
            elif event == "rerecord":
                terminate_episode = True
                rerecord_episode = True
                self._success_active = False

        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_active,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


def make_teleoperator_from_config(config: TeleoperatorConfig) -> "Teleoperator":
    # TODO(Steven): Consider just using the make_device_from_device_class for all types
    if config.type == "keyboard":
        from .keyboard import KeyboardTeleop

        return KeyboardTeleop(config)
    elif config.type == "koch_leader":
        from .koch_leader import KochLeader

        return KochLeader(config)
    elif config.type == "omx_leader":
        from .omx_leader import OmxLeader

        return OmxLeader(config)
    elif config.type == "so100_leader":
        from .so_leader import SO100Leader

        return SO100Leader(config)
    elif config.type == "so101_leader":
        from .so_leader import SO101Leader

        return SO101Leader(config)
    elif config.type == "mock_teleop":
        from tests.mocks.mock_teleop import MockTeleop

        return MockTeleop(config)
    elif config.type == "gamepad":
        from .gamepad.teleop_gamepad import GamepadTeleop

        return GamepadTeleop(config)
    elif config.type == "keyboard_ee":
        from .keyboard.teleop_keyboard import KeyboardEndEffectorTeleop

        return KeyboardEndEffectorTeleop(config)
    elif config.type == "homunculus_glove":
        from .homunculus import HomunculusGlove

        return HomunculusGlove(config)
    elif config.type == "homunculus_arm":
        from .homunculus import HomunculusArm

        return HomunculusArm(config)
    elif config.type == "unitree_g1":
        from .unitree_g1 import UnitreeG1Teleoperator

        return UnitreeG1Teleoperator(config)
    elif config.type == "bi_so_leader":
        from .bi_so_leader import BiSOLeader

        return BiSOLeader(config)
    elif config.type == "reachy2_teleoperator":
        from .reachy2_teleoperator import Reachy2Teleoperator

        return Reachy2Teleoperator(config)
    elif config.type == "openarm_leader":
        from .openarm_leader import OpenArmLeader

        return OpenArmLeader(config)
    elif config.type == "bi_openarm_leader":
        from .bi_openarm_leader import BiOpenArmLeader

        return BiOpenArmLeader(config)
    elif config.type == "openarm_mini":
        from .openarm_mini import OpenArmMini

        return OpenArmMini(config)
    else:
        try:
            return cast("Teleoperator", make_device_from_device_class(config))
        except Exception as e:
            raise ValueError(f"Error creating robot with config {config}: {e}") from e
