# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Console status display for rollout strategies.

One subclass per strategy — static states/controls are declared as class
constants; runtime-dependent values are passed to ``__init__``.

In each strategy's ``setup()``:

    self._display = DAggerDisplay(
        record_autonomous=self.config.record_autonomous,
        num_episodes=self.config.num_episodes,
        episode_duration_s=self._episode_duration_s,
        input_device=self.config.input_device,
        pause_key="SPACE",
        correction_key="TAB",
        upload_key="ENTER",
    )
    self._display.show_banner()

On each state transition:

    self._display.show_state("correcting")
"""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class _C:
    """ANSI escape codes."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[1;92m"
    YELLOW = "\033[1;93m"
    RED = "\033[1;91m"
    CYAN = "\033[1;96m"
    WHITE = "\033[1;97m"
    GRAY = "\033[2;37m"


@dataclass
class StateConfig:
    """One named rollout state.

    ``key`` must match the string passed to ``RolloutStatusDisplay.show_state()``.
    """

    key: str
    emoji: str
    label: str
    description: str
    color: str = _C.WHITE


@dataclass
class ControlConfig:
    """One keyboard/pedal binding shown in the startup banner."""

    key: str
    description: str


# ---------------------------------------------------------------------------
# Base display class
# ---------------------------------------------------------------------------


class RolloutStatusDisplay:
    """Unified console status display.  Subclass once per strategy."""

    def __init__(
        self,
        strategy: str,
        states: list[StateConfig],
        controls: list[ControlConfig],
        info: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self._states = {s.key: s for s in states}
        self._controls = controls
        self._info = info or []
        self._use_color = _supports_color()

    def _c(self, code: str, text: str) -> str:
        if not self._use_color:
            return text
        return f"{code}{text}{_C.RESET}"

    def show_banner(self) -> None:
        """Print startup banner: strategy name, states, controls, config info."""
        width = 62
        sep = self._c(_C.BOLD, "═" * width)

        print(f"\n{sep}")
        print(self._c(_C.BOLD, f"  lerobot-rollout  │  {self.strategy}"))

        if self._states:
            print()
            for state in self._states.values():
                label = self._c(state.color, f"{state.label:<14}")
                desc = self._c(_C.GRAY, state.description)
                print(f"  {state.emoji}  {label}  {desc}")

        if self._controls:
            print()
            key_width = max(len(c.key) for c in self._controls)
            for ctrl in self._controls:
                key_str = self._c(_C.CYAN, f"[{ctrl.key:<{key_width}}]")
                print(f"  {key_str}  {ctrl.description}")

        if self._info:
            print()
            for item in self._info:
                print(f"  {item}")

        print(f"{sep}\n")

    def show_state(self, state_key: str | enum.Enum) -> None:
        """Print the current state and available controls - call this on every transition."""
        key = state_key.value if isinstance(state_key, enum.Enum) else state_key
        state = self._states.get(key)
        if state is None:
            return
        label = self._c(state.color, f"{state.label:<14}")
        desc = self._c(_C.GRAY, state.description)
        print(f"\n  {state.emoji}  {label}  {desc}\n")

        if self._controls:
            key_width = max(len(c.key) for c in self._controls)
            for ctrl in self._controls:
                key_str = self._c(_C.CYAN, f"[{ctrl.key:<{key_width}}]")
                print(f"  {key_str}  {ctrl.description}")
            print()


# ---------------------------------------------------------------------------
# One display subclass per strategy
# ---------------------------------------------------------------------------


class BaseDisplay(RolloutStatusDisplay):
    """Status display for the base (eval-only, no recording) strategy."""

    _STATES = [StateConfig("running", "🟢", "RUNNING", "autonomous rollout — no recording", _C.GREEN)]
    _CONTROLS = [ControlConfig("Ctrl+C", "stop session")]

    def __init__(self, duration: float = 0) -> None:
        info = ["No recording — evaluation only."]
        if duration > 0:
            info.append(f"Duration: {duration:.0f}s")
        super().__init__("base", self._STATES, self._CONTROLS, info)


class SentryDisplay(RolloutStatusDisplay):
    """Status display for the sentry (continuous autonomous recording) strategy."""

    _STATES = [StateConfig("recording", "🟢", "RECORDING", "continuous autonomous recording", _C.GREEN)]
    _CONTROLS = [ControlConfig("Ctrl+C", "stop session")]

    def __init__(self, episode_duration_s: float, upload_every_n_episodes: int) -> None:
        info = [
            f"Episode rotation: ~{episode_duration_s:.0f}s  |  "
            f"Upload every {upload_every_n_episodes} episodes",
        ]
        super().__init__("sentry", self._STATES, self._CONTROLS, info)


class HighlightDisplay(RolloutStatusDisplay):
    """Status display for the highlight (ring-buffer on-demand save) strategy."""

    def __init__(self, ring_buffer_seconds: float, save_key: str, push_key: str) -> None:
        states = [
            StateConfig(
                "buffering",
                "⚪",
                "BUFFERING",
                f"ring buffer active — last {ring_buffer_seconds:.0f}s captured",
                _C.WHITE,
            ),
            StateConfig("recording", "🔴", "RECORDING", "live recording — press [s] to save episode", _C.RED),
        ]
        controls = [
            ControlConfig(save_key, "BUFFERING ↔ RECORDING  start recording / save episode"),
            ControlConfig(push_key, "push dataset to Hub (background)"),
            ControlConfig("ESC", "stop session"),
        ]
        super().__init__("highlight", states, controls)


class DAggerDisplay(RolloutStatusDisplay):
    """Status display for the dagger (human-in-the-loop) strategy."""

    _PAUSED_STATE = StateConfig("paused", "🟡", "PAUSED", "holding last position — awaiting input", _C.YELLOW)
    _CORRECTING_STATE = StateConfig(
        "correcting", "🔴", "CORRECTING", "human teleop active — recording correction", _C.RED
    )

    def __init__(
        self,
        record_autonomous: bool,
        num_episodes: int,
        episode_duration_s: float,
        input_device: str,
        pause_key: str,
        correction_key: str,
        upload_key: str,
    ) -> None:
        mode = "continuous recording" if record_autonomous else "corrections only"
        auto_desc = "policy running — recording" if record_autonomous else "policy running — no recording"
        states = [
            StateConfig("autonomous", "🟢", "AUTONOMOUS", auto_desc, _C.GREEN),
            self._PAUSED_STATE,
            self._CORRECTING_STATE,
        ]
        controls = [
            ControlConfig(pause_key, "AUTONOMOUS ↔ PAUSED    pause / resume policy"),
            ControlConfig(correction_key, "PAUSED ↔ CORRECTING   start / stop correction"),
            ControlConfig(upload_key, "push dataset to Hub"),
            ControlConfig("ESC", "stop session"),
        ]
        info = [f"Target: {num_episodes} episodes  |  Input: {input_device}"]
        if record_autonomous:
            info.append(f"Episode rotation: ~{episode_duration_s:.0f}s")
        super().__init__(f"dagger  [{mode}]", states, controls, info)


if __name__ == "__main__":
    dagger_display = DAggerDisplay(
        record_autonomous=False,
        num_episodes=20,
        episode_duration_s=30,
        input_device="keyboard",
        pause_key="SPACE",
        correction_key="TAB",
        upload_key="ENTER",
    )
    dagger_display.show_banner()
    dagger_display.show_state("paused")
    dagger_display.show_state("correcting")
    dagger_display.show_state("paused")
    dagger_display.show_state("autonomous")
