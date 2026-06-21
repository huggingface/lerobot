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

from __future__ import annotations

########################################################################################
# Utilities
########################################################################################
import logging
import os
import select
import subprocess
import sys
import termios
import threading
import time
import traceback
import webbrowser
from contextlib import nullcontext
from copy import copy
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.utils.import_utils import _deepdiff_available, require_package

if TYPE_CHECKING or _deepdiff_available:
    from deepdiff import DeepDiff
else:
    DeepDiff = None

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset


class _TerminalKeyboardListener:
    """Fallback listener for terminals where pynput does not receive macOS key events."""

    def __init__(self, events: dict[str, bool]):
        self.events = events
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_settings = None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            return False

        try:
            import tty

            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception as exc:
            logging.debug("Could not start terminal keyboard fallback: %s", exc)
            return False

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._fd is not None and self._old_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception as exc:
                logging.debug("Could not restore terminal settings: %s", exc)

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                char = sys.stdin.read(1)
                if char != "\x1b":
                    continue

                sequence = char
                for _ in range(2):
                    readable, _, _ = select.select([sys.stdin], [], [], 0.02)
                    if not readable:
                        break
                    sequence += sys.stdin.read(1)

                if sequence == "\x1b[C":
                    print("Right arrow key pressed. Exiting loop...")
                    self.events["exit_early"] = True
                elif sequence == "\x1b[D":
                    print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                    self.events["rerecord_episode"] = True
                    self.events["exit_early"] = True
                elif sequence == "\x1b":
                    print("Escape key pressed. Stopping data recording...")
                    self.events["stop_recording"] = True
                    self.events["exit_early"] = True
            except Exception as exc:
                logging.debug("Error handling terminal key press: %s", exc)


class _CombinedKeyboardListener:
    def __init__(self, *listeners):
        self.listeners = [listener for listener in listeners if listener is not None]

    def stop(self) -> None:
        for listener in self.listeners:
            try:
                listener.stop()
            except Exception as exc:
                logging.debug("Could not stop listener %s: %s", listener, exc)


class _RecordControlWindow:
    """Small click UI for episode controls when keyboard hooks are unreliable."""

    def __init__(self, events: dict[str, bool]):
        self.events = events
        self._command_fpath = f"/tmp/lerobot_record_controls_{os.getpid()}.txt"
        self._process: subprocess.Popen | None = None
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None

    def start(self) -> bool:
        script = r'''
import pathlib
import sys
import tkinter as tk

command_fpath = pathlib.Path(sys.argv[1])

def write_command(command):
    command_fpath.write_text(command)

root = tk.Tk()
root.title("LeRobot Recording Controls")
root.attributes("-topmost", True)
root.resizable(False, False)
root.geometry("+80+120")

frame = tk.Frame(root, padx=16, pady=14)
frame.pack()

tk.Label(frame, text="LeRobot Recording Controls", font=("Helvetica", 16, "bold")).pack(pady=(0, 10))
tk.Label(frame, text="Click buttons to control episodes.", font=("Helvetica", 12)).pack(pady=(0, 12))

tk.Button(frame, text="Finish Episode", command=lambda: write_command("finish"), width=24, height=2).pack(pady=(0, 8))
tk.Button(frame, text="Rerecord Episode", command=lambda: write_command("rerecord"), width=24, height=2).pack(pady=(0, 8))
tk.Button(frame, text="Stop Recording", command=lambda: write_command("stop"), width=24, height=2).pack()

root.lift()
root.focus_force()
root.mainloop()
'''
        try:
            try:
                os.remove(self._command_fpath)
            except FileNotFoundError:
                pass
            self._process = subprocess.Popen([sys.executable, "-c", script, self._command_fpath])
            self._poll_thread = threading.Thread(target=self._poll_commands, daemon=True)
            self._poll_thread.start()
            print("LeRobot Recording Controls popup opened.")
            return True
        except Exception as exc:
            logging.warning("Could not open recording control UI: %s", exc)
            return False

    def stop(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=0.5)
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        try:
            os.remove(self._command_fpath)
        except FileNotFoundError:
            pass

    def _poll_commands(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not os.path.exists(self._command_fpath):
                    time.sleep(0.1)
                    continue
                with open(self._command_fpath) as f:
                    command = f.read().strip()
                os.remove(self._command_fpath)
                if command == "finish":
                    print("Finish Episode clicked. Exiting loop...")
                    self.events["exit_early"] = True
                elif command == "rerecord":
                    print("Rerecord Episode clicked. Exiting loop and rerecording the last episode...")
                    self.events["rerecord_episode"] = True
                    self.events["exit_early"] = True
                elif command == "stop":
                    print("Stop Recording clicked. Stopping data recording...")
                    self.events["stop_recording"] = True
                    self.events["exit_early"] = True
            except Exception as exc:
                logging.debug("Error reading recording control UI command: %s", exc)
            time.sleep(0.1)


class _RecordControlWebServer:
    """Local browser controls for episode recording."""

    def __init__(self, events: dict[str, bool]):
        self.events = events
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url: str | None = None

    def start(self) -> bool:
        events = self.events

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.startswith("/finish"):
                    print("Finish Episode clicked. Exiting loop...")
                    events["exit_early"] = True
                    self._send_page("Finish Episode clicked.")
                elif self.path.startswith("/rerecord"):
                    print("Rerecord Episode clicked. Exiting loop and rerecording the last episode...")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                    self._send_page("Rerecord Episode clicked.")
                elif self.path.startswith("/stop"):
                    print("Stop Recording clicked. Stopping data recording...")
                    events["stop_recording"] = True
                    events["exit_early"] = True
                    self._send_page("Stop Recording clicked.")
                else:
                    self._send_page("Ready.")

            def log_message(self, format: str, *args) -> None:
                return

            def _send_page(self, status: str) -> None:
                html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LeRobot Recording Controls</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; max-width: 460px; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    p {{ color: #444; }}
    a {{ display: block; margin: 12px 0; padding: 16px 18px; border-radius: 8px; text-decoration: none;
         color: white; background: #2563eb; font-size: 18px; font-weight: 700; text-align: center; }}
    a.secondary {{ background: #d97706; }}
    a.danger {{ background: #dc2626; }}
    .status {{ margin-top: 18px; padding: 12px; background: #f1f5f9; border-radius: 8px; }}
  </style>
</head>
<body tabindex="0">
  <h1>LeRobot Recording Controls</h1>
  <p>Press Space or click Finish Episode for both task completion and reset completion.</p>
  <a href="/finish" id="finish">Finish Episode</a>
  <a class="secondary" href="/rerecord">Rerecord Episode</a>
  <a class="danger" href="/stop">Stop Recording</a>
  <div class="status" id="status">{status}</div>
  <script>
    document.body.focus();
    let sending = false;
    async function finishEpisode() {{
      if (sending) return;
      sending = true;
      document.getElementById("status").textContent = "Finish Episode sent.";
      try {{
        await fetch("/finish");
      }} finally {{
        setTimeout(() => {{ sending = false; }}, 500);
      }}
    }}
    document.addEventListener("keydown", (event) => {{
      if (event.code === "Space") {{
        event.preventDefault();
        finishEpisode();
      }}
    }});
    window.addEventListener("focus", () => document.body.focus());
  </script>
</body>
</html>"""
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        try:
            try:
                self._server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
            except OSError:
                self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            port = self._server.server_address[1]
            self.url = f"http://127.0.0.1:{port}"
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            print(f"LeRobot browser recording controls: {self.url}", flush=True)
            webbrowser.open(self.url)
            return True
        except Exception as exc:
            logging.warning("Could not start browser recording controls: %s", exc)
            return False

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import Robot
from lerobot.types import PolicyAction


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def init_keyboard_listener():
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (right arrow, left arrow, escape) to control
    the program flow during execution, such as stopping recording or exiting loops. It gracefully
    handles headless environments where keyboard listening is not possible.

    Returns:
        A tuple containing:
        - The `pynput.keyboard.Listener` instance, or `None` if in a headless environment.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    pynput_listener = None
    if not is_headless():
        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    if key == keyboard.Key.right:
                        print("Right arrow key pressed. Exiting loop...")
                        events["exit_early"] = True
                    elif key == keyboard.Key.left:
                        print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                        events["rerecord_episode"] = True
                        events["exit_early"] = True
                    elif key == keyboard.Key.esc:
                        print("Escape key pressed. Stopping data recording...")
                        events["stop_recording"] = True
                        events["exit_early"] = True
                except Exception as e:
                    print(f"Error handling key press: {e}")

            pynput_listener = keyboard.Listener(on_press=on_press)
            pynput_listener.start()
        except Exception as exc:
            logging.warning("pynput keyboard controls unavailable: %s", exc)
    else:
        logging.warning("Headless keyboard detection failed; browser recording controls will still be enabled.")

    terminal_listener = _TerminalKeyboardListener(events)
    if terminal_listener.start():
        logging.info("Terminal keyboard fallback enabled for arrow-key recording controls.")
    else:
        terminal_listener = None

    control_window = None

    web_controls = _RecordControlWebServer(events)
    if web_controls.start():
        logging.info("Browser recording controls enabled at %s.", web_controls.url)
    else:
        web_controls = None

    listener = _CombinedKeyboardListener(pynput_listener, terminal_listener, control_window, web_controls)

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    require_package("deepdiff", extra="deepdiff-dep")

    from lerobot.utils.constants import DEFAULT_FEATURES

    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )


########################################################################################
# Teleoperator smooth handover helpers
# NOTE(Maxime): These functions use minimal type hints to maintain compatibility with utils
# being a root module.
########################################################################################


def teleop_supports_feedback(teleop) -> bool:
    """Return True when the teleop can receive position feedback (is actuated).

    Actuated teleops (e.g. SO-101, OpenArmMini) have non-empty ``feedback_features``
    and expose ``enable_torque`` / ``disable_torque`` motor-control methods.

    TODO(Maxime): See if it is possible to unify this interface across teleops instead of duck-typing.
    """
    return (
        bool(teleop.feedback_features)
        and hasattr(teleop, "disable_torque")
        and hasattr(teleop, "enable_torque")
    )


def teleop_smooth_move_to(teleop, target_pos: dict, duration_s: float = 2.0, fps: int = 30) -> None:
    """Smoothly move an actuated teleop to ``target_pos`` via linear interpolation.

    Requires the teleoperator to support feedback (i.e. have non-empty
    ``feedback_features`` and implement ``disable_torque`` / ``enable_torque``).

    ``target_pos`` is expected to be in the teleop's action/feedback key space.
    For homogeneous setups (e.g. SO-101 leader + SO-101 follower) this matches
    the robot action key space directly.

    TODO(Maxime): This blocks up to ``duration_s`` seconds; during this time the
    follower robot does not receive new actions, which could be an issue on LeKiwi.
    """
    teleop.enable_torque()
    current = teleop.get_action()
    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {
            k: current[k] * (1 - t) + target_pos[k] * t if k in target_pos else current[k] for k in current
        }
        teleop.send_feedback(interp)
        time.sleep(1 / fps)


def follower_smooth_move_to(
    robot, current: dict, target: dict, duration_s: float = 1.0, fps: int = 30
) -> None:
    """Smoothly move the follower robot from ``current`` to ``target`` action.

    Used when the teleop is non-actuated: instead of driving the leader arm to
    the follower, the follower is brought to the teleop's current pose so the
    robot meets the operator's hand rather than jumping to it on the first frame.

    Both ``current`` and ``target`` must be in the robot action key space
    (i.e. the output of ``robot_action_processor``).
    """
    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {k: current[k] * (1 - t) + target[k] * t if k in target else current[k] for k in current}
        robot.send_action(interp)
        time.sleep(1 / fps)
