#!/usr/bin/env python
"""Interactively teleoperate a LIBERO simulation while scoring a reward model.

Example:

```bash
python examples/tree_search/interactive_libero_reward_teleop.py \
    --checkpoint=/path/to/best_reward_model.pt \
    --suite=libero_object \
    --task_id=7 \
    --device=cuda
```
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

# LIBERO / robosuite creates an offscreen MuJoCo context. Default to EGL before
# importing the LIBERO env stack; otherwise local GUI sessions can fall back to
# GLX/GLFW and collide with pygame/OpenCV windows.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from lerobot.envs.libero import LiberoEnv, _get_suite

from policy_inference_api import (
    RewardModelScorer,
    _load_task_language_map,
    _translate_task_language,
)
from vlm_sequence_prompt_probe import _get_libero_task

logger = logging.getLogger(__name__)


def _make_scorer(args: argparse.Namespace, device: torch.device) -> RewardModelScorer:
    cfg = SimpleNamespace(
        reward_model_checkpoint=args.checkpoint,
        reward_scene_image_keys=args.reward_scene_image_keys,
        reward_wrist_image_keys=args.reward_wrist_image_keys,
        reward_use_rendered_fallback=True,
    )
    return RewardModelScorer(cfg, device=device)


def _make_env(args: argparse.Namespace) -> LiberoEnv:
    suite = _get_suite(args.suite)
    return LiberoEnv(
        task_suite=suite,
        task_id=args.task_id,
        task_suite_name=args.suite,
        episode_length=args.episode_length,
        camera_name=args.camera_name,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=args.observation_width,
        observation_height=args.observation_height,
        visualization_width=args.visualization_width,
        visualization_height=args.visualization_height,
        init_states=args.init_states,
        episode_index=args.init_state_index,
        n_envs=1,
        camera_name_mapping=None,
        num_steps_wait=args.num_steps_wait,
        control_mode=args.control_mode,
        is_libero_plus=args.is_libero_plus,
    )


def _task_text(args: argparse.Namespace) -> tuple[str, str]:
    task_metadata = _get_libero_task(args.suite, args.task_id)
    original = str(task_metadata["language"])
    translations = _load_task_language_map(args.task_language_map)
    translated = _translate_task_language(
        original,
        suite=args.suite,
        task_id=args.task_id,
        translations=translations,
    )
    return original, translated


def _action_for_key(key: int, args: argparse.Namespace, gripper: float) -> tuple[np.ndarray, float, str | None]:
    action = np.zeros(7, dtype=np.float32)
    action[6] = gripper
    command: str | None = None

    if key < 0:
        return action, gripper, command

    key_char = chr(key & 0xFF).lower()
    t = float(args.translation_scale)
    r = float(args.rotation_scale)

    if key_char == "w":
        action[0] = t
    elif key_char == "s":
        action[0] = -t
    elif key_char == "a":
        action[1] = t
    elif key_char == "d":
        action[1] = -t
    elif key_char == "q":
        action[2] = t
    elif key_char == "e":
        action[2] = -t
    elif key_char == "u":
        action[3] = r
    elif key_char == "o":
        action[3] = -r
    elif key_char == "i":
        action[4] = r
    elif key_char == "k":
        action[4] = -r
    elif key_char == "j":
        action[5] = r
    elif key_char == "l":
        action[5] = -r
    elif key_char == "z":
        gripper = float(args.close_gripper_value)
        action[6] = gripper
    elif key_char == "x":
        gripper = float(args.open_gripper_value)
        action[6] = gripper
    elif key_char == "r":
        command = "reset"
    elif key_char == "h":
        command = "help"
    elif key == 27:
        command = "quit"

    return np.clip(action, -1.0, 1.0), gripper, command


def _overlay_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    canvas = frame.copy()
    line_height = 22
    pad = 8
    box_height = pad * 2 + line_height * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], box_height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0.0, dst=canvas)
    for i, line in enumerate(lines):
        y = pad + 16 + i * line_height
        cv2.putText(
            canvas,
            line,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


class OpenCvDisplay:
    def __init__(self, args: argparse.Namespace) -> None:
        import cv2

        self.cv2 = cv2
        self.args = args
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, args.visualization_width, args.visualization_height)

    def step(self, frame_rgb: np.ndarray, lines: list[str], gripper: float) -> tuple[np.ndarray, float, str | None]:
        display_bgr = self.cv2.cvtColor(frame_rgb, self.cv2.COLOR_RGB2BGR)
        self.cv2.imshow(self.args.window_name, _overlay_text(display_bgr, lines))
        key = self.cv2.waitKey(max(1, int(self.args.wait_ms)))
        return _action_for_key(key, self.args, gripper)

    def close(self) -> None:
        self.cv2.destroyAllWindows()


class PygameDisplay:
    def __init__(self, args: argparse.Namespace) -> None:
        import pygame

        self.pygame = pygame
        self.args = args
        pygame.init()
        pygame.display.set_caption(args.window_name)
        self.screen = pygame.display.set_mode((args.visualization_width, args.visualization_height))
        self.font = pygame.font.Font(None, 22)

    def _action_from_pressed(self, gripper: float) -> tuple[np.ndarray, float, str | None]:
        pygame = self.pygame
        action = np.zeros(7, dtype=np.float32)
        command: str | None = None
        t = float(self.args.translation_scale)
        r = float(self.args.rotation_scale)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                command = "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    command = "quit"
                elif event.key == pygame.K_r:
                    command = "reset"
                elif event.key == pygame.K_h:
                    command = "help"

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_w]:
            action[0] += t
        if pressed[pygame.K_s]:
            action[0] -= t
        if pressed[pygame.K_a]:
            action[1] += t
        if pressed[pygame.K_d]:
            action[1] -= t
        if pressed[pygame.K_q]:
            action[2] += t
        if pressed[pygame.K_e]:
            action[2] -= t
        if pressed[pygame.K_u]:
            action[3] += r
        if pressed[pygame.K_o]:
            action[3] -= r
        if pressed[pygame.K_i]:
            action[4] += r
        if pressed[pygame.K_k]:
            action[4] -= r
        if pressed[pygame.K_j]:
            action[5] += r
        if pressed[pygame.K_l]:
            action[5] -= r
        if pressed[pygame.K_z]:
            gripper = float(self.args.close_gripper_value)
        if pressed[pygame.K_x]:
            gripper = float(self.args.open_gripper_value)
        action[6] = gripper

        return np.clip(action, -1.0, 1.0), gripper, command

    def step(self, frame_rgb: np.ndarray, lines: list[str], gripper: float) -> tuple[np.ndarray, float, str | None]:
        pygame = self.pygame
        frame = np.ascontiguousarray(frame_rgb)
        surface = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
        surface = pygame.transform.smoothscale(
            surface,
            (self.args.visualization_width, self.args.visualization_height),
        )
        self.screen.blit(surface, (0, 0))

        overlay = pygame.Surface((self.args.visualization_width, 24 * len(lines) + 12), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        for i, line in enumerate(lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (8, 8 + i * 24))
        pygame.display.flip()
        pygame.time.wait(max(1, int(self.args.wait_ms)))
        return self._action_from_pressed(gripper)

    def close(self) -> None:
        self.pygame.quit()


class TkDisplay:
    def __init__(self, args: argparse.Namespace) -> None:
        import tkinter as tk
        from PIL import Image, ImageDraw, ImageFont, ImageTk

        self.tk = tk
        self.image_module = Image
        self.image_draw = ImageDraw
        self.image_font = ImageFont
        self.image_tk = ImageTk
        self.args = args
        self.root = tk.Tk()
        self.root.title(args.window_name)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.held_keys: set[str] = set()
        self.command: str | None = None
        self.photo = None
        self.font = ImageFont.load_default()

        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.focus_force()

    def _on_key_press(self, event: Any) -> None:
        key = str(event.keysym).lower()
        self.held_keys.add(key)
        if key == "escape":
            self.command = "quit"
        elif key == "r":
            self.command = "reset"
        elif key == "h":
            self.command = "help"

    def _on_key_release(self, event: Any) -> None:
        self.held_keys.discard(str(event.keysym).lower())

    def _on_close(self) -> None:
        self.command = "quit"

    def _action_from_pressed(self, gripper: float) -> tuple[np.ndarray, float, str | None]:
        action = np.zeros(7, dtype=np.float32)
        t = float(self.args.translation_scale)
        r = float(self.args.rotation_scale)

        if "w" in self.held_keys:
            action[0] += t
        if "s" in self.held_keys:
            action[0] -= t
        if "a" in self.held_keys:
            action[1] += t
        if "d" in self.held_keys:
            action[1] -= t
        if "q" in self.held_keys:
            action[2] += t
        if "e" in self.held_keys:
            action[2] -= t
        if "u" in self.held_keys:
            action[3] += r
        if "o" in self.held_keys:
            action[3] -= r
        if "i" in self.held_keys:
            action[4] += r
        if "k" in self.held_keys:
            action[4] -= r
        if "j" in self.held_keys:
            action[5] += r
        if "l" in self.held_keys:
            action[5] -= r
        if "z" in self.held_keys:
            gripper = float(self.args.close_gripper_value)
        if "x" in self.held_keys:
            gripper = float(self.args.open_gripper_value)
        action[6] = gripper

        command = self.command
        self.command = None
        return np.clip(action, -1.0, 1.0), gripper, command

    def _draw_overlay(self, frame_rgb: np.ndarray, lines: list[str]) -> Any:
        image = self.image_module.fromarray(np.ascontiguousarray(frame_rgb))
        image = image.resize((self.args.visualization_width, self.args.visualization_height))
        overlay_height = 22 * len(lines) + 12
        overlay = self.image_module.new("RGBA", image.size, (0, 0, 0, 0))
        draw = self.image_draw.Draw(overlay)
        draw.rectangle((0, 0, image.size[0], overlay_height), fill=(0, 0, 0, 150))
        for i, line in enumerate(lines):
            draw.text((8, 8 + i * 22), line, fill=(255, 255, 255, 255), font=self.font)
        return self.image_module.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def step(self, frame_rgb: np.ndarray, lines: list[str], gripper: float) -> tuple[np.ndarray, float, str | None]:
        self.root.update_idletasks()
        self.root.update()
        image = self._draw_overlay(frame_rgb, lines)
        self.photo = self.image_tk.PhotoImage(image=image)
        self.label.configure(image=self.photo)
        self.root.update_idletasks()
        self.root.update()
        time.sleep(max(1, int(self.args.wait_ms)) / 1000.0)
        return self._action_from_pressed(gripper)

    def close(self) -> None:
        with suppress(Exception):
            self.root.destroy()


def _make_display(args: argparse.Namespace) -> OpenCvDisplay | PygameDisplay | TkDisplay:
    if args.display_backend == "tk":
        return TkDisplay(args)
    if args.display_backend == "opencv":
        return OpenCvDisplay(args)
    if args.display_backend == "pygame":
        return PygameDisplay(args)
    raise ValueError(f"Unsupported display backend: {args.display_backend}")


def _print_help() -> None:
    print(
        "\nControls:\n"
        "  W/S: action[0] +/- translation\n"
        "  A/D: action[1] +/- translation\n"
        "  Q/E: action[2] +/- translation\n"
        "  U/O: action[3] +/- rotation\n"
        "  I/K: action[4] +/- rotation\n"
        "  J/L: action[5] +/- rotation\n"
        "  Z/X: close/open gripper persistently\n"
        "  R: reset episode\n"
        "  H: print this help\n"
        "  ESC: quit\n",
        flush=True,
    )


def _write_score(log_file: Any, payload: dict[str, Any]) -> None:
    if log_file is None:
        return
    log_file.write(json.dumps(payload) + "\n")
    log_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--task_language_map", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episode_length", type=int, default=None)
    parser.add_argument("--init_states", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init_state_index", type=int, default=0)
    parser.add_argument("--is_libero_plus", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--control_mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--camera_name", default="agentview_image,robot0_eye_in_hand_image")
    parser.add_argument("--observation_width", type=int, default=360)
    parser.add_argument("--observation_height", type=int, default=360)
    parser.add_argument("--visualization_width", type=int, default=640)
    parser.add_argument("--visualization_height", type=int, default=480)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--translation_scale", type=float, default=0.75)
    parser.add_argument("--rotation_scale", type=float, default=0.75)
    parser.add_argument("--open_gripper_value", type=float, default=-1.0)
    parser.add_argument("--close_gripper_value", type=float, default=1.0)
    parser.add_argument("--score_every_steps", type=int, default=1)
    parser.add_argument("--reward_scene_image_keys", default="image,base_0_rgb")
    parser.add_argument("--reward_wrist_image_keys", default="image2,left_wrist_0_rgb")
    parser.add_argument(
        "--display_backend",
        choices=["tk", "pygame", "opencv"],
        default="tk",
        help=(
            "Interactive viewer backend. Tk is the default because it avoids OpenGL/GLX; "
            "pygame and OpenCV can collide with robosuite EGL on some desktops."
        ),
    )
    parser.add_argument("--window_name", default="LIBERO reward teleop")
    parser.add_argument("--wait_ms", type=int, default=30)
    parser.add_argument("--output_log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    device = torch.device(args.device)
    env = _make_env(args)
    scorer = _make_scorer(args, device)
    display_backend = None
    original_task, task = _task_text(args)
    print(f"Task: {task}", flush=True)
    if task != original_task:
        print(f"Original LIBERO task: {original_task}", flush=True)
    _print_help()

    log_file = None
    if args.output_log is not None:
        args.output_log.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.output_log.open("w")

    step = 0
    last_score = 0.0
    last_reason = "not scored yet"
    gripper = float(args.open_gripper_value)
    scene_history: list[np.ndarray] = []

    try:
        # Let robosuite create and bind its offscreen EGL context before any
        # GUI toolkit initializes a windowing context.
        observation, info = env.reset()
        rendered = env.render()
        current_scene = scorer.current_scene_image(image=rendered, observation=observation)
        scene_history = scorer.extend_scene_history([], current_scene)
        display_backend = _make_display(args)

        while True:
            rendered = env.render()
            should_score = step % max(1, int(args.score_every_steps)) == 0
            if should_score:
                score_result = scorer.score(
                    image=rendered,
                    observation=observation,
                    task=task,
                    success=bool(info.get("is_success", False)),
                    metadata={"step": step, "interactive": True},
                    scene_history=scorer.prior_history_for_current(scene_history),
                )
                last_score = score_result.score
                last_reason = score_result.reason
                _write_score(
                    log_file,
                    {
                        "time_s": time.time(),
                        "step": step,
                        "score": last_score,
                        "reason": last_reason,
                        "success": bool(info.get("is_success", False)),
                        "task": task,
                        "original_task": original_task,
                        "gripper": gripper,
                    },
                )

            display = scorer.current_scene_image(image=rendered, observation=observation)
            lines = [
                f"step={step} score={last_score:.3f} success={bool(info.get('is_success', False))}",
                f"gripper={gripper:.2f} reason={last_reason}",
                f"task={task[:90]}",
                "WASD/QE move | UO/IK/JL rotate | Z/X grip | R reset | H help | ESC quit",
            ]
            action, gripper, command = display_backend.step(display, lines, gripper)
            if command == "quit":
                break
            if command == "help":
                _print_help()
                continue
            if command == "reset":
                observation, info = env.reset()
                step = 0
                last_score = 0.0
                last_reason = "reset"
                gripper = float(args.open_gripper_value)
                rendered = env.render()
                current_scene = scorer.current_scene_image(image=rendered, observation=observation)
                scene_history = scorer.extend_scene_history([], current_scene)
                continue

            observation, _env_reward, terminated, truncated, info = env.step_no_reset(action)
            step += 1
            rendered_after_step = env.render()
            current_scene = scorer.current_scene_image(image=rendered_after_step, observation=observation)
            scene_history = scorer.extend_scene_history(scene_history, current_scene)
            if terminated or truncated:
                print(
                    f"Episode ended: step={step} terminated={terminated} truncated={truncated} "
                    f"success={bool(info.get('is_success', False))}",
                    flush=True,
                )
    finally:
        if log_file is not None:
            log_file.close()
        # Robosuite / PyOpenGL EGL objects are noisy if Python tears them down
        # after the graphics backend has already been finalized. Close and
        # collect while the process is still in a known-good state.
        with suppress(Exception):
            env.close()
        env = None
        gc.collect()
        if display_backend is not None:
            with suppress(Exception):
                display_backend.close()
        display_backend = None
        gc.collect()


if __name__ == "__main__":
    main()
