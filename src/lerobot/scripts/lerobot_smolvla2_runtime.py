#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""``lerobot-smolvla2-runtime`` — interactive REPL for trained SmolVLA2.

Drives the multi-rate runtime defined in
:mod:`lerobot.policies.smolvla2.inference`. Stdin becomes the user
channel: type a task, then natural-language interjections / questions.
The runtime prints state changes (plan / subtask / memory / vqa /
speech) as they happen.

Examples
--------

Dry run on a Hub checkpoint, no robot connected — useful for sanity-
checking text generation::

    uv run lerobot-smolvla2-runtime \\
        --policy.path=pepijn223/smolvla2_hirobot_super_poulain_tool2 \\
        --no_robot \\
        --task="please clean the kitchen"

Same, but feed real frames from an annotated dataset so plan / subtask
/ memory / VQA generation runs against actual video + state::

    uv run lerobot-smolvla2-runtime \\
        --policy.path=pepijn223/smolvla2_hirobot_super_poulain_tool2 \\
        --dataset.repo_id=pepijn223/super_poulain_annotated \\
        --dataset.episode=0 \\
        --no_robot \\
        --task="please clean the kitchen"

With a real robot::

    uv run lerobot-smolvla2-runtime \\
        --policy.path=... \\
        --robot.type=so101 --robot.port=/dev/tty.usbmodem... \\
        --tts.voice=alba

``--policy.path`` accepts either a local directory or a Hugging Face
Hub repo id. ``--dataset.repo_id`` likewise.

Tool dispatch (TTS via ``SayTool``) is enabled by default when
``pocket-tts`` is installed; pass ``--no_tts`` to disable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Callable

logger = logging.getLogger("lerobot.smolvla2.runtime")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="lerobot-smolvla2-runtime",
        description="Interactive REPL runtime for a trained SmolVLA2 checkpoint.",
    )
    p.add_argument(
        "--policy.path",
        dest="policy_path",
        type=str,
        required=True,
        help=(
            "Local directory or Hugging Face Hub repo id pointing at a "
            "trained SmolVLA2 ``pretrained_model``."
        ),
    )
    p.add_argument(
        "--dataset.repo_id",
        dest="dataset_repo_id",
        type=str,
        default=None,
        help=(
            "Optional dataset (local path or Hub repo id) used to drive "
            "observations during dry-run inference. When set, the runtime "
            "reads camera frames + state from the chosen episode and feeds "
            "them into all forward passes — so plan / subtask / memory / "
            "VQA generation see the same visual context the policy was "
            "trained on."
        ),
    )
    p.add_argument(
        "--dataset.episode",
        dest="dataset_episode",
        type=int,
        default=0,
        help="Episode index to walk through (default: 0).",
    )
    p.add_argument(
        "--dataset.start_frame",
        dest="dataset_start_frame",
        type=int,
        default=0,
        help="Frame index within the episode to start from (default: 0).",
    )
    p.add_argument(
        "--dataset.advance_per_tick",
        dest="dataset_advance_per_tick",
        type=int,
        default=1,
        help=(
            "How many dataset frames to advance per runtime tick. The "
            "default of 1 means the runtime walks the episode forward "
            "frame by frame; set to 0 to freeze on ``start_frame``."
        ),
    )
    p.add_argument(
        "--task",
        dest="task",
        type=str,
        default=None,
        help="Initial task. If omitted, the first stdin line is treated as the task.",
    )
    p.add_argument(
        "--no_robot",
        action="store_true",
        help="Skip robot connection — language-only / dry-run mode.",
    )
    # --- Real-robot mode args ----------------------------------------
    # Setting ``--robot.type`` flips the runtime into autonomous mode:
    # it connects to the robot, builds an observation provider that
    # reads ``robot.get_observation()`` instead of dataset frames, and
    # an action executor that postprocesses (denormalises) the policy's
    # output and calls ``robot.send_action(...)`` at ``--ctrl_hz``. The
    # high-level REPL-style stdin still works in a background thread
    # for interjections / VQA.
    p.add_argument(
        "--robot.type",
        dest="robot_type",
        type=str,
        default=None,
        help=(
            "Robot config choice (e.g. ``so101``, ``so101_follower``). "
            "When set, the runtime drives the actual robot at "
            "``--ctrl_hz`` instead of running the dataset-driven dry-run "
            "REPL. Implies ``--autonomous`` unless ``--no_robot`` is also "
            "passed (in which case the flag is ignored). See "
            "``lerobot.robots`` for available choices."
        ),
    )
    p.add_argument(
        "--robot.port",
        dest="robot_port",
        type=str,
        default=None,
        help="Serial port for the robot (e.g. ``/dev/tty.usbmodem...``).",
    )
    p.add_argument(
        "--robot.id",
        dest="robot_id",
        type=str,
        default=None,
        help="Optional robot identifier (passed through to ``RobotConfig.id``).",
    )
    p.add_argument(
        "--robot.cameras",
        dest="robot_cameras",
        type=str,
        default=None,
        help=(
            "Optional JSON dict describing camera configs to attach to "
            "the robot (e.g. ``'{\"top\": {\"type\": \"opencv\", \"index\": 0}}'``). "
            "Camera keys MUST match the ``observation.images.*`` features "
            "the policy was trained on."
        ),
    )
    p.add_argument(
        "--robot.max_relative_target",
        dest="robot_max_relative_target",
        type=str,
        default=None,
        help=(
            "Safety clip on per-motor relative motion, passed through to "
            "``RobotConfig.max_relative_target``. Accepts either a float "
            "(applied to every motor — e.g. ``5.0`` degrees) or a JSON "
            "object mapping motor names to caps "
            "(e.g. ``'{\"shoulder_pan\": 5, \"gripper\": 30}'``). The "
            "robot driver clips each commanded position relative to the "
            "current measured position before sending — same kill-switch "
            "``lerobot-record`` uses. Default ``None`` = no clipping."
        ),
    )
    p.add_argument(
        "--auto_start",
        action="store_true",
        help=(
            "Skip the ``Press ENTER to start`` confirmation prompt before "
            "the autonomous control loop begins. Off by default — having "
            "to confirm catches a lot of stupid mistakes (wrong policy, "
            "wrong robot, robot not at home pose)."
        ),
    )
    p.add_argument(
        "--no_tts",
        action="store_true",
        help="Disable the ``say`` tool dispatch.",
    )
    p.add_argument(
        "--tts.voice",
        dest="tts_voice",
        type=str,
        default="alba",
        help="Pocket-tts voice name (or path to a .wav for cloning).",
    )
    p.add_argument(
        "--chunk_hz",
        type=float,
        default=1.0,
        help=(
            "Action-chunk generation rate (Hz). Default ``1.0`` — one "
            "new chunk per second. Lower = less inference cost / "
            "smoother behaviour but longer reaction time to changes. "
            "Higher = fresher actions / more inference cost; cap at "
            "~1/(forward-pass latency)."
        ),
    )
    p.add_argument(
        "--ctrl_hz", type=float, default=50.0, help="Action dispatch rate."
    )
    p.add_argument(
        "--high_level_hz",
        type=float,
        default=1.0,
        help="High-level subtask generation rate.",
    )
    p.add_argument(
        "--max_ticks",
        type=int,
        default=None,
        help="Stop after N ticks (debug / smoke-test).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def _load_policy_and_preprocessor(
    policy_path: str,
    dataset_repo_id: str | None,
) -> tuple[Any, Any, Any, Any]:
    """Load a SmolVLA2 checkpoint (local path or Hub repo id).

    Returns ``(policy, preprocessor, postprocessor, ds_meta)``.
    ``preprocessor`` / ``postprocessor`` / ``ds_meta`` are ``None``
    when no dataset is provided (rare — needed for autonomous robot
    mode to have action-denormalisation stats).
    """
    from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: PLC0415

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path

    ds_meta = None
    preprocessor = None
    postprocessor = None
    if dataset_repo_id is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: PLC0415

        ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
        policy = make_policy(cfg, ds_meta=ds_meta)
        # ``pretrained_path=None`` rebuilds fresh — the saved
        # ``policy_preprocessor.json`` doesn't round-trip
        # ``RenderMessagesStep.recipe``. Stats come from the dataset
        # the user is feeding through, so normalisation is consistent.
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=None,
            dataset_stats=ds_meta.stats,
        )
    else:
        from lerobot.policies.factory import get_policy_class  # noqa: PLC0415

        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(policy_path, config=cfg)
        policy.to(cfg.device)

    policy.eval()
    return policy, preprocessor, postprocessor, ds_meta


def _build_observation_provider(
    *,
    dataset_repo_id: str,
    episode: int,
    start_frame: int,
    advance_per_tick: int,
    preprocessor: Any,
    device: str,
) -> Callable[[], dict | None]:
    """Build a closure that feeds dataset frames into the runtime.

    Each call returns a preprocessed observation batch (images +
    state, batched, on the policy's device, normalized) suitable for
    ``policy.select_action`` and ``policy.select_message``. The
    closure walks the chosen episode forward by ``advance_per_tick``
    frames per call, looping back to the episode start when it falls
    off the end.

    The dataset's ``language_persistent`` / ``language_events``
    columns are stripped before the sample reaches the preprocessor,
    so ``RenderMessagesStep`` and ``SmolVLA2ChatTokenizerStep`` are
    no-ops; the runtime supplies its own messages from current state.
    """
    import torch  # noqa: PLC0415

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    ds = LeRobotDataset(dataset_repo_id, episodes=[episode])
    if len(ds) == 0:
        raise ValueError(
            f"Dataset {dataset_repo_id!r} episode {episode} is empty."
        )

    state = {"cursor": max(0, min(start_frame, len(ds) - 1))}

    def _provider() -> dict | None:
        idx = state["cursor"]
        if advance_per_tick > 0:
            state["cursor"] = (idx + advance_per_tick) % len(ds)

        sample = ds[idx]
        # Strip the language columns so the preprocessor's render step
        # is a no-op — the runtime drives messages itself.
        for k in ("language_persistent", "language_events"):
            sample.pop(k, None)

        if preprocessor is not None:
            sample = preprocessor(sample)

        # Keep only observation keys; the runtime's text path will
        # merge these with its own lang_tokens / lang_masks.
        observation = {
            k: v
            for k, v in sample.items()
            if isinstance(k, str) and k.startswith("observation.")
        }
        # Defensive: if something further upstream forgot the batch
        # dim, add it now so downstream Tensor ops don't crash.
        for k, v in list(observation.items()):
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] != 1:
                # ``add_batch_dim`` already ran inside the preprocessor;
                # an unbatched tensor at this point means a step
                # somewhere produced an unbatched output. Best-effort
                # fix.
                if v.shape[0] != 1 and v.ndim < 4 and "image" not in k:
                    observation[k] = v.unsqueeze(0)
        # Move to device (the preprocessor's DeviceProcessorStep should
        # already have done this when ``preprocessor is not None``;
        # this is a belt-and-braces no-op in the common case).
        for k, v in list(observation.items()):
            if isinstance(v, torch.Tensor):
                observation[k] = v.to(device)
        return observation

    return _provider


def _bootstrap_state_from_dataset(
    *,
    dataset_repo_id: str,
    episode: int,
    start_frame: int,
) -> dict[str, str]:
    """Pull task / active plan / active memory / active subtask at ``start_frame``.

    The model is heavily memorised on the exact training prompts the
    recipe rendered from this dataset (canonical task wording,
    persistent atoms emitted earlier in the episode). Reconstructing
    that state at REPL startup lets the runtime's first prompt line
    up with what training looked like — without it the model sees an
    out-of-distribution prompt and falls back to its dominant
    training mode (VQA JSON spam).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    ds = LeRobotDataset(dataset_repo_id, episodes=[episode])
    if len(ds) == 0:
        return {}
    idx = max(0, min(start_frame, len(ds) - 1))
    sample = ds[idx]

    out: dict[str, str] = {}
    task = sample.get("task")
    if isinstance(task, str) and task.strip():
        out["task"] = task

    persistent = sample.get("language_persistent") or []
    # ``persistent`` is the broadcast slice of the episode; pick the
    # *latest* row of each style whose ``timestamp`` is ≤ the
    # frame's timestamp (matches the renderer's ``active_at``
    # semantics).
    try:
        frame_ts = (
            float(sample["timestamp"])
            if not hasattr(sample["timestamp"], "item")
            else sample["timestamp"].item()
        )
    except Exception:  # noqa: BLE001
        frame_ts = float("inf")

    by_style: dict[str, tuple[float, str]] = {}
    for row in persistent:
        style = row.get("style")
        ts = row.get("timestamp")
        content = row.get("content")
        if not (style and content) or ts is None:
            continue
        try:
            ts_f = float(ts)
        except (TypeError, ValueError):
            continue
        if ts_f > frame_ts:
            continue
        prev = by_style.get(style)
        if prev is None or ts_f >= prev[0]:
            by_style[style] = (ts_f, content)
    for style, (_, content) in by_style.items():
        if style in {"plan", "memory", "subtask"}:
            out[style] = content
    return out


def _build_robot(
    *,
    robot_type: str,
    robot_port: str | None,
    robot_id: str | None,
    robot_cameras_json: str | None,
    robot_max_relative_target: str | None,
):
    """Build and connect a robot from CLI args.

    Mirrors how ``lerobot-record`` builds a robot but takes the args
    flat from argparse instead of through draccus, so the runtime
    keeps its plain ``--key=value`` CLI surface. ``max_relative_target``
    is passed through to the RobotConfig — the driver itself clips each
    commanded joint position relative to the current measured one
    before issuing it on the bus.
    """
    import importlib  # noqa: PLC0415
    import json  # noqa: PLC0415
    import pkgutil  # noqa: PLC0415

    import lerobot.robots as _robots_pkg  # noqa: PLC0415
    from lerobot.robots import (  # noqa: PLC0415
        RobotConfig,
        make_robot_from_config,
    )

    # ``RobotConfig._choice_registry`` is populated lazily — each robot's
    # ``config_<name>.py`` calls ``@RobotConfig.register_subclass`` at
    # import time. ``lerobot.robots/__init__.py`` doesn't import the
    # individual robot packages, so ``get_choice_class(robot_type)``
    # raises ``KeyError`` until at least one robot module has been
    # imported. Mirror what ``make_robot_from_config`` does internally:
    # walk the robots package's submodules and import each so the
    # decorator side-effect runs. Slow only on the first call (~200ms
    # for ~10 dataclass modules); negligible for an autonomous run that
    # then loops at ctrl_hz for minutes.
    for _modinfo in pkgutil.iter_modules(_robots_pkg.__path__):
        if _modinfo.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"lerobot.robots.{_modinfo.name}")
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not import lerobot.robots.%s: %s", _modinfo.name, exc)

    try:
        cls = RobotConfig.get_choice_class(robot_type)
    except KeyError as exc:
        available = sorted(RobotConfig._choice_registry.keys())
        raise ValueError(
            f"Unknown robot type {robot_type!r}. Available choices: {available}"
        ) from exc
    kwargs: dict[str, Any] = {}
    if robot_port:
        kwargs["port"] = robot_port
    if robot_id:
        kwargs["id"] = robot_id
    if robot_cameras_json:
        try:
            cameras_raw = json.loads(robot_cameras_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"--robot.cameras must be a JSON object, got {robot_cameras_json!r}: {exc}"
            ) from exc
        # ``RobotConfig`` expects ``cameras: dict[str, CameraConfig]`` —
        # each inner value must be an actual ``CameraConfig`` subclass
        # instance, not a raw dict. Look up the matching subclass via
        # ``CameraConfig.get_choice_class(<type>)`` (registered by
        # ``@CameraConfig.register_subclass`` decorators on each camera
        # backend's config) and instantiate it. Mirror the lazy-import
        # pattern from above so the registry is populated.
        import lerobot.cameras as _cameras_pkg  # noqa: PLC0415
        from lerobot.cameras import CameraConfig  # noqa: PLC0415

        for _modinfo in pkgutil.iter_modules(_cameras_pkg.__path__):
            if _modinfo.name.startswith("_"):
                continue
            try:
                importlib.import_module(f"lerobot.cameras.{_modinfo.name}")
            except Exception as exc:  # noqa: BLE001
                logger.debug("could not import lerobot.cameras.%s: %s", _modinfo.name, exc)

        cameras: dict[str, Any] = {}
        for cam_name, cam_dict in cameras_raw.items():
            if not isinstance(cam_dict, dict):
                raise ValueError(
                    f"camera {cam_name!r} value must be a dict, got {cam_dict!r}"
                )
            cam_dict = dict(cam_dict)  # don't mutate caller's parsed JSON
            cam_type = cam_dict.pop("type", None)
            if cam_type is None:
                raise ValueError(
                    f"camera {cam_name!r} is missing a 'type' field "
                    f"(e.g. 'opencv', 'intelrealsense')"
                )
            try:
                cam_cls = CameraConfig.get_choice_class(cam_type)
            except KeyError as exc:
                available = sorted(CameraConfig._choice_registry.keys())
                raise ValueError(
                    f"camera {cam_name!r}: unknown type {cam_type!r}. "
                    f"Available choices: {available}"
                ) from exc
            cameras[cam_name] = cam_cls(**cam_dict)
        kwargs["cameras"] = cameras
    if robot_max_relative_target:
        # Accept either a bare float (uniform cap) or a JSON object
        # (per-motor cap). Matches ``RobotConfig.max_relative_target``'s
        # ``float | dict[str, float] | None`` shape.
        s = robot_max_relative_target.strip()
        try:
            if s.startswith("{"):
                kwargs["max_relative_target"] = json.loads(s)
            else:
                kwargs["max_relative_target"] = float(s)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"--robot.max_relative_target must be a float or JSON dict, "
                f"got {robot_max_relative_target!r}: {exc}"
            ) from exc
    cfg = cls(**kwargs)
    robot = make_robot_from_config(cfg)
    robot.connect()
    return robot


def _build_robot_observation_provider(
    *,
    robot,
    preprocessor: Any,
    device: str,
    task: str | None,
) -> Callable[[], dict | None]:
    """Closure that reads from the robot, runs the policy preprocessor.

    Each call: ``robot.get_observation()`` (raw numpy dict) →
    ``prepare_observation_for_inference`` (tensor / batch dim / device) →
    wrap in an ``EnvTransition`` (the preprocessor pipeline is
    transition-shaped, keyed by ``TransitionKey``) → preprocessor
    (rename, render-messages no-op when no language columns, chat
    tokenizer no-op when no messages, normalise) → unwrap and return
    the flat observation batch ``policy.select_action`` /
    ``policy.select_message`` consume.
    """
    import torch  # noqa: PLC0415

    from lerobot.policies.utils import prepare_observation_for_inference  # noqa: PLC0415
    from lerobot.types import TransitionKey  # noqa: PLC0415

    torch_device = torch.device(device) if isinstance(device, str) else device
    robot_type = getattr(robot, "robot_type", None) or getattr(
        getattr(robot, "config", None), "type", None
    )

    def _provider() -> dict | None:
        try:
            raw = robot.get_observation()
        except Exception as exc:  # noqa: BLE001
            logger.warning("robot.get_observation failed: %s", exc)
            return None

        # Strip language-column leakage just in case (the runtime
        # supplies messages itself).
        for k in ("language_persistent", "language_events"):
            raw.pop(k, None)

        try:
            obs_tensors = prepare_observation_for_inference(
                raw, torch_device, task=task, robot_type=robot_type
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("prepare_observation_for_inference failed: %s", exc)
            return None

        if preprocessor is not None:
            transition: dict[str, Any] = {
                TransitionKey.OBSERVATION.value: obs_tensors,
                TransitionKey.ACTION.value: None,
                TransitionKey.REWARD.value: None,
                TransitionKey.DONE.value: None,
                TransitionKey.TRUNCATED.value: None,
                TransitionKey.INFO.value: None,
                TransitionKey.COMPLEMENTARY_DATA.value: {},
            }
            try:
                transition = preprocessor(transition)
            except Exception as exc:  # noqa: BLE001
                logger.warning("preprocessor failed on robot observation: %s", exc)
                return None
            obs_tensors = transition.get(TransitionKey.OBSERVATION.value) or {}

        observation = {
            k: v
            for k, v in obs_tensors.items()
            if isinstance(k, str) and k.startswith("observation.")
        }
        for k, v in list(observation.items()):
            if isinstance(v, torch.Tensor):
                observation[k] = v.to(torch_device)
        return observation

    return _provider


def _build_robot_action_executor(
    *,
    robot,
    postprocessor: Any,
    ds_meta: Any,
) -> Callable[[Any], None]:
    """Closure that postprocesses an action and dispatches to the robot.

    Mirrors ``lerobot-record``'s ``predict_action`` tail: postprocess
    (denormalise) → ``make_robot_action`` (tensor → ``{joint: value}``
    dict) → ``robot.send_action(...)``. Safety clipping happens *inside*
    ``robot.send_action`` via the driver's ``max_relative_target``
    cap (passed in at ``RobotConfig`` construction time) — same place
    ``lerobot-record`` enforces it.
    """
    import torch  # noqa: PLC0415

    from lerobot.policies.utils import make_robot_action  # noqa: PLC0415

    def _executor(action: Any) -> None:
        try:
            if postprocessor is not None:
                action = postprocessor(action)
            if isinstance(action, torch.Tensor):
                if action.ndim > 1 and action.shape[0] == 1:
                    action = action.squeeze(0)
                action_dict = make_robot_action(action, ds_meta.features)
            elif isinstance(action, dict):
                action_dict = action
            else:
                logger.warning("unsupported action type %r — skipping", type(action))
                return
            robot.send_action(action_dict)
        except Exception as exc:  # noqa: BLE001
            logger.error("robot.send_action failed: %s", exc, exc_info=True)

    return _executor


def _run_autonomous(
    runtime: Any,
    *,
    robot,
    auto_start: bool,
    initial_task: str | None,
    max_ticks: int | None,
) -> int:
    """Drive the runtime continuously at ``ctrl_hz`` while accepting
    stdin events in the foreground.

    Different from ``_run_repl`` (dataset dry-run): the policy needs
    to keep generating action chunks at ``chunk_hz`` and dispatching
    them at ``ctrl_hz`` regardless of whether the user is typing, so
    ``runtime.run()`` runs in a background thread and stdin handling
    happens here in the main thread.
    """
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415

    if not auto_start:
        try:
            input(
                "[smolvla2] Robot connected. Press ENTER to start the autonomous "
                "control loop, Ctrl+C to abort. "
            )
        except (EOFError, KeyboardInterrupt):
            print("\n[smolvla2] aborted before start", flush=True)
            return 130

    if initial_task:
        runtime.set_task(initial_task)

    thread = threading.Thread(
        target=runtime.run,
        kwargs={"max_ticks": max_ticks},
        name="smolvla2-runtime-loop",
        daemon=True,
    )
    thread.start()
    print(
        "[smolvla2] autonomous loop running. Type interjections / "
        "questions on stdin (Ctrl+C to stop).",
        flush=True,
    )

    try:
        while thread.is_alive():
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            lower = line.lower()
            if lower in {"stop", "quit", "exit"}:
                break
            if not runtime.state.get("task"):
                runtime.set_task(line[5:].strip() if lower.startswith("task:") else line)
                continue
            if lower.endswith("?"):
                runtime.state["recent_vqa_query"] = line
                runtime.state.setdefault("events_this_tick", []).append("user_vqa_query")
            else:
                runtime.state["recent_interjection"] = line
                runtime.state.setdefault("events_this_tick", []).append("user_interjection")
    except KeyboardInterrupt:
        print("\n[smolvla2] interrupt — stopping", flush=True)
    finally:
        runtime.stop()
        # Give the loop a moment to drain.
        for _ in range(10):
            if not thread.is_alive():
                break
            time.sleep(0.1)
        try:
            robot.disconnect()
            print("[smolvla2] robot disconnected", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[smolvla2] WARNING: robot.disconnect raised {exc}", flush=True)

    return 0


def _build_tools(no_tts: bool, tts_voice: str) -> dict[str, Any]:
    """Instantiate the tools declared on this dataset/policy."""
    if no_tts:
        return {}
    try:
        from lerobot.tools import SayTool  # noqa: PLC0415

        return {"say": SayTool(voice=tts_voice)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not initialise SayTool (%s) — speech disabled.", exc)
        return {}


def _silence_noisy_loggers() -> None:
    """Drop chatty third-party loggers down to WARNING.

    HuggingFace / httpx / urllib3 emit one log line per HTTP request,
    which the REPL has to print between the state block and the
    prompt — completely unreadable. We never need that detail in the
    REPL and the user can opt back into it via ``-v`` (verbose mode
    keeps DEBUG on the lerobot loggers but still gates the noisy ones
    here unless they explicitly want them).
    """
    for name in (
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.proxy",
        "httpx",
        "urllib3",
        "urllib3.connectionpool",
        "huggingface_hub",
        "huggingface_hub.repocard",
        "huggingface_hub.file_download",
        "transformers",
        "transformers.modeling_utils",
        "transformers.tokenization_utils_base",
        "datasets",
        "filelock",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _silence_noisy_loggers()

    autonomous_mode = bool(args.robot_type) and not args.no_robot
    if autonomous_mode and not args.dataset_repo_id:
        print(
            "[smolvla2] ERROR: autonomous robot mode requires --dataset.repo_id "
            "for action-denormalisation stats and feature shapes. Pass the "
            "same dataset the policy was trained on.",
            file=sys.stderr,
        )
        return 2

    print(f"[smolvla2] loading policy from {args.policy_path}", flush=True)
    policy, preprocessor, postprocessor, ds_meta = _load_policy_and_preprocessor(
        args.policy_path, args.dataset_repo_id
    )

    # Bootstrap canonical task / plan / memory / subtask from the
    # dataset whenever one is provided — both REPL dry-run and
    # autonomous robot mode benefit, since the model is memorised on
    # the exact training prompts and matching wording is what gets
    # recall to fire.
    bootstrap_state: dict[str, str] = {}
    if args.dataset_repo_id is not None:
        bootstrap_state = _bootstrap_state_from_dataset(
            dataset_repo_id=args.dataset_repo_id,
            episode=args.dataset_episode,
            start_frame=args.dataset_start_frame,
        )
        if bootstrap_state.get("task") and not args.task:
            args.task = bootstrap_state["task"]
            print(
                f"[smolvla2] using canonical task from dataset: {args.task!r}",
                flush=True,
            )

    observation_provider: Callable[[], dict | None] | None = None
    robot_executor: Callable[[Any], None] | None = None
    robot = None

    if autonomous_mode:
        print(
            f"[smolvla2] connecting to robot.type={args.robot_type} "
            f"port={args.robot_port}",
            flush=True,
        )
        robot = _build_robot(
            robot_type=args.robot_type,
            robot_port=args.robot_port,
            robot_id=args.robot_id,
            robot_cameras_json=args.robot_cameras,
            robot_max_relative_target=args.robot_max_relative_target,
        )
        observation_provider = _build_robot_observation_provider(
            robot=robot,
            preprocessor=preprocessor,
            device=str(getattr(policy.config, "device", "cpu")),
            task=args.task,
        )
        robot_executor = _build_robot_action_executor(
            robot=robot,
            postprocessor=postprocessor,
            ds_meta=ds_meta,
        )
    elif args.dataset_repo_id is not None:
        print(
            f"[smolvla2] streaming observations from {args.dataset_repo_id} "
            f"episode={args.dataset_episode} "
            f"start_frame={args.dataset_start_frame}",
            flush=True,
        )
        observation_provider = _build_observation_provider(
            dataset_repo_id=args.dataset_repo_id,
            episode=args.dataset_episode,
            start_frame=args.dataset_start_frame,
            advance_per_tick=args.dataset_advance_per_tick,
            preprocessor=preprocessor,
            device=str(getattr(policy.config, "device", "cpu")),
        )

    tools = _build_tools(args.no_tts, args.tts_voice)
    if tools:
        print(f"[smolvla2] tools loaded: {list(tools)}", flush=True)

    from lerobot.policies.smolvla2.inference import SmolVLA2Runtime  # noqa: PLC0415

    runtime = SmolVLA2Runtime(
        policy=policy,
        tools=tools,
        observation_provider=observation_provider,
        robot_executor=robot_executor,
        # No background event collector — the REPL drives ticks
        # synchronously after each user input (REPL mode). Autonomous
        # mode runs ``runtime.run()`` in a thread; stdin events are
        # injected from the foreground.
        event_collector=None,
        chunk_hz=args.chunk_hz,
        ctrl_hz=args.ctrl_hz,
        high_level_hz=args.high_level_hz,
    )
    if args.task:
        runtime.set_task(args.task)
    # Seed plan/memory/subtask so the first prompt the runtime builds
    # mirrors what training rendered (task + active plan + active
    # memory + optional current subtask). Without this the runtime
    # starts empty, which only matched the very-early frames during
    # training and is an out-of-distribution prompt for the rest.
    if bootstrap_state.get("plan"):
        runtime.state["current_plan"] = bootstrap_state["plan"]
    if bootstrap_state.get("memory"):
        runtime.state["current_memory"] = bootstrap_state["memory"]
    if bootstrap_state.get("subtask"):
        runtime.state["current_subtask"] = bootstrap_state["subtask"]

    if autonomous_mode:
        return _run_autonomous(
            runtime,
            robot=robot,
            auto_start=args.auto_start,
            initial_task=args.task,
            max_ticks=args.max_ticks,
        )
    return _run_repl(runtime, initial_task=args.task, max_ticks=args.max_ticks)


def _run_repl(runtime: Any, *, initial_task: str | None, max_ticks: int | None) -> int:
    """Claude-Code-style block REPL.

    Each turn redraws a status block (task / subtask / plan / memory)
    at the top, prints any robot log lines that came in since the last
    turn, then asks for input on a clean ``> `` prompt at the bottom.
    No live region, no panel re-renders, no rendering races with HTTP
    log lines — just clear-screen + reprint each turn, the way a
    chat-style REPL is meant to look.
    """
    try:
        from rich.console import Console  # noqa: PLC0415
    except ImportError:
        print(
            "[smolvla2] rich is required for the interactive REPL. "
            "`pip install rich` and re-run.",
            file=sys.stderr,
        )
        return 2

    console = Console(highlight=False)

    def _redraw(robot_lines: list[str] | None = None) -> None:
        # ANSI clear screen + home cursor. Falls back gracefully on
        # dumb terminals — they just see scrolled output, which is
        # fine.
        console.clear()
        console.rule("[bold]SmolVLA2[/] · dry-run", style="cyan")
        st = runtime.state
        for key, label in (
            ("task", "task"),
            ("current_subtask", "subtask"),
            ("current_plan", "plan"),
            ("current_memory", "memory"),
        ):
            value = st.get(key)
            if value:
                console.print(f"  [bold cyan]{label:<8}[/] {value}")
            else:
                console.print(f"  [dim]{label:<8} (not set)[/]")
        queue_len = (
            len(st["action_queue"])
            if isinstance(st.get("action_queue"), (list, tuple))
            or hasattr(st.get("action_queue"), "__len__")
            else 0
        )
        pending = len(st.get("tool_calls_pending") or [])
        console.print(
            f"  [dim]queued actions: {queue_len}    pending tool calls: {pending}[/]"
        )
        console.rule(style="cyan")
        if robot_lines:
            for line in robot_lines:
                console.print(f"  [magenta]{line.strip()}[/]")
            console.print()
        # Help line under the divider when nothing is set yet.
        if not st.get("task"):
            console.print(
                "  [dim]Type the task to begin. Lines ending in '?' are VQA, "
                "anything else is an interjection. Type 'stop' to exit.[/]"
            )

    last_logs: list[str] = []
    _redraw()
    if initial_task is None:
        # Already shown the help line in _redraw when task is None.
        pass
    ticks_done = 0
    try:
        while True:
            try:
                line = console.input("[bold cyan]> [/]").strip()
            except EOFError:
                break
            if not line:
                _redraw(last_logs)
                continue
            lower = line.lower()
            if lower in {"stop", "quit", "exit"}:
                break

            # Inject the user input as the right kind of event,
            # then run a single pipeline tick to consume it.
            if not runtime.state.get("task"):
                task = line[5:].strip() if lower.startswith("task:") else line
                runtime.set_task(task)
            elif lower.endswith("?"):
                runtime.state["recent_vqa_query"] = line
                runtime.state.setdefault("events_this_tick", []).append(
                    "user_vqa_query"
                )
            else:
                runtime.state["recent_interjection"] = line
                runtime.state.setdefault("events_this_tick", []).append(
                    "user_interjection"
                )

            last_logs = runtime.step_once() or []
            _redraw(last_logs)

            ticks_done += 1
            if max_ticks is not None and ticks_done >= max_ticks:
                break
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/]")
    console.print("[dim]runtime stopped[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
