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
            "Local directory or Hugging Face Hub repo id pointing at a trained SmolVLA2 ``pretrained_model``."
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
        "--dataset.augment_at_inference",
        dest="dataset_augment_at_inference",
        action="store_true",
        help=(
            "Apply the same torchvision-v2 ColorJitter / SharpnessJitter "
            "/ RandomAffine pipeline that training used to each dataset "
            "frame fed to the policy. Use to test whether the LM head "
            "generalises under the augmentation distribution it was "
            "supervised on — if dry-run still produces coherent subtask "
            "text with this flag on, the head has learned beyond exact "
            "frames; if it collapses to '\\n' the head is hyper-specific "
            "to the unperturbed training samples."
        ),
    )
    p.add_argument(
        "--task",
        dest="task",
        type=str,
        default=None,
        help=(
            "Initial task. When given, the startup task picker is skipped "
            "and this task is used directly. If omitted, the picker is "
            "shown (or the first stdin line is treated as the task)."
        ),
    )
    p.add_argument(
        "--mode",
        dest="mode",
        type=str,
        choices=["action", "paused"],
        default=None,
        help=(
            "Start-up run mode. 'action' runs the robot immediately on "
            "--task; 'paused' (the default) comes up at the command line "
            "with the robot idle. Flip any time with /action and /pause."
        ),
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
            'the robot (e.g. ``\'{"top": {"type": "opencv", "index": 0}}\'``). '
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
            '(e.g. ``\'{"shoulder_pan": 5, "gripper": 30}\'``). The '
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
    p.add_argument("--ctrl_hz", type=float, default=50.0, help="Action dispatch rate.")
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
    p.add_argument(
        "--text_min_new_tokens",
        type=int,
        default=0,
        help=(
            "Debug knob for under-trained checkpoints: force the LM head "
            "to emit at least N non-EOS tokens before EOS is allowed. "
            "Use when the head's prior at position 0 still favours EOS "
            "(short training run on a chat-pretrained backbone). 3-5 "
            "is usually enough to reveal whether the model has real "
            "subtask-token mass under the EOS argmax."
        ),
    )
    p.add_argument(
        "--text_temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature for high-level text gen. 0 = greedy "
            "argmax (default, matches training). Set 0.3-0.7 with an "
            "under-trained checkpoint to escape stuck-at-EOS argmax."
        ),
    )
    p.add_argument(
        "--text_top_p",
        type=float,
        default=1.0,
        help="Nucleus filtering for high-level text gen.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def _log_obs_tensors_once(label: str, obs: Any, flag: dict) -> None:
    """Print shape / dtype / per-channel stats of every observation tensor
    going into the policy, exactly once per provider lifetime.

    Used to bisect train/inference mismatches: if the dry-run path
    and the robot path produce identifiably different tensors here
    (e.g. one is batched twice, one has a different range, one is on
    a different device), the LM head's collapse on the live robot is
    a tensor-shape bug, not a distribution-shift problem. If the
    tensors *do* match byte-for-byte and the head still collapses,
    only then is the scene-content OOD hypothesis the right one.
    """
    if flag.get("done") or not isinstance(obs, dict):
        return
    flag["done"] = True
    import torch as _torch  # noqa: PLC0415

    for k, v in obs.items():
        if not isinstance(k, str) or not k.startswith("observation."):
            continue
        if isinstance(v, _torch.Tensor):
            try:
                stats = (
                    f"min={float(v.min()):.4f} max={float(v.max()):.4f} "
                    f"mean={float(v.mean()):.4f} std={float(v.float().std()):.4f}"
                )
            except Exception:  # noqa: BLE001
                stats = "(stats unavailable)"
            logger.warning(
                "obs[%s] %-30s shape=%s dtype=%s device=%s %s",
                label,
                k,
                tuple(v.shape),
                v.dtype,
                v.device,
                stats,
            )
        else:
            logger.warning("obs[%s] %-30s type=%s value=%r", label, k, type(v).__name__, v)


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
    augment: bool = False,
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
        raise ValueError(f"Dataset {dataset_repo_id!r} episode {episode} is empty.")

    # Optional: apply the same torchvision-v2 augmentation pipeline
    # that training used, so dry-run sees frames from the augmented
    # support region (not just the unperturbed dataset frames). When
    # the LM head still generates coherent text under this, it has
    # learned over the augmentation distribution — the *opposite* of
    # the "memorised one specific frame per supervision" failure
    # mode. When it collapses to ``\n`` here too, the head is hyper-
    # specific to the unperturbed training samples and only the
    # retrain can help.
    inference_aug = None
    if augment:
        from lerobot.transforms import (  # noqa: PLC0415
            ImageTransforms,
            ImageTransformsConfig,
        )

        aug_cfg = ImageTransformsConfig(enable=True)
        inference_aug = ImageTransforms(aug_cfg)
        ds.set_image_transforms(inference_aug)
        logger.warning(
            "dry-run augmentation ENABLED — frames will be jittered "
            "(brightness/contrast/saturation/hue/sharpness/affine) "
            "before going to the policy"
        )

    state = {"cursor": max(0, min(start_frame, len(ds) - 1))}
    _logged = {"done": False}

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

        _log_obs_tensors_once("dry-run", sample, _logged)

        # Keep only observation keys; the runtime's text path will
        # merge these with its own lang_tokens / lang_masks.
        observation = {k: v for k, v in sample.items() if isinstance(k, str) and k.startswith("observation.")}
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
        raise ValueError(f"Unknown robot type {robot_type!r}. Available choices: {available}") from exc
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
                raise ValueError(f"camera {cam_name!r} value must be a dict, got {cam_dict!r}")
            cam_dict = dict(cam_dict)  # don't mutate caller's parsed JSON
            cam_type = cam_dict.pop("type", None)
            if cam_type is None:
                raise ValueError(
                    f"camera {cam_name!r} is missing a 'type' field (e.g. 'opencv', 'intelrealsense')"
                )
            try:
                cam_cls = CameraConfig.get_choice_class(cam_type)
            except KeyError as exc:
                available = sorted(CameraConfig._choice_registry.keys())
                raise ValueError(
                    f"camera {cam_name!r}: unknown type {cam_type!r}. Available choices: {available}"
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
    ds_features: dict[str, Any] | None,
) -> Callable[[], dict | None]:
    """Closure that reads from the robot, runs the policy preprocessor.

    Each call: ``robot.get_observation()`` (raw per-joint + per-camera
    dict, possibly with scalar floats) → ``build_inference_frame``
    (extract the keys the dataset declared, reshape per-joint floats
    into a single ``observation.state`` vector, prefix camera keys
    with ``observation.images.``, convert to tensors with batch dim
    on device) → wrap in an ``EnvTransition`` (the preprocessor
    pipeline is transition-shaped, keyed by ``TransitionKey``) →
    preprocessor (rename, normalise) → unwrap and return the flat
    observation batch ``policy.select_action`` / ``policy.select_message``
    consume.
    """
    import torch  # noqa: PLC0415

    from lerobot.policies.utils import (  # noqa: PLC0415
        build_inference_frame,
        prepare_observation_for_inference,
    )

    torch_device = torch.device(device) if isinstance(device, str) else device
    robot_type = getattr(robot, "robot_type", None) or getattr(getattr(robot, "config", None), "type", None)

    # Pre-compute the camera-key → target (H, W) map from
    # ``ds_features``. The training distribution sees frames at the
    # recorded resolution (e.g. 480×640); a live Mac/USB camera will
    # almost always hand us a different native size (720p / 1080p).
    # SmolVLA's internal ``resize_with_pad(512, 512)`` does pad the
    # input to a fixed canvas, but the *geometry* of that pad differs
    # by input aspect ratio — top/left padding varies, so the visual
    # tokens at each tile carry different content than what the model
    # saw at training. The action expert tolerates this (flow head
    # rides broad geometry); the LM head, supervised much more
    # tightly on visual features, goes out of distribution and the
    # head's distribution at position 0 collapses to its dominant
    # mode (a memorised ``\n``-only run in this checkpoint).
    _resize_logged = {"done": False}
    _obs_logged = {"done": False}
    target_image_shapes: dict[str, tuple[int, int]] = {}
    if ds_features:
        for fkey, fmeta in ds_features.items():
            if not isinstance(fmeta, dict):
                continue
            dtype = fmeta.get("dtype")
            if dtype not in ("image", "video"):
                continue
            shape = fmeta.get("shape")
            if not shape or len(shape) != 3:
                continue
            names = fmeta.get("names") or []
            # Feature schema stores either (H, W, C) or (C, H, W);
            # disambiguate by the ``names`` ordering when present.
            if names and len(names) == 3 and names[0] == "channels":
                _, h, w = shape
            else:
                h, w, _ = shape
            cam_key = fkey.removeprefix("observation.images.")
            target_image_shapes[cam_key] = (int(h), int(w))

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

        # Force-match the training-time visual distribution:
        # every camera frame the model trained on came from the
        # dataset at its recorded (H, W). Resize the live frame to
        # that exact shape so the downstream resize_with_pad geometry
        # matches training. Without this the LM head is OOD on every
        # tick.
        if target_image_shapes:
            try:
                import cv2 as _cv2  # noqa: PLC0415
                import numpy as _np  # noqa: PLC0415

                for cam_key, (target_h, target_w) in target_image_shapes.items():
                    img = raw.get(cam_key)
                    if img is None or not isinstance(img, _np.ndarray):
                        continue
                    if img.ndim != 3:
                        continue
                    cur_h, cur_w = img.shape[:2]
                    if not _resize_logged["done"]:
                        logger.warning(
                            "camera %s: live=%dx%d, training=%dx%d (resize=%s)",
                            cam_key,
                            cur_h,
                            cur_w,
                            target_h,
                            target_w,
                            "yes" if (cur_h, cur_w) != (target_h, target_w) else "no — already matched",
                        )
                    if (cur_h, cur_w) == (target_h, target_w):
                        continue
                    raw[cam_key] = _cv2.resize(img, (target_w, target_h), interpolation=_cv2.INTER_AREA)
                _resize_logged["done"] = True
                # Also print the state vector once so the operator
                # can eyeball it against the dataset's stats. State
                # OOD is a real failure mode for VLAs — the prefix
                # carries state via the projection layer, and a
                # neutral home pose can easily sit a couple σ off
                # the supervised support region.
                if "observation.state" in (ds_features or {}):
                    state_names = ds_features["observation.state"].get("names") or []
                    state_vals = [raw.get(n) for n in state_names]
                    logger.warning(
                        "robot state at startup: %s",
                        {
                            n: round(v, 2) if isinstance(v, float) else v
                            for n, v in zip(state_names, state_vals, strict=False)
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("camera resize to dataset shape failed: %s", exc)

        try:
            if ds_features:
                # Use the dataset's feature schema to pick the right
                # raw keys and fold per-joint scalars into a single
                # ``observation.state`` tensor. Then tensor-ise +
                # device-place + add batch dim.
                obs_tensors = build_inference_frame(
                    raw,
                    torch_device,
                    ds_features=ds_features,
                    task=task,
                    robot_type=robot_type,
                )
            else:
                # No dataset features available — fall back to the
                # generic numpy-only path; only works when the robot
                # already returns dataset-shaped keys.
                obs_tensors = prepare_observation_for_inference(
                    raw,
                    torch_device,
                    task=task,
                    robot_type=robot_type,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("observation prep failed: %s", exc)
            return None

        if preprocessor is not None:
            # ``PolicyProcessorPipeline`` defaults its ``to_transition``
            # to ``batch_to_transition``, which expects a *flat batch
            # dict* keyed by ``observation.*`` / ``action`` / etc., and
            # wraps it into an ``EnvTransition`` itself. Pre-wrapping
            # here would just have ``batch_to_transition`` look for
            # ``observation.*`` keys at top level, find none (they'd
            # be nested under ``TransitionKey.OBSERVATION``), and
            # produce an empty observation → ``ObservationProcessorStep``
            # bails. Pass the flat dict straight in; ``to_output``
            # gives us a flat dict back.
            try:
                processed = preprocessor(obs_tensors)
            except Exception as exc:  # noqa: BLE001
                logger.warning("preprocessor failed on robot observation: %s", exc)
                return None
            obs_tensors = processed if isinstance(processed, dict) else {}

        _log_obs_tensors_once("robot", obs_tensors, _obs_logged)

        observation = {
            k: v for k, v in obs_tensors.items() if isinstance(k, str) and k.startswith("observation.")
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


def _print_runtime_help() -> None:
    """Print the slash-command reference."""
    print(
        "[smolvla2] commands (arguments need no quotes):\n"
        "  /action <task>     run the robot; an argument switches to that task\n"
        "  /action            resume the robot on the current task\n"
        "  /action <seconds>  run the robot for N seconds, then auto-pause\n"
        "  /pause             pause the action loop — robot holds position\n"
        "  /question <text>   pause and answer one VQA question\n"
        "  /help              show this help\n"
        "  stop | quit | exit end the session\n"
        "\n"
        "  VQA examples:\n"
        "    /question point to the yellow cube     -> point overlay\n"
        "    /question detect the blue cube         -> bounding-box overlay",
        flush=True,
    )


def _is_number(text: str) -> bool:
    """True if ``text`` parses as a float (a ``/action`` duration arg)."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _strip_quotes(text: str) -> str:
    """Strip one pair of surrounding quotes from a command argument."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def _clear_action_queue(runtime: Any) -> None:
    """Drop any queued action chunk so nothing fires while paused."""
    queue = runtime.state.get("action_queue")
    if hasattr(queue, "clear"):
        queue.clear()


def _handle_slash_command(runtime: Any, line: str) -> bool:
    """Dispatch the runtime slash commands.

      ``/action ["task"]``  run the robot; a quoted/bare argument sets a
                            new task, a bare number is a timed burst
                            (seconds), no argument resumes the current
                            task.
      ``/pause``            pause the action loop — the robot holds.
      ``/question "text"``  pause and answer one VQA question.
      ``/help``             print the command reference.

    Returns ``True`` when ``line`` was a recognised command (consumed).
    """
    stripped = line.strip()
    if not stripped.startswith("/"):
        return False
    head, _, rest = stripped.partition(" ")
    cmd = head.lower()
    rest = _strip_quotes(rest)

    if cmd in {"/action", "/act", "/run"}:
        runtime.state["mode"] = "action"
        if rest and _is_number(rest):
            import time as _time  # noqa: PLC0415

            secs = float(rest)
            runtime.state["action_deadline"] = _time.monotonic() + secs
            print(
                f"[smolvla2] action — running {secs:g}s, then auto-pause",
                flush=True,
            )
        else:
            runtime.state["action_deadline"] = None
            if rest:
                runtime.set_task(rest)
                # New task → drop the stale subtask so the high-level
                # loop regenerates one for the new goal.
                runtime.state["current_subtask"] = None
                print(f"[smolvla2] action — task: {rest!r}", flush=True)
            elif runtime.state.get("task"):
                print(
                    f"[smolvla2] action — resuming: {runtime.state['task']!r}",
                    flush=True,
                )
            else:
                runtime.state["mode"] = "paused"
                print(
                    "[smolvla2] no task set — use /action <your task>",
                    flush=True,
                )
        return True

    if cmd in {"/pause", "/p"}:
        runtime.state["mode"] = "paused"
        runtime.state["action_deadline"] = None
        _clear_action_queue(runtime)
        print("[smolvla2] paused — robot holding position", flush=True)
        return True

    if cmd in {"/question", "/q", "/ask", "/vqa", "/vlm"}:
        # A question always pauses the action loop first so the policy
        # is not used concurrently by the background runtime thread.
        runtime.state["mode"] = "paused"
        runtime.state["action_deadline"] = None
        _clear_action_queue(runtime)
        if not rest:
            print(
                "[smolvla2] usage: /question <your question>  "
                "(e.g. /question point to the yellow cube)",
                flush=True,
            )
            return True
        _run_vqa_query(runtime, rest)
        return True

    if cmd in {"/help", "/?"}:
        _print_runtime_help()
        return True
    return False


def _run_vqa_query(runtime: Any, question: str) -> None:
    """Run one interactive VQA question against the runtime's policy.

    Invoked by ``/question`` — the action loop is paused first so the
    policy is free for a synchronous VQA call.
    """
    from lerobot.policies.smolvla2.inference.vqa import handle_vqa_query  # noqa: PLC0415

    handle_vqa_query(
        policy=runtime.policy,
        observation_provider=runtime.observation_provider,
        question=question,
        state=runtime.state,
    )


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

    # Only gate on ENTER when the robot will actually move at startup
    # (``--mode=action``). The default is paused — the command line
    # comes up immediately and nothing moves until ``/action``.
    if not auto_start and runtime.state.get("mode", "paused") == "action":
        try:
            input(
                "[smolvla2] Robot connected — starting in ACTION mode. "
                "Press ENTER to begin, Ctrl+C to abort. "
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

    # Capture log lines flushed by the runtime each tick into a
    # bounded scrollback that the panel renderer prints inside the
    # rule block. Without this, ``runtime._flush_logs`` just calls
    # ``print(...)`` which the 2 Hz panel redraw clears immediately —
    # so failure messages from generation (e.g. ``[warn] subtask gen
    # failed: ...``) flash for ≤ 0.5 s and disappear, leaving the
    # operator with no idea why ``last_raw`` stays empty.
    _scrollback: list[str] = []
    _scrollback_max = 12

    def _flush_into_scrollback() -> None:
        for line in runtime.state.get("log_lines") or []:
            _scrollback.append(line)
        # Trim to the cap so the panel doesn't grow unbounded.
        if len(_scrollback) > _scrollback_max:
            del _scrollback[: len(_scrollback) - _scrollback_max]

    runtime._flush_logs = _flush_into_scrollback  # type: ignore[method-assign]

    redraw = _make_state_panel_renderer(runtime, mode_label="autonomous", scrollback=_scrollback)
    redraw()
    print(
        "  [autonomous] /action <task> to run  ·  /pause to stop  ·  "
        "/question <text> to ask  ·  /help  ·  stop",
        flush=True,
    )

    # Background panel-redraw thread so state changes from the runtime
    # loop (subtask refresh, plan update, etc.) are visible without the
    # user typing anything.
    #
    # In ``/vlm`` mode the action loop is paused — nothing changes in the
    # background — so the timer redraw is suspended entirely. That keeps
    # the screen stable while the operator types a VQA question and the
    # interactive camera prompt, instead of the panel clearing the
    # prompt every tick.
    _panel_stop = threading.Event()

    def _panel_loop() -> None:
        while not _panel_stop.is_set():
            st = runtime.state
            if st.get("mode", "action") == "action":
                # Timed burst (``/action <seconds>``): once the deadline
                # passes, auto-revert to question mode and clear the
                # queue so the robot stops.
                deadline = st.get("action_deadline")
                if deadline is not None and time.monotonic() >= deadline:
                    st["mode"] = "paused"
                    st["action_deadline"] = None
                    queue = st.get("action_queue")
                    if hasattr(queue, "clear"):
                        queue.clear()
                    print(
                        "\n[smolvla2] timed action elapsed — paused",
                        flush=True,
                    )
                else:
                    try:
                        redraw()
                        # Re-print the prompt the redraw just cleared so
                        # the operator always has a visible ``> ``.
                        print("> ", end="", flush=True)
                    except Exception:  # noqa: BLE001
                        pass
            _panel_stop.wait(0.7)

    panel_thread = threading.Thread(target=_panel_loop, name="smolvla2-panel-redraw", daemon=True)
    panel_thread.start()

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
            # The runtime is command-driven: /action "task", /pause,
            # /question "...", /help. ``_handle_slash_command`` runs the
            # VQA query inline for /question (the action loop is paused
            # first, so the policy isn't in concurrent use).
            if _handle_slash_command(runtime, line):
                try:
                    redraw()
                except Exception:  # noqa: BLE001
                    pass
                continue
            # A bare (non-slash) line is treated as a user interjection
            # — the trained ``user_interjection_response`` path. ``stop``
            # already handled above; everything else routes here.
            if runtime.state.get("task"):
                runtime.state["recent_interjection"] = line
                runtime.state.setdefault("events_this_tick", []).append("user_interjection")
            else:
                print(
                    "[smolvla2] no task yet — use /action <your task> to start",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[smolvla2] interrupt — stopping", flush=True)
    finally:
        _panel_stop.set()
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


def _make_state_panel_renderer(
    runtime: Any,
    *,
    mode_label: str,
    scrollback: list[str] | None = None,
) -> Callable[[list[str] | None], None]:
    """Return a closure that prints the task/subtask/plan/memory panel.

    Used by both ``_run_repl`` (dry-run, called per user input) and
    ``_run_autonomous`` (real robot, called on a 2 Hz timer +
    whenever the user types). Centralises the visual format so the
    two modes look identical.
    """
    from rich.console import Console  # noqa: PLC0415

    console = Console(highlight=False)

    def _redraw(robot_lines: list[str] | None = None) -> None:
        console.clear()
        st = runtime.state
        run_mode = st.get("mode", "action")
        mode_tag = "[green]mode: action[/]" if run_mode == "action" else "[yellow]mode: paused[/]"
        console.rule(f"[bold]SmolVLA2[/] · {mode_label} · {mode_tag}", style="cyan")
        # Always-visible command hint so the operator never has to
        # remember the slash commands.
        if run_mode == "action":
            console.print(
                "  [dim]commands:[/] [bold]/pause[/] stop  ·  "
                "[bold]/question[/] <text> ask  ·  [bold]/help[/]  ·  [bold]stop[/]"
            )
        else:
            console.print(
                "  [dim]commands:[/] [bold]/action[/] <task> run  ·  "
                "[bold]/question[/] <text> ask  ·  [bold]/help[/]  ·  [bold]stop[/]"
            )
        # Reference VQA prompts — the two answer shapes that draw an
        # overlay (point + bounding box). No quotes needed.
        console.print(
            "  [dim]vqa examples:[/] /question point to the yellow cube  ·  "
            "/question detect the blue cube"
        )
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
            if isinstance(st.get("action_queue"), (list, tuple)) or hasattr(st.get("action_queue"), "__len__")
            else 0
        )
        pending = len(st.get("tool_calls_pending") or [])
        dispatched = int(st.get("actions_dispatched") or 0)
        console.print(
            f"  [dim]queued actions: {queue_len}    "
            f"dispatched: {dispatched}    "
            f"pending tool calls: {pending}[/]"
        )

        # Overfit / memorisation diagnostics. The high-level steps
        # surface the raw generation each time they fire (even when
        # rejected as gibberish or unchanged), plus repeat/rejection
        # counters. Rule of thumb:
        #
        #   * subtask repeat ≥ ~5 and queue_len cycles fully → model
        #     can't move past current subtask (memorised one phase
        #     of the task — classic overfit signature)
        #   * subtask gibberish climbing → LM head collapsed to
        #     chat-template fragments / one-token salads
        #   * last raw differs from accepted → at least the LM is
        #     varying, the gibberish filter is doing its job
        raw_subtask = st.get("last_subtask_raw")
        sub_rep = int(st.get("subtask_repeat_count") or 0)
        sub_gib = int(st.get("subtask_gibberish_count") or 0)
        sub_empty = int(st.get("subtask_empty_count") or 0)
        if raw_subtask is not None or sub_rep or sub_gib or sub_empty:
            raw_display = (raw_subtask or "(empty)")[:80]
            color = "yellow" if (sub_rep >= 3 or sub_gib >= 3 or sub_empty >= 3) else "dim"
            console.print(
                f"  [{color}]subtask diag    repeat:{sub_rep}  "
                f"gibberish:{sub_gib}  empty:{sub_empty}  "
                f"last_raw: {raw_display!r}[/]"
            )

        # Same diagnostics for memory and plan when available.
        mem_gib = int(st.get("memory_gibberish_count") or 0)
        plan_gib = int(st.get("plan_gibberish_count") or 0)
        if mem_gib or plan_gib:
            console.print(f"  [dim]gen rejects     memory:{mem_gib}  plan:{plan_gib}[/]")
        console.rule(style="cyan")
        # Runtime scrollback — log lines pushed from generation steps
        # (warnings, gibberish rejections, plan/say speech, vqa
        # answers). Last N lines, oldest first.
        if scrollback:
            for line in scrollback:
                console.print(f"  [magenta]{line.rstrip()}[/]")
            console.rule(style="cyan")
        if robot_lines:
            for line in robot_lines:
                console.print(f"  [magenta]{line.strip()}[/]")
            console.print()
        if not st.get("task"):
            console.print(
                "  [dim]Type [bold]/action <your task>[/bold] to begin, "
                "[bold]/question <text>[/bold] to ask, /help for commands, "
                "stop to exit.[/]"
            )

    return _redraw


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

    # The robot's relative-goal-position clamp warning fires *every*
    # dispatch tick on a memorised model — the LM is trying to jump
    # the wrist far past where max_relative_target allows, so the
    # warning floods the panel at ~30 Hz. Promote it from WARNING to
    # DEBUG: the dispatch counter on the panel already tells the
    # operator the loop is running, and the panel itself shows
    # whether motion is happening. If anyone needs the per-action
    # clamp detail, ``-v`` puts it back via DEBUG.
    logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)


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

    # Bootstrap the canonical task from the dataset whenever one is
    # provided, so ``/action`` (no argument) has a sensible task to
    # resume. The model is memorised on the exact training wording, so
    # matching it is what gets recall to fire.
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
                f"[smolvla2] canonical task from dataset: {args.task!r}",
                flush=True,
            )

    # No startup prompts — the runtime is command-driven. It comes up at
    # the command line in ``paused`` mode (robot idle) unless ``--mode``
    # forces a mode. The operator drives it with /action, /pause and
    # /question.
    startup_mode = args.mode or "paused"

    observation_provider: Callable[[], dict | None] | None = None
    robot_executor: Callable[[Any], None] | None = None
    robot = None

    if autonomous_mode:
        print(
            f"[smolvla2] connecting to robot.type={args.robot_type} port={args.robot_port}",
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
            ds_features=ds_meta.features if ds_meta is not None else None,
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
            augment=getattr(args, "dataset_augment_at_inference", False),
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
    # Stash text-gen knobs on the state dict so the high-level steps
    # (which read state) can pick them up and forward them to
    # policy.select_message. Letting the operator try
    # ``--text_min_new_tokens=5 --text_temperature=0.6`` on an
    # under-trained checkpoint without recompiling.
    runtime.state["text_gen_min_new_tokens"] = int(getattr(args, "text_min_new_tokens", 0) or 0)
    runtime.state["text_gen_temperature"] = float(getattr(args, "text_temperature", 0.0) or 0.0)
    runtime.state["text_gen_top_p"] = float(getattr(args, "text_top_p", 1.0) or 1.0)
    # Apply the startup mode chosen above the task picker.
    runtime.state["mode"] = startup_mode
    if args.task:
        runtime.set_task(args.task)
    # Seed the current subtask from the dataset so the first chunk —
    # before ``HighLevelSubtaskFwd`` has run — has a real subtask to
    # condition the action expert on instead of falling back to the
    # bare task. Plan and memory are NOT seeded: the current recipe
    # trains neither, no inference step consumes them, and seeding
    # them only put a stale plan in the status panel that does
    # nothing.
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
    # Fire one full pipeline tick at startup so the obs diagnostic
    # *and* the subtask generation actually run before the REPL
    # blocks on stdin. The REPL otherwise only ticks on user input,
    # which made the dry-run bisection test (does the LM head produce
    # text at start_frame=0?) require typing something. Doing
    # ``step_once`` here means the diag row populates without any
    # manual interaction.
    if observation_provider is not None:
        try:
            startup_logs = runtime.step_once()
        except Exception as exc:  # noqa: BLE001
            logger.warning("startup tick failed: %s", exc)
            startup_logs = []
        for line in startup_logs or []:
            print(f"[smolvla2] {line}", flush=True)
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
            "[smolvla2] rich is required for the interactive REPL. `pip install rich` and re-run.",
            file=sys.stderr,
        )
        return 2

    _redraw = _make_state_panel_renderer(runtime, mode_label="dry-run")
    # Keep a local ``console`` just for the styled input prompt; the
    # state panel is owned by the shared renderer.
    console = Console(highlight=False)

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

            # Command-driven: /action "task", /pause, /question "...",
            # /help. ``_handle_slash_command`` runs the VQA query inline
            # for /question (single-threaded REPL — no concurrency).
            if _handle_slash_command(runtime, line):
                last_logs = list(runtime.state.get("log_lines") or [])
                _redraw(last_logs)
                ticks_done += 1
                if max_ticks is not None and ticks_done >= max_ticks:
                    break
                continue

            # A bare (non-slash) line is a user interjection — needs a
            # task to be meaningful.
            if not runtime.state.get("task"):
                print(
                    "[smolvla2] no task yet — use /action <your task>",
                    flush=True,
                )
                _redraw(last_logs)
                continue
            runtime.state["recent_interjection"] = line
            runtime.state.setdefault("events_this_tick", []).append("user_interjection")

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
