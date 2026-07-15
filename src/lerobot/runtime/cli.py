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
"""Interactive REPL for a language-conditioned robot policy.

Policy-agnostic CLI over :class:`lerobot.runtime.LanguageConditionedRuntime`.
A policy wires it up with :func:`run`, passing an adapter factory
(``policy -> LanguageConditionedPolicyAdapter``); see
``lerobot.scripts.lerobot_language_runtime`` for the entry point.

Stdin is the user channel: type a task, then natural-language
interjections. The runtime prints state changes (plan / subtask /
memory) as they happen.

Examples
--------

Dry run on a Hub checkpoint, no robot connected — useful for sanity-
checking text generation::

    uv run lerobot-rollout --language \\
        --policy.path=<repo-or-dir> \\
        --no_robot \\
        --task="please clean the kitchen"

Same, but feed real frames from an annotated dataset so plan / subtask
/ memory generation runs against actual video + state::

    uv run lerobot-rollout --language \\
        --policy.path=<repo-or-dir> \\
        --dataset.repo_id=<annotated-dataset> \\
        --dataset.episode=0 \\
        --no_robot \\
        --task="please clean the kitchen"

With a real robot::

    uv run lerobot-rollout --language \\
        --policy.path=... \\
        --robot.type=so101 --robot.port=/dev/tty.usbmodem...

``--policy.path`` accepts either a local directory or a Hugging Face
Hub repo id. ``--dataset.repo_id`` likewise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from typing import Any

from .adapter import GenerationConfig
from .language_runtime import LanguageConditionedPolicyAdapter, LanguageConditionedRuntime
from .repl import _emit

logger = logging.getLogger("lerobot.runtime")


def _parse_args(argv: list[str] | None = None, *, prog: str | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        prog=prog,
        description="Interactive REPL runtime for a language-conditioned robot policy.",
    )
    p.add_argument(
        "--policy.path",
        dest="policy_path",
        type=str,
        required=True,
        help="Local directory or Hugging Face Hub repo id pointing at a trained ``pretrained_model``.",
    )
    p.add_argument(
        "--policy.device",
        dest="policy_device",
        type=str,
        default=None,
        help=(
            "Override the checkpoint's ``config.device`` (e.g. ``cuda``, ``cpu``). "
            "Some checkpoints ship ``device=cpu``; pass ``cuda`` to run on GPU."
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
            "them into all forward passes — so plan / subtask / memory "
            "generation see the same visual context the policy was "
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
    # for interjections.
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
        "--rerun",
        action="store_true",
        help="Live rerun viewer for the robot cameras (real-robot mode). Serves a "
        "headless web viewer; forward --rerun.web_port and --rerun.grpc_port over SSH.",
    )
    p.add_argument(
        "--rerun.web_port",
        dest="rerun_web_port",
        type=int,
        default=9090,
        help="rerun web-viewer port (default 9090).",
    )
    p.add_argument(
        "--rerun.grpc_port",
        dest="rerun_grpc_port",
        type=int,
        default=9876,
        help="rerun gRPC data port (default 9876).",
    )
    p.add_argument(
        "--direct_subtask",
        action="store_true",
        help="Direct-subtask mode (sim OR robot): your typed text IS the subtask "
        "fed to the action expert; the LM subtask generator is disabled.",
    )
    # --- RoboCasa simulation mode args -------------------------------
    # Setting ``--sim`` flips the runtime into simulation mode: instead of
    # a real robot it drives a single RoboCasa mujoco scene, feeding the
    # eval observation/action pipeline. The operator still types prompts
    # (/action <prompt>) that the policy executes inside the chosen scene.
    # Mutually exclusive with ``--robot.type``.
    p.add_argument(
        "--sim",
        action="store_true",
        help=(
            "Run the policy in the RoboCasa simulator instead of on a real "
            "robot. Select the scene with --sim.task; type prompts with "
            "/action <prompt> to have the policy execute them in that scene."
        ),
    )
    p.add_argument(
        "--sim.task",
        dest="sim_task",
        type=str,
        default="CloseFridge",
        help="RoboCasa task/scene to instantiate (e.g. OpenDrawer, LoadDishwasher).",
    )
    p.add_argument(
        "--sim.split",
        dest="sim_split",
        type=str,
        default="pretrain",
        help="RoboCasa scene split (all/pretrain/target). Default: pretrain.",
    )
    p.add_argument(
        "--sim.obj_registries",
        dest="sim_obj_registries",
        type=str,
        default="objaverse,lightwheel",
        help="Comma-separated object-mesh registries. Default: objaverse,lightwheel.",
    )
    p.add_argument(
        "--sim.seed",
        dest="sim_seed",
        type=int,
        default=1000,
        help="Seed for RoboCasa scene reset (default: 1000, matches eval).",
    )
    p.add_argument(
        "--sim.record",
        dest="sim_record",
        type=str,
        choices=["mp4", "off"],
        default="mp4",
        help="Record an annotated mp4 (task/subtask/memory overlay) of the sim session. Default: mp4.",
    )
    p.add_argument(
        "--sim.output_dir",
        dest="sim_output_dir",
        type=str,
        default="outputs/runtime_sim",
        help="Directory for the recorded sim video (default: outputs/runtime_sim).",
    )
    p.add_argument(
        "--sim.render_size",
        dest="sim_render_size",
        type=int,
        default=384,
        help=(
            "Resolution (px) of the observation cameras used for the display "
            "(default 384; try 512 for sharper, 256 for faster). The policy is "
            "unaffected — it resizes to 224 internally."
        ),
    )
    p.add_argument(
        "--sim.views",
        dest="sim_views",
        type=str,
        default="robot0_agentview_left,robot0_eye_in_hand,robot0_agentview_right",
        help=(
            "Comma-separated camera views to show side by side. Default shows "
            "left, wrist (eye-in-hand), right. Use e.g. 'robot0_eye_in_hand' "
            "for wrist-only."
        ),
    )
    p.add_argument(
        "--sim.stream_port",
        dest="sim_stream_port",
        type=int,
        default=8010,
        help=(
            "Port for the live MJPEG viewer (default: 8010; 0 disables). "
            "Open http://localhost:<port> in a browser; over SSH forward it with "
            "ssh -L <port>:localhost:<port> <host>."
        ),
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
        "--sim.direct_subtask",
        dest="sim_direct_subtask",
        action="store_true",
        help=(
            "Direct-subtask mode: what you type IS the subtask fed to the action "
            "expert (no LM subtask generation). Good when the model's subtask "
            "head is weak — you steer the policy with exact imperatives."
        ),
    )
    p.add_argument(
        "--disable_memory",
        action="store_true",
        help=(
            "Skip the memory-note generation on subtask change. Use for "
            "subtask-only checkpoints (no memory head) — avoids a wasted LM "
            "decode and a meaningless memory line."
        ),
    )
    p.add_argument(
        "--fp8",
        action="store_true",
        help=(
            "PI052 only: enable FlashRT FP8 MLP kernels. Calibrates on the "
            "first inference call and swaps every Gemma + SigLIP MLP to fused "
            "FP8 (needs the `kernels` package and CUDA SM>=8.9; degrades to "
            "BF16 otherwise). Speeds up the forward pass at a small accuracy "
            "cost — see policies/pi052/flashrt_fp8.py."
        ),
    )
    p.add_argument(
        "--subtask_chunks_per_gen",
        type=int,
        default=1,
        help=(
            "Throttle subtask gen to once every N action-chunk boundaries. "
            "Default 1 = regenerate the subtask on every chunk refresh. "
            "Set to 5 to run ~5 flow-matching action chunks per LM-head "
            "subtask gen — saves compute and avoids re-planning trajectories "
            "mid-grasp when a subtask is still valid across multiple chunks."
        ),
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
    args, unknown = p.parse_known_args(raw_argv)
    unsupported = [arg for arg in unknown if not arg.startswith(("--robot.", "--policy."))]
    if unsupported:
        p.error(f"unrecognized arguments: {' '.join(unsupported)}")
    args.raw_argv = raw_argv
    return args


# Columns the runtime supplies itself via its own message stream — strip
# them so the recipe render + text-tokenizer processor steps are no-ops.
_RUNTIME_OWNED_LANGUAGE_COLS = ("language_persistent", "language_events")


def _strip_runtime_owned_language_cols(sample: dict) -> None:
    """In-place drop of language columns the runtime owns at inference."""
    for k in _RUNTIME_OWNED_LANGUAGE_COLS:
        sample.pop(k, None)


# Model-input keys some policies emit OUTSIDE the ``observation.*`` namespace and
# still need at inference. MolmoAct2's processor packs its prompt + images into
# these top-level keys; PI0-family policies never produce them, so keeping the
# allowlist is a no-op for them.
_MODEL_INPUT_PASSTHROUGH_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
    "pixel_values_videos",
    "video_token_pooling",
    "video_grids",
)


def _select_observation_to_device(sample: dict, device: Any) -> dict:
    """Keep ``observation.*`` (+ model-input passthrough) keys, move tensors to ``device``."""
    import torch  # noqa: PLC0415

    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in sample.items()
        if isinstance(k, str) and (k.startswith("observation.") or k in _MODEL_INPUT_PASSTHROUGH_KEYS)
    }


def _load_policy_and_preprocessor(
    policy_path: str,
    dataset_repo_id: str | None,
    *,
    load_processors_from_checkpoint: bool = False,
    fp8: bool = False,
    device: str | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Load a policy checkpoint (local path or Hub repo id).

    Returns ``(policy, preprocessor, postprocessor, ds_meta)``.
    ``preprocessor`` / ``postprocessor`` / ``ds_meta`` are ``None``
    when no dataset is provided (rare — needed for autonomous robot
    mode to have action-denormalisation stats).

    When ``load_processors_from_checkpoint`` is set and no dataset is
    given, the pre/post processors are loaded from the checkpoint exactly
    like ``lerobot-eval`` (normalizer stats from the saved safetensors,
    recipe from ``cfg.recipe_path``). This is what the RoboCasa sim
    backend uses so it needs no dataset to match eval-time processing.
    """
    from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: PLC0415

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path

    # Optional device override — some checkpoints ship device=cpu.
    if device:
        cfg.device = device

    # Inference-only overrides (mirror lerobot-eval). torch.compile recompiles
    # whenever the prompt length changes (every subtask switch) — catastrophic
    # in the interactive runtime — and gradient checkpointing only slows the
    # forward pass. Neither is wanted for serving.
    if getattr(cfg, "compile_model", False):
        cfg.compile_model = False
    if getattr(cfg, "gradient_checkpointing", False):
        cfg.gradient_checkpointing = False

    # Opt-in FP8: only PI052 has the FlashRT MLP swap. The policy calibrates and
    # swaps on its first predict_action_chunk when this flag is set.
    if fp8:
        if hasattr(cfg, "use_flashrt_fp8_mlp"):
            cfg.use_flashrt_fp8_mlp = True
            logging.info("[runtime] FP8 MLP kernels enabled (FlashRT) for %s", cfg.type)
        else:
            logging.warning(
                "[runtime] --fp8 ignored: %s has no use_flashrt_fp8_mlp (PI052 only).",
                cfg.type,
            )

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
        if load_processors_from_checkpoint:
            # Eval-matching processors: stats from the checkpoint safetensors,
            # recipe from cfg.recipe_path. No dataset needed.
            preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=cfg.pretrained_path)

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
    """Closure feeding preprocessed dataset frames to the runtime, advancing
    ``advance_per_tick`` frames per call and looping at episode end.

    Language columns are stripped first — the runtime supplies its own
    messages from current state, not the dataset's annotations.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    ds = LeRobotDataset(dataset_repo_id, episodes=[episode])
    if len(ds) == 0:
        raise ValueError(f"Dataset {dataset_repo_id!r} episode {episode} is empty.")

    # Optional: replay training's augmentation pipeline so dry-run probes the
    # augmented support region — coherent text under jitter means the LM head
    # generalized; collapse to "\n" means it memorised unperturbed frames.
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

    def _provider() -> dict | None:
        idx = state["cursor"]
        if advance_per_tick > 0:
            state["cursor"] = (idx + advance_per_tick) % len(ds)

        sample = ds[idx]
        _strip_runtime_owned_language_cols(sample)

        if preprocessor is not None:
            sample = preprocessor(sample)

        return _select_observation_to_device(sample, device)

    return _provider


def _bootstrap_state_from_dataset(
    *,
    dataset_repo_id: str,
    episode: int,
    start_frame: int,
) -> dict[str, str]:
    """Pull task / active plan / memory / subtask at ``start_frame``, so the
    runtime's first prompt matches the canonical training prompts (an OOD
    prompt makes the model fall back to its dominant training mode).
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


def _select_task_interactively(
    *,
    ds_meta: Any,
    bootstrap_task: str | None,
) -> str | None:
    """Interactive task picker: numbered menu of dataset tasks (bootstrap task
    as default) plus a custom-input option; plain prompt without a dataset.
    Non-TTY runs skip the prompt and return the bootstrap task. Returns
    ``None`` when the operator declines (Ctrl-D / empty + no default).
    """
    options: list[str] = []
    seen: set[str] = set()
    if bootstrap_task:
        options.append(bootstrap_task)
        seen.add(bootstrap_task)
    if ds_meta is not None and getattr(ds_meta, "tasks", None) is not None:
        try:
            for t in list(ds_meta.tasks.index):
                if isinstance(t, str) and t and t not in seen:
                    options.append(t)
                    seen.add(t)
        except Exception as exc:  # noqa: BLE001 — defensive: tasks shape varies
            logger.debug("could not enumerate dataset tasks: %s", exc)

    if not sys.stdin.isatty():
        # Scripted / piped run: no interactive prompt; fall back to the
        # bootstrap default (may be None — REPL handles that).
        return bootstrap_task

    print("\n[runtime] Select startup task:", flush=True)
    if options:
        for i, opt in enumerate(options, 1):
            marker = "  (dataset default)" if opt == bootstrap_task else ""
            print(f"  [{i}] {opt}{marker}", flush=True)
        print("  [c] type a custom task", flush=True)
        prompt = "Choice [1]: " if bootstrap_task else "Choice: "
    else:
        print("  (no tasks available from dataset)", flush=True)
        prompt = "Enter task: "

    while True:
        try:
            choice = input(prompt).strip()
        except EOFError:
            print(flush=True)
            return bootstrap_task

        # No dataset options at all: the entered line *is* the task.
        if not options:
            return choice or None

        # Empty input: take the default (item 1) when there is one.
        if not choice:
            return options[0] if bootstrap_task else None

        if choice.lower() in ("c", "custom"):
            try:
                free = input("Enter task: ").strip()
            except EOFError:
                print(flush=True)
                return bootstrap_task
            if free:
                return free
            # Empty free-form input → loop back to the menu.
            continue

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]

        print(
            f"  invalid choice {choice!r}; pick 1–{len(options)} or 'c'.",
            flush=True,
        )


def _build_language_rollout_context(args: argparse.Namespace) -> Any:
    """Build the canonical rollout context for a language-controlled robot."""
    import threading  # noqa: PLC0415

    import draccus  # noqa: PLC0415

    from lerobot.configs import parser  # noqa: PLC0415
    from lerobot.rollout import RolloutConfig, build_rollout_context  # noqa: PLC0415

    # Importing the rollout entry point registers every bundled camera and
    # robot config choice used by Draccus. Third-party choices were registered
    # by the top-level entry point before reaching this function.
    from lerobot.scripts import lerobot_rollout as _rollout_registrations  # noqa: F401, PLC0415

    rollout_argv = [arg for arg in args.raw_argv if arg.startswith(("--policy.", "--robot."))]
    if args.task:
        rollout_argv.append(f"--task={args.task}")
    rollout_argv.extend(
        (
            "--strategy.type=base",
            f"--fps={args.ctrl_hz}",
            "--return_to_initial_position=false",
        )
    )

    previous_argv = sys.argv
    try:
        # RolloutConfig resolves --policy.path and policy overrides through the
        # shared parser helpers, which intentionally read sys.argv.
        sys.argv = [previous_argv[0], *rollout_argv]
        parsed_argv = parser.filter_path_args(RolloutConfig.__get_path_fields__(), rollout_argv)
        cfg = draccus.parse(config_class=RolloutConfig, args=parsed_argv)
    finally:
        sys.argv = previous_argv

    if getattr(cfg.policy, "compile_model", False):
        cfg.policy.compile_model = False
    if getattr(cfg.policy, "gradient_checkpointing", False):
        cfg.policy.gradient_checkpointing = False
    if args.fp8:
        if hasattr(cfg.policy, "use_flashrt_fp8_mlp"):
            cfg.policy.use_flashrt_fp8_mlp = True
        else:
            logger.warning("--fp8 ignored: %s does not support it", cfg.policy.type)

    return build_rollout_context(cfg, threading.Event())


def _build_rollout_runtime_io(
    ctx: Any,
    *,
    rerun_log: bool,
    get_task: Callable[[], str | None],
) -> tuple[Callable[[], dict | None], Callable[[Any], None]]:
    """Adapt a rollout context to the language runtime's observation/action API."""
    import torch  # noqa: PLC0415

    from lerobot.policies.utils import (  # noqa: PLC0415
        make_robot_action,
        prepare_observation_for_inference,
    )
    from lerobot.utils.feature_utils import build_dataset_frame  # noqa: PLC0415

    robot = ctx.hardware.robot_wrapper
    device = torch.device(ctx.runtime.cfg.device or "cpu")
    latest_raw: dict[str, Any] = {}

    def _provider() -> dict | None:
        try:
            raw = robot.get_observation()
            latest_raw.clear()
            latest_raw.update(raw)
            if rerun_log:
                from lerobot.runtime import rerun_viz  # noqa: PLC0415

                camera_keys = list(robot.cameras)
                state = {k: v for k, v in raw.items() if isinstance(v, (int, float))}
                rerun_viz.log_robot_frame(raw, camera_keys, state=state, task=get_task())
            _strip_runtime_owned_language_cols(raw)
            processed = ctx.processors.robot_observation_processor(raw)
            observation = build_dataset_frame(ctx.data.dataset_features, processed, prefix="observation")
            observation = prepare_observation_for_inference(
                observation,
                device,
                task=get_task(),
                robot_type=robot.robot_type,
            )
            observation = ctx.policy.preprocessor(observation)
            return _select_observation_to_device(observation, device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("robot observation pipeline failed: %s", exc)
            return None

    def _executor(action: Any) -> None:
        try:
            processed_action = ctx.policy.postprocessor(action)
            if isinstance(processed_action, torch.Tensor):
                if processed_action.ndim == 1:
                    processed_action = processed_action.unsqueeze(0)
                action_dict = make_robot_action(processed_action, ctx.data.dataset_features)
            elif isinstance(processed_action, dict):
                action_dict = processed_action
            else:
                logger.warning("unsupported action type %r — skipping", type(processed_action))
                return
            raw = latest_raw or robot.get_observation()
            robot_action = ctx.processors.robot_action_processor((action_dict, raw))
            robot.send_action(robot_action)
            if rerun_log:
                from lerobot.runtime import rerun_viz  # noqa: PLC0415

                rerun_viz.log_cameras(robot)
        except Exception as exc:  # noqa: BLE001
            logger.error("robot action pipeline failed: %s", exc, exc_info=True)

    return _provider, _executor


def _print_runtime_help() -> None:
    """Print the slash-command reference."""
    print(
        "[runtime] commands (arguments need no quotes):\n"
        "  /action <task>     run the robot; an argument switches to that task\n"
        "  /action            resume the robot on the current task\n"
        "  /action <seconds>  run the robot for N seconds, then auto-pause\n"
        "  /pause             pause the action loop — robot holds position\n"
        "  /help              show this help\n"
        "  stop | quit | exit end the session",
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
                f"[runtime] action — running {secs:g}s, then auto-pause",
                flush=True,
            )
        else:
            runtime.state["action_deadline"] = None
            if rest:
                runtime.set_task(rest)
                # New task → drop the stale subtask so the high-level
                # loop regenerates one for the new goal.
                runtime.state["current_subtask"] = None
                print(f"[runtime] action — task: {rest!r}", flush=True)
            elif runtime.state.get("task"):
                print(
                    f"[runtime] action — resuming: {runtime.state['task']!r}",
                    flush=True,
                )
            else:
                runtime.state["mode"] = "paused"
                print(
                    "[runtime] no task set — use /action <your task>",
                    flush=True,
                )
        return True

    if cmd in {"/pause", "/p"}:
        runtime.state["mode"] = "paused"
        runtime.state["action_deadline"] = None
        _clear_action_queue(runtime)
        print("[runtime] paused — robot holding position", flush=True)
        return True

    if cmd in {"/help", "/?"}:
        _print_runtime_help()
        return True
    return False


def _make_state_panel_renderer(
    runtime: Any,
    *,
    mode_label: str,
    panel_label: str = "Runtime",
    scrollback: list[str] | None = None,
) -> Callable[[list[str] | None], None]:
    """Return a closure that prints the task/subtask/plan/memory panel.

    Used by ``_run_repl`` for dataset-driven dry runs.
    """
    from rich.console import Console  # noqa: PLC0415

    console = Console(highlight=False)

    def _redraw(robot_lines: list[str] | None = None) -> None:
        console.clear()
        st = runtime.state
        run_mode = st.get("mode", "action")
        mode_tag = "[green]mode: action[/]" if run_mode == "action" else "[yellow]mode: paused[/]"
        console.rule(f"[bold]{panel_label}[/] · {mode_label} · {mode_tag}", style="cyan")
        # Always-visible command hint so the operator never has to
        # remember the slash commands.
        if run_mode == "action":
            console.print("  [dim]commands:[/] [bold]/pause[/] stop  ·  [bold]/help[/]  ·  [bold]stop[/]")
        else:
            console.print(
                "  [dim]commands:[/] [bold]/action[/] <task> run  ·  [bold]/help[/]  ·  [bold]stop[/]"
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
        dispatched = int(st.get("actions_dispatched") or 0)
        console.print(f"  [dim]queued actions: {queue_len}    dispatched: {dispatched}[/]")

        # Overfit / memorisation diagnostics from the adapter. High repeat
        # + fully cycling queue ⇒ stuck on one subtask (memorised a phase);
        # climbing gibberish ⇒ LM head collapsed to chat-template salads.
        diag = getattr(runtime.policy_adapter, "diag", None)
        if diag is not None:
            raw_subtask = diag.last_raw.get("subtask")
            sub_rep = int(diag.repeat)
            sub_gib = int(diag.gibberish.get("subtask", 0))
            sub_empty = int(diag.empty.get("subtask", 0))
            if raw_subtask is not None or sub_rep or sub_gib or sub_empty:
                raw_display = (raw_subtask or "(empty)")[:80]
                color = "yellow" if (sub_rep >= 3 or sub_gib >= 3 or sub_empty >= 3) else "dim"
                console.print(
                    f"  [{color}]subtask diag    repeat:{sub_rep}  "
                    f"gibberish:{sub_gib}  empty:{sub_empty}  "
                    f"last_raw: {raw_display!r}[/]"
                )
            mem_gib = int(diag.gibberish.get("memory", 0))
            if mem_gib:
                console.print(f"  [dim]gen rejects     memory:{mem_gib}[/]")
        console.rule(style="cyan")
        # Runtime scrollback — log lines pushed from generation steps
        # (warnings, gibberish rejections, plan speech). Last N lines,
        # oldest first.
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
                "  [dim]Type [bold]/action <your task>[/bold] to begin, /help for commands, stop to exit.[/]"
            )

    return _redraw


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


def run(
    argv: list[str] | None = None,
    *,
    adapter_factory: Callable[[Any, GenerationConfig], LanguageConditionedPolicyAdapter] | None = None,
    panel_label: str | None = None,
    prog: str = "lerobot-language-runtime",
) -> int:
    """Run the interactive language-conditioned runtime CLI.

    ``adapter_factory`` turns ``(policy, GenerationConfig)`` into a
    :class:`LanguageConditionedPolicyAdapter` (typically the adapter class).
    When ``None`` it is resolved from :mod:`lerobot.runtime.registry` by the
    loaded policy's type, so the ``lerobot-rollout`` entry
    point serves every registered policy. ``panel_label`` defaults to the
    policy type.
    """
    args = _parse_args(argv, prog=prog)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _silence_noisy_loggers()

    sim_mode = bool(getattr(args, "sim", False)) and not args.no_robot
    autonomous_mode = bool(args.robot_type) and not args.no_robot
    if sim_mode and autonomous_mode:
        print(
            "[runtime] ERROR: --sim and --robot.type are mutually exclusive "
            "(pick a simulator scene OR a real robot).",
            file=sys.stderr,
        )
        return 2
    # Autonomous robot mode can run without a dataset: normalization stats are
    # loaded from the checkpoint (same as lerobot-rollout and sim mode) and the
    # observation/action feature schema is derived from the connected robot. A
    # dataset is still honoured when given — its stats then take precedence.
    if autonomous_mode and not args.dataset_repo_id:
        logger.info(
            "autonomous robot mode without --dataset.repo_id: loading "
            "normalization stats from the checkpoint and deriving the feature "
            "schema from the robot."
        )

    # Create the sim env subprocess BEFORE the policy initialises CUDA — the
    # env worker inherits a corrupt EGL/GL context if forked from a CUDA parent
    # (dark/garbled renders). This mirrors eval's make_env-before-make_policy.
    sim_env = None
    sim_obs = None
    sim_stream_server = None
    sim_holder: dict[str, Any] = {"backend": None}
    if sim_mode:
        from lerobot.runtime.sim_robocasa import create_sim_env, start_mjpeg_server  # noqa: PLC0415

        # Start the live viewer first so the port listens during the ~60s model
        # load (browsers get a loading page instead of connection-refused).
        if args.sim_stream_port:
            sim_stream_server = start_mjpeg_server(
                args.sim_stream_port,
                lambda: sim_holder["backend"]._latest_frame if sim_holder["backend"] else None,
            )
        print(
            f"[runtime] starting RoboCasa sim scene={args.sim_task!r} split={args.sim_split!r}",
            flush=True,
        )
        sim_env, sim_obs = create_sim_env(
            task=args.sim_task,
            split=args.sim_split,
            obj_registries=[r.strip() for r in args.sim_obj_registries.split(",") if r.strip()],
            seed=args.sim_seed,
            render_size=args.sim_render_size,
        )

    rollout_ctx = None
    if autonomous_mode:
        print("[runtime] building rollout context (policy, processors, robot)", flush=True)
        rollout_ctx = _build_language_rollout_context(args)
        policy = rollout_ctx.policy.policy
        preprocessor = rollout_ctx.policy.preprocessor
        postprocessor = rollout_ctx.policy.postprocessor
        ds_meta = None
        if args.dataset_repo_id is not None:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: PLC0415

            ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)
    else:
        print(f"[runtime] loading policy from {args.policy_path}", flush=True)
        policy, preprocessor, postprocessor, ds_meta = _load_policy_and_preprocessor(
            args.policy_path,
            args.dataset_repo_id,
            load_processors_from_checkpoint=sim_mode,
            fp8=args.fp8,
            device=args.policy_device,
        )

    policy_type = getattr(policy.config, "type", None)
    if adapter_factory is None:
        from .registry import get_language_adapter_factory  # noqa: PLC0415

        adapter_factory = get_language_adapter_factory(policy_type)
    if panel_label is None:
        panel_label = str(policy_type or "runtime").upper()

    # Bootstrap the canonical task from the dataset whenever one is
    # provided, so the interactive picker below can offer it as the
    # default. The model is memorised on the exact training wording, so
    # matching it is what gets recall to fire.
    bootstrap_state: dict[str, str] = {}
    if args.dataset_repo_id is not None:
        bootstrap_state = _bootstrap_state_from_dataset(
            dataset_repo_id=args.dataset_repo_id,
            episode=args.dataset_episode,
            start_frame=args.dataset_start_frame,
        )

    # Interactive task picker. Skipped when ``--task`` is already set on
    # the CLI (scripted runs and explicit overrides win). When no task
    # was passed, prompt the operator: pick from the dataset's tasks or
    # type a custom one. Non-TTY runs fall back to the bootstrap task
    # silently — the existing "first stdin line becomes task" flow in
    # ``_run_repl`` still handles the no-default case.
    if not args.task:
        chosen = _select_task_interactively(
            ds_meta=ds_meta,
            bootstrap_task=bootstrap_state.get("task"),
        )
        if chosen:
            args.task = chosen
            print(f"[runtime] task: {args.task!r}", flush=True)

    # No startup prompts — the runtime is command-driven. It comes up at
    # the command line in ``paused`` mode (robot idle) unless ``--mode``
    # forces a mode. The operator drives it with /action, /pause and
    # /question.
    startup_mode = args.mode or "paused"

    observation_provider: Callable[[], dict | None] | None = None
    robot_executor: Callable[[Any], None] | None = None
    robot = None
    sim_backend = None
    # Late-bound handle to the runtime so the robot observation provider can read
    # the live task/subtask each frame (the runtime is created further below).
    runtime_box: dict[str, Any] = {}

    def _live_task() -> str | None:
        rt = runtime_box.get("rt")
        if rt is None:
            return args.task
        return rt.state.get("current_subtask") or rt.state.get("task") or args.task

    if sim_mode:
        from lerobot.runtime.sim_robocasa import RoboCasaSimBackend  # noqa: PLC0415

        sim_backend = RoboCasaSimBackend(
            env=sim_env,
            last_obs=sim_obs,
            task=args.sim_task,
            seed=args.sim_seed,
            device=str(getattr(policy.config, "device", "cpu")),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            record=(args.sim_record == "mp4"),
            output_dir=args.sim_output_dir,
            view_cams=[v.strip() for v in args.sim_views.split(",") if v.strip()],
        )
        observation_provider = sim_backend.observation_provider
        robot_executor = sim_backend.action_executor
        robot = sim_backend
        # Point the already-running live viewer at the backend and hand it the
        # server so disconnect() shuts it down cleanly.
        sim_holder["backend"] = sim_backend
        if sim_stream_server is not None:
            sim_backend.attach_stream_server(sim_stream_server)
    elif autonomous_mode:
        if args.rerun:
            from lerobot.runtime.rerun_viz import start_rerun  # noqa: PLC0415

            start_rerun(
                app_name=f"lerobot_{policy_type or 'runtime'}",
                grpc_port=args.rerun_grpc_port,
                web_port=args.rerun_web_port,
            )
        robot = rollout_ctx.hardware.robot_wrapper.inner
        print(f"[runtime] connected to {robot.name}", flush=True)
        observation_provider, robot_executor = _build_rollout_runtime_io(
            rollout_ctx,
            rerun_log=bool(args.rerun),
            get_task=_live_task,
        )
    elif args.dataset_repo_id is not None:
        print(
            f"[runtime] streaming observations from {args.dataset_repo_id} "
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

    # Text-generation knobs are fixed config, passed to the adapter at
    # construction — not smuggled through per-tick runtime state. Lets the
    # operator try e.g. ``--text_temperature=0.6 --subtask_chunks_per_gen=5``
    # on an under-trained checkpoint without recompiling.
    gen_config = GenerationConfig(
        min_new_tokens=int(args.text_min_new_tokens or 0),
        temperature=float(args.text_temperature or 0.0),
        top_p=float(args.text_top_p or 1.0),
        chunks_per_regen=max(1, int(args.subtask_chunks_per_gen or 1)),
        enable_memory=not bool(getattr(args, "disable_memory", False)),
        enable_subtask=not _direct_subtask_enabled(args),
    )
    runtime = LanguageConditionedRuntime(
        policy_adapter=adapter_factory(policy, gen_config),
        observation_provider=observation_provider,
        action_executor=robot_executor,
        # No background event collector — the REPL drives ticks
        # synchronously after each user input (REPL mode). Autonomous
        # mode runs ``runtime.run()`` in a thread; stdin events are
        # injected from the foreground.
        event_collector=None,
        chunk_hz=args.chunk_hz,
        ctrl_hz=args.ctrl_hz,
        high_level_hz=args.high_level_hz,
    )
    # Let the robot observation provider read the live task/subtask each frame.
    runtime_box["rt"] = runtime
    # Apply the startup mode chosen above the task picker.
    runtime.state["mode"] = startup_mode
    if args.task:
        runtime.set_task(args.task)
    # Seed the current subtask from the dataset so the first chunk —
    # before the adapter has generated one — has a real subtask to
    # condition the action expert on instead of falling back to the
    # bare task. Plan and memory are NOT seeded: the current recipe
    # trains neither, no inference step consumes them, and seeding
    # them only put a stale plan in the status panel that does
    # nothing.
    if bootstrap_state.get("subtask"):
        runtime.state["current_subtask"] = bootstrap_state["subtask"]

    # Let the sim backend read live task/subtask/memory for the video overlay.
    if sim_backend is not None:
        sim_backend.bind_runtime(runtime)
        # Sim runs its control/render loop in the MAIN thread (see
        # _run_sim_interactive) — background-thread rendering corrupts EGL.
        return _run_sim_interactive(
            runtime,
            sim_backend,
            initial_task=args.task,
            max_ticks=args.max_ticks,
            panel_label=panel_label,
            direct_subtask=_direct_subtask_enabled(args),
        )

    if autonomous_mode:
        return _run_robot_interactive(
            runtime,
            robot,
            initial_task=args.task,
            max_ticks=args.max_ticks,
            direct_subtask=_direct_subtask_enabled(args),
            panel_label=panel_label,
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
            print(f"[runtime] {line}", flush=True)
    return _run_repl(runtime, initial_task=args.task, max_ticks=args.max_ticks, panel_label=panel_label)


def _direct_subtask_enabled(args: Any) -> bool:
    """Direct-subtask mode via either the general or sim-scoped flag."""
    return bool(getattr(args, "direct_subtask", False) or getattr(args, "sim_direct_subtask", False))


def _run_sim_interactive(
    runtime: Any,
    sim_backend: Any,
    *,
    initial_task: str | None,
    max_ticks: int | None,
    panel_label: str = "Runtime",
    direct_subtask: bool = False,
) -> int:
    """Main-thread control loop for the RoboCasa sim backend.

    The tick loop — and therefore MuJoCo's EGL rendering — runs in the MAIN
    thread. Driving the sim render from a background thread intermittently
    corrupts the offscreen GL context (dark/garbled frames); main-thread
    stepping matches ``lerobot-eval`` and renders cleanly. Stdin is polled
    non-blockingly so typed commands still work while the sim runs.
    """
    import select  # noqa: PLC0415
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    if initial_task:
        runtime.set_task(initial_task)
        # In direct-subtask mode the typed text IS the subtask; otherwise clear
        # it so the model generates one.
        runtime.state["current_subtask"] = initial_task if direct_subtask else None
        runtime.state["mode"] = "action"

    # Clean chat-style prompt. The control loop steps in the MAIN thread (clean
    # EGL rendering); the browser live-view shows the rollout, so the terminal
    # stays a quiet command line. Nothing is printed mid-step, so typing is never
    # clobbered — you can queue the next command any time.
    _mode_line = (
        "  Mode: DIRECT subtask (your text drives the action expert as-is)\n"
        if direct_subtask
        else "  Mode: task (the model generates a subtask from your text)\n"
    )
    print(
        f"\n{'=' * 64}\n"
        f"  {panel_label} — RoboCasa interactive sim (one persistent kitchen)\n"
        f"{_mode_line}"
        f"  Type a command + Enter to run it, e.g.  open the fridge\n"
        f"  Commands:  /pause  ·  /resume  ·  /reset (new kitchen)  ·  stop\n"
        f"{'=' * 64}",
        flush=True,
    )

    def _prompt() -> None:
        print("\n> ", end="", flush=True)

    _prompt()
    ticks_done = 0
    stdin_open = True
    try:
        while True:
            # Non-blocking stdin: a full line (canonical-mode terminal) is read
            # only when Enter is pressed, so line editing works normally.
            if stdin_open and select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if line == "":  # EOF — keep running the sim, stop reading stdin
                    stdin_open = False
                else:
                    cmd = line.strip()
                    if cmd:
                        low = cmd.lower()
                        if low in {"stop", "quit", "exit"}:
                            break
                        elif low in {"/pause", "pause", "/p"}:
                            runtime.state["mode"] = "paused"
                            _clear_action_queue(runtime)
                            print("[paused] robot holding", flush=True)
                        elif low in {"/resume", "resume", "/run"}:
                            runtime.state["mode"] = "action"
                            print("[running]", flush=True)
                        elif low in {"/reset", "reset"}:
                            sim_backend.reset_scene()
                            _clear_action_queue(runtime)
                            runtime.state["current_subtask"] = None
                            if hasattr(runtime.policy, "reset"):
                                runtime.policy.reset()
                            print("[reset] new kitchen scene", flush=True)
                        else:
                            # A bare line is a new command: switch the robot to it
                            # immediately (clear the in-flight chunk + subtask) and
                            # force the subtask to regenerate on the very next tick
                            # (reset the adapter throttle + high-level rate gate).
                            runtime.set_task(cmd)
                            # Direct mode: the typed text is the subtask itself;
                            # otherwise clear it so the model regenerates one.
                            runtime.state["current_subtask"] = cmd if direct_subtask else None
                            _clear_action_queue(runtime)
                            adapter = getattr(runtime, "policy_adapter", None)
                            if adapter is not None and hasattr(adapter, "_chunks_until_regen"):
                                adapter._chunks_until_regen = 0
                            gate = getattr(runtime, "_language_gate", None)
                            if gate is not None and hasattr(gate, "rearm"):
                                gate.rearm()
                            runtime.state["mode"] = "action"
                            print(f"[running] {cmd}", flush=True)
                    _prompt()

            # One tick in the MAIN thread: subtask/action gen + env.step + render.
            # inference_mode matches lerobot-eval's forward context.
            if runtime.state.get("mode", "paused") == "action":
                with torch.inference_mode():
                    runtime.step_once()
                ticks_done += 1
            else:
                time.sleep(0.05)  # idle only while paused (robot not moving)
            if runtime.state.stop:
                break
            if max_ticks is not None and ticks_done >= max_ticks:
                break
    except KeyboardInterrupt:
        print("\n[stopping]", flush=True)
    finally:
        runtime.stop()
        try:
            sim_backend.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"[runtime] WARNING: sim disconnect raised {exc}", flush=True)
    return 0


def _run_robot_interactive(
    runtime: Any,
    robot: Any,
    *,
    initial_task: str | None,
    max_ticks: int | None,
    direct_subtask: bool = False,
    panel_label: str = "Runtime",
) -> int:
    """Real-robot interactive loop.

    The control loop runs at real-time rates in a background thread
    (``runtime.run()`` — a robot must be driven at a steady ``ctrl_hz``), while
    the foreground is a clean chat prompt: type a command to run it (generate- or
    direct-subtask mode), ``/pause`` / ``/resume`` / ``stop``. Starts PAUSED so
    the arm doesn't move until you issue a command.
    """
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415

    if initial_task:
        runtime.set_task(initial_task)
        runtime.state["current_subtask"] = initial_task if direct_subtask else None
        # A task was given (via --task or the startup picker) => start running it
        # immediately. Without an initial task we stay paused until the first
        # typed command (which switches to action). No flag needed.
        runtime.state["mode"] = "action"

    mode_line = (
        "DIRECT subtask (your text drives the action expert)"
        if direct_subtask
        else "task (the model generates a subtask from your text)"
    )
    starting_action = runtime.state.get("mode", "paused") == "action"
    start_line = (
        f"  Starting in ACTION — the ARM WILL MOVE NOW on: {initial_task!r}\n"
        if starting_action
        else "  Starts PAUSED. Type a command + Enter to run it — the ARM WILL MOVE.\n"
    )
    print(
        f"\n{'=' * 64}\n"
        f"  {panel_label} — OMX robot runtime  ·  Mode: {mode_line}\n"
        f"{start_line}"
        f"  Commands:  /pause  ·  /resume  ·  stop\n"
        f"{'=' * 64}",
        flush=True,
    )

    thread = threading.Thread(
        target=runtime.run, kwargs={"max_ticks": max_ticks}, name="runtime-loop", daemon=True
    )
    thread.start()
    try:
        while thread.is_alive():
            try:
                line = input("\n> ").strip()
            except EOFError:
                break
            if not line:
                continue
            low = line.lower()
            if low in {"stop", "quit", "exit"}:
                break
            elif low in {"/pause", "pause", "/p"}:
                runtime.state["mode"] = "paused"
                _clear_action_queue(runtime)
                print("[paused] robot holding", flush=True)
            elif low in {"/resume", "resume", "/run"}:
                runtime.state["mode"] = "action"
                print("[running]", flush=True)
            else:
                # New command: switch task/subtask immediately and regenerate.
                runtime.set_task(line)
                runtime.state["current_subtask"] = line if direct_subtask else None
                _clear_action_queue(runtime)
                adapter = getattr(runtime, "policy_adapter", None)
                if adapter is not None and hasattr(adapter, "_chunks_until_regen"):
                    adapter._chunks_until_regen = 0
                gate = getattr(runtime, "_language_gate", None)
                if gate is not None and hasattr(gate, "rearm"):
                    gate.rearm()
                runtime.state["mode"] = "action"
                print(f"[running] {line}", flush=True)
    except KeyboardInterrupt:
        print("\n[stopping]", flush=True)
    finally:
        runtime.stop()
        for _ in range(10):
            if not thread.is_alive():
                break
            time.sleep(0.1)
        try:
            robot.disconnect()
            print("[runtime] robot disconnected", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[runtime] WARNING: robot.disconnect raised {exc}", flush=True)
    return 0


def _run_repl(
    runtime: Any, *, initial_task: str | None, max_ticks: int | None, panel_label: str = "Runtime"
) -> int:
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
            "[runtime] rich is required for the interactive REPL. `pip install rich` and re-run.",
            file=sys.stderr,
        )
        return 2

    _redraw = _make_state_panel_renderer(runtime, mode_label="dry-run", panel_label=panel_label)
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
                    "[runtime] no task yet — use /action <your task>",
                    flush=True,
                )
                _redraw(last_logs)
                continue
            runtime.state["recent_interjection"] = line
            _emit(runtime.state, "user_interjection")

            last_logs = runtime.step_once() or []
            _redraw(last_logs)

            ticks_done += 1
            if max_ticks is not None and ticks_done >= max_ticks:
                break
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/]")
    console.print("[dim]runtime stopped[/]")
    return 0
