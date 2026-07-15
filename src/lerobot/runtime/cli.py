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
"""Interactive CLI for language-conditioned policy rollouts.

It supports a text-only REPL, real robots, and RoboCasa with local or Hub checkpoints.
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
        "--task",
        dest="task",
        type=str,
        default=None,
        help=("Initial task. If omitted, enter a task at the interactive prompt."),
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
        help="Skip robot connection and open a language-only REPL.",
    )
    # ``--robot.type`` enables real-time control while stdin remains interactive.
    p.add_argument(
        "--robot.type",
        dest="robot_type",
        type=str,
        default=None,
        help=(
            "Robot config choice (e.g. ``so101``, ``so101_follower``). "
            "When set, the runtime drives the actual robot at "
            "``--ctrl_hz`` instead of the no-robot REPL. Implies "
            "``--autonomous`` unless ``--no_robot`` is also "
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
    # ``--sim`` uses the eval pipeline and is mutually exclusive with a robot.
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


# Non-observation model inputs emitted by processors such as MolmoAct2's.
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
    *,
    load_processors_from_checkpoint: bool = False,
    fp8: bool = False,
    device: str | None = None,
) -> tuple[Any, Any, Any]:
    """Load a local or Hub policy, optionally with its eval processors."""
    from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: PLC0415

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path

    # Optional device override — some checkpoints ship device=cpu.
    if device:
        cfg.device = device

    # Variable prompts trigger recompilation, and checkpointing only adds inference overhead.
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

    preprocessor = None
    postprocessor = None
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg)
    policy.to(cfg.device)
    if load_processors_from_checkpoint:
        preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=cfg.pretrained_path)

    policy.eval()
    return policy, preprocessor, postprocessor


def _build_language_rollout_context(args: argparse.Namespace) -> Any:
    """Build the canonical rollout context for a language-controlled robot."""
    import threading  # noqa: PLC0415

    import draccus  # noqa: PLC0415

    from lerobot.configs import parser  # noqa: PLC0415
    from lerobot.rollout import RolloutConfig, build_rollout_context  # noqa: PLC0415

    # Import for bundled Draccus camera and robot registrations.
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

    Used by ``_run_repl`` for the no-robot language REPL.
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

        # Surface repeated or rejected generations as overfitting diagnostics.
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
        # Show recent generation warnings and speech oldest-first.
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
    """Keep request-level third-party logs out of the interactive prompt."""
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

    # Clamp warnings can fire every control tick and flood the panel.
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
    # Fork the simulator before CUDA initialization to avoid inherited EGL corruption.
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
    else:
        print(f"[runtime] loading policy from {args.policy_path}", flush=True)
        policy, preprocessor, postprocessor = _load_policy_and_preprocessor(
            args.policy_path,
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

    # Default to idle until the operator supplies a command.
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
    # Generation settings belong to the adapter rather than mutable runtime state.
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
        event_collector=None,
        chunk_hz=args.chunk_hz,
        ctrl_hz=args.ctrl_hz,
        high_level_hz=args.high_level_hz,
    )
    # Let the robot observation provider read the live task/subtask each frame.
    runtime_box["rt"] = runtime
    # Apply the configured startup mode.
    runtime.state["mode"] = startup_mode
    if args.task:
        runtime.set_task(args.task)

    # Let the sim backend read live task/subtask/memory for the video overlay.
    if sim_backend is not None:
        sim_backend.bind_runtime(runtime)
        # Keep EGL rendering on the main thread.
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
    """Keep RoboCasa rendering on the main thread while polling stdin."""
    import select  # noqa: PLC0415
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    if initial_task:
        runtime.set_task(initial_task)
        # In direct-subtask mode the typed text IS the subtask; otherwise clear
        # it so the model generates one.
        runtime.state["current_subtask"] = initial_task if direct_subtask else None
        runtime.state["mode"] = "action"

    # Keep the terminal quiet while the browser renders the rollout.
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
                            # Clear queued actions and rearm generation for a new command.
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

            # Match lerobot-eval's inference context on the main thread.
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
    """Run steady robot control in the background and commands in the foreground."""
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415

    if initial_task:
        runtime.set_task(initial_task)
        runtime.state["current_subtask"] = initial_task if direct_subtask else None
        # An explicit initial task starts immediately; otherwise the robot stays paused.
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
    """Redraw the status block and logs once per REPL turn."""
    try:
        from rich.console import Console  # noqa: PLC0415
    except ImportError:
        print(
            "[runtime] rich is required for the interactive REPL. `pip install rich` and re-run.",
            file=sys.stderr,
        )
        return 2

    _redraw = _make_state_panel_renderer(runtime, mode_label="no robot", panel_label=panel_label)
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

            # Slash commands, including VQA questions, run inline.
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
