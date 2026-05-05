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
        "--chunk_hz", type=float, default=4.0, help="Action-chunk generation rate."
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
) -> tuple[Any, Any, Any]:
    """Load a SmolVLA2 checkpoint (local path or Hub repo id).

    When ``dataset_repo_id`` is provided, the dataset's metadata is used
    to derive policy features (matching the standard
    ``make_policy(cfg, ds_meta=...)`` flow used by ``lerobot-train`` and
    ``lerobot-record``). When it isn't, we fall back to instantiating
    the policy directly via ``from_pretrained`` — this skips the
    feature-derivation path that ``make_policy`` insists on, but also
    means we can't load the saved preprocessor pipeline (which depends
    on ``input_features`` / ``output_features``). For inference-only
    dry-runs this is fine; the policy still loads.

    Returns ``(policy, preprocessor, ds_meta)`` where ``preprocessor``
    and ``ds_meta`` may be ``None`` if no dataset was provided.
    """
    from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: PLC0415

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path

    ds_meta = None
    preprocessor = None
    if dataset_repo_id is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: PLC0415

        ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
        policy = make_policy(cfg, ds_meta=ds_meta)
        # NOTE: we deliberately pass ``pretrained_path=None`` here even
        # though the checkpoint ships a ``policy_preprocessor.json``.
        # ``RenderMessagesStep`` carries a ``TrainingRecipe`` field that
        # isn't faithfully serialized into that JSON, so the saved
        # pipeline can't currently be round-tripped via
        # ``PolicyProcessorPipeline.from_pretrained`` — it crashes with
        # ``RenderMessagesStep.__init__() missing 1 required argument:
        # 'recipe'``. Building fresh from ``cfg`` re-runs
        # ``make_smolvla2_pre_post_processors``, which loads the recipe
        # YAML referenced by ``cfg.recipe_path`` and wires it back into
        # ``RenderMessagesStep`` correctly. Normalization stats come
        # from ``ds_meta.stats`` (the same dataset the user is feeding
        # into the runtime), so no quality loss in practice.
        preprocessor, _ = make_pre_post_processors(
            cfg,
            pretrained_path=None,
            dataset_stats=ds_meta.stats,
        )
    else:
        # No dataset: instantiate the policy class directly so we don't
        # need ds_meta. This bypasses ``make_policy``'s feature-shape
        # derivation, which is fine for a pretrained checkpoint where
        # the saved config already carries those shapes.
        from lerobot.policies.factory import get_policy_class  # noqa: PLC0415

        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(policy_path, config=cfg)
        policy.to(cfg.device)

    policy.eval()
    return policy, preprocessor, ds_meta


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print(f"[smolvla2] loading policy from {args.policy_path}", flush=True)
    policy, preprocessor, _ds_meta = _load_policy_and_preprocessor(
        args.policy_path, args.dataset_repo_id
    )

    observation_provider: Callable[[], dict | None] | None = None
    if args.dataset_repo_id is not None:
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

    robot_executor = None
    if not args.no_robot:
        print(
            "[smolvla2] WARNING: real-robot integration is a follow-up. "
            "Running in dry-run mode for now (no actions executed).",
            flush=True,
        )

    from lerobot.policies.smolvla2.inference import SmolVLA2Runtime  # noqa: PLC0415

    runtime = SmolVLA2Runtime(
        policy=policy,
        tools=tools,
        observation_provider=observation_provider,
        robot_executor=robot_executor,
        # No background event collector — the REPL drives ticks
        # synchronously after each user input. The runtime's own
        # ``run()`` loop is bypassed here in favour of ``step_once()``
        # so the input prompt and the live state panel co-exist
        # cleanly.
        event_collector=None,
        chunk_hz=args.chunk_hz,
        ctrl_hz=args.ctrl_hz,
        high_level_hz=args.high_level_hz,
    )
    if args.task:
        runtime.set_task(args.task)

    return _run_repl(runtime, initial_task=args.task, max_ticks=args.max_ticks)


def _run_repl(runtime: Any, *, initial_task: str | None, max_ticks: int | None) -> int:
    """Two-zone TUI: chat scrollback above a persistent state panel.

    Uses :class:`rich.live.Live` to keep the state panel rendered
    below the chat history. ``console.input`` for the prompt — Live
    auto-suspends repaint while the user is typing.
    """
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.live import Live  # noqa: PLC0415
    except ImportError:
        print(
            "[smolvla2] rich is required for the interactive REPL. "
            "`pip install rich` and re-run.",
            file=sys.stderr,
        )
        return 2

    from lerobot.policies.smolvla2.inference import (  # noqa: PLC0415
        make_state_panel,
        print_robot_lines,
        print_user_line,
    )

    console = Console()
    console.print(
        "[bold]SmolVLA2[/] ready. "
        "Type a task to begin, then any line for an interjection, "
        "a line ending in '?' for VQA, or 'stop' to exit.",
    )
    if initial_task is None:
        console.print("[dim]No --task provided; first stdin line will be used.[/]")

    panel = make_state_panel(runtime.state)
    ticks_done = 0
    with Live(
        panel,
        console=console,
        refresh_per_second=4,
        transient=False,
        screen=False,
    ) as live:
        try:
            while True:
                try:
                    line = console.input("[bold cyan]>[/] ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                lower = line.lower()
                if lower in {"stop", "quit", "exit"}:
                    break

                print_user_line(live.console, line)

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

                logs = runtime.step_once()
                if logs:
                    print_robot_lines(live.console, logs)
                live.update(make_state_panel(runtime.state))

                ticks_done += 1
                if max_ticks is not None and ticks_done >= max_ticks:
                    break
        except KeyboardInterrupt:
            console.print("\n[smolvla2] interrupted", style="dim")
    print("[smolvla2] runtime stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
