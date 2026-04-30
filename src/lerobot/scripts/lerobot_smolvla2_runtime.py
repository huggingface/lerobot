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

Dry run on a checkpoint, no robot connected — useful for sanity-
checking text generation::

    uv run lerobot-smolvla2-runtime \\
        --policy.path=outputs/train/smolvla2_super_poulain/000020000/pretrained_model \\
        --no_robot \\
        --task="please clean the kitchen"

With a real robot::

    uv run lerobot-smolvla2-runtime \\
        --policy.path=... \\
        --robot.type=so101 --robot.port=/dev/tty.usbmodem... \\
        --tts.voice=alba

Tool dispatch (TTS via ``SayTool``) is enabled by default when
``pocket-tts`` is installed; pass ``--no_tts`` to disable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("lerobot.smolvla2.runtime")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="lerobot-smolvla2-runtime",
        description="Interactive REPL runtime for a trained SmolVLA2 checkpoint.",
    )
    p.add_argument(
        "--policy.path",
        dest="policy_path",
        type=Path,
        required=True,
        help="Path to a trained SmolVLA2 ``pretrained_model`` directory.",
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


def _load_policy(path: Path):  # noqa: ANN202
    """Load a SmolVLA2 checkpoint from ``path``."""
    from lerobot.policies.factory import make_policy_from_path  # noqa: PLC0415

    policy = make_policy_from_path(str(path))
    policy.eval()
    return policy


def _build_tools(policy_path: Path, no_tts: bool, tts_voice: str) -> dict[str, Any]:
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

    if not args.policy_path.exists():
        print(f"[smolvla2] policy path not found: {args.policy_path}", file=sys.stderr)
        return 1

    print(f"[smolvla2] loading policy from {args.policy_path}", flush=True)
    policy = _load_policy(args.policy_path)

    tools = _build_tools(args.policy_path, args.no_tts, args.tts_voice)
    if tools:
        print(f"[smolvla2] tools loaded: {list(tools)}", flush=True)

    # Robot wiring is left as a follow-up — for v1 we run language-only
    # / dry-run so REPL development doesn't require a connected robot.
    observation_provider = None
    robot_executor = None
    if not args.no_robot:
        print(
            "[smolvla2] WARNING: real-robot integration is a follow-up. "
            "Running in dry-run mode for now (no actions executed).",
            flush=True,
        )

    from lerobot.policies.smolvla2.inference import (  # noqa: PLC0415
        SmolVLA2Runtime,
        StdinReader,
    )

    runtime = SmolVLA2Runtime(
        policy=policy,
        tools=tools,
        observation_provider=observation_provider,
        robot_executor=robot_executor,
        event_collector=StdinReader().poll,
        chunk_hz=args.chunk_hz,
        ctrl_hz=args.ctrl_hz,
        high_level_hz=args.high_level_hz,
    )
    if args.task:
        runtime.set_task(args.task)
    print(
        "[smolvla2] runtime ready. Type a task to begin, then any line for "
        "interjections, questions ending in '?' for VQA, or 'stop' to exit.",
        flush=True,
    )
    try:
        runtime.run(max_ticks=args.max_ticks)
    except KeyboardInterrupt:
        runtime.stop()
        print("\n[smolvla2] interrupted by user", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
