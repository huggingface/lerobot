#!/usr/bin/env python

"""Create an MP4 from PushT search trace JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from plot_search_trace import (
    TraceChoice,
    load_rgb_image,
    render_trace_preview,
    resolve_base_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render RUN_DIR/search_images traces with the same styling controls as "
            "plot_search_trace.py, then write them as an MP4."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing search_images/")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output MP4 path. Defaults to RUN_DIR/plot_videos/search_trace.mp4.",
    )
    parser.add_argument("--episode", type=int, default=None, help="Render only one episode index.")
    parser.add_argument("--step", type=int, default=None, help="Render only one environment step.")
    parser.add_argument("--start-step", type=int, default=None, help="First environment step to include.")
    parser.add_argument("--end-step", type=int, default=None, help="Last environment step to include.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum video frames to render. Use 0 for all.")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video frames per second.")
    parser.add_argument(
        "--base-source",
        choices=("auto", "frames", "search_image"),
        default="auto",
        help=(
            "Background image source. auto prefers frames/episode_XXX/frame_YYYYY.png "
            "and falls back to the paired search image."
        ),
    )
    parser.add_argument("--dot-radius", type=int, default=6, help="Dot radius in pixels.")
    parser.add_argument(
        "--render-mode",
        choices=("dots", "line", "both"),
        default="both",
        help="Draw traces as dots only, lines only, or both.",
    )
    parser.add_argument(
        "--max-dots",
        type=int,
        default=10,
        help="Maximum number of leading dots to draw per candidate trace. Use 0 to draw all points.",
    )
    parser.add_argument("--line-thickness", type=int, default=2, help="Connecting line thickness.")
    parser.add_argument("--line-alpha", type=float, default=0.35, help="Alpha for connecting lines.")
    parser.add_argument(
        "--fade-dots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fade later dots and line segments along each trace. Enabled by default.",
    )
    parser.add_argument("--dot-start-alpha", type=float, default=1.0, help="Opacity of the first dot.")
    parser.add_argument("--dot-end-alpha", type=float, default=0.0, help="Opacity of the last dot.")
    parser.add_argument(
        "--fade-last-n",
        type=int,
        default=0,
        help=(
            "Only fade the last N rendered points. Use 0 to fade across the full trace, "
            "which preserves the plot_search_trace.py default."
        ),
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if "episode_index" not in payload or "env_step" not in payload:
        raise ValueError(f"Malformed search trace JSON: {path}")
    return payload


def should_include(
    payload: dict[str, Any],
    *,
    episode: int | None,
    step: int | None,
    start_step: int | None,
    end_step: int | None,
) -> bool:
    episode_index = int(payload["episode_index"])
    env_step = int(payload["env_step"])
    if episode is not None and episode_index != episode:
        return False
    if step is not None and env_step != step:
        return False
    if start_step is not None and env_step < start_step:
        return False
    if end_step is not None and env_step > end_step:
        return False
    return True


def collect_trace_choices(
    run_dir: Path,
    *,
    episode: int | None,
    step: int | None,
    start_step: int | None,
    end_step: int | None,
    max_frames: int,
) -> list[TraceChoice]:
    search_dir = run_dir / "search_images"
    choices: list[TraceChoice] = []
    for json_path in sorted(search_dir.glob("episode_*/step_*.json")):
        payload = load_payload(json_path)
        if should_include(
            payload,
            episode=episode,
            step=step,
            start_step=start_step,
            end_step=end_step,
        ):
            choices.append(TraceChoice(json_path=json_path, image_path=json_path.with_suffix(".png"), payload=payload))
            if max_frames > 0 and len(choices) >= max_frames:
                break

    choices.sort(key=lambda choice: (int(choice.payload["episode_index"]), int(choice.payload["env_step"])))
    if not choices:
        raise FileNotFoundError(
            f"No matching search trace JSON files found under {search_dir}. "
            "Run with --dump_search_images=true first."
        )
    return choices


def render_choice(run_dir: Path, choice: TraceChoice, args: argparse.Namespace) -> np.ndarray:
    base_path = resolve_base_image(run_dir, choice, args.base_source)
    base_image = load_rgb_image(base_path)
    return render_trace_preview(
        image=base_image,
        payload=choice.payload,
        dot_radius=args.dot_radius,
        max_dots=args.max_dots,
        line_thickness=args.line_thickness,
        line_alpha=args.line_alpha,
        fade_dots=args.fade_dots,
        dot_start_alpha=args.dot_start_alpha,
        dot_end_alpha=args.dot_end_alpha,
        fade_last_n=args.fade_last_n,
        render_mode=args.render_mode,
    )


def open_video_writer(path: Path, *, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def write_video(run_dir: Path, choices: list[TraceChoice], args: argparse.Namespace, output_path: Path) -> None:
    first_frame = render_choice(run_dir, choices[0], args)
    height, width = first_frame.shape[:2]
    writer = open_video_writer(output_path, width=width, height=height, fps=args.fps)
    try:
        for frame_ix, choice in enumerate(choices):
            frame = first_frame if frame_ix == 0 else render_choice(run_dir, choice, args)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    run_dir = args.run_dir.expanduser().resolve()
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path
        else run_dir / "plot_videos" / "search_trace.mp4"
    )
    choices = collect_trace_choices(
        run_dir,
        episode=args.episode,
        step=args.step,
        start_step=args.start_step,
        end_step=args.end_step,
        max_frames=args.max_frames,
    )
    write_video(run_dir, choices, args, output_path)
    first = choices[0].payload
    last = choices[-1].payload
    print(f"output_video={output_path}")
    print(f"frames={len(choices)} fps={args.fps:g}")
    print(
        "range="
        f"episode_{int(first['episode_index']):03d}/step_{int(first['env_step']):05d}"
        "..."
        f"episode_{int(last['episode_index']):03d}/step_{int(last['env_step']):05d}"
    )


if __name__ == "__main__":
    main()
