#!/usr/bin/env python

"""Create an animated MP4 from PushT search trace JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from plot_search_trace import (
    CHOSEN_GRADIENT,
    ORIGINAL_GRADIENT,
    OTHER_GRADIENT,
    TraceChoice,
    draw_trace,
    limit_points,
    load_rgb_image,
    resolve_base_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate RUN_DIR/search_images traces by progressively drawing the nominal "
            "ACT chunk, candidate chunks, and selected chunk."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing search_images/")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output MP4 path. Defaults to RUN_DIR/plot_videos/eps_N_animated.mp4 "
            "when --episode=N is set, otherwise RUN_DIR/plot_videos/search_trace_animated.mp4."
        ),
    )
    parser.add_argument("--episode", type=int, default=None, help="Render only one episode index.")
    parser.add_argument("--step", type=int, default=None, help="Render only one environment step.")
    parser.add_argument("--start-step", type=int, default=None, help="First environment step to include.")
    parser.add_argument("--end-step", type=int, default=None, help="Last environment step to include.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum decision points to render. Use 0 for all.")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video frames per second.")
    parser.add_argument(
        "--base-source",
        choices=("auto", "frames", "search_image"),
        default="auto",
        help=(
            "Background image source. auto prefers frames/episode_XXX/frame_YYYYY.png "
            "and falls back to the paired search image."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Maximum non-policy candidates to animate per decision point. Use 0 for all.",
    )
    parser.add_argument(
        "--points-per-frame",
        type=int,
        default=1,
        help="Number of new trace points revealed per animation frame.",
    )
    parser.add_argument("--intro-frames", type=int, default=4, help="Base-image frames before drawing starts.")
    parser.add_argument("--candidate-hold-frames", type=int, default=2, help="Hold frames after each candidate.")
    parser.add_argument("--selected-hold-frames", type=int, default=8, help="Hold frames after selected highlight.")
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
        help="Only fade the last N rendered points. Use 0 to fade across the full trace.",
    )
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw compact episode, step, phase, and selected-candidate text.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if "episode_index" not in payload or "env_step" not in payload or "traces" not in payload:
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
    return not (end_step is not None and env_step > end_step)


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


def default_output_path(run_dir: Path, *, episode: int | None) -> Path:
    filename = f"eps_{episode}_animated.mp4" if episode is not None else "search_trace_animated.mp4"
    return run_dir / "plot_videos" / filename


def open_video_writer(path: Path, *, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def trace_points(trace: dict[str, Any], max_dots: int) -> list[tuple[int, int]]:
    return limit_points([tuple(map(int, point)) for point in trace["pixel_points"]], max_dots)


def draw_one_trace(
    image: np.ndarray,
    trace: dict[str, Any],
    colors: tuple[tuple[int, int, int], ...],
    args: argparse.Namespace,
    *,
    points: list[tuple[int, int]] | None = None,
    selected_style: bool = False,
) -> None:
    draw_trace(
        image,
        trace_points(trace, args.max_dots) if points is None else points,
        colors,
        dot_radius=args.dot_radius + (1 if selected_style else 0),
        line_thickness=args.line_thickness + (1 if selected_style else 0),
        line_alpha=min(1.0, args.line_alpha + (0.15 if selected_style else 0.0)),
        fade_dots=args.fade_dots,
        dot_start_alpha=args.dot_start_alpha,
        dot_end_alpha=args.dot_end_alpha,
        fade_last_n=args.fade_last_n,
        render_mode=args.render_mode,
    )


def selected_index(payload: dict[str, Any]) -> int | None:
    selected = next((trace for trace in payload["traces"] if trace.get("is_selected")), None)
    return None if selected is None else int(selected["candidate_index"])


def trace_label(trace: dict[str, Any]) -> str:
    index = int(trace["candidate_index"])
    if trace.get("is_original"):
        return "ACT policy"
    if trace.get("is_selected"):
        return f"selected candidate {index}"
    return f"candidate {index}"


def annotate(image: np.ndarray, payload: dict[str, Any], phase: str, *, enabled: bool) -> np.ndarray:
    if not enabled:
        return image
    annotated = np.ascontiguousarray(image.copy())
    episode_index = int(payload["episode_index"])
    env_step = int(payload["env_step"])
    selected = selected_index(payload)
    lines = [
        f"episode={episode_index} step={env_step}",
        f"phase={phase}",
        f"selected={selected if selected is not None else 'none'} original=0",
    ]
    pad = 8
    line_height = 22
    box_width = min(annotated.shape[1], 430)
    box_height = pad * 2 + line_height * len(lines)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
    for line_ix, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (pad, pad + 16 + line_ix * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def write_frame(writer: cv2.VideoWriter, frame: np.ndarray, *, size: tuple[int, int]) -> None:
    width, height = size
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def render_completed(
    base_image: np.ndarray,
    completed: list[tuple[dict[str, Any], tuple[tuple[int, int, int], ...], bool]],
    args: argparse.Namespace,
) -> np.ndarray:
    frame = np.ascontiguousarray(base_image.copy())
    for trace, colors, selected_style in completed:
        draw_one_trace(frame, trace, colors, args, selected_style=selected_style)
    return frame


def trace_colors(trace: dict[str, Any], *, selected: bool = False) -> tuple[tuple[int, int, int], ...]:
    if selected:
        return CHOSEN_GRADIENT
    if trace.get("is_original"):
        return ORIGINAL_GRADIENT
    return OTHER_GRADIENT


def candidate_order(payload: dict[str, Any], *, max_candidates: int) -> list[dict[str, Any]]:
    traces = sorted(payload["traces"], key=lambda item: int(item["candidate_index"]))
    original = [trace for trace in traces if trace.get("is_original")]
    others = [trace for trace in traces if not trace.get("is_original")]
    if max_candidates > 0:
        selected = [trace for trace in others if trace.get("is_selected")]
        non_selected = [trace for trace in others if not trace.get("is_selected")]
        others = non_selected[:max_candidates]
        for trace in selected:
            if trace not in others:
                others.append(trace)
    return original + others


def animate_trace(
    *,
    writer: cv2.VideoWriter,
    base_image: np.ndarray,
    completed: list[tuple[dict[str, Any], tuple[tuple[int, int, int], ...], bool]],
    trace: dict[str, Any],
    payload: dict[str, Any],
    args: argparse.Namespace,
    size: tuple[int, int],
    selected_style: bool = False,
) -> None:
    points = trace_points(trace, args.max_dots)
    if not points:
        return
    stride = max(1, int(args.points_per_frame))
    colors = trace_colors(trace, selected=selected_style)
    phase = trace_label(trace) if not selected_style else f"highlight {trace_label(trace)}"

    for end in range(stride, len(points) + stride, stride):
        frame = render_completed(base_image, completed, args)
        draw_one_trace(
            frame,
            trace,
            colors,
            args,
            points=points[: min(end, len(points))],
            selected_style=selected_style,
        )
        write_frame(writer, annotate(frame, payload, phase, enabled=args.overlay), size=size)


def hold_frame(
    *,
    writer: cv2.VideoWriter,
    frame: np.ndarray,
    payload: dict[str, Any],
    phase: str,
    count: int,
    args: argparse.Namespace,
    size: tuple[int, int],
) -> None:
    for _ in range(max(0, int(count))):
        write_frame(writer, annotate(frame, payload, phase, enabled=args.overlay), size=size)


def animate_choice(
    *,
    writer: cv2.VideoWriter,
    run_dir: Path,
    choice: TraceChoice,
    args: argparse.Namespace,
    size: tuple[int, int],
) -> None:
    base_path = resolve_base_image(run_dir, choice, args.base_source)
    base_image = load_rgb_image(base_path)
    payload = choice.payload
    completed: list[tuple[dict[str, Any], tuple[tuple[int, int, int], ...], bool]] = []

    hold_frame(
        writer=writer,
        frame=base_image,
        payload=payload,
        phase="current frame",
        count=args.intro_frames,
        args=args,
        size=size,
    )

    for trace in candidate_order(payload, max_candidates=args.max_candidates):
        animate_trace(
            writer=writer,
            base_image=base_image,
            completed=completed,
            trace=trace,
            payload=payload,
            args=args,
            size=size,
        )
        completed.append((trace, trace_colors(trace), False))
        frame = render_completed(base_image, completed, args)
        hold_frame(
            writer=writer,
            frame=frame,
            payload=payload,
            phase=f"hold {trace_label(trace)}",
            count=args.candidate_hold_frames,
            args=args,
            size=size,
        )

    selected = next((trace for trace in payload["traces"] if trace.get("is_selected")), None)
    if selected is not None:
        animate_trace(
            writer=writer,
            base_image=base_image,
            completed=completed,
            trace=selected,
            payload=payload,
            args=args,
            size=size,
            selected_style=True,
        )
        completed.append((selected, CHOSEN_GRADIENT, True))

    final_frame = render_completed(base_image, completed, args)
    hold_frame(
        writer=writer,
        frame=final_frame,
        payload=payload,
        phase="execute selected chunk",
        count=args.selected_hold_frames,
        args=args,
        size=size,
    )


def write_video(run_dir: Path, choices: list[TraceChoice], args: argparse.Namespace, output_path: Path) -> None:
    first_base_path = resolve_base_image(run_dir, choices[0], args.base_source)
    first_frame = load_rgb_image(first_base_path)
    height, width = first_frame.shape[:2]
    writer = open_video_writer(output_path, width=width, height=height, fps=args.fps)
    try:
        for choice in choices:
            animate_choice(writer=writer, run_dir=run_dir, choice=choice, args=args, size=(width, height))
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.points_per_frame <= 0:
        raise ValueError("--points-per-frame must be positive.")
    if args.max_candidates < 0:
        raise ValueError("--max-candidates must be non-negative.")

    run_dir = args.run_dir.expanduser().resolve()
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path
        else default_output_path(run_dir, episode=args.episode)
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
    print(f"decision_points={len(choices)} fps={args.fps:g}")
    print(
        "range="
        f"episode_{int(first['episode_index']):03d}/step_{int(first['env_step']):05d}"
        "..."
        f"episode_{int(last['episode_index']):03d}/step_{int(last['env_step']):05d}"
    )


if __name__ == "__main__":
    main()
