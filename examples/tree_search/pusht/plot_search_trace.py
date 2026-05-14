#!/usr/bin/env python

"""Offline renderer for PushT search trace JSON files."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CHOSEN_GRADIENT = (
    (255, 245, 105),
    (255, 190, 64),
    (248, 91, 65),
    (228, 47, 155),
    (83, 34, 190),
)
ORIGINAL_GRADIENT = (
    (117, 223, 255),
    (63, 164, 246),
    (34, 86, 216),
)
OTHER_GRADIENT = (
    (214, 214, 214),
    (158, 158, 158),
    (88, 88, 88),
)


@dataclass(frozen=True)
class TraceChoice:
    json_path: Path
    image_path: Path
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick a PushT search trace from RUN_DIR/search_images and render an "
            "offline styling preview."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing search_images/")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for preview PNGs. Defaults to RUN_DIR/plot_previews.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for random trace selection.")
    parser.add_argument("--episode", type=int, default=None, help="Select a specific episode index.")
    parser.add_argument("--step", type=int, default=None, help="Select a specific environment step.")
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
        "--max-dots",
        type=int,
        default=10,
        help="Maximum number of dots to draw per candidate trace. Use 0 to draw all points.",
    )
    parser.add_argument("--line-thickness", type=int, default=2, help="Connecting line thickness.")
    parser.add_argument("--line-alpha", type=float, default=0.35, help="Alpha for connecting lines.")
    return parser.parse_args()


def load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write image: {path}")


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a_i + (b_i - a_i) * t)) for a_i, b_i in zip(a, b, strict=True))


def gradient_color(colors: tuple[tuple[int, int, int], ...], index: int, count: int) -> tuple[int, int, int]:
    if count <= 1:
        return colors[-1]
    position = index / (count - 1)
    scaled = position * (len(colors) - 1)
    left = int(np.floor(scaled))
    right = min(left + 1, len(colors) - 1)
    return lerp_color(colors[left], colors[right], scaled - left)


def draw_alpha_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int,
    alpha: float,
) -> None:
    if thickness <= 0 or alpha <= 0:
        return
    overlay = image.copy()
    cv2.line(overlay, start, end, color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)


def draw_trace(
    image: np.ndarray,
    points: list[tuple[int, int]],
    colors: tuple[tuple[int, int, int], ...],
    *,
    dot_radius: int,
    line_thickness: int,
    line_alpha: float,
) -> None:
    if not points:
        return

    for point_ix in range(1, len(points)):
        color = gradient_color(colors, point_ix, len(points))
        draw_alpha_line(
            image,
            points[point_ix - 1],
            points[point_ix],
            color,
            thickness=line_thickness,
            alpha=line_alpha,
        )

    for point_ix, point in enumerate(points):
        color = gradient_color(colors, point_ix, len(points))
        # cv2.circle(image, point, dot_radius + 1, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(image, point, dot_radius, color, thickness=-1, lineType=cv2.LINE_AA)


def limit_points(points: list[tuple[int, int]], max_dots: int) -> list[tuple[int, int]]:
    if max_dots <= 0 or len(points) <= max_dots:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_dots, dtype=np.int64)
    return [points[int(index)] for index in indices]


def json_matches(path: Path, *, episode: int | None, step: int | None) -> bool:
    if episode is not None and path.parent.name != f"episode_{episode:03d}":
        return False
    if step is not None and path.stem != f"step_{step:05d}":
        return False
    return True


def choose_trace(run_dir: Path, *, episode: int | None, step: int | None, seed: int | None) -> TraceChoice:
    search_dir = run_dir / "search_images"
    candidates = sorted(
        path for path in search_dir.glob("episode_*/step_*.json") if json_matches(path, episode=episode, step=step)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No matching search trace JSON files found under {search_dir}. "
            "Run with --dump_search_images=true first."
        )

    rng = random.Random(seed)
    json_path = rng.choice(candidates)
    with json_path.open() as f:
        payload = json.load(f)
    image_path = json_path.with_suffix(".png")
    return TraceChoice(json_path=json_path, image_path=image_path, payload=payload)


def resolve_base_image(run_dir: Path, choice: TraceChoice, base_source: str) -> Path:
    episode_index = int(choice.payload["episode_index"])
    env_step = int(choice.payload["env_step"])
    frame_path = run_dir / "frames" / f"episode_{episode_index:03d}" / f"frame_{env_step:05d}.png"

    if base_source == "frames":
        return frame_path
    if base_source == "search_image":
        return choice.image_path
    return frame_path if frame_path.exists() else choice.image_path


def render_trace_preview(
    *,
    image: np.ndarray,
    payload: dict[str, Any],
    dot_radius: int,
    max_dots: int,
    line_thickness: int,
    line_alpha: float,
) -> np.ndarray:
    rendered = np.ascontiguousarray(image.copy())
    traces = sorted(payload["traces"], key=lambda item: int(item["candidate_index"]))

    for trace in traces:
        if trace["is_selected"] or trace["is_original"]:
            continue
        points = limit_points([tuple(map(int, point)) for point in trace["pixel_points"]], max_dots)
        draw_trace(
            rendered,
            points,
            OTHER_GRADIENT,
            dot_radius=dot_radius,
            line_thickness=line_thickness,
            line_alpha=line_alpha,
        )

    original = next((trace for trace in traces if trace["is_original"]), None)
    if original is not None:
        draw_trace(
            rendered,
            limit_points([tuple(map(int, point)) for point in original["pixel_points"]], max_dots),
            ORIGINAL_GRADIENT,
            dot_radius=dot_radius,
            line_thickness=line_thickness,
            line_alpha=line_alpha,
        )

    selected = next((trace for trace in traces if trace["is_selected"]), None)
    if selected is not None:
        draw_trace(
            rendered,
            limit_points([tuple(map(int, point)) for point in selected["pixel_points"]], max_dots),
            CHOSEN_GRADIENT,
            dot_radius=dot_radius + 1,
            line_thickness=line_thickness + 1,
            line_alpha=min(1.0, line_alpha + 0.15),
        )

    return rendered


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "plot_previews"

    choice = choose_trace(run_dir, episode=args.episode, step=args.step, seed=args.seed)
    base_path = resolve_base_image(run_dir, choice, args.base_source)
    base_image = load_rgb_image(base_path)
    preview = render_trace_preview(
        image=base_image,
        payload=choice.payload,
        dot_radius=args.dot_radius,
        max_dots=args.max_dots,
        line_thickness=args.line_thickness,
        line_alpha=args.line_alpha,
    )

    episode_index = int(choice.payload["episode_index"])
    env_step = int(choice.payload["env_step"])
    output_path = output_dir / f"episode_{episode_index:03d}_step_{env_step:05d}_trace_preview.png"
    write_rgb_image(output_path, preview)

    selected = next((trace for trace in choice.payload["traces"] if trace["is_selected"]), None)
    selected_index = None if selected is None else selected["candidate_index"]
    print(f"trace_json={choice.json_path}")
    print(f"base_image={base_path}")
    print(f"output_image={output_path}")
    print(f"selected_candidate={selected_index}")


if __name__ == "__main__":
    main()
