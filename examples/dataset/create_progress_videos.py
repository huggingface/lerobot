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

"""Draw the scores written by ``lerobot-score`` over episode videos.

For each episode, the script reads the named scores from ``<dataset root>/scores/`` (or from the
Hub, if they were pushed), draws them over one camera's video and writes
``<output-dir>/<repo>_ep<episode>_<scores>.mp4``, plus an animated ``.webp`` preview with
``--webp``. The y-axis follows each signal's descriptor: its bounds when it has them, such as
``[0, 1]`` for progress, otherwise the episode's values in the signal's unit, such as seconds.

Every score is checked for every episode before any video is downloaded or rendered.

Usage:
    # Every signal of one score.
    python examples/dataset/create_progress_videos.py \
        --repo-id lerobot/libero_10 \
        --episodes 0 1 \
        --scores robometer

    # One signal of each score, plus a WebP preview.
    python examples/dataset/create_progress_videos.py \
        --repo-id lerobot/libero_10 \
        --dataset-root /path/to/local/dataset \
        --episodes 0 \
        --scores topreward robometer:reward.robometer.progress \
        --webp
"""

from __future__ import annotations

import argparse
import logging
import math
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageDraw, ImageFont

from lerobot.rewards.sarm.sarm_utils import find_stage_and_tau, normalize_stage_tau
from lerobot.rewards.scoring import SignalDescriptor, load_signal_descriptors
from lerobot.utils.constants import HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE
from lerobot.utils.import_utils import (
    _av_available,
    _datasets_available,
    get_safe_default_video_backend,
    require_package,
)

if TYPE_CHECKING or (_datasets_available and _av_available):
    import av
    from av.video.reformatter import ColorRange, Colorspace

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.scores import load_scores, load_scores_info
    from lerobot.datasets.utils import task_name
    from lerobot.datasets.video_utils import decode_video_frames

Color = tuple[int, int, int]
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

# LeRobot brand colors.
_BACKGROUND: Color = (18, 17, 13)
_CREAM: Color = (243, 240, 228)
_MUTED: Color = (168, 163, 142)
_INK: Color = (21, 20, 15)
_SIGNAL_COLORS: list[Color] = [(255, 210, 30), (143, 168, 255), (255, 157, 0), _CREAM]
_GROUND_TRUTH: Color = (34, 197, 94)
# Subtask colors (Tableau 10).
_SUBTASK_COLORS: list[Color] = [
    (78, 121, 167),
    (242, 142, 43),
    (225, 87, 89),
    (118, 183, 178),
    (89, 161, 79),
    (237, 201, 72),
    (176, 122, 161),
    (255, 157, 167),
    (156, 117, 95),
    (186, 176, 172),
]

# Layout in pixels of a 768-pixel frame, scaled to the output size.
_REFERENCE_SIZE = 768
_MARGIN = 24
_PADDING = 12
_RADIUS = 8
_PLOT_SIDE = 48
_PLOT_TOP = 120
_PLOT_BOTTOM = 104
_TASK_WIDTH = 460
_LEGEND_ROW = 22
_FOOTER = 29
_LOGO_HEIGHT = 28
_CURVE_WIDTH = 3
_STRIP = 6  # height of a subtask timeline strip
_TRACK_ROW = 23  # a strip with its label above it, under the plot

# Brand fonts (Open Font License), downloaded once, and their sizes in layout pixels.
_FONT_URLS = {
    "regular": "https://raw.githubusercontent.com/adobe-fonts/source-sans/release/TTF/SourceSans3-Regular.ttf",
    "bold": "https://raw.githubusercontent.com/adobe-fonts/source-sans/release/TTF/SourceSans3-Bold.ttf",
    "mono": "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexmono/IBMPlexMono-SemiBold.ttf",
}
_FONT_STYLES = {
    "label": ("mono", 11),
    "task": ("bold", 17),
    "model": ("bold", 14),
    "signal": ("regular", 14),
    "value": ("mono", 14),
    "footer": ("mono", 13),
    "logo": ("bold", 20),
}
_FONT_DIR = HF_LEROBOT_HOME / "fonts"
# The LeRobot mark, from the official brand assets.
_LOGO_REPO, _LOGO_FILE = "lerobot/LeRobot-branding", "images/LeRobot Logos/logo.png"

_SUPERSAMPLING = 4  # shapes are drawn this many times larger, then shrunk, for smooth edges
_DECODE_CHUNK = 64
_MP4_CRF = 17  # H.264 quality, lower is better
_WEBP_WIDTH = 720
_WEBP_MAX_FPS = 15


@dataclass
class Signal:
    """One signal of an episode, with its values per frame."""

    model: str
    name: str
    values: np.ndarray
    descriptor: SignalDescriptor
    color: Color


@dataclass
class SubtaskTrack:
    """An episode's subtasks over time: each frame's index into ``names``, or -1 for none."""

    label: str
    names: list[str]
    index: np.ndarray


def _load_signals(dataset: LeRobotDataset, episode: int, specs: list[str]) -> list[Signal]:
    """Load an episode's signals from ``specs`` such as ``robometer`` or ``robometer:<signal>``."""
    num_frames = int(dataset.meta.episodes[episode]["length"])
    signals: list[Signal] = []
    for spec in specs:
        score, _, only = spec.partition(":")
        try:
            descriptors = load_signal_descriptors(dataset, score)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"No score {score!r} in {dataset.root / 'scores'} or on the Hub: run lerobot-score first, "
                "or pass the --dataset-root it wrote to"
            ) from error
        names = [only] if only else list(descriptors)
        unknown = [name for name in names if name not in descriptors]
        if unknown:
            raise ValueError(f"Score {score!r} has no signal {unknown}; it has {list(descriptors)}")

        provenance = load_scores_info(dataset, score).get("provenance", {})
        model = provenance.get("model", {})
        model_name = Path(model["id"]).name if model.get("id") else model.get("type") or score

        rows = load_scores(dataset, score, columns=["frame_index", *names], episodes=[episode])
        rows = rows.with_format("numpy")[:]
        if len(rows["frame_index"]) == 0:
            raise ValueError(f"Score {score!r} has no rows for episode {episode}")
        for name in names:
            values = np.full(num_frames, np.nan)
            values[rows["frame_index"]] = rows[name]
            descriptor = descriptors[name]
            label = name.rsplit(".", 1)[-1]
            if descriptor.unit:
                label = label.removesuffix(f"_{descriptor.unit}")
            label = label.replace("_", " ")
            curves = sum(signal.descriptor.labels is None for signal in signals)
            color = _SIGNAL_COLORS[curves % len(_SIGNAL_COLORS)]
            signals.append(Signal(model_name, label, values, descriptor, color))
            if descriptor.labels is not None and model.get("type") == "sarm":
                head = name.rsplit(".", 1)[-1].removesuffix("_stage")
                options = provenance.get("scorer", {}).get("options", {})
                ground_truth = _sarm_ground_truth(dataset, episode, head, list(descriptor.labels), options)
                if ground_truth is not None:
                    signals.append(Signal("annotations", f"{head} progress", *ground_truth, _GROUND_TRUTH))
    return signals


def _sarm_ground_truth(
    dataset: LeRobotDataset, episode: int, head: str, names: list[str], options: dict
) -> tuple[np.ndarray, SignalDescriptor] | None:
    """Return SARM's target progress for each frame, from the episode's subtask annotations.

    As SARM is trained: the annotated subtask and how far through it each frame is, mapped onto
    the subtask time shares recorded in the score's config.
    """
    annotations = _annotations(dataset, episode, [head])
    proportions = options.get(f"{head}_temporal_proportions")
    if annotations is None or not proportions:
        return None
    num_frames = int(dataset.meta.episodes[episode]["length"])
    values = np.array(
        [
            normalize_stage_tau(
                find_stage_and_tau(frame, num_frames, *annotations, names, None, return_combined=True),
                temporal_proportions=proportions,
                subtask_names=names,
            )
            for frame in range(num_frames)
        ]
    )
    descriptor = SignalDescriptor(
        "Progress from the subtask annotations", greater_is_better=True, bounds=(0.0, 1.0)
    )
    return values, descriptor


def _annotations(
    dataset: LeRobotDataset, episode: int, kinds: Sequence[str] = ("dense", "sparse")
) -> tuple[list[str], list[int], list[int]] | None:
    """Return the episode's annotated subtask names, start frames and end frames, if any.

    ``kinds`` lists the annotation types to look for, in order: ``dense`` or ``sparse``, as SARM's
    annotation tool writes them (sparse annotations may also have no prefix).
    """
    row = dataset.meta.episodes[episode]
    prefixes = [f"{kind}_" for kind in kinds] + ([""] if "sparse" in kinds else [])
    for prefix in prefixes:
        columns = [f"{prefix}subtask_{suffix}" for suffix in ("names", "start_frames", "end_frames")]
        names, starts, ends = (row.get(column) for column in columns)
        if names is not None and starts is not None and ends is not None and len(names) > 0:
            return [str(name) for name in names], [int(v) for v in starts], [int(v) for v in ends]
    return None


def _scored_camera(dataset: LeRobotDataset, specs: list[str]) -> str | None:
    """Return the camera the first score was computed on, if the dataset has its video."""
    score = specs[0].partition(":")[0]
    options = load_scores_info(dataset, score).get("provenance", {}).get("scorer", {}).get("options", {})
    image_key = options.get("image_key")
    return image_key if image_key in dataset.meta.video_keys else None


def _episode_task(dataset: LeRobotDataset, episode: int) -> str:
    """Return the episode's task."""
    rows = dataset.hf_dataset.with_format("numpy").select_columns(["episode_index", "task_index"])[:]
    return task_name(dataset.meta.tasks, rows["task_index"][rows["episode_index"] == episode][0])


def _axis_range(signals: list[Signal]) -> tuple[float, float]:
    """Return the y-axis range: the signals' bounds, else zero to a rounded maximum."""
    bounds = {signal.descriptor.bounds for signal in signals}
    if len(bounds) == 1 and None not in bounds:
        low, high = bounds.pop()
        return float(low), float(high)
    units = {signal.descriptor.unit for signal in signals}
    if len(units) > 1:
        raise ValueError(f"Signals in different units cannot share one plot: {sorted(map(str, units))}")
    values = np.concatenate([signal.values for signal in signals])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("The signals have no values for this episode")
    low, high = min(0.0, float(values.min())), float(values.max())
    if high <= low:
        return low, low + 1.0
    # Round the top up to a readable number.
    magnitude = 10 ** math.floor(math.log10(high - low))
    step = next(m for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if low + m * magnitude >= high)
    return low, low + step * magnitude


def _format_value(value: float, descriptor: SignalDescriptor) -> str:
    """Format a value with its unit, or a dash when there is no score."""
    if not np.isfinite(value):
        return "-"
    return f"{value:.1f} {descriptor.unit}" if descriptor.unit else f"{value:.2f}"


def _load_font(name: str, size: float) -> Font:
    """Load a brand font, downloading it on first use, or Pillow's default font when offline."""
    path = _FONT_DIR / Path(_FONT_URLS[name]).name
    if not path.exists():
        try:
            _FONT_DIR.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".part")
            urllib.request.urlretrieve(_FONT_URLS[name], partial)  # nosec B310: fixed https URL
            partial.replace(path)
        except OSError as error:
            logging.warning("Using Pillow's default font: cannot download %s (%s)", _FONT_URLS[name], error)
            return ImageFont.load_default(size)
    return ImageFont.truetype(str(path), round(size))


def _load_logo() -> Image.Image | None:
    """Download the LeRobot logo from the Hub, or return None when offline."""
    try:
        return Image.open(hf_hub_download(_LOGO_REPO, _LOGO_FILE, repo_type="space")).convert("RGBA")
    except OSError as error:
        logging.warning("Leaving out the logo: cannot download it (%s)", error)
        return None


def _draw_text(
    image: Image.Image, xy: tuple[float, float], text: str, font: Font, color: Color, anchor: str = "la"
) -> None:
    """Draw text on an RGBA image through a glyph mask, so its edges don't darken on transparent pixels."""
    left, top, right, bottom = font.getbbox(text, anchor=anchor)
    x, y = math.floor(xy[0] + left), math.floor(xy[1] + top)
    mask = Image.new("L", (math.ceil(right - left) + 2, math.ceil(bottom - top) + 2), 0)
    ImageDraw.Draw(mask).text((xy[0] - x, xy[1] - y), text, font=font, fill=255, anchor=anchor)
    glyphs = Image.new("RGBA", mask.size, color)
    glyphs.putalpha(mask)
    image.alpha_composite(glyphs, (max(x, 0), max(y, 0)))


def _wrap_text(text: str, font: Font, width: float, max_lines: int = 3) -> list[str]:
    """Wrap text to a width, ending with an ellipsis when it doesn't fit."""
    lines, words = [], text.split()
    while words and len(lines) < max_lines:
        line = words.pop(0)
        while words and font.getlength(f"{line} {words[0]}") <= width:
            line = f"{line} {words.pop(0)}"
        lines.append(line)
    if words:
        while lines[-1] and font.getlength(f"{lines[-1]}...") > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] = f"{lines[-1].rstrip()}..."
    return lines


def _supersampled(size: tuple[int, int], draw: Callable[[ImageDraw.ImageDraw, int], None]) -> Image.Image:
    """Call ``draw(draw, k)`` on a transparent canvas ``k`` times larger, then shrink it."""
    k = _SUPERSAMPLING
    canvas = Image.new("RGBA", (size[0] * k, size[1] * k), (0, 0, 0, 0))
    draw(ImageDraw.Draw(canvas), k)
    return canvas.resize(size, Image.Resampling.LANCZOS)


class ProgressOverlay:
    """Draw an episode's signals over its frames.

    Static parts are drawn once; ``render`` adds the marker, values and time to each frame.
    Each of ``tracks`` adds a row under the task with the current subtask, and a strip under the
    plot with its subtasks over time. ``boundaries`` marks the frames where annotated subtasks start.
    """

    def __init__(
        self,
        signals: list[Signal],
        size: tuple[int, int],
        fps: int,
        task: str,
        caption: str,
        logo: Image.Image | None,
        tracks: Sequence[SubtaskTrack] = (),
        boundaries: Sequence[int] = (),
    ):
        self.signals = signals
        self.boundaries = boundaries
        self.tracks = list(tracks)
        self.size = size
        self.fps = fps
        self.num_frames = len(signals[0].values)
        self.scale = min(size) / _REFERENCE_SIZE
        self.low, self.high = _axis_range(signals)
        width, height = size
        self.fonts = {
            role: _load_font(name, self._px(points)) for role, (name, points) in _FONT_STYLES.items()
        }
        self.logo = None
        if logo is not None:
            # The LeRobot mark followed by the name, as the logo for dark backgrounds.
            logo_height = round(self._px(_LOGO_HEIGHT))
            mark = logo.resize(
                (round(logo.width * logo_height / logo.height), logo_height), Image.Resampling.LANCZOS
            )
            name_x = mark.width + round(self._px(6))
            self.logo = Image.new(
                "RGBA", (name_x + math.ceil(self.fonts["logo"].getlength("LeRobot")) + 2, logo_height)
            )
            self.logo.alpha_composite(mark)
            _draw_text(self.logo, (name_x, logo_height / 2), "LeRobot", self.fonts["logo"], _CREAM, "lm")

        # Legend: model and signal-name columns, and a width that fits the longest of each.
        self.model_x = self._px(_PADDING + 18)
        self.name_x = (
            self.model_x + max(self.fonts["model"].getlength(s.model) for s in signals) + self._px(8)
        )
        legend_width = (
            self.name_x + max(self.fonts["signal"].getlength(s.name) for s in signals) + self._px(_PADDING)
        )
        if len(signals) > 1:
            values = [_format_value(v, s.descriptor) for s in signals for v in (self.low, self.high)]
            legend_width += self._px(16) + max(self.fonts["value"].getlength(text) for text in values)
        rows = len(signals) if len(signals) > 1 else 1 + (signals[0].descriptor.greater_is_better is False)
        top = self._px(_MARGIN)
        self.legend = (
            width - self._px(_MARGIN) - legend_width,
            top,
            width - self._px(_MARGIN),
            top + self._px(14 + _LEGEND_ROW * rows),
        )

        task_width = min(self._px(_TASK_WIDTH), self.legend[0] - self._px(2 * _MARGIN))
        self.task_lines = _wrap_text(task, self.fonts["task"], task_width - self._px(2 * _PADDING))
        self.task_box = (
            self._px(_MARGIN),
            top,
            self._px(_MARGIN) + task_width,
            top
            + self._px(40 + 21 * len(self.task_lines) + (6 + 21 * len(self.tracks) if self.tracks else 0)),
        )
        # The plot starts below the panels, which grow with the task and subtask rows.
        plot_top = max(self._px(_PLOT_TOP), self.task_box[3] + self._px(20), self.legend[3] + self._px(20))
        self.plot = (
            self._px(_PLOT_SIDE),
            plot_top,
            width - self._px(_PLOT_SIDE),
            height - self._px(_PLOT_BOTTOM + _TRACK_ROW * len(self.tracks)),
        )

        # Subtask rows: a label, a swatch of the subtask's color, then its text cut to the panel width.
        names = dict.fromkeys(name for track in self.tracks for name in track.names)
        self.subtask_colors = {
            name: _SUBTASK_COLORS[i % len(_SUBTASK_COLORS)] for i, name in enumerate(names)
        }
        label_width = max((self.fonts["label"].getlength(track.label) for track in self.tracks), default=0)
        self.subtask_x = self._px(_PADDING + 10) + label_width
        text_width = task_width - self.subtask_x - self._px(_PADDING + 16)
        self.subtask_texts = [
            {
                i: _wrap_text(f"{i + 1}/{len(track.names)}  {name}", self.fonts["signal"], text_width, 1)[0]
                for i, name in enumerate(track.names)
            }
            for track in self.tracks
        ]

        self.background = self._draw_background(caption)
        self.curves = _supersampled(size, self._draw_curves)
        self.curves.alpha_composite(self._draw_tracks(alpha=255))
        self.markers = {signal.color: self._draw_marker(signal.color) for signal in signals}

    def _px(self, length: float) -> float:
        """Scale a length from the 768-pixel layout to the output."""
        return length * self.scale

    def _x_at(self, frame: float) -> float:
        left, _, right, _ = self.plot
        return left + frame / max(self.num_frames - 1, 1) * (right - left)

    def _y_at(self, value: float) -> float:
        _, top, _, bottom = self.plot
        return bottom - (value - self.low) / (self.high - self.low) * (bottom - top)

    def _legend_row_y(self, row: int) -> float:
        return self.legend[1] + self._px(7 + _LEGEND_ROW * (row + 0.5))

    def _subtask_row_y(self, row: int) -> float:
        return self.task_box[1] + self._px(46 + 21 * (len(self.task_lines) + row))

    def _track_top(self, row: int) -> float:
        """Return the top of a subtask strip's row under the plot: its label, then the strip."""
        return self.plot[3] + self._px(6 + _TRACK_ROW * row)

    def _dashed_vline(self, draw: ImageDraw.ImageDraw, x: float, y0: float, y1: float, alpha: int) -> None:
        """Draw a dashed one-pixel vertical line from ``y0`` up to ``y1``."""
        dash = self._px(7)
        for y in np.arange(y0, y1, -dash):
            draw.line(
                [(round(x), round(y)), (round(x), round(max(y - dash * 3 / 7, y1)))], fill=(*_CREAM, alpha)
            )

    def _points(self, values: np.ndarray, k: int) -> list[list[tuple[float, float]]]:
        """Return a signal's polylines in ``k``-times-larger pixels, split where it has no score."""
        lines: list[list[tuple[float, float]]] = [[]]
        for frame, value in enumerate(values):
            if np.isfinite(value):
                lines[-1].append((self._x_at(frame) * k, self._y_at(value) * k))
            elif lines[-1]:
                lines.append([])
        return [line for line in lines if line]

    def _draw_background(self, caption: str) -> Image.Image:
        """Draw what every frame shares: backdrop, grid, faint full curves, panels, labels and logo."""
        width, height = self.size
        left, top, right, bottom = self.plot
        layer = Image.new("RGBA", self.size, (*_BACKGROUND, 102))

        # One-pixel grid lines on whole pixels stay sharp.
        draw = ImageDraw.Draw(layer)
        for y in np.linspace(top, bottom, 5).round():
            draw.line([(round(left), y), (round(right), y)], fill=(*_CREAM, 30))
        for x in np.linspace(left, right, 7).round():
            draw.line([(x, round(top)), (x, round(bottom))], fill=(*_CREAM, 30))
        for start in self.boundaries:
            if 0 < start < self.num_frames:
                self._dashed_vline(draw, self._x_at(start), bottom, top, alpha=110)

        def draw_shapes(draw: ImageDraw.ImageDraw, k: int) -> None:
            for signal in self.signals:
                for line in self._points(signal.values, k):
                    draw.line(line, fill=(*signal.color, 80), width=round(self._px(1.5) * k), joint="curve")
            for box in (self.task_box, self.legend):
                draw.rounded_rectangle(
                    [v * k for v in box], radius=self._px(_RADIUS) * k, fill=(*_BACKGROUND, 210)
                )
            for row, signal in enumerate(self.signals):
                x, y = self.legend[0] + self._px(_PADDING), self._legend_row_y(row)
                draw.ellipse(
                    [v * k for v in (x, y - self._px(5), x + self._px(10), y + self._px(5))],
                    fill=signal.color,
                )

        layer.alpha_composite(_supersampled(self.size, draw_shapes))

        label, unit = self.fonts["label"], self.signals[0].descriptor.unit
        for value, y in (
            (self.high, top),
            ((self.low + self.high) / 2, (top + bottom) / 2),
            (self.low, bottom),
        ):
            tick = f"{value:g} {unit}" if unit else ("0" if value == 0 else f"{value:.1f}")
            _draw_text(layer, (left + self._px(6), y - self._px(5)), tick, label, _CREAM, anchor="ls")
        time_y = bottom + self._px(20 + _TRACK_ROW * len(self.tracks))
        _draw_text(layer, (left, time_y), "0 s", label, _CREAM, anchor="ls")
        duration = f"{self.num_frames / self.fps:.1f} s"
        _draw_text(layer, (right, time_y), duration, label, _CREAM, anchor="rs")
        layer.alpha_composite(self._draw_tracks(alpha=70))
        for row, track in enumerate(self.tracks):
            _draw_text(layer, (left, self._track_top(row) + self._px(9)), track.label, label, _MUTED, "ls")

        x, y = self.task_box[0] + self._px(_PADDING), self.task_box[1]
        _draw_text(layer, (x, y + self._px(10)), "TASK", label, _MUTED)
        for i, line in enumerate(self.task_lines):
            _draw_text(layer, (x, y + self._px(28 + 21 * i)), line, self.fonts["task"], _CREAM)
        for row, track in enumerate(self.tracks):
            _draw_text(layer, (x, self._subtask_row_y(row)), track.label, label, _MUTED, "lm")

        for row, signal in enumerate(self.signals):
            y = self._legend_row_y(row)
            _draw_text(
                layer, (self.legend[0] + self.model_x, y), signal.model, self.fonts["model"], _CREAM, "lm"
            )
            _draw_text(
                layer, (self.legend[0] + self.name_x, y), signal.name, self.fonts["signal"], _MUTED, "lm"
            )
        if len(self.signals) == 1 and self.signals[0].descriptor.greater_is_better is False:
            position = (self.legend[0] + self._px(_PADDING), self._legend_row_y(1))
            _draw_text(layer, position, "lower is better", self.fonts["signal"], _MUTED, "lm")

        footer_y = height - self._px(_FOOTER)
        _draw_text(layer, (self._px(_MARGIN), footer_y), caption, self.fonts["footer"], _MUTED, "lm")
        if self.logo is not None:
            position = (
                round(width - self._px(_MARGIN) - self.logo.width),
                round(footer_y - self.logo.height / 2),
            )
            layer.alpha_composite(self.logo, position)
        return layer

    def _draw_tracks(self, alpha: int) -> Image.Image:
        """Draw a strip under the plot for each subtask track, colored by subtask over time."""
        layer = Image.new("RGBA", self.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        left, _, right, _ = self.plot
        for row, track in enumerate(self.tracks):
            y0 = self._track_top(row) + self._px(12)
            changes = np.flatnonzero(np.diff(track.index)) + 1
            for start, end in zip([0, *changes], [*changes, len(track.index)], strict=True):
                current = int(track.index[start])
                if current < 0:
                    continue
                x0, x1 = max(self._x_at(start - 0.5), left), min(self._x_at(end - 0.5), right)
                color = self.subtask_colors[track.names[current]]
                draw.rectangle(
                    [round(x0), round(y0), round(x1) - 1, round(y0 + self._px(_STRIP)) - 1],
                    fill=(*color, alpha),
                )
        return layer

    def _draw_curves(self, draw: ImageDraw.ImageDraw, k: int) -> None:
        """Draw the solid curves, which each frame reveals up to its position."""
        for signal in self.signals:
            for line in self._points(signal.values, k):
                draw.line(
                    line * (2 if len(line) == 1 else 1),
                    fill=signal.color,
                    width=round(self._px(_CURVE_WIDTH) * k),
                    joint="curve",
                )

    def _draw_marker(self, color: Color) -> Image.Image:
        """Draw the current-frame dot, with a cream ring."""
        size = math.ceil(self._px(14))
        return _supersampled(
            (size, size),
            lambda draw, k: draw.ellipse(
                [0, 0, size * k - 1, size * k - 1], fill=color, outline=_CREAM, width=round(self._px(2) * k)
            ),
        )

    def render(self, frame: np.ndarray, index: int) -> Image.Image:
        """Return the output frame for video ``frame`` (uint8 RGB, any size) at ``index``."""
        width, height = self.size
        left, top, right, bottom = self.plot
        image = Image.fromarray(frame).resize(self.size, Image.Resampling.LANCZOS).convert("RGBA")
        image.alpha_composite(self.background)
        x = self._x_at(index)
        image.alpha_composite(
            self.curves.crop((0, 0, min(math.ceil(x + self._px(_CURVE_WIDTH)), width), height))
        )

        values = [signal.values[index] for signal in self.signals]
        scored = [(s, v) for s, v in zip(self.signals, values, strict=True) if np.isfinite(v)]
        if scored:
            # Playhead from the plot bottom up to the highest marker.
            highest = min(self._y_at(v) for _, v in scored)
            self._dashed_vline(ImageDraw.Draw(image), x, bottom, highest, alpha=190)
        for signal, value in scored:
            marker = self.markers[signal.color]
            image.alpha_composite(
                marker, (round(x - marker.width / 2), round(self._y_at(value) - marker.height / 2))
            )

        if len(self.signals) > 1:
            for row, (signal, value) in enumerate(zip(self.signals, values, strict=True)):
                position = (self.legend[2] - self._px(_PADDING), self._legend_row_y(row))
                _draw_text(
                    image,
                    position,
                    _format_value(value, signal.descriptor),
                    self.fonts["value"],
                    _CREAM,
                    "rm",
                )
        elif scored:
            # Value label above the marker, or below it near the top of the plot.
            signal, value = scored[0]
            text, font = _format_value(value, signal.descriptor), self.fonts["value"]
            box_width, box_height = math.ceil(font.getlength(text) + self._px(16)), math.ceil(self._px(26))
            badge = _supersampled(
                (box_width, box_height),
                lambda draw, k: draw.rounded_rectangle(
                    [0, 0, box_width * k - 1, box_height * k - 1],
                    radius=self._px(_RADIUS) * k,
                    fill=signal.color,
                ),
            )
            _draw_text(badge, (box_width / 2, box_height / 2), text, font, _INK, "mm")
            y, gap = self._y_at(value), self._px(14)
            badge_y = y - gap - box_height if y - gap - box_height >= top else y + gap
            badge_x = min(max(x - box_width / 2, left), right - box_width)
            image.alpha_composite(badge, (round(badge_x), round(badge_y)))

        draw = ImageDraw.Draw(image)
        for row, track in enumerate(self.tracks):
            x0, y = self.task_box[0] + self.subtask_x, self._subtask_row_y(row)
            current = int(track.index[index])
            if current < 0:
                _draw_text(image, (x0, y), "-", self.fonts["signal"], _CREAM, "lm")
                continue
            swatch = self._px(10)
            box = [round(x0), round(y - swatch / 2), round(x0 + swatch), round(y + swatch / 2)]
            draw.rectangle(box, fill=self.subtask_colors[track.names[current]])
            text = self.subtask_texts[row][current]
            _draw_text(image, (x0 + self._px(16), y), text, self.fonts["signal"], _CREAM, "lm")

        time = f"{index / self.fps:.1f} s / {self.num_frames / self.fps:.1f} s"
        time_right = width - self._px(_MARGIN) - (self.logo.width + self._px(16) if self.logo else 0)
        _draw_text(image, (time_right, height - self._px(_FOOTER)), time, self.fonts["footer"], _CREAM, "rm")
        return image.convert("RGB")


def _download_videos(dataset: LeRobotDataset, episodes: list[int], camera_key: str, local_root: bool) -> None:
    """Download the episodes' video files of one camera that are not on disk yet.

    ``local_root`` tells whether the dataset lives in a folder passed with ``--dataset-root``
    rather than in the Hub cache.
    """
    paths = {str(dataset.meta.get_video_file_path(episode, camera_key)) for episode in episodes}
    missing = sorted(path for path in paths if not (dataset.root / path).exists())
    if missing:
        location = {"local_dir": dataset.root} if local_root else {"cache_dir": HF_LEROBOT_HUB_CACHE}
        snapshot_download(
            dataset.repo_id,
            repo_type="dataset",
            revision=dataset.revision,
            allow_patterns=missing,
            **location,
        )


def _iter_episode_frames(dataset: LeRobotDataset, episode: int, camera_key: str) -> Iterator[np.ndarray]:
    """Yield an episode's frames as ``(height, width, 3)`` uint8 arrays."""
    episode_meta = dataset.meta.episodes[episode]
    video_path = dataset.root / dataset.meta.get_video_file_path(episode, camera_key)
    start = float(episode_meta[f"videos/{camera_key}/from_timestamp"])
    num_frames = int(episode_meta["length"])
    backend = get_safe_default_video_backend()
    for chunk_start in range(0, num_frames, _DECODE_CHUNK):
        timestamps = [
            start + i / dataset.fps for i in range(chunk_start, min(chunk_start + _DECODE_CHUNK, num_frames))
        ]
        frames = decode_video_frames(
            video_path, timestamps, dataset.tolerance_s, backend=backend, return_uint8=True
        )
        yield from frames.permute(0, 2, 3, 1).numpy()


def render_episode(
    dataset: LeRobotDataset,
    episode: int,
    camera_key: str,
    signals: list[Signal],
    task: str,
    path: Path,
    height: int,
    webp: bool,
    logo: Image.Image | None,
) -> None:
    """Render one episode to an MP4 at ``path``, and to a WebP next to it when ``webp`` is set.

    Signals with labels, such as SARM's predicted subtask, are shown as text under the task, next
    to the episode's annotated subtask when the dataset has one.
    """
    camera = dataset.meta.features[camera_key]
    names = camera.get("names") or ["height", "width", "channels"]
    aspect = camera["shape"][names.index("width")] / camera["shape"][names.index("height")]
    size = (round(height * aspect / 2) * 2, height)
    fps = dataset.fps

    curves = [signal for signal in signals if signal.descriptor.labels is None]
    stages = [signal for signal in signals if signal.descriptor.labels is not None]
    frames = np.arange(len(curves[0].values))
    tracks = []
    heads = [stage.name.split()[0] for stage in stages]
    annotated = _annotations(dataset, episode, heads + ["dense", "sparse"])
    if annotated is not None:
        names, starts, _ = annotated
        tracks.append(SubtaskTrack("SUBTASK", names, np.searchsorted(starts, frames, side="right") - 1))
    for stage in stages:
        label = "PREDICTED" if len(stages) == 1 else f"PREDICTED {stage.name.split()[0].upper()}"
        index = np.where(np.isfinite(stage.values), stage.values, -1).astype(int)
        tracks.append(SubtaskTrack(label, list(stage.descriptor.labels or ()), index))
    overlay = ProgressOverlay(
        curves,
        size,
        fps,
        task,
        f"{dataset.repo_id} | episode {episode}",
        logo,
        tracks=tracks,
        boundaries=annotated[1] if annotated is not None else (),
    )

    preview_step = max(1, round(fps / _WEBP_MAX_FPS))
    previews = []
    with av.open(str(path), "w", options={"movflags": "faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        # Tag the video as BT.709, the colors players assume for HD video, and convert to match.
        stream.options = {
            "crf": str(_MP4_CRF),
            "preset": "slow",
            "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        }
        for index, frame in enumerate(_iter_episode_frames(dataset, episode, camera_key)):
            image = overlay.render(frame, index)
            video_frame = av.VideoFrame.from_image(image).reformat(
                format="yuv420p", dst_colorspace=Colorspace.ITU709, dst_color_range=ColorRange.MPEG
            )
            container.mux(stream.encode(video_frame))
            if webp and index % preview_step == 0:
                previews.append(
                    image.resize(
                        (_WEBP_WIDTH, round(_WEBP_WIDTH * size[1] / size[0])), Image.Resampling.LANCZOS
                    )
                )
        container.mux(stream.encode())
    print(f"Wrote {path}")

    if previews:
        preview_path = path.with_suffix(".webp")
        duration = round(1000 * preview_step / fps)
        previews[0].save(
            preview_path, save_all=True, append_images=previews[1:], duration=duration, loop=0, quality=85
        )
        print(f"Wrote {preview_path}")


def main() -> None:
    require_package("datasets", extra="dataset")
    require_package("av", extra="dataset")
    parser = argparse.ArgumentParser(
        description="Render episode videos with reward-model scores drawn over them."
    )
    parser.add_argument(
        "--repo-id", required=True, help="Dataset repository ID, such as 'lerobot/libero_10'."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Local dataset folder with the scores (default: the Hub cache).",
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=[0], help="Episodes to render (default: 0)."
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        required=True,
        help="Scores to draw, such as 'robometer' or 'robometer:reward.robometer.progress'.",
    )
    parser.add_argument(
        "--camera-key",
        default=None,
        help="Camera to draw on (default: the camera the first score was computed on, else the first camera).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("progress_videos"), help="Output folder.")
    parser.add_argument(
        "--height", type=int, default=1080, help="Output height in pixels, even (default: 1080)."
    )
    parser.add_argument("--webp", action="store_true", help="Also write an animated WebP preview.")
    args = parser.parse_args()
    if args.height <= 0 or args.height % 2:
        parser.error("--height must be a positive even number, as H.264 requires")

    # Check everything before downloading videos or rendering.
    dataset = LeRobotDataset(
        args.repo_id, root=args.dataset_root, episodes=args.episodes, download_videos=False
    )
    episodes: dict[int, tuple[list[Signal], str]] = {}
    for episode in args.episodes:
        signals = _load_signals(dataset, episode, args.scores)
        curves = [signal for signal in signals if signal.descriptor.labels is None]
        if not curves:
            raise ValueError(
                f"Scores {args.scores} have no signal to plot, only labeled ones such as subtasks"
            )
        _axis_range(curves)
        episodes[episode] = signals, _episode_task(dataset, episode)
    video_keys = dataset.meta.video_keys
    camera_key = args.camera_key or _scored_camera(dataset, args.scores) or next(iter(video_keys), None)
    if camera_key not in video_keys:
        raise ValueError(f"No video {camera_key!r}; the dataset's videos are {video_keys}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _download_videos(dataset, args.episodes, camera_key, local_root=args.dataset_root is not None)
    logo = _load_logo()
    scores = "+".join(spec.partition(":")[0] for spec in args.scores)
    for episode, (signals, task) in episodes.items():
        path = args.output_dir / f"{args.repo_id.replace('/', '_')}_ep{episode}_{scores}.mp4"
        render_episode(dataset, episode, camera_key, signals, task, path, args.height, args.webp, logo)


if __name__ == "__main__":
    main()
