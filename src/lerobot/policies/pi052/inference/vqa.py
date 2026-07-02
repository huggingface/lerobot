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
"""Interactive VQA for the PI052 runtime.

In ``/vlm`` mode a typed line is treated as a VQA question. This module
runs the full interactive flow:

  1. pull the current observation and list available cameras,
  2. ask the operator which camera to ground the question on,
  3. generate the answer with the VLM conditioned on that one camera,
  4. parse the JSON answer; if it carries a bounding box (``bbox``) or a
     point (``keypoint``), draw the overlay on the camera frame, save a
     PNG to ``./vqa_overlays/`` and auto-open it.

VQA answer schemas mirror the annotation pipeline's ``VQA_ANSWER_SHAPES``
(see ``lerobot.annotations.steerable_pipeline.validator``):

  * ``bbox``     — ``{"detections": [{"label", "bbox_format": "xyxy",
                    "bbox": [x1, y1, x2, y2]}, ...]}``
  * ``keypoint`` — ``{"label", "point_format": "xy", "point": [x, y]}``
  * ``count`` / ``attribute`` / ``spatial`` — text-only, no overlay.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import webbrowser
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IMAGE_PREFIX = "observation.images."

# PaliGemma detection / pointing vocabulary. PI052 trains spatial VQA
# answers in this native ``<locNNNN>`` format (index in [0, 1023],
# normalized to the image axis) instead of pixel-coordinate JSON, so the
# answer string the runtime parses can be e.g.
# ``<loc0512><loc0301> blue cube`` (point) or
# ``<loc0100><loc0080><loc0400><loc0360> blue cube`` (box).
_LOC_RE = re.compile(r"<loc(\d{1,4})>")

# Iteration order for shape matching — most specific keys first so an
# answer is classified deterministically.
_SHAPE_ORDER = ("bbox", "keypoint", "count", "attribute", "spatial")

_BBOX_COLOR = (255, 64, 64)
_POINT_COLOR = (64, 220, 64)


# ---------------------------------------------------------------------------
# Camera selection
# ---------------------------------------------------------------------------


def available_cameras(observation: dict | None) -> list[str]:
    """Return the sorted ``observation.images.*`` keys present in ``observation``."""
    if not observation:
        return []
    return sorted(k for k in observation if isinstance(k, str) and k.startswith(_IMAGE_PREFIX))


def camera_short_name(camera_key: str) -> str:
    """Strip the ``observation.images.`` prefix for display."""
    return camera_key[len(_IMAGE_PREFIX) :] if camera_key.startswith(_IMAGE_PREFIX) else camera_key


def prompt_camera_choice(
    cameras: list[str],
    *,
    input_fn: Any = input,
    print_fn: Any = print,
) -> str | None:
    """Ask the operator which camera frame to draw a VQA overlay on.

    Accepts either the menu number or the (short or full) camera name.
    A single-camera setup auto-selects without prompting. Returns the
    chosen ``observation.images.*`` key, or ``None`` if the operator
    cancels / gives an invalid answer.
    """
    if not cameras:
        return None
    if len(cameras) == 1:
        return cameras[0]
    print_fn("Draw the result on which camera?")
    for i, cam in enumerate(cameras, 1):
        print_fn(f"  [{i}] {camera_short_name(cam)}")
    try:
        raw = str(input_fn("camera> ")).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return cameras[0]
    if raw.isdigit():
        idx = int(raw) - 1
        return cameras[idx] if 0 <= idx < len(cameras) else None
    for cam in cameras:
        if raw == cam or raw == camera_short_name(cam):
            return cam
    return None


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def _loc_to_norm(idx: int) -> float:
    """PaliGemma ``<locNNNN>`` index → normalized [0, 1] axis coordinate."""
    return max(0.0, min(1023.0, float(idx))) / 1023.0


def parse_loc_answer(answer: str) -> dict | None:
    """Parse a PaliGemma ``<loc>``-format spatial VQA answer.

    Point: ``<label> <locY><locX>``; box: ``<label> <locY0><locX0><locY1><locX1>``;
    multiple boxes joined by `` ; `` (label/loc order irrelevant). Returns
    ``{"kind", "payload", "normalized": True}`` with [0, 1] coords mirroring the
    JSON shapes (shared overlay code), or ``None`` without ``<loc>`` tokens.
    """
    if not answer or "<loc" not in answer:
        return None
    segments = [seg for seg in answer.split(";") if "<loc" in seg]
    points: list[tuple[float, float, str]] = []
    boxes: list[tuple[float, float, float, float, str]] = []
    for seg in segments:
        locs = [int(m) for m in _LOC_RE.findall(seg)]
        label = _LOC_RE.sub("", seg).strip()
        if len(locs) == 2:
            y, x = (_loc_to_norm(v) for v in locs[:2])
            points.append((x, y, label))
        elif len(locs) >= 4:
            y1, x1, y2, x2 = (_loc_to_norm(v) for v in locs[:4])
            boxes.append((x1, y1, x2, y2, label))
    if boxes:
        detections = [
            {"label": lbl, "bbox_format": "xyxy", "bbox": [x1, y1, x2, y2]} for (x1, y1, x2, y2, lbl) in boxes
        ]
        return {"kind": "bbox", "payload": {"detections": detections}, "normalized": True}
    if len(points) == 1:
        x, y, lbl = points[0]
        return {
            "kind": "keypoint",
            "payload": {"label": lbl, "point_format": "xy", "point": [x, y]},
            "normalized": True,
        }
    if points:  # several bare points → treat as detections-as-points
        detections = [{"label": lbl, "bbox_format": "xyxy", "bbox": [x, y, x, y]} for (x, y, lbl) in points]
        return {"kind": "bbox", "payload": {"detections": detections}, "normalized": True}
    return None


def parse_vqa_answer(answer: str) -> dict | None:
    """Parse a VQA answer (``<loc>`` text or JSON) into ``{"kind", "payload"}``.

    ``kind`` is a ``VQA_ANSWER_SHAPES`` name or ``"unknown"``; ``<loc>`` answers
    are tried first. Returns ``None`` when neither format parses.
    """
    if not answer or not answer.strip():
        return None
    loc_parsed = parse_loc_answer(answer)
    if loc_parsed is not None:
        return loc_parsed
    try:
        payload = json.loads(answer)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    try:
        from lerobot.annotations.steerable_pipeline.validator import (  # noqa: PLC0415
            VQA_ANSWER_SHAPES,
        )

        shapes = VQA_ANSWER_SHAPES
    except ImportError:  # pragma: no cover - annotation extra not installed
        shapes = {
            "bbox": {"detections"},
            "keypoint": {"label", "point_format", "point"},
            "count": {"label", "count"},
            "attribute": {"label", "attribute", "value"},
            "spatial": {"subject", "relation", "object"},
        }

    keys = set(payload)
    for kind in _SHAPE_ORDER:
        required = shapes.get(kind)
        if required and required <= keys:
            return {"kind": kind, "payload": payload}
    return {"kind": "unknown", "payload": payload}


def answer_has_overlay(parsed: dict | None) -> bool:
    """True iff ``parsed`` carries drawable spatial coordinates."""
    return bool(parsed) and parsed.get("kind") in ("bbox", "keypoint")


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------


def observation_image_to_pil(image_tensor: Any) -> Any:
    """Convert an ``observation.images.*`` tensor to a PIL RGB image.

    The runtime observation stores images as ``(1, C, H, W)`` (or
    ``(C, H, W)``) float tensors in ``[0, 1]``. Reuses
    ``image_array_to_pil_image`` which handles the CHW→HWC transpose and
    the float→uint8 scaling.
    """
    from lerobot.datasets.image_writer import image_array_to_pil_image  # noqa: PLC0415

    arr = image_tensor
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    while arr.ndim > 3:  # drop leading batch dim(s)
        arr = arr[0]
    return image_array_to_pil_image(arr).convert("RGB")


def draw_vqa_overlay(image: Any, parsed: dict) -> Any:
    """Draw ``bbox`` / ``keypoint`` answers onto a copy of ``image``.

    Non-spatial answers (``count`` / ``attribute`` / ``spatial`` /
    ``unknown``) are returned as an unmodified copy. When ``parsed`` has
    ``normalized=True`` (PaliGemma ``<loc>`` answers) the [0, 1]
    coordinates are scaled to the image's pixel size.
    """
    from PIL import ImageDraw  # noqa: PLC0415

    img = image.convert("RGB").copy()
    kind = parsed.get("kind")
    payload = parsed.get("payload") or {}
    draw = ImageDraw.Draw(img)
    w, h = img.size
    sx, sy = (w, h) if parsed.get("normalized") else (1, 1)

    if kind == "bbox":
        for det in payload.get("detections") or []:
            if not isinstance(det, dict):
                continue
            box = det.get("bbox")
            if not (isinstance(box, list | tuple) and len(box) == 4):
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            draw.rectangle([x1, y1, x2, y2], outline=_BBOX_COLOR, width=3)
            label = str(det.get("label", "")).strip()
            if label:
                draw.text((x1 + 3, max(0.0, y1 - 12)), label, fill=_BBOX_COLOR)
    elif kind == "keypoint":
        point = payload.get("point")
        if isinstance(point, list | tuple) and len(point) == 2:
            try:
                x, y = float(point[0]) * sx, float(point[1]) * sy
            except (TypeError, ValueError):
                return img
            r = 6
            draw.ellipse([x - r, y - r, x + r, y + r], outline=_POINT_COLOR, width=3)
            draw.line([x - 2 * r, y, x + 2 * r, y], fill=_POINT_COLOR, width=2)
            draw.line([x, y - 2 * r, x, y + 2 * r], fill=_POINT_COLOR, width=2)
            label = str(payload.get("label", "")).strip()
            if label:
                draw.text((x + r + 3, y - r), label, fill=_POINT_COLOR)
    return img


def _open_file(path: Path) -> None:
    """Best-effort open ``path`` in the OS default viewer."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)  # nosec B607
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False)  # nosec B607
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606  # nosec B606
        else:  # pragma: no cover - exotic platform
            webbrowser.open(path.resolve().as_uri())
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not auto-open %s: %s", path, exc)


def save_and_open_overlay(image: Any, out_dir: str | Path = "./vqa_overlays") -> Path:
    """Save ``image`` as a timestamped PNG under ``out_dir`` and auto-open it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"vqa_{int(time.time() * 1000)}.png"
    image.save(path)
    _open_file(path)
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def handle_vqa_query(
    *,
    policy_adapter: Any | None = None,
    policy: Any | None = None,
    observation_provider: Any,
    question: str,
    state: Any,
    input_fn: Any = input,
    print_fn: Any = print,
) -> None:
    """Run one interactive VQA question end to end.

    Called synchronously from the input layer while the runtime is in
    ``/question`` mode (the action loop is gated off, so the policy is
    not in concurrent use). Progress is reported via both
    ``state.log`` (REPL panel scrollback) and ``print_fn`` (direct stdout)
    — in autonomous question mode the panel redraw is suspended,
    so the direct print is what the operator actually sees.
    """
    if policy_adapter is None and policy is not None:
        from .pi052_adapter import PI052PolicyAdapter  # noqa: PLC0415

        policy_adapter = PI052PolicyAdapter(policy)

    def report(line: str) -> None:
        """Surface a line both to the panel scrollback and to stdout."""
        if hasattr(state, "log"):
            state.log(line)
        else:
            state.setdefault("log_lines", []).append(line)
        with suppress(Exception):
            print_fn(line)

    if policy_adapter is None:
        report("  [warn] vqa: no policy adapter — skipping")
        return

    observation: dict | None = None
    if observation_provider is not None:
        try:
            observation = observation_provider()
        except Exception as exc:  # noqa: BLE001
            logger.debug("observation_provider raised %s", exc)

    # Feed the FULL observation (every camera + state) to the VLM. The
    # ``ask_vqa_*`` recipes look single-camera, but the image *block* is
    # stripped before tokenization — the actual frames reach the model
    # via PI052's ``OBS_IMAGES_*`` channels, and ``embed_prefix``
    # consumes *all* ``config.image_features`` regardless of which
    # camera the sub-recipe was tagged for. So the model always sees
    # every camera; the operator never has to name one to ask.
    result = policy_adapter.answer_vqa(question, None, observation, state)
    answer = result.answer
    if not answer:
        report("  [info] vqa gen returned empty")
        return
    report(f"  vqa: {answer}")

    parsed = result.parsed if result.parsed is not None else parse_vqa_answer(answer)
    if not answer_has_overlay(parsed):
        if parsed is None:
            report("  [info] vqa answer is not JSON — no overlay")
        return

    # The answer carries a bounding box / point. Its pixel coordinates
    # are camera-specific and the text answer doesn't say which camera,
    # so ask the operator *now* — only when there is actually something
    # to draw — which camera frame to render the overlay on.
    cameras = available_cameras(observation)
    if observation is None or not cameras:
        report("  [info] no camera image — cannot draw overlay")
        return
    chosen = prompt_camera_choice(cameras, input_fn=input_fn, print_fn=print_fn)
    if chosen is None:
        report("  [info] overlay skipped — no camera selected")
        return
    try:
        pil = observation_image_to_pil(observation[chosen])
        overlay = draw_vqa_overlay(pil, parsed)
        path = save_and_open_overlay(overlay)
        report(f"  vqa overlay ({camera_short_name(chosen)}) saved: {path}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("vqa overlay failed: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        report(f"  [warn] vqa overlay failed: {type(exc).__name__}: {exc}")
