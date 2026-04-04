# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Pluggable VLM-based object detector with local (Florence-2), cloud, and Gemini backends.

The detector takes an RGB image and a natural-language query, and returns
bounding boxes with labels.  It also provides a convenience method for
generating a segmentation mask from the bounding box using OpenCV GrabCut.

Supported backends:
- ``"local"``: Florence-2 via HuggingFace transformers (runs on your GPU)
- ``"cloud"``: OpenAI-compatible API (GPT-4o, etc.)
- ``"gemini"``: Google Gemini via its OpenAI-compatible endpoint
- ``"claude"``: Anthropic Claude Vision API (claude-3-5-sonnet, claude-3-5-haiku, etc.)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# (model_id, device) -> (processor, model); avoids reloading SAM on every bbox.
_sam2_model_cache: dict[tuple[str, str], tuple[object, object]] = {}


def _phrase_for_phrase_grounding(query: str) -> str:
    """Strip common robot verbs so Florence-2 phrase grounding sees an object caption.

    CAPTION_TO_PHRASE_GROUNDING echoes phrases from the caption; feeding the full
    task (e.g. 'pick up the red cube') yields one box labeled like the command.
    """
    s = query.strip()
    if not s:
        return s
    lowered = s.lower()
    for prefix in (
        "pick up ",
        "pickup ",
        "grab ",
        "get ",
        "take ",
        "move ",
        "place ",
        "put ",
        "stack ",
    ):
        if lowered.startswith(prefix):
            s = s[len(prefix) :].lstrip()
            lowered = s.lower()
    return s.strip() or query.strip()


@dataclass
class Detection:
    """A single detected object."""

    label: str
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float = 1.0
    mask: np.ndarray | None = None


def mask_from_bbox_grabcut(
    rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    iterations: int = 5,
) -> np.ndarray:
    """Refine a bounding box into a binary mask using OpenCV GrabCut."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = (
        max(0, int(bbox_xyxy[0])),
        max(0, int(bbox_xyxy[1])),
        min(w, int(bbox_xyxy[2])),
        min(h, int(bbox_xyxy[3])),
    )
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((h, w), dtype=np.uint8)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)

    try:
        cv2.grabCut(bgr, gc_mask, rect, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return binary


def mask_from_bbox_sam(
    rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    model_id: str = "facebook/sam2-hiera-large",
) -> np.ndarray:
    """Refine a bounding box into a binary mask using a SAM2 image model if available.

    ``AutoModel`` maps ``facebook/sam2-hiera-*`` to ``Sam2VideoModel``, which does not
    accept ``pixel_values`` for single-image box prompts; we use ``Sam2Model`` instead.

    Falls back to GrabCut when the SAM dependencies or weights are not available.
    """
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except Exception as e:
        # SAM not installed; fall back to GrabCut.
        logger.warning(f"SAM backend not available ({e}); falling back to GrabCut.")
        return mask_from_bbox_grabcut(rgb, bbox_xyxy)

    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((h, w), dtype=np.uint8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        cache_key = (model_id, device)
        if cache_key not in _sam2_model_cache:
            processor = Sam2Processor.from_pretrained(model_id)
            model = Sam2Model.from_pretrained(model_id).to(device)
            model.eval()
            _sam2_model_cache[cache_key] = (processor, model)
        processor, model = _sam2_model_cache[cache_key]

        pil_image = Image.fromarray(rgb)
        # SAM2 processors expect input_boxes to have 3 nesting levels:
        # [image_index][box_index][coords], i.e. shape (num_images, num_boxes, 4).
        box = [float(x1), float(y1), float(x2), float(y2)]
        inputs = processor(
            images=pil_image,
            input_boxes=[[box]],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
    except Exception as e:
        logger.warning(f"SAM model '{model_id}' call failed ({e}); falling back to GrabCut.")
        return mask_from_bbox_grabcut(rgb, bbox_xyxy)

    logits = getattr(outputs, "pred_masks", None)
    if logits is None and isinstance(outputs, dict):
        logits = outputs.get("pred_masks")
    if logits is None:
        for key in ("masks", "segmentation"):
            if hasattr(outputs, key):
                logits = getattr(outputs, key)
                break
            if isinstance(outputs, dict) and key in outputs:
                logits = outputs[key]
                break

    if logits is None:
        logger.warning("SAM backend did not return masks; falling back to GrabCut.")
        return mask_from_bbox_grabcut(rgb, bbox_xyxy)

    # Sam2 image head: [B, num_prompts, num_candidate_masks, H, W]
    if logits.dim() == 5:
        candidates = logits[0, 0]
        scores = getattr(outputs, "iou_scores", None)
        if scores is not None and scores.dim() >= 2:
            idx = int(scores[0, 0].argmax().item())
        else:
            idx = 0
        mask_logits = candidates[idx]
    elif logits.dim() == 4:
        mask_logits = logits[0, 0]
    elif logits.dim() == 3:
        mask_logits = logits[0]
    else:
        mask_logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])[0]

    mask_prob = torch.sigmoid(mask_logits.float()).cpu().numpy()
    if mask_prob.ndim == 3:
        mask_prob = mask_prob[0]
    mask_resized = cv2.resize(mask_prob, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (mask_resized > 0.5).astype(np.uint8) * 255
    return binary


class _Florence2Backend:
    """Local Florence-2 inference via HuggingFace transformers."""

    def __init__(self, model_id: str = "microsoft/Florence-2-base", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.float16 if "cuda" in device else torch.float32

        logger.info(f"Loading Florence-2 from {model_id} on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Florence-2 hub code predates newer transformers SDPA hooks; eager avoids
        # AttributeError: ... has no attribute '_supports_sdpa' during init.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
        logger.info("Florence-2 loaded.")

    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        import torch

        pil_image = Image.fromarray(rgb)
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        phrase = _phrase_for_phrase_grounding(query)
        prompt = task + phrase

        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype) if v.dtype in (torch.float32, torch.float16) else v.to(self.device) for k, v in inputs.items()}

        # Newer transformers uses a cache layout Florence-2 hub code does not handle
        # (prepare_inputs_for_generation assumes past_key_values[0][0] is a tensor).
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            use_cache=False,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil_image.width, pil_image.height),
        )

        result = parsed.get(task, {})
        bboxes = result.get("bboxes", [])
        labels = result.get("labels", [])

        detections: list[Detection] = []
        n = min(len(bboxes), len(labels))
        for i in range(n):
            bbox = bboxes[i]
            label = labels[i].strip()
            # Single box: use object phrase so the scene graph matches the task object.
            if n == 1:
                label = phrase
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            mask = mask_from_bbox_sam(rgb, (x1, y1, x2, y2))
            detections.append(Detection(label=label, bbox_xyxy=(x1, y1, x2, y2), mask=mask))

        return detections


class _CloudVLMBackend:
    """Cloud VLM backend using OpenAI-compatible API (works with GPT-4o, Gemini, etc.)."""

    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        is_gemini: bool = False,
    ):
        import os

        if is_gemini:
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
            self.base_url = base_url or self.GEMINI_BASE_URL
            self.model = model if model != "gpt-4o" else "gemini-2.5-flash"
            env_hint = "GEMINI_API_KEY"
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self.base_url = base_url
            self.model = model
            env_hint = "OPENAI_API_KEY"

        if not self.api_key:
            raise ValueError(
                f"Cloud VLM backend requires an API key. Set {env_hint} or pass api_key=."
            )

    def _encode_image(self, rgb: np.ndarray) -> str:
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        b64 = self._encode_image(rgb)
        h, w = rgb.shape[:2]

        system_prompt = (
            "You are an object detection assistant. Given an image and a query, "
            "return a JSON array of detected objects. Each object should have: "
            '"label" (string), "bbox" (array of [x1, y1, x2, y2] in pixel coordinates). '
            f"The image is {w}x{h} pixels. Only return the JSON array, no other text."
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Detect objects matching: {query}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            max_tokens=1024,
        )

        text = response.choices[0].message.content or ""
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            logger.warning(f"Cloud VLM returned no parseable JSON: {text[:200]}")
            return self._fallback_parse_detections(text, rgb)

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse cloud VLM response as JSON: {text[:200]}")
            items = self._fallback_parse_detections(text, rgb)
            if not items:
                return []

        detections: list[Detection] = []
        for item in items:
            label = str(item.get("label", "unknown"))
            bbox = item.get("bbox", [0, 0, 0, 0])
            sanitized = self._sanitize_bbox(bbox, rgb.shape[1], rgb.shape[0])
            if sanitized is None:
                continue
            x1, y1, x2, y2 = sanitized
            # For cloud/Gemini/Claude backends, use a lightweight, dependency-free
            # mask refinement to avoid tight coupling to specific SAM model APIs.
            mask = mask_from_bbox_grabcut(rgb, (x1, y1, x2, y2))
            detections.append(Detection(label=label, bbox_xyxy=(x1, y1, x2, y2), mask=mask))

        return detections

    def _sanitize_bbox(self, bbox: object, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
        """Convert a bbox payload into a valid in-bounds xyxy box.

        Cloud VLMs sometimes return coordinates in a wrong scale (e.g. assuming a
        square image) or swapped dimensions. Here we apply conservative
        heuristics to map it into the actual image size so downstream mask/3D
        extraction doesn't silently yield empty objects.
        """
        # Parse bbox into floats.
        vals: list[float] = []
        if isinstance(bbox, (list, tuple)):
            for v in list(bbox)[:4]:
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(0.0)
        elif isinstance(bbox, str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", bbox)
            for n in nums[:4]:
                with np.errstate(all="ignore"):
                    vals.append(float(n))
            while len(vals) < 4:
                vals.append(0.0)
        else:
            vals = [0.0, 0.0, 0.0, 0.0]

        x1f, y1f, x2f, y2f = vals[:4]

        # If bbox looks like normalized [0,1], scale up.
        if 0.0 <= x2f <= 1.5 and 0.0 <= y2f <= 1.5 and max(x1f, y1f, x2f, y2f) <= 1.5:
            x1f *= img_w
            x2f *= img_w
            y1f *= img_h
            y2f *= img_h

        # If bbox exceeds image bounds, rescale to fit (common when model assumes different resolution).
        max_x = max(x1f, x2f, 1.0)
        max_y = max(y1f, y2f, 1.0)
        if max_x > img_w or max_y > img_h:
            sx = img_w / max_x if max_x > 0 else 1.0
            sy = img_h / max_y if max_y > 0 else 1.0
            x1f *= sx
            x2f *= sx
            y1f *= sy
            y2f *= sy

        # Order and clamp.
        x1 = int(round(min(x1f, x2f)))
        x2 = int(round(max(x1f, x2f)))
        y1 = int(round(min(y1f, y2f)))
        y2 = int(round(max(y1f, y2f)))

        x1 = max(0, min(img_w - 1, x1))
        x2 = max(0, min(img_w - 1, x2))
        y1 = max(0, min(img_h - 1, y1))
        y2 = max(0, min(img_h - 1, y2))

        bw, bh = x2 - x1, y2 - y1
        # Do not fabricate a tiny corner box: that used to force 4×4px at (0,0) and broke depth.
        _min_side, _min_area = 10, 400
        if bw < _min_side or bh < _min_side or bw * bh < _min_area:
            logger.warning(
                "Cloud VLM bbox rejected after sanitize (too small for depth/mask): "
                "raw=%s → xyxy=(%d,%d,%d,%d) on %dx%d",
                bbox,
                x1,
                y1,
                x2,
                y2,
                img_w,
                img_h,
            )
            return None

        return x1, y1, x2, y2

    def _fallback_parse_detections(self, text: str, rgb: np.ndarray) -> list[dict]:
        """Heuristic fallback parser for slightly malformed JSON from cloud VLMs.

        Handles cases like:
        ```json
        [
          {"label": "red cube",
           "bbox": [376, 525, 627, 629]
        ]
        ```
        by extracting label / bbox pairs with regex instead of strict JSON.
        """
        # Strip common Markdown code fences if present.
        code_match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if code_match:
            snippet = code_match.group(1)
        else:
            snippet = text

        label_matches = re.findall(r'"label"\s*:\s*"([^"]+)"', snippet)
        bbox_matches = re.findall(r'"bbox"\s*:\s*\[([^\]]+)\]', snippet)

        if not label_matches or not bbox_matches:
            return []

        items: list[dict] = []
        for i, label in enumerate(label_matches):
            if i >= len(bbox_matches):
                break
            nums_str = bbox_matches[i]
            try:
                nums = [float(v.strip()) for v in nums_str.split(",") if v.strip()]
            except ValueError:
                continue
            if len(nums) < 4:
                continue
            x1, y1, x2, y2 = nums[:4]
            items.append({"label": label, "bbox": [x1, y1, x2, y2]})

        if not items:
            return []

        logger.warning(
            "Using fallback regex parser for cloud VLM detections; raw text was: %s",
            text[:200],
        )
        return items


class _ClaudeBackend:
    """Claude Vision API backend using Anthropic SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        import os

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Claude backend requires an API key. Set ANTHROPIC_API_KEY or pass api_key=."
            )
        self.model = model

    def _encode_image(self, rgb: np.ndarray) -> str:
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        b64 = self._encode_image(rgb)
        h, w = rgb.shape[:2]

        system_prompt = (
            "You are an object detection assistant. Given an image and a query, "
            "return a JSON array of detected objects. Each object should have: "
            '"label" (string), "bbox" (array of [x1, y1, x2, y2] in pixel coordinates). '
            f"The image is {w}x{h} pixels. Only return the JSON array, no other text."
        )

        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": f"Detect objects matching: {query}"},
                    ],
                },
            ],
        )

        text = ""
        if response.content:
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    text += getattr(block, "text", "") or ""

        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            logger.warning(f"Claude returned no parseable JSON: {text[:200]}")
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Claude response as JSON: {text[:200]}")
            return []

        detections: list[Detection] = []
        for item in items:
            label = str(item.get("label", "unknown"))
            bbox = item.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            mask = mask_from_bbox_sam(rgb, (x1, y1, x2, y2))
            detections.append(Detection(label=label, bbox_xyxy=(x1, y1, x2, y2), mask=mask))

        return detections


class VLMDetector:
    """Pluggable VLM object detector with local, cloud, Gemini, and Claude backends.

    Example::

        detector = VLMDetector(backend="gemini")
        detections = detector.detect(rgb_image, "a red cube on the table")
        for det in detections:
            print(det.label, det.bbox_xyxy)
    """

    def __init__(
        self,
        backend: str = "gemini",
        model_id: str = "microsoft/Florence-2-base",
        device: str | None = None,
        api_key: str | None = None,
        cloud_model: str = "gpt-4o",
        cloud_base_url: str | None = None,
    ):
        """
        Args:
            backend: ``"local"`` for Florence-2, ``"cloud"`` for OpenAI API,
                ``"gemini"`` for Google Gemini, or ``"claude"`` for Anthropic Claude.
            model_id: HuggingFace model ID for local backend.
            device: Torch device for local backend (auto-detected if None).
            api_key: API key for cloud/gemini/claude backend. If None, reads from
                ``OPENAI_API_KEY`` (cloud), ``GEMINI_API_KEY`` (gemini), or
                ``ANTHROPIC_API_KEY`` (claude).
            cloud_model: Model name for cloud backend. Defaults to ``"gpt-4o"`` for
                cloud, ``"gemini-2.5-flash"`` for gemini, ``"claude-sonnet-4-20250514"``
                for claude.
            cloud_base_url: Optional base URL override for cloud API (cloud/gemini only).
        """
        self.backend_name = backend
        if backend == "local":
            self._backend = _Florence2Backend(model_id=model_id, device=device)
        elif backend == "cloud":
            self._backend = _CloudVLMBackend(
                api_key=api_key, model=cloud_model, base_url=cloud_base_url, is_gemini=False
            )
        elif backend == "gemini":
            self._backend = _CloudVLMBackend(
                api_key=api_key, model=cloud_model, base_url=cloud_base_url, is_gemini=True
            )
        elif backend == "claude":
            claude_model = cloud_model if cloud_model != "gpt-4o" else "claude-sonnet-4-20250514"
            self._backend = _ClaudeBackend(api_key=api_key, model=claude_model)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Use 'local', 'cloud', 'gemini', or 'claude'."
            )

    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        """Detect objects in an RGB image matching a natural-language query.

        Args:
            rgb: (H, W, 3) uint8 RGB image.
            query: Natural language description of what to find, e.g.
                ``"a red cube and a blue cup"``.

        Returns:
            List of Detection objects with labels, bounding boxes, and masks.
        """
        return self._backend.detect(rgb, query)
