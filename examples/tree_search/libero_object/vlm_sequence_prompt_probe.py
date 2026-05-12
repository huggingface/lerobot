#!/usr/bin/env python
"""Probe a VLM reward prompt on successful LIBERO demonstration sequences.

This script samples frames from a successful LeRobotDataset episode, scores
each frame independently with an OpenAI-compatible VLM endpoint, and writes an
annotated video with the score rendered in red at the bottom-left.

Example:

```bash
export OAI_KEY=EMPTY

uv run python examples/tree_search/vlm_sequence_prompt_probe.py \
    --dataset_repo_id=HuggingFaceVLA/libero \
    --suite=libero_object \
    --task_id=7 \
    --frame_stride=10 \
    --max_frames=32 \
    --output_dir=outputs/tree_search/vlm_probe_task7 \
    --video_path=outputs/tree_search/vlm_probe_task7.mp4 \
    --vlm_model=Qwen/Qwen3.5-2B \
    --vlm_base_url=http://localhost:8000/v1 \
    --vlm_api_key_env=OAI_KEY
```

To test a prompt variant without editing this file:

```bash
uv run python examples/tree_search/vlm_sequence_prompt_probe.py \
    ... \
    --prompt_template_path=prompt_variant.txt
```

Supported template variables:
`{task}`, `{image_description}`, `{robot_state_json}`, `{task_metadata_json}`.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.image_writer import write_image
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE
from lerobot.utils.io_utils import write_video

logger = logging.getLogger(__name__)
TREE_SEARCH_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass
class VLMScore:
    score: float
    reason: str
    prompt: str
    raw_response: Any | None


@dataclass
class FrameScoreRecord:
    sample_index: int
    episode_index: int
    dataset_index: int
    local_index: int
    frame_index: int | None
    timestamp: float | None
    task: str
    score: float
    reason: str
    annotated_image_path: str
    vlm_input_images: list[dict[str, str]]
    temporal_context_local_indices: list[int]
    prompt: str
    raw_response: Any | None


def _normalize_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    return value


def _coerce_scalar(value: Any) -> float | int | bool | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return None
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return None
    if isinstance(value, bool | int | float):
        return value
    return None


def _image_to_uint8_hwc(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        image_path = value.get("path")
        if image_path:
            with Image.open(image_path) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        for key in ("array", "data"):
            if key in value:
                return _image_to_uint8_hwc(value[key])
        raise ValueError(f"Unsupported image dictionary keys: {sorted(value)}")
    if isinstance(value, bytes | bytearray):
        with Image.open(BytesIO(value)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image.shape}.")
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 1.0
        if max_value <= 1.0:
            image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _get_libero_task(suite_name: str, task_id: int) -> dict[str, Any]:
    try:
        from libero.libero import benchmark, get_libero_path
    except ImportError as exc:
        raise SystemExit("LIBERO is required for suite/task_id lookup.") from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise SystemExit(f"Unknown suite '{suite_name}'. Available: {sorted(benchmark_dict)}")

    suite = benchmark_dict[suite_name]()
    task = suite.get_task(task_id)
    metadata: dict[str, Any] = {
        "suite": suite_name,
        "task_id": task_id,
        "name": str(task.name),
        "language": str(task.language),
        "bddl_file": str(task.bddl_file),
    }

    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if bddl_path.exists():
        bddl_text = bddl_path.read_text()
        metadata["objects"] = _parse_bddl_objects(bddl_text)
        metadata["objects_of_interest"] = _parse_bddl_object_interest(bddl_text)
        metadata["goal"] = _compact_lisp(_find_lisp_section(bddl_text, "goal"))
    return metadata


def _strip_lisp_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def _compact_lisp(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_lisp_comments(text)).strip()


def _find_lisp_section(text: str, section_name: str) -> str:
    text = _strip_lisp_comments(text)
    match = re.search(r"\(:" + re.escape(section_name) + r"\b", text)
    if match is None:
        return ""
    depth = 0
    for idx in range(match.start(), len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                return text[match.start() : idx + 1]
    return text[match.start() :]


def _parse_bddl_objects(text: str) -> list[dict[str, str]]:
    objects: list[dict[str, str]] = []
    for raw_line in _find_lisp_section(text, "objects").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("(:objects") or line == ")" or " - " not in line:
            continue
        instances_raw, category = line.split(" - ", 1)
        category = category.strip().strip(")")
        for instance in instances_raw.split():
            if instance and not instance.startswith("("):
                objects.append(
                    {
                        "instance": instance,
                        "category": category,
                        "label": category.replace("_", " "),
                    }
                )
    return objects


def _parse_bddl_object_interest(text: str) -> list[str]:
    section = _find_lisp_section(text, "obj_of_interest")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", section)
    return [token for token in tokens if token != "obj_of_interest"]


def _episode_tasks(meta: LeRobotDatasetMetadata, episode: dict[str, Any]) -> list[str]:
    tasks = episode.get("tasks")
    if isinstance(tasks, str):
        return [tasks]
    if isinstance(tasks, list | tuple):
        return [str(task) for task in tasks]
    return []


def _find_episode_for_task(
    meta: LeRobotDatasetMetadata,
    *,
    task_metadata: dict[str, Any],
    episode_index: int | None,
) -> int:
    if episode_index is not None:
        if episode_index < 0 or episode_index >= meta.total_episodes:
            raise SystemExit(f"episode_index={episode_index} is out of range [0, {meta.total_episodes - 1}].")
        return episode_index

    targets = {
        _normalize_text(str(task_metadata.get("language", ""))),
        _normalize_text(str(task_metadata.get("name", ""))),
    }
    targets.discard("")

    seen_tasks: set[str] = set()
    for ep_idx in range(meta.total_episodes):
        episode = meta.episodes[ep_idx]
        for task in _episode_tasks(meta, episode):
            seen_tasks.add(task)
            if _normalize_text(task) in targets:
                return ep_idx

    sample = "\n".join(f"- {task}" for task in sorted(seen_tasks)[:25])
    raise SystemExit(
        "Could not find a dataset episode matching the LIBERO task language/name. "
        "Pass --episode_index explicitly, or check dataset task strings.\n"
        f"Task targets: {sorted(targets)}\nSample dataset tasks:\n{sample}"
    )


def _parquet_index_range(path: Path) -> tuple[int, int] | None:
    try:
        index_column = pd.read_parquet(path, columns=["index"])["index"]
    except Exception:
        return None
    if index_column.empty:
        return None
    return int(index_column.min()), int(index_column.max())


def _find_episode_data_path(
    root: Path,
    *,
    dataset_from_index: int,
    dataset_to_index: int,
) -> Path:
    candidates = sorted((root / "data").glob("*/*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")

    for path in candidates:
        index_range = _parquet_index_range(path)
        if index_range is None:
            continue
        min_index, max_index = index_range
        if dataset_from_index >= min_index and dataset_from_index <= max_index:
            return path
        if dataset_to_index - 1 >= min_index and dataset_to_index - 1 <= max_index:
            return path

    ranges = []
    for path in candidates[:8]:
        index_range = _parquet_index_range(path)
        if index_range is not None:
            ranges.append(f"{path.relative_to(root)}={index_range}")
    raise FileNotFoundError(
        f"Could not find a data parquet containing global frame range "
        f"[{dataset_from_index}, {dataset_to_index}). Sample available ranges: {ranges}"
    )


def _infer_video_path_from_data_path(root: Path, data_path: Path, camera_key: str) -> Path:
    relative = data_path.relative_to(root)
    parts = list(relative.parts)
    if len(parts) < 3 or parts[0] != "data":
        raise ValueError(f"Unexpected data parquet path: {data_path}")
    return root / "videos" / camera_key / parts[1] / f"{Path(parts[2]).stem}.mp4"


class EpisodeFrameReader:
    """Direct reader for one episode's parquet rows and video frames.

    This avoids the Hugging Face Datasets filtered parquet path, which can fail
    with "Instruction train corresponds to no data" on partially cached nested
    LeRobot datasets.
    """

    def __init__(
        self,
        *,
        meta: LeRobotDatasetMetadata,
        episode_index: int,
        video_backend: str | None,
        tolerance_s: float = 1e-4,
    ) -> None:
        self.meta = meta
        self.root = meta.root
        self.episode_index = episode_index
        self.episode = meta.episodes[episode_index]
        self.dataset_from_index = int(self.episode["dataset_from_index"])
        self.dataset_to_index = int(self.episode["dataset_to_index"])
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s
        data_path = _find_episode_data_path(
            self.root,
            dataset_from_index=self.dataset_from_index,
            dataset_to_index=self.dataset_to_index,
        )
        if not data_path.exists():
            raise FileNotFoundError(f"Episode data parquet is missing: {data_path}")
        self.data_path = data_path

        df = pd.read_parquet(data_path)
        if "index" not in df.columns:
            raise ValueError(f"Parquet file has no global index column: {data_path}")
        df = df[
            (df["index"] >= self.dataset_from_index) & (df["index"] < self.dataset_to_index)
        ].reset_index(drop=True)
        if df.empty:
            min_index = pd.read_parquet(data_path, columns=["index"])["index"].min()
            max_index = pd.read_parquet(data_path, columns=["index"])["index"].max()
            raise ValueError(
                f"Metadata row {episode_index} has no frame rows in {data_path} for global index range "
                f"[{self.dataset_from_index}, {self.dataset_to_index}). File index range is "
                f"[{min_index}, {max_index}]."
            )
        self.df = df

    @property
    def camera_keys(self) -> list[str]:
        return list(self.meta.camera_keys)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, local_index: int) -> dict[str, Any]:
        row = self.df.iloc[int(local_index)].to_dict()
        item = dict(row)
        timestamp = float(row["timestamp"])

        for camera_key in self.meta.video_keys:
            from_timestamp = float(self.episode[f"videos/{camera_key}/from_timestamp"])
            video_path = _infer_video_path_from_data_path(self.root, self.data_path, camera_key)
            if not video_path.exists():
                video_path = self.root / self.meta.get_video_file_path(self.episode_index, camera_key)
            if not video_path.exists():
                raise FileNotFoundError(f"Episode video file is missing: {video_path}")
            frames = decode_video_frames(
                video_path,
                [from_timestamp + timestamp],
                self.tolerance_s,
                self.video_backend,
                return_uint8=True,
            )
            item[camera_key] = frames.squeeze(0)

        frame_index = int(_coerce_scalar(row.get("frame_index")) or int(local_index))
        for camera_key in self.meta.image_keys:
            raw_value = item.get(camera_key)
            if raw_value is not None:
                with contextlib.suppress(Exception):
                    item[camera_key] = _image_to_uint8_hwc(raw_value)
                    continue

            image_path = Path(
                DEFAULT_IMAGE_PATH.format(
                    image_key=camera_key,
                    episode_index=self.episode_index,
                    frame_index=frame_index,
                )
            )
            candidates = [self.root / image_path]
            raw_path = row.get(camera_key)
            if isinstance(raw_path, str):
                raw = Path(raw_path)
                candidates.extend([raw, self.root / raw])
            for candidate in candidates:
                if candidate.exists():
                    item[camera_key] = np.asarray(Image.open(candidate).convert("RGB"))
                    break
            else:
                raise FileNotFoundError(
                    f"Episode image file is missing for camera_key={camera_key} "
                    f"episode={self.episode_index} frame={frame_index}. Tried: {candidates}"
                )

        task_index = _coerce_scalar(row.get("task_index"))
        if task_index is not None:
            item["task"] = str(self.meta.tasks.iloc[int(task_index)].name)
        return item


def _resolve_camera_keys(reader: EpisodeFrameReader, requested: str) -> list[str]:
    available = list(reader.camera_keys)
    if requested.strip().lower() == "all":
        return available
    keys = [key.strip() for key in requested.split(",") if key.strip()]
    missing = [key for key in keys if key not in available]
    if missing:
        raise SystemExit(f"Unknown camera key(s): {missing}. Available camera keys: {available}")
    return keys


def _sample_local_indices(length: int, *, frame_stride: int, max_frames: int | None) -> list[int]:
    if length <= 0:
        return []
    indices = list(range(0, length, max(1, frame_stride)))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = [indices[int(pos)] for pos in positions]
    return sorted(set(indices))


def _reference_image_matches_task(path: Path, task_id: int) -> bool:
    task_tokens = {str(task_id), f"{task_id:02d}", f"{task_id:03d}"}
    for part in [path.stem, *path.parent.parts]:
        text = part.lower()
        compact = re.sub(r"[^a-z0-9]+", "", text)
        normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        for token in task_tokens:
            marker_forms = {
                token,
                f"task{token}",
                f"task_{token}",
                f"task-{token}",
                f"taskid{token}",
                f"task_id_{token}",
                f"task-id-{token}",
            }
            compact_marker_forms = {form.replace("_", "").replace("-", "") for form in marker_forms}
            if normalized in marker_forms or compact in compact_marker_forms:
                return True
            if normalized.startswith((f"{token}_", f"{token}-", f"task_{token}_", f"task_{token}-")):
                return True
            if normalized.startswith((f"task{token}_", f"task{token}-")):
                return True
    return False


def _load_reference_images(
    *,
    reference_image_dir: Path | None,
    reference_image_paths: str | None,
    reference_image_glob: str,
    reference_image_max_size: int,
    task_id: int,
    filter_by_task: bool,
) -> list[tuple[str, np.ndarray]]:
    paths: list[Path] = []
    for token in [part.strip() for part in (reference_image_paths or "").split(",") if part.strip()]:
        path = Path(token)
        if path.is_dir():
            paths.extend(
                child
                for child in sorted(path.glob(reference_image_glob))
                if child.is_file() and child.suffix.lower() in REFERENCE_IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in REFERENCE_IMAGE_SUFFIXES:
            paths.append(path)
        else:
            raise SystemExit(f"Reference image path does not exist or is not a supported image: {path}")

    if reference_image_dir is not None and reference_image_dir.exists():
        paths.extend(
            child
            for child in sorted(reference_image_dir.glob(reference_image_glob))
            if child.is_file() and child.suffix.lower() in REFERENCE_IMAGE_SUFFIXES
        )

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)

    if filter_by_task:
        all_paths = unique_paths
        unique_paths = [path for path in all_paths if _reference_image_matches_task(path, task_id)]
        if all_paths and not unique_paths:
            logger.warning("Found reference images, but none matched task_id=%s.", task_id)

    images: list[tuple[str, np.ndarray]] = []
    for ix, path in enumerate(unique_paths):
        with Image.open(path) as pil_image:
            image = pil_image.convert("RGB")
            if reference_image_max_size > 0:
                image.thumbnail((reference_image_max_size, reference_image_max_size), Image.Resampling.LANCZOS)
            label = f"reference.target.task_{task_id}.{ix:02d}.{path.stem}"
            images.append((label, np.asarray(image, dtype=np.uint8)))
    return images


def _extract_robot_state(item: dict[str, Any]) -> dict[str, Any] | None:
    state = item.get("observation.state")
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    state = np.asarray(state).reshape(-1)
    if state.size < 8:
        return {"observation_state": _to_jsonable(state)}
    return {
        "eef_position": _to_jsonable(state[:3]),
        "eef_axisangle": _to_jsonable(state[3:6]),
        "gripper_qpos": _to_jsonable(state[6:8]),
    }


def _image_to_data_url(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _parse_score_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}.")
    return parsed


def _error_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None) if response is not None else None
    if status_code is not None:
        with contextlib.suppress(TypeError, ValueError):
            return int(status_code)

    match = re.search(r"Error code:\s*(\d+)", str(exc))
    return int(match.group(1)) if match else None


def _is_rate_limit_error(exc: Exception) -> bool:
    if _error_status_code(exc) == 429:
        return True
    return "rate limit" in str(exc).lower()


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    if retry_after is None:
        return None
    with contextlib.suppress(TypeError, ValueError):
        return max(0.0, float(retry_after))
    return None


class VLMFrameScorer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model = args.vlm_model
        self.base_url = args.vlm_base_url or args.vlm_api_url
        self.api_key_env = args.vlm_api_key_env
        self.max_tokens = args.vlm_max_tokens
        self.temperature = args.vlm_temperature
        self.top_p = args.vlm_top_p
        self.presence_penalty = args.vlm_presence_penalty
        self.top_k = args.vlm_top_k
        self.min_p = args.vlm_min_p
        self.repetition_penalty = args.vlm_repetition_penalty
        self.timeout_s = args.vlm_timeout_s
        self.requests_per_minute = args.vlm_requests_per_minute
        self.max_retries = args.vlm_max_retries
        self.retry_sleep_s = args.vlm_retry_sleep_s
        self.rate_limit_sleep_s = args.vlm_rate_limit_sleep_s
        self.log_response_chars = args.vlm_log_response_chars
        self.verbose = args.vlm_verbose
        self.prompt_template = (
            args.prompt_template_path.read_text() if args.prompt_template_path is not None else None
        )
        self._client = None
        self._last_request_monotonic: float | None = None

    def score(
        self,
        *,
        task: str,
        images: list[tuple[str, np.ndarray]],
        robot_state: dict[str, Any] | None,
        task_metadata: dict[str, Any],
    ) -> VLMScore:
        prompt = self.build_prompt(
            task=task,
            image_labels=[label for label, _ in images],
            robot_state=robot_state,
            task_metadata=task_metadata,
        )
        if self.model is None:
            return VLMScore(
                score=0.0,
                reason="VLM model disabled.",
                prompt=prompt,
                raw_response=None,
            )

        client = self._get_client()
        content: list[dict[str, Any]] = []
        for label, image in images:
            content.append({"type": "text", "text": f"Image label: {label}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(image)},
                }
            )
        content.append({"type": "text", "text": prompt})

        extra_body: dict[str, Any] = {
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }

        last_error: Exception | None = None
        last_message = ""
        raw_response: Any | None = None
        max_attempts = self.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                if self.verbose:
                    logger.info(
                        "VLM request attempt=%s/%s model=%s image_count=%s image_labels=%s",
                        attempt,
                        max_attempts,
                        self.model,
                        len(images),
                        [label for label, _ in images],
                    )
                self._wait_for_rate_limit()
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    extra_body=extra_body,
                    stream=False,
                )
            except Exception as exc:
                last_error = exc
                if _is_rate_limit_error(exc) and attempt < max_attempts:
                    sleep_s = self._rate_limit_retry_sleep_s(exc)
                    logger.warning(
                        "VLM rate limit hit on attempt %s/%s; retrying in %.1fs: %s",
                        attempt,
                        max_attempts,
                        sleep_s,
                        exc,
                    )
                    time.sleep(sleep_s)
                    continue

                logger.warning("VLM request failed on attempt %s/%s: %s", attempt, max_attempts, exc)
                break

            raw_response = completion.model_dump() if hasattr(completion, "model_dump") else str(completion)
            last_message = completion.choices[0].message.content or ""
            response_tail = self._response_tail(last_message)
            try:
                parsed = _parse_score_json(last_message)
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts:
                    sleep_s = self._parse_retry_sleep_s()
                    logger.warning(
                        "VLM response parse failed on attempt %s/%s; response_tail=%r; retrying in %.1fs: %s",
                        attempt,
                        max_attempts,
                        response_tail,
                        sleep_s,
                        exc,
                    )
                    time.sleep(sleep_s)
                    continue

                logger.warning(
                    "VLM response parse failed on final attempt %s/%s; response_tail=%r: %s",
                    attempt,
                    max_attempts,
                    response_tail,
                    exc,
                )
                break

            score = min(1.0, max(0.0, float(parsed.get("score", 0.0))))
            if self.verbose:
                logger.info(
                    "VLM response parsed attempt=%s/%s score=%.3f reason=%r response_tail=%r",
                    attempt,
                    max_attempts,
                    score,
                    str(parsed.get("reason", ""))[:160],
                    response_tail,
                )
            return VLMScore(
                score=score,
                reason=str(parsed.get("reason", "")),
                prompt=prompt,
                raw_response=raw_response,
            )

        response_tail = self._response_tail(last_message)
        reason = f"VLM scoring failed after {max_attempts} attempt(s): {last_error}; response_tail={response_tail!r}"
        return VLMScore(score=0.0, reason=reason, prompt=prompt, raw_response=raw_response)

    def _min_request_interval_s(self) -> float:
        if self.requests_per_minute <= 0:
            return 0.0
        return 60.0 / float(self.requests_per_minute)

    def _wait_for_rate_limit(self) -> None:
        min_interval_s = self._min_request_interval_s()
        if min_interval_s <= 0:
            return
        now = time.monotonic()
        if self._last_request_monotonic is not None:
            elapsed_s = now - self._last_request_monotonic
            sleep_s = min_interval_s - elapsed_s
            if sleep_s > 0:
                if self.verbose:
                    logger.info(
                        "VLM throttle sleeping %.1fs for %.2f requests/minute limit.",
                        sleep_s,
                        self.requests_per_minute,
                    )
                time.sleep(sleep_s)
        self._last_request_monotonic = time.monotonic()

    def _response_tail(self, message: str) -> str:
        if not message:
            return "<empty>"
        return message[-self.log_response_chars :]

    def _parse_retry_sleep_s(self) -> float:
        return max(self.retry_sleep_s, self._min_request_interval_s())

    def _rate_limit_retry_sleep_s(self, exc: Exception) -> float:
        retry_after_s = _retry_after_seconds(exc)
        if retry_after_s is not None:
            return retry_after_s
        return max(self.rate_limit_sleep_s, self.retry_sleep_s, self._min_request_interval_s())

    def build_prompt(
        self,
        *,
        task: str,
        image_labels: list[str],
        robot_state: dict[str, Any] | None,
        task_metadata: dict[str, Any],
    ) -> str:
        image_description = "\n".join(f"{ix}. {label}" for ix, label in enumerate(image_labels, start=1))
        robot_state_json = json.dumps(_to_jsonable(robot_state), indent=2) if robot_state else "null"
        task_metadata_json = json.dumps(_to_jsonable(task_metadata), indent=2)
        if self.prompt_template is not None:
            return self.prompt_template.format(
                task=task,
                image_description=image_description,
                robot_state_json=robot_state_json,
                task_metadata_json=task_metadata_json,
            )
        return (
            f'Task: "{task}"\n\n'
            "You are scoring progress in a robot manipulation task from the supplied images. "
            "The images whose labels start with `reference.target.` are target-object reference images. "
            "They are not current state images; use them only to identify what the target object looks like "
            "in the scene. The images whose labels start with `temporal.scene.` are consecutive whole-scene "
            "views from a successful demonstration. The final `temporal.scene.current` image is the state to "
            "score. Earlier temporal images provide context only. Return one score for the current image, "
            "indicating how close it is to the final successful state. Do not score progress toward a "
            "distractor object unless it matches the task and target reference.\n\n"
            f"Images are supplied in this order:\n{image_description}\n\n"
            "Return JSON only with keys `score` and `reason`.\n"
            "Score meaning: 0.0 = no visible progress toward success, 0.3 = gripper is near or aligned with "
            "the target object, 0.5 = target object is grasped or being lifted, 0.8 = target object is near "
            "or over the goal, 1.0 = task appears complete in the current image."
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install `openai` to use VLM scoring.") from exc

        import os

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.api_key_env}")
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout_s}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client


def _make_composite_frame(
    camera_images: list[tuple[str, np.ndarray]],
    *,
    score: float,
    reason: str,
    local_index: int,
    dataset_index: int,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    tiles: list[Image.Image] = []
    font = ImageFont.load_default()
    for label, image in camera_images:
        tile = Image.fromarray(image).resize((tile_width, tile_height), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile_width, 18), fill=(0, 0, 0))
        draw.text((5, 4), label, fill=(255, 255, 255), font=font)
        tiles.append(tile)

    width = tile_width * len(tiles)
    height = tile_height
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    for ix, tile in enumerate(tiles):
        canvas.paste(tile, (ix * tile_width, 0))

    draw = ImageDraw.Draw(canvas)
    score_text = f"score={score:.3f}  frame={local_index}  idx={dataset_index}"
    reason_text = reason[:110]
    margin = 8
    y_score = height - 34
    y_reason = height - 18
    draw.rectangle((0, height - 42, width, height), fill=(0, 0, 0))
    draw.text((margin, y_score), score_text, fill=(255, 0, 0), font=font)
    draw.text((margin, y_reason), reason_text, fill=(255, 80, 80), font=font)
    return np.asarray(canvas, dtype=np.uint8)


def _save_vlm_inputs(
    *,
    output_dir: Path,
    sample_index: int,
    images: list[tuple[str, np.ndarray]],
) -> list[dict[str, str]]:
    saved: list[dict[str, str]] = []
    frame_dir = output_dir / "vlm_inputs" / f"frame_{sample_index:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for image_ix, (label, image) in enumerate(images):
        path = frame_dir / f"{image_ix:02d}_{_safe_name(label)}.png"
        write_image(image, path)
        saved.append({"label": label, "path": path.relative_to(output_dir).as_posix()})
    return saved


def _write_records(records: list[FrameScoreRecord], output_dir: Path, summary: dict[str, Any]) -> None:
    payload = {
        "summary": _to_jsonable(summary),
        "frames": [_to_jsonable(asdict(record)) for record in records],
    }
    (output_dir / "scores.json").write_text(json.dumps(payload, indent=2))

    with (output_dir / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "episode_index",
                "dataset_index",
                "local_index",
                "frame_index",
                "timestamp",
                "score",
                "reason",
                "annotated_image_path",
                "temporal_context_local_indices",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_index": record.sample_index,
                    "episode_index": record.episode_index,
                    "dataset_index": record.dataset_index,
                    "local_index": record.local_index,
                    "frame_index": record.frame_index,
                    "timestamp": record.timestamp,
                    "score": record.score,
                    "reason": record.reason,
                    "annotated_image_path": record.annotated_image_path,
                    "temporal_context_local_indices": record.temporal_context_local_indices,
                }
            )


def _score_monotonicity(scores: list[float], tolerance: float) -> dict[str, Any]:
    violations = []
    for ix in range(1, len(scores)):
        if scores[ix] + tolerance < scores[ix - 1]:
            violations.append(
                {
                    "prev_index": ix - 1,
                    "index": ix,
                    "prev_score": scores[ix - 1],
                    "score": scores[ix],
                }
            )
    return {
        "first_score": scores[0] if scores else None,
        "last_score": scores[-1] if scores else None,
        "delta": (scores[-1] - scores[0]) if len(scores) >= 2 else None,
        "monotonic_violations": violations,
        "monotonic_violation_count": len(violations),
    }


def _ensure_episode_files(
    args: argparse.Namespace,
    meta: LeRobotDatasetMetadata,
    *,
    episode_index: int,
) -> None:
    episode = meta.episodes[episode_index]
    from_idx = int(episode["dataset_from_index"])
    to_idx = int(episode["dataset_to_index"])
    data_path: Path | None = None
    with contextlib.suppress(FileNotFoundError):
        data_path = _find_episode_data_path(meta.root, dataset_from_index=from_idx, dataset_to_index=to_idx)

    data_patterns = ["data/**/*.parquet"] if data_path is None or args.force_cache_sync else []
    video_patterns: list[str] = []
    if data_path is not None and args.download_videos:
        for video_key in meta.video_keys:
            inferred = _infer_video_path_from_data_path(meta.root, data_path, video_key)
            if args.force_cache_sync or not inferred.exists():
                video_patterns.append(str(inferred.relative_to(meta.root)))

    files = sorted(set(data_patterns + video_patterns))
    if not files:
        return

    logger.info("Downloading %d dataset file pattern(s) for episode=%s.", len(files), episode_index)
    if args.dataset_root is None:
        meta.root = Path(
            snapshot_download(
                args.dataset_repo_id,
                repo_type="dataset",
                revision=meta.revision,
                cache_dir=HF_LEROBOT_HUB_CACHE,
                allow_patterns=files,
            )
        )
    else:
        args.dataset_root.mkdir(exist_ok=True, parents=True)
        snapshot_download(
            args.dataset_repo_id,
            repo_type="dataset",
            revision=meta.revision,
            local_dir=args.dataset_root,
            allow_patterns=files,
        )
        meta.root = args.dataset_root

    if data_path is None or args.force_cache_sync:
        data_path = _find_episode_data_path(meta.root, dataset_from_index=from_idx, dataset_to_index=to_idx)

    expected_files: list[Path] = [data_path]
    if args.download_videos:
        expected_files.extend(
            _infer_video_path_from_data_path(meta.root, data_path, video_key) for video_key in meta.video_keys
        )
    still_missing = [path for path in expected_files if not path.exists()]
    if still_missing:
        raise FileNotFoundError(
            "Could not find selected episode files after download:\n"
            + "\n".join(str(path) for path in still_missing[:20])
        )


def _load_episode_reader(
    args: argparse.Namespace,
    meta: LeRobotDatasetMetadata,
    episode_index: int,
) -> EpisodeFrameReader:
    _ensure_episode_files(args, meta, episode_index=episode_index)
    return EpisodeFrameReader(
        meta=meta,
        episode_index=episode_index,
        video_backend=args.video_backend,
    )


def _temporal_context_indices(
    sampled_indices: list[int],
    sample_index: int,
    *,
    context_size: int,
) -> list[int]:
    context_size = max(1, int(context_size))
    start = max(0, sample_index - context_size + 1)
    return sampled_indices[start : sample_index + 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_repo_id", default="HuggingFaceVLA/libero")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument(
        "--camera_keys",
        default="observation.images.image,observation.images.image2",
        help="'all' or comma-separated camera keys.",
    )
    parser.add_argument(
        "--scene_camera_key",
        default="observation.images.image",
        help="Whole-scene camera used for temporal VLM context.",
    )
    parser.add_argument(
        "--temporal_context",
        type=int,
        default=3,
        help="Number of consecutive whole-scene frames to send to the VLM, including current.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--video_path", type=Path, default=None)
    parser.add_argument("--video_fps", type=int, default=2)
    parser.add_argument("--tile_width", type=int, default=256)
    parser.add_argument("--tile_height", type=int, default=256)
    parser.add_argument("--save_vlm_inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video_backend", default=None)
    parser.add_argument("--download_videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force_cache_sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force re-downloading selected episode data/video files from the Hub.",
    )
    parser.add_argument("--reference_image_dir", type=Path, default=TREE_SEARCH_DIR / "references")
    parser.add_argument("--reference_image_paths", default=None)
    parser.add_argument("--reference_image_glob", default="**/*")
    parser.add_argument("--reference_image_max_size", type=int, default=512)
    parser.add_argument("--reference_filter_by_task", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vlm_api_url", default=None)
    parser.add_argument("--vlm_base_url", default=None)
    parser.add_argument("--vlm_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--vlm_model", default=None)
    parser.add_argument("--vlm_max_tokens", type=int, default=512)
    parser.add_argument("--vlm_temperature", type=float, default=0.6)
    parser.add_argument("--vlm_top_p", type=float, default=0.95)
    parser.add_argument("--vlm_presence_penalty", type=float, default=0.0)
    parser.add_argument("--vlm_top_k", type=int, default=20)
    parser.add_argument("--vlm_min_p", type=float, default=0.0)
    parser.add_argument("--vlm_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--vlm_timeout_s", type=float, default=30.0)
    parser.add_argument(
        "--vlm_requests_per_minute",
        type=float,
        default=10.0,
        help="Throttle VLM request starts. Use 0 to disable throttling.",
    )
    parser.add_argument(
        "--vlm_max_retries",
        type=int,
        default=3,
        help="Retry count for 429 rate limits and malformed/empty VLM responses.",
    )
    parser.add_argument(
        "--vlm_retry_sleep_s",
        type=float,
        default=6.0,
        help="Minimum sleep before retrying a malformed/empty VLM response.",
    )
    parser.add_argument(
        "--vlm_rate_limit_sleep_s",
        type=float,
        default=60.0,
        help="Fallback sleep after HTTP 429 when the server does not provide Retry-After.",
    )
    parser.add_argument(
        "--vlm_log_response_chars",
        type=int,
        default=200,
        help="Number of response characters to include in VLM success/failure logs.",
    )
    parser.add_argument(
        "--vlm_verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log every VLM request, throttle sleep, and successful response tail.",
    )
    parser.add_argument("--prompt_template_path", type=Path, default=None)
    parser.add_argument("--monotonic_tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "annotated_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.video_path is None:
        args.video_path = args.output_dir / "scored_sequence.mp4"
    args.video_path.parent.mkdir(parents=True, exist_ok=True)

    task_metadata = _get_libero_task(args.suite, args.task_id)
    task_language = str(task_metadata["language"])

    logger.info("Loading dataset metadata: %s", args.dataset_repo_id)
    meta = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
    )
    episode_index = _find_episode_for_task(meta, task_metadata=task_metadata, episode_index=args.episode_index)
    episode = meta.episodes[episode_index]
    from_idx = int(episode["dataset_from_index"])
    to_idx = int(episode["dataset_to_index"])
    length = max(0, to_idx - from_idx)
    local_indices = _sample_local_indices(
        length,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    logger.info(
        "Selected episode=%s task='%s' length=%s sampled_frames=%s",
        episode_index,
        task_language,
        length,
        len(local_indices),
    )

    reader = _load_episode_reader(args, meta, episode_index)
    length = len(reader)
    local_indices = _sample_local_indices(
        length,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    logger.info("Loaded direct episode reader with %s frame rows.", length)
    camera_keys = _resolve_camera_keys(reader, args.camera_keys)
    if args.scene_camera_key not in reader.camera_keys:
        raise SystemExit(
            f"Unknown scene_camera_key '{args.scene_camera_key}'. Available camera keys: {reader.camera_keys}"
        )
    reference_images = _load_reference_images(
        reference_image_dir=args.reference_image_dir,
        reference_image_paths=args.reference_image_paths,
        reference_image_glob=args.reference_image_glob,
        reference_image_max_size=args.reference_image_max_size,
        task_id=args.task_id,
        filter_by_task=args.reference_filter_by_task,
    )
    logger.info("Loaded %d reference image(s).", len(reference_images))

    scorer = VLMFrameScorer(args)
    records: list[FrameScoreRecord] = []
    video_frames: list[np.ndarray] = []

    for sample_index, local_index in enumerate(local_indices):
        item = reader[local_index]
        dataset_index = int(_coerce_scalar(item.get("index")) or (from_idx + local_index))
        frame_index = _coerce_scalar(item.get("frame_index"))
        timestamp = _coerce_scalar(item.get("timestamp"))
        task = str(item.get("task") or task_language)

        current_images: list[tuple[str, np.ndarray]] = []
        for camera_key in camera_keys:
            current_images.append((f"current.{camera_key}", _image_to_uint8_hwc(item[camera_key])))

        temporal_indices = _temporal_context_indices(
            local_indices,
            sample_index,
            context_size=args.temporal_context,
        )
        temporal_images: list[tuple[str, np.ndarray]] = []
        for context_ix, context_local_index in enumerate(temporal_indices):
            context_item = item if context_local_index == local_index else reader[context_local_index]
            label = (
                f"temporal.scene.current.{args.scene_camera_key}"
                if context_local_index == local_index
                else f"temporal.scene.prev_{len(temporal_indices) - context_ix - 1}.{args.scene_camera_key}"
            )
            temporal_images.append((label, _image_to_uint8_hwc(context_item[args.scene_camera_key])))

        vlm_images = [*reference_images, *temporal_images]
        robot_state = _extract_robot_state(item)
        score = scorer.score(
            task=task,
            images=vlm_images,
            robot_state=robot_state,
            task_metadata=task_metadata,
        )

        annotated = _make_composite_frame(
            current_images,
            score=score.score,
            reason=score.reason,
            local_index=local_index,
            dataset_index=dataset_index,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
        )
        annotated_path = frames_dir / f"frame_{sample_index:04d}_idx_{dataset_index:08d}.png"
        write_image(annotated, annotated_path)
        video_frames.append(annotated)
        saved_inputs = (
            _save_vlm_inputs(output_dir=args.output_dir, sample_index=sample_index, images=vlm_images)
            if args.save_vlm_inputs
            else []
        )

        record = FrameScoreRecord(
            sample_index=sample_index,
            episode_index=episode_index,
            dataset_index=dataset_index,
            local_index=local_index,
            frame_index=int(frame_index) if frame_index is not None else None,
            timestamp=float(timestamp) if timestamp is not None else None,
            task=task,
            score=score.score,
            reason=score.reason,
            annotated_image_path=annotated_path.relative_to(args.output_dir).as_posix(),
            vlm_input_images=saved_inputs,
            temporal_context_local_indices=temporal_indices,
            prompt=score.prompt,
            raw_response=score.raw_response,
        )
        records.append(record)
        logger.info(
            "sample=%s local=%s dataset=%s score=%.3f reason=%s",
            sample_index,
            local_index,
            dataset_index,
            score.score,
            score.reason,
        )

    if not video_frames:
        raise SystemExit("No frames were sampled.")

    write_video(args.video_path, video_frames, fps=args.video_fps)
    summary = {
        "dataset_repo_id": args.dataset_repo_id,
        "suite": args.suite,
        "task_id": args.task_id,
        "task": task_language,
        "episode_index": episode_index,
        "episode_dataset_from_index": from_idx,
        "episode_dataset_to_index": to_idx,
        "camera_keys": camera_keys,
        "scene_camera_key": args.scene_camera_key,
        "temporal_context": args.temporal_context,
        "reference_image_labels": [label for label, _ in reference_images],
        "video_path": str(args.video_path),
        **_score_monotonicity([record.score for record in records], args.monotonic_tolerance),
    }
    _write_records(records, args.output_dir, summary)
    logger.info("Wrote video: %s", args.video_path)
    logger.info("Wrote scores: %s", args.output_dir / "scores.json")
    logger.info(
        "Score delta=%s monotonic_violations=%s",
        summary["delta"],
        summary["monotonic_violation_count"],
    )


if __name__ == "__main__":
    main()
