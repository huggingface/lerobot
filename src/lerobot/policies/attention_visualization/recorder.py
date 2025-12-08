from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.policies.attention_visualization.hooks import AttentionSample, resolve_attention_context
from lerobot.policies.pretrained import PreTrainedPolicy


def _to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return repr(value)


def _extract_images_from_obs_frame(observation_frame: dict[str, Any]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for key, value in observation_frame.items():
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            continue
        if arr.ndim != 3:
            continue
        img = arr
        if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = np.ascontiguousarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images[key] = img_bgr
    return images


def _compose_side_by_side(frames: list[np.ndarray]) -> np.ndarray | None:
    if len(frames) == 0:
        return None
    if len(frames) == 1:
        return frames[0]
    target_height = max(frame.shape[0] for frame in frames)
    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        if h != target_height:
            scale = target_height / h
            frame = cv2.resize(frame, (int(w * scale), target_height), interpolation=cv2.INTER_LINEAR)
        resized.append(frame)
    return np.concatenate(resized, axis=1)


@dataclass
class EpisodeBuffer:
    episode_idx: int
    frames: list[dict[str, Any]] = field(default_factory=list)


class AttnVideoRecorder:
    def __init__(self, output_path: Path, fps: int):
        self.output_path = Path(output_path)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (w, h))
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class AttentionRecordingManager:
    def __init__(self, policy: PreTrainedPolicy, output_root: Path, repo_id: str, fps: int):
        self.context = resolve_attention_context(policy)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.fps = fps
        self._writer: AttnVideoRecorder | None = None
        self._episode_buffer: EpisodeBuffer | None = None
        self._episodes: list[EpisodeBuffer] = []
        self._policy = policy

    def start_episode(self, episode_idx: int) -> None:
        if self.context is None:
            return
        self._episode_buffer = EpisodeBuffer(episode_idx=episode_idx)
        video_name = f"{self.repo_id.replace('/', '_')}_ep{episode_idx:06d}.mp4"
        self._writer = AttnVideoRecorder(self.output_root / video_name, fps=self.fps)

    def log_frame(
        self,
        observation_frame: dict[str, Any],
        action_values: Any,
        frame_idx: int,
        timestamp: float | None = None,
    ) -> None:
        if self.context is None or self._episode_buffer is None:
            return

        images_bgr = _extract_images_from_obs_frame(observation_frame)
        attn_samples = self.context.collect_attentions(self._policy, images_bgr)

        overlays: list[np.ndarray] = []
        overlay_keys: list[str] = []
        if attn_samples:
            overlays = [sample.overlay_bgr for sample in attn_samples]
            overlay_keys = [sample.camera_key for sample in attn_samples]
        else:
            ordered_keys = [
                key for key in getattr(self._policy.config, "image_features", {}) if key in images_bgr
            ]
            overlay_keys = ordered_keys[:2] if ordered_keys else list(images_bgr.keys())[:2]
            overlays = [images_bgr[key] for key in overlay_keys if key in images_bgr]

        combined = _compose_side_by_side(overlays)
        if combined is not None and self._writer is not None:
            self._writer.add_frame(combined)

        attn_entries = {
            sample.camera_key: {
                "attention_patches": _to_serializable(sample.attention_patches),
            }
            for sample in attn_samples
        }

        frame_record = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "action": _to_serializable(action_values),
            "attention": attn_entries,
        }
        self._episode_buffer.frames.append(frame_record)

    def finish_episode(self) -> None:
        if self.context is None or self._episode_buffer is None:
            return
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._episodes.append(self._episode_buffer)
        self._episode_buffer = None

    def finalize(self) -> None:
        if self.context is None:
            return
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        payload = {
            "repo_id": self.repo_id,
            "episodes": [
                {"episode": ep.episode_idx, "frames": ep.frames, "fps": self.fps} for ep in self._episodes
            ],
        }
        values_path = self.output_root / "values.json"
        values_path.write_text(json.dumps(payload), encoding="utf-8")
