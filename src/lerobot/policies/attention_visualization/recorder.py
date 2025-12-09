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


def _annotate_overlay(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    annotated = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    pad = 8
    x0, y0 = 10, 10
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x1 = x0 + text_w + 2 * pad
    y1 = y0 + text_h + 2 * pad
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    text_org = (x0 + pad, y0 + pad + text_h - baseline)
    # 白文字＋薄い影で視認性アップ
    cv2.putText(annotated, text, (text_org[0] + 1, text_org[1] + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(annotated, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return annotated


def _draw_label(img: np.ndarray, text: str, x0: int, y0: int = 10) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    pad = 8
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x1 = x0 + text_w + 2 * pad
    y1 = y0 + text_h + 2 * pad
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    text_org = (x0 + pad, y0 + pad + text_h - baseline)
    cv2.putText(img, text, (text_org[0] + 1, text_org[1] + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


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
        self._cam_writers: dict[str, AttnVideoRecorder] = {}
        self._cam_indices: dict[str, int] = {}
        self._episode_buffer: EpisodeBuffer | None = None
        self._policy = policy
        self._chunk_actions: list[Any] = []
        self._chunk_start_idx: int | None = None
        self._chunk_start_ts: float | None = None
        self._chunk_attn: dict[str, Any] | None = None
        self._chunk_scores: dict[str, Any] | None = None
        self._chunk_size = getattr(policy.config, "n_action_steps", 1)

    def start_episode(self, episode_idx: int) -> None:
        if self.context is None:
            return
        self._episode_buffer = EpisodeBuffer(episode_idx=episode_idx)
        video_name = f"attn_all_cameras_episode_{episode_idx}.mp4"
        self._writer = AttnVideoRecorder(self.output_root / video_name, fps=self.fps)
        self._cam_writers = {}
        self._cam_indices = {}
        self._chunk_actions = []
        self._chunk_start_idx = None
        self._chunk_start_ts = None
        self._chunk_attn = None
        self._chunk_scores = None

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
        attn_scores: dict[str, dict[str, float]] = {}
        combined_overlay = None
        if attn_samples:
            # まずはテキスト付きの個別オーバーレイを作成（後段で使うため保持）
            for sample in attn_samples:
                disp_key = sample.camera_key.split(".")[-1]
                attn_scores[sample.camera_key] = {
                    "max_raw": sample.raw_max,
                    "mean_raw": sample.raw_mean,
                    "sum_raw": sample.raw_sum,
                }
                text = f"{disp_key}: sum {sample.raw_sum:.2f} mean {sample.raw_mean:.3f}"
                overlays.append(_annotate_overlay(sample.overlay_bgr, text))
                overlay_keys.append(sample.camera_key)

            # 生アテンション値で2枚を横結合し、全体で正規化したヒートマップを生成
            raw_maps_resized: list[np.ndarray] = []
            ordered_images: list[np.ndarray] = []
            x_offsets: list[int] = []
            x_offset = 0
            for sample in attn_samples:
                img = images_bgr.get(sample.camera_key)
                if img is None:
                    continue
                ordered_images.append(img)
                x_offsets.append(x_offset)
                x_offset += img.shape[1]
                raw_map = sample.attention_raw_patches
                raw_maps_resized.append(
                    cv2.resize(raw_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                )

            if ordered_images and raw_maps_resized:
                combined_img = _compose_side_by_side(ordered_images)
                concat_map = np.concatenate(raw_maps_resized, axis=1)
                concat_map = concat_map - concat_map.min()
                maxv = concat_map.max()
                if maxv > 0:
                    concat_map = concat_map / maxv
                attn_uint8 = (concat_map * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
                combined_overlay = cv2.addWeighted(combined_img, 0.5, heatmap, 0.5, 0.0)
                # 各カメラ領域の左上にラベルを描画
                for sample, x0 in zip(attn_samples, x_offsets):
                    disp_key = sample.camera_key.split(".")[-1]
                    text = f"{disp_key}: sum {sample.raw_sum:.2f} mean {sample.raw_mean:.3f}"
                    _draw_label(combined_overlay, text, x0 + 10, 10)
        else:
            ordered_keys = [
                key for key in getattr(self._policy.config, "image_features", {}) if key in images_bgr
            ]
            overlay_keys = ordered_keys[:2] if ordered_keys else list(images_bgr.keys())[:2]
            overlays = [images_bgr[key] for key in overlay_keys if key in images_bgr]

        combined = combined_overlay if combined_overlay is not None else _compose_side_by_side(overlays)
        if combined is not None and self._writer is not None:
            self._writer.add_frame(combined)

        # カメラ別のオーバーレイ／ヒート動画を書き出す
        for sample in attn_samples:
            disp_key = sample.camera_key.split(".")[-1]
            safe_key = disp_key.replace(".", "_")
            cam_idx = self._cam_indices.setdefault(sample.camera_key, len(self._cam_indices))
            # オーバーレイ
            if sample.camera_key not in self._cam_writers:
                path = self.output_root / f"attn_camera{cam_idx}_episode_{self._episode_buffer.episode_idx}.mp4"
                self._cam_writers[sample.camera_key] = AttnVideoRecorder(path, self.fps)
            self._cam_writers[sample.camera_key].add_frame(sample.overlay_bgr)

        # チャンク単位でログをまとめる
        chunk_step = frame_idx % self._chunk_size if self._chunk_size > 0 else 0
        if chunk_step == 0 or self._chunk_start_idx is None:
            self._chunk_actions = []
            self._chunk_start_idx = frame_idx
            self._chunk_start_ts = timestamp
            self._chunk_attn = {
                sample.camera_key: {
                    "attention_patches": _to_serializable(sample.attention_patches),
                }
                for sample in attn_samples
            }
            self._chunk_scores = attn_scores

        self._chunk_actions.append(_to_serializable(action_values))

        chunk_end = (chunk_step == self._chunk_size - 1) or (self._chunk_size <= 1)
        if chunk_end and self._chunk_start_idx is not None:
            frame_record = {
                "frame_idx": self._chunk_start_idx,
                "timestamp": self._chunk_start_ts,
                "actions": self._chunk_actions,
                "attention": self._chunk_attn or {},
                "attention_score": self._chunk_scores or {},
            }
            self._episode_buffer.frames.append(frame_record)
            self._chunk_actions = []
            self._chunk_start_idx = None
            self._chunk_start_ts = None
            self._chunk_attn = None
            self._chunk_scores = None

    def finish_episode(self) -> None:
        if self.context is None or self._episode_buffer is None:
            return
        if self._chunk_actions:
            frame_record = {
                "frame_idx": self._chunk_start_idx,
                "timestamp": self._chunk_start_ts,
                "actions": self._chunk_actions,
                "attention": self._chunk_attn or {},
                "attention_score": self._chunk_scores or {},
            }
            self._episode_buffer.frames.append(frame_record)
            self._chunk_actions = []
            self._chunk_start_idx = None
            self._chunk_start_ts = None
            self._chunk_attn = None
            self._chunk_scores = None
        # 先に JSON を書き出す
        values_path = self.output_root / f"episode_values_{self._episode_buffer.episode_idx}.json"
        payload = {
            "repo_id": self.repo_id,
            "episode": self._episode_buffer.episode_idx,
            "fps": self.fps,
            "frames": self._episode_buffer.frames,
        }
        values_path.write_text(json.dumps(payload), encoding="utf-8")

        if self._writer is not None:
            self._writer.close()
            self._writer = None
        for rec in self._cam_writers.values():
            rec.close()
        self._cam_writers = {}
        self._episode_buffer = None

    def finalize(self) -> None:
        if self.context is None:
            return
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        for rec in self._cam_writers.values():
            rec.close()
        self._cam_writers = {}
