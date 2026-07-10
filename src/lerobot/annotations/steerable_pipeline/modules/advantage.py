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

"""Advantage scoring module for RECAP.

Computes per-frame advantage values using a frozen distributional value function,
binarizes them into improvement indicators (I_t), and emits ``style="advantage"``
persistent rows for policy conditioning.

Paper reference: pi*0.6, Section IV-B and Appendix F.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..config import AdvantageConfig
from ..frames import VideoFrameProvider, null_provider
from ..reader import EpisodeRecord
from ..staging import EpisodeStaging

logger = logging.getLogger(__name__)


@dataclass
class AdvantageModule:
    """Compute advantage indicators and emit persistent annotation rows.

    The module loads a frozen distributional value function and scores each
    frame in an episode. Advantages are binarized into ``positive``/``negative``
    indicators using a per-task threshold, then written as ``style="advantage"``
    persistent rows into the staging area.

    Requires ``mc_return`` column in the dataset (from lerobot-compute-returns).
    """

    config: AdvantageConfig
    frame_provider: Any = None
    _model: Any = field(default=None, init=False, repr=False)
    _preprocessor: Any = field(default=None, init=False, repr=False)
    _threshold: float | None = field(default=None, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the frozen value function on first use."""
        if self._model is not None:
            return

        from lerobot.rewards import (
            make_reward_model,
            make_reward_model_config,
            make_reward_pre_post_processors,
        )

        cfg = make_reward_model_config(
            "distributional_value_function",
            pretrained_path=self.config.value_function_path,
            device=self.config.device,
        )
        self._model = make_reward_model(cfg)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._preprocessor, _ = make_reward_pre_post_processors(cfg)
        logger.info("Loaded frozen VF from %s on %s", self.config.value_function_path, self.config.device)

    def compute_advantages_for_episode(self, record: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        """Compute raw advantage values for all frames in an episode.

        Returns:
            (advantages, intervention_mask) both shape [num_frames].
            advantages[t] = A_t, intervention_mask[t] = True if frame is intervention.
        """
        self._ensure_model_loaded()

        df = record.frames_df()
        num_frames = len(df)

        mc_return_key = self.config.mc_return_key
        if mc_return_key not in df.columns:
            raise KeyError(
                f"Column '{mc_return_key}' not found in episode {record.episode_index}. "
                "Run lerobot-compute-returns first."
            )

        mc_returns = df[mc_return_key].values.astype(np.float32)

        intervention_mask = np.zeros(num_frames, dtype=bool)
        if self.config.intervention_key in df.columns:
            intervention_mask = df[self.config.intervention_key].values.astype(bool)

        # Skip VF inference on intervention frames — they're always "positive"
        # regardless of advantage value, so V(s_t) is never used for them.
        skip_mask = intervention_mask if self.config.force_positive_on_intervention else None
        values = self._compute_values(record, skip_mask=skip_mask)

        if self.config.n_step is None:
            advantages = mc_returns - values
        else:
            advantages = self._compute_n_step_advantages(mc_returns, values, record, n=self.config.n_step)

        return advantages, intervention_mask

    def _compute_values(self, record: EpisodeRecord, skip_mask: np.ndarray | None = None) -> np.ndarray:
        """Run frozen VF over all frames to get V(s_t) predictions.

        Supports both image datasets (columns in parquet) and video datasets
        (frames decoded from .mp4 via the shared VideoFrameProvider).

        Args:
            record: Episode data.
            skip_mask: Optional boolean mask [num_frames]. Frames where True are
                skipped (left as 0.0) to avoid unnecessary inference.
        """
        df = record.frames_df()
        num_frames = len(df)
        values = np.zeros(num_frames, dtype=np.float32)

        # Determine which frame indices actually need inference
        infer_indices = np.where(~skip_mask)[0] if skip_mask is not None else np.arange(num_frames)
        if len(infer_indices) == 0:
            return values

        # Try parquet image columns first, fall back to video decoding
        image_key = self._resolve_image_key(df)
        video_frames = None

        if image_key is None:
            image_key, video_frames = self._decode_video_frames(record, infer_indices)
            if image_key is None:
                logger.warning(
                    "No image/video key found for episode %d; returning zero values.", record.episode_index
                )
                return values

        task_text = record.episode_task

        for batch_start in range(0, len(infer_indices), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(infer_indices))
            batch_indices = infer_indices[batch_start:batch_end]
            batch_images = []

            for local_i in range(len(batch_indices)):
                if video_frames is not None:
                    img_tensor = video_frames[batch_start + local_i].float()
                else:
                    idx = batch_indices[local_i]
                    img_val = df.iloc[idx][image_key]
                    if isinstance(img_val, np.ndarray):
                        img_tensor = torch.from_numpy(img_val).float()
                    elif isinstance(img_val, torch.Tensor):
                        img_tensor = img_val.float()
                    else:
                        img_tensor = torch.zeros(3, 224, 224)
                batch_images.append(img_tensor)

            batch_images_tensor = torch.stack(batch_images)
            batch_size = batch_images_tensor.shape[0]

            raw_batch = {
                image_key: batch_images_tensor,
                "task": [task_text] * batch_size,
            }

            processed = self._preprocessor(raw_batch)

            with torch.no_grad():
                v_values = self._model.compute_reward(processed)

            values[batch_indices] = v_values.cpu().numpy()

        return values

    def _decode_video_frames(
        self, record: EpisodeRecord, infer_indices: np.ndarray
    ) -> tuple[str | None, torch.Tensor | None]:
        """Decode video frames using the existing VideoFrameProvider infrastructure.

        Returns (image_key, decoded_frames_tensor) or (None, None) on failure.
        """
        dataset_root = record.data_path.parent.parent.parent

        if not hasattr(self, "_frame_provider") or self._frame_provider is None:
            try:
                self._frame_provider = VideoFrameProvider(root=dataset_root)
            except Exception:
                self._frame_provider = null_provider()

        if not self._frame_provider.camera_keys:
            return None, None

        camera_key = self._frame_provider.camera_keys[0]
        timestamps = [float(record.frame_timestamps[i]) for i in infer_indices]

        frames = self._frame_provider.frames_at(record, timestamps, camera_key=camera_key)
        if not frames:
            return None, None

        frames_tensor = torch.stack(frames)
        return camera_key, frames_tensor

    def _compute_n_step_advantages(
        self, mc_returns: np.ndarray, values: np.ndarray, record: EpisodeRecord, n: int
    ) -> np.ndarray:
        """Compute N-step advantage: A_t = Σ r_{t:t+N-1} + V(s_{t+N}) - V(s_t).

        When t+N exceeds episode length, truncates to MC (uses mc_return directly).
        """
        num_frames = len(values)
        advantages = np.zeros(num_frames, dtype=np.float32)

        for t in range(num_frames):
            if t + n >= num_frames:
                advantages[t] = mc_returns[t] - values[t]
            else:
                n_step_return = mc_returns[t] - mc_returns[t + n]
                advantages[t] = n_step_return + values[t + n] - values[t]

        return advantages

    def _resolve_image_key(self, df) -> str | None:
        """Find the first image observation key in the dataframe columns."""
        for col in df.columns:
            if col.startswith("observation.images."):
                return col
        return None

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        """Score one episode and write advantage rows to staging."""
        if self.config.constant_value:
            self._run_constant_mode(record, staging)
            return

        if not self.config.value_function_path:
            logger.warning("No value_function_path or constant_value configured; skipping advantage scoring.")
            return

        advantages, intervention_mask = self.compute_advantages_for_episode(record)
        num_frames = len(advantages)

        threshold = self._compute_threshold(advantages, intervention_mask)

        rng = np.random.default_rng(seed=self.config.seed + record.episode_index)

        rows: list[dict[str, Any]] = []
        for t in range(num_frames):
            if rng.random() < self.config.dropout_rate:
                continue

            if (
                self.config.force_positive_on_intervention
                and intervention_mask[t]
                or advantages[t] > threshold
            ):
                indicator = "positive"
            else:
                indicator = "negative"

            timestamp = float(record.frame_timestamps[t]) if t < len(record.frame_timestamps) else 0.0

            rows.append(
                {
                    "role": "user",
                    "content": indicator,
                    "style": "advantage",
                    "timestamp": timestamp,
                    "camera": None,
                    "tool_calls": None,
                }
            )

        staging.write("advantage", rows)
        logger.debug(
            "Episode %d: %d/%d frames scored (threshold=%.4f, %d positive, %d negative)",
            record.episode_index,
            len(rows),
            num_frames,
            threshold,
            sum(1 for r in rows if r["content"] == "positive"),
            sum(1 for r in rows if r["content"] == "negative"),
        )

    def _run_constant_mode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        """Emit a fixed advantage value for every frame (with dropout for CFG)."""
        num_frames = len(record.frame_timestamps)
        rng = np.random.default_rng(seed=self.config.seed + record.episode_index)

        rows: list[dict[str, Any]] = []
        for t in range(num_frames):
            if rng.random() < self.config.dropout_rate:
                continue

            rows.append(
                {
                    "role": "user",
                    "content": self.config.constant_value,
                    "style": "advantage",
                    "timestamp": float(record.frame_timestamps[t]),
                    "camera": None,
                    "tool_calls": None,
                }
            )

        staging.write("advantage", rows)
        logger.debug(
            "Episode %d: %d/%d frames labeled constant '%s' (dropout=%.2f)",
            record.episode_index,
            len(rows),
            num_frames,
            self.config.constant_value,
            self.config.dropout_rate,
        )

    def _compute_threshold(self, advantages: np.ndarray, intervention_mask: np.ndarray) -> float:
        """Compute the binarization threshold as the configured percentile of advantages."""
        non_intervention = advantages[~intervention_mask] if intervention_mask.any() else advantages
        if len(non_intervention) == 0:
            return 0.0
        return float(np.percentile(non_intervention, self.config.threshold_percentile * 100))
