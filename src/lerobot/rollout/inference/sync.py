# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy, unpack_action_output
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, RelativeActionsProcessorStep

from .base import InferenceEngine

logger = logging.getLogger(__name__)


# Relative-action support: a predicted chunk of offsets is anchored to the robot
# state at prediction time, but the sync engine reruns the pre/post pipeline every
# tick, so ``RelativeActionsProcessorStep`` would re-anchor cached actions to the
# current (moved) state and drift through the chunk. We pin the anchor per chunk:
# a probe on the policy's public ``predict_action_chunk`` flags the ticks that
# predict a fresh chunk; on the others the engine restores the anchor the relative
# step overwrote. ``select_action`` stays on the hot path, so per-tick side effects
# (e.g. LingBot-VA keyframe feedback) are preserved.


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a
    CPU action tensor reordered to match the dataset action keys.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
        visualize_predictions: bool = False,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type

        # Find an enabled RelativeActionsProcessorStep to pin its anchor per chunk
        # (see module comment), mirroring the RTC engine.
        self._relative_step = next(
            (
                s
                for s in getattr(preprocessor, "steps", ())
                if isinstance(s, RelativeActionsProcessorStep) and s.enabled
            ),
            None,
        )
        # Set by the probe for the current tick / ever, respectively.
        self._chunk_predicted = False
        self._ever_predicted_chunk = False
        self._original_predict_action_chunk = None  # set while the probe is installed
        if self._relative_step is not None:
            # ``action_names`` is optional on the step; fill it lazily from the
            # policy/dataset so the relative<->absolute mask is built correctly. This is
            # a deliberate engine->step side effect (the step is configured by its consumer).
            if self._relative_step.action_names is None:
                cfg_names = getattr(policy.config, "action_feature_names", None)
                self._relative_step.action_names = list(cfg_names) if cfg_names else list(ordered_action_keys)
            self._install_chunk_probe()
            logger.info("Relative actions enabled: chunk anchor pinned per predicted chunk")

        # Intermediate-prediction visualization (e.g. a world model's imagined video). When on,
        # ``get_action`` requests predictions and keeps the current chunk's frame stacks; a playhead
        # (``get_intermediate_predictions``) advances one step per tick, paced across the chunk's tick
        # span so the imagined clip stays wall-clock aligned with execution.
        self._visualize_predictions = visualize_predictions
        self._pred_stacks: dict = {}  # key -> [T, H, W, 3] frame stack for the current chunk
        self._pred_cursor = 0  # ticks elapsed since the current chunk's frames arrived
        self._ticks_per_chunk = getattr(getattr(policy, "config", None), "chunk_size", None)

        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d, visualize_predictions=%s)",
            self._device,
            len(ordered_action_keys),
            self._visualize_predictions,
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        # Undo the probe so the policy object isn't left permanently patched
        # (it may outlive this engine or be reused by another).
        if self._original_predict_action_chunk is not None:
            self._policy.predict_action_chunk = self._original_predict_action_chunk
            self._original_predict_action_chunk = None
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        # New episode: the next tick predicts a fresh chunk and re-anchors.
        self._chunk_predicted = False
        self._ever_predicted_chunk = False
        self._pred_stacks = {}
        self._pred_cursor = 0

    def _install_chunk_probe(self) -> None:
        """Wrap the policy's public ``predict_action_chunk`` so we learn which ticks
        predict a fresh chunk (when the anchor must advance) without introspecting any
        private action queue. Chunking policies call it from ``select_action``.

        Wraps whatever callable is currently bound (e.g. an already-``torch.compile``d
        one, since ``build_rollout_context`` compiles before building the engine); undone
        in ``stop()``."""
        self._original_predict_action_chunk = self._policy.predict_action_chunk
        inner = self._original_predict_action_chunk

        def probe(*args, **kwargs):
            self._chunk_predicted = True
            self._ever_predicted_chunk = True
            return inner(*args, **kwargs)

        self._policy.predict_action_chunk = probe

    def get_intermediate_predictions(self) -> dict | None:
        """Serve one imagined frame per key for this tick, advancing the playhead.

        Maps the current chunk's ``T`` decoded frames onto its ``ticks_per_chunk`` control ticks so
        the imagined video plays back in step with execution (falls back to one frame/tick, clamped,
        when the chunk's tick span is unknown). Returns ``None`` until a chunk with frames arrives.
        """
        if not self._pred_stacks:
            return None
        tick = self._pred_cursor
        span = self._ticks_per_chunk
        out: dict = {}
        for key, stack in self._pred_stacks.items():
            n = len(stack)
            if n == 0:
                continue
            idx = round(tick / (span - 1) * (n - 1)) if span and span > 1 else tick
            idx = min(max(idx, 0), n - 1)
            frame = stack[idx]
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            out[key] = frame
        self._pred_cursor += 1
        return out or None

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if obs_frame is None:
            return None
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        # Snapshot the chunk anchor before the preprocessor overwrites it with this
        # tick's state; restore it below if this tick only served a cached action.
        # ``clone`` so the snapshot survives even if the cached tensor is ever mutated
        # in place (today it is only rebound, but the copy is cheap for a state vector).
        anchor_before = None
        if self._relative_step is not None:
            cached = self._relative_step.get_cached_state()
            anchor_before = cached.clone() if cached is not None else None
        self._chunk_predicted = False
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            if self._visualize_predictions:
                action, predictions = unpack_action_output(
                    self._policy.select_action(observation, return_intermediate_predictions=True)
                )
                if predictions:
                    # A fresh chunk was predicted this tick — store its frame stacks and restart the playhead.
                    self._pred_stacks = predictions
                    self._pred_cursor = 0
            else:
                action = self._policy.select_action(observation)
            # Hold the anchor only for a chunking policy serving a cached action this
            # tick; policies that never chunk or that recomputed keep refreshing.
            if self._relative_step is not None and self._ever_predicted_chunk and not self._chunk_predicted:
                self._relative_step.set_cached_state(anchor_before)
            action = self._postprocessor(action)
        action_tensor = action.squeeze(0).cpu()

        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])
