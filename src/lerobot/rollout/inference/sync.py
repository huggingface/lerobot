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
import queue
from collections import deque
from contextlib import nullcontext
from copy import copy
from threading import Event, Thread

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)

# Hard timeout for joining the prefetch worker on stop().
_PREFETCH_JOIN_TIMEOUT_S: float = 5.0
# Poll interval for the worker's request-queue wait (bounds shutdown latency).
_PREFETCH_POLL_S: float = 0.1


# NOTE: ``chunk_action_steps`` enables the "chunked action cache" fast path
# (see ``_get_action_chunked``).  It is opt-in via ``SyncInferenceConfig`` and
# the factory only enables it for policies where it is provably
# behaviour-preserving (currently ACT without temporal ensembling).  It
# bypasses ``select_action`` — which is unsafe for SAC
# (``predict_action_chunk`` raises), ACT temporal ensembling (the ensembler
# lives in ``select_action``), and Diffusion-family policies (obs-history
# queues are populated as a side effect of ``select_action``).
#
# TODO(Steven): support relative-action policies.  The per-tick flow refreshes
# ``RelativeActionsProcessorStep._last_state`` every call, so cached chunk
# actions popped on later ticks get reanchored to the *current* robot state and
# absolute targets drift through the chunk.  Relative-action policies are
# rejected at context-build time today; RTC postprocesses the whole chunk and
# is unaffected.


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a
    CPU action tensor reordered to match the dataset action keys.

    When ``chunk_action_steps`` is set, ``get_action`` instead serves actions
    from a locally cached chunk (see :meth:`_get_action_chunked`): the policy
    is queried via ``predict_action_chunk`` only when the cache is empty, so
    the per-tick observation upload + normalization is skipped while cached
    actions remain.

    When ``prefetch_watermark`` is additionally set, the chunk computation
    moves off the control thread entirely (see :meth:`_get_action_prefetch`):
    once the cache drops to the watermark, the observation is snapshotted and
    the next chunk is computed by a background worker while the control loop
    keeps serving cached actions.  Time alignment is preserved by skipping the
    first ``len(cache-at-snapshot)`` actions of the new chunk: those indices
    correspond to ticks that are served from the old cache before the new
    chunk starts, so serving them again would replay the past.  If the worker
    is slower than the remaining runway the cache empties and ``get_action``
    returns ``None`` (the robot holds its last pose) until the chunk lands —
    the same time-shift semantics as an inline stall, never a jump.
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
        chunk_action_steps: int | None = None,
        prefetch_watermark: int | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._chunk_action_steps = chunk_action_steps
        self._prefetch_watermark = prefetch_watermark
        # Local cache of postprocessed, CPU-side, reordered action rows.  Only
        # used when ``chunk_action_steps`` is set.
        self._action_cache: deque[torch.Tensor] = deque()
        # Background prefetch state (only used when ``prefetch_watermark`` is set).
        self._worker: Thread | None = None
        self._worker_stop = Event()
        # Single-slot request queue: at most one chunk computation in flight.
        self._request_q: queue.Queue = queue.Queue(maxsize=1)
        self._result_q: queue.Queue = queue.Queue()
        self._inflight = False
        # Bumped on reset() so results computed from pre-reset observations are discarded.
        self._generation = 0
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d, chunk_steps=%s, "
            "prefetch_watermark=%s)",
            self._device,
            len(ordered_action_keys),
            chunk_action_steps,
            prefetch_watermark,
        )

    def start(self) -> None:
        """Start the prefetch worker (if enabled); otherwise nothing to do."""
        if self._prefetch_watermark is None:
            logger.info("SyncInferenceEngine started (inline mode — no background thread)")
            return
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker_stop.clear()
        self._worker = Thread(target=self._prefetch_loop, daemon=True, name="SyncChunkPrefetch")
        self._worker.start()
        logger.info(
            "SyncInferenceEngine started (chunk prefetch worker running, watermark=%d)",
            self._prefetch_watermark,
        )

    def stop(self) -> None:
        """Stop the prefetch worker (if running)."""
        if self._worker is not None:
            self._worker_stop.set()
            self._worker.join(timeout=_PREFETCH_JOIN_TIMEOUT_S)
            if self._worker.is_alive():
                logger.warning("Chunk prefetch worker did not join within %.1fs", _PREFETCH_JOIN_TIMEOUT_S)
            self._worker = None
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._action_cache.clear()
        # Invalidate any in-flight prefetch: its observation predates the reset.
        self._generation += 1
        self._inflight = False

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return the next action, reordered to match the dataset action keys."""
        if self._chunk_action_steps is not None:
            if self._prefetch_watermark is not None:
                return self._get_action_prefetch(obs_frame)
            return self._get_action_chunked(obs_frame)
        return self._get_action_single(obs_frame)

    def _get_action_single(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return one action."""
        if obs_frame is None:
            return None
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        with torch.inference_mode(), self._autocast_ctx():
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
        action_tensor = action.squeeze(0).cpu()

        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

    def _get_action_chunked(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Serve the next action from the local chunk cache, refilling when empty.

        This mirrors ACT's internal ``_action_queue`` semantics
        (``predict_action_chunk`` sliced to ``n_action_steps``) but caches the
        *postprocessed* actions here.  On ticks served from the cache, the
        expensive observation upload + normalization is skipped entirely.
        This is behaviour-preserving for policies whose ``select_action``
        ignores the observation while its internal queue is non-empty
        (validated for ACT without temporal ensembling in the factory).
        """
        if not self._action_cache:
            if obs_frame is None:
                return None
            self._action_cache.extend(self._run_policy_chunk(obs_frame))
        if not self._action_cache:
            return None
        return self._action_cache.popleft()

    def _run_policy_chunk(self, obs_frame: dict) -> list[torch.Tensor]:
        """Run the policy once and return a list of postprocessed, reordered action rows."""
        observation = copy(obs_frame)
        with torch.inference_mode(), self._autocast_ctx():
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            actions = self._policy.predict_action_chunk(observation)
            actions = actions[:, : self._chunk_action_steps]
            actions = self._postprocessor(actions)
        # (B=1, T, A) -> (T, A) on CPU.
        actions = actions.squeeze(0).cpu()

        rows: list[torch.Tensor] = []
        for step in range(actions.shape[0]):
            action_dict = make_robot_action(actions[step], self._dataset_features)
            rows.append(torch.tensor([action_dict[k] for k in self._ordered_action_keys]))
        return rows

    # ------------------------------------------------------------------
    # Background chunk prefetch
    # ------------------------------------------------------------------

    def _get_action_prefetch(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Serve cached actions while the next chunk is computed off-thread.

        Per call: (1) fold in any finished chunk (dropping the first ``skip``
        actions so the new chunk stays time-aligned with what the old cache
        already served), (2) if the cache is at/below the watermark and nothing
        is in flight, snapshot ``obs_frame`` and request the next chunk,
        (3) pop one cached action, or ``None`` if the cache is empty (the
        caller then sends nothing and the robot holds its last pose).
        """
        self._drain_prefetch_results()

        if (
            not self._inflight
            and obs_frame is not None
            and len(self._action_cache) <= self._prefetch_watermark
        ):
            # ``skip`` = actions still pending at snapshot time (including the
            # one popped below this tick).  The new chunk's indices [0, skip)
            # cover the same ticks, so they must not be served again.
            skip = len(self._action_cache)
            try:
                self._request_q.put_nowait((self._generation, skip, copy(obs_frame)))
                self._inflight = True
            except queue.Full:
                # A stale (pre-reset) request is still queued; the worker will
                # pick it up and its result will be discarded by generation.
                pass

        if not self._action_cache:
            return None
        return self._action_cache.popleft()

    def _drain_prefetch_results(self) -> None:
        """Fold finished prefetch results into the action cache (non-blocking)."""
        while True:
            try:
                status, generation, payload = self._result_q.get_nowait()
            except queue.Empty:
                return
            self._inflight = False
            if generation != self._generation:
                logger.debug("Discarding stale prefetched chunk (generation %d)", generation)
                continue
            if status == "error":
                raise RuntimeError("Chunk prefetch worker failed during policy inference") from payload
            skip, rows = payload
            self._action_cache.extend(rows[skip:])

    def _prefetch_loop(self) -> None:
        """Background worker: compute one chunk per request."""
        logger.info("Chunk prefetch worker started")
        while not self._worker_stop.is_set():
            try:
                generation, skip, obs_frame = self._request_q.get(timeout=_PREFETCH_POLL_S)
            except queue.Empty:
                continue
            try:
                rows = self._run_policy_chunk(obs_frame)
                self._result_q.put(("ok", generation, (skip, rows)))
            except Exception as e:  # propagated to the control thread on next get_action
                logger.error("Chunk prefetch worker error: %s", e)
                self._result_q.put(("error", generation, e))
        logger.info("Chunk prefetch worker exiting")

    def _autocast_ctx(self):
        """AMP autocast on CUDA when the policy enables it; a no-op otherwise."""
        if self._device.type == "cuda" and self._policy.config.use_amp:
            return torch.autocast(device_type=self._device.type)
        return nullcontext()
