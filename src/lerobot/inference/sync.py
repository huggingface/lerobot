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

"""Synchronous inference: predict a chunk inline when the previous one has executed.

One engine runs every synchronous loop: robot rollouts (``get_action``), simulated
evaluation over batched environments and user scripts (``step``). It drives the
:class:`~lerobot.inference.execution.ChunkRuntime` and
:func:`~lerobot.inference.prediction.predict_chunk` the asynchronous engines use,
on the caller's clock: each chunk executes completely before the next is predicted.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from contextlib import nullcontext
from copy import copy
from typing import Any

import torch
from torch import Tensor

from lerobot.lerobot_types import EnvTransition, PolicyPrediction
from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.processor import (
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    create_transition,
)
from lerobot.utils import OBS_STR
from lerobot.utils.constants import ACTION, OBS_PREFIX
from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine, InferenceRobot, PolicyQuery
from .contracts import ActionChunk, ActionProvenance, ExecutionMode, ObservationSnapshot
from .execution import ChunkRuntime
from .history import ExecutionRecord, apply_history, policy_history, recorded_names, rename_history
from .prediction import chunk_inference_context, predict_chunk, rename_prediction

logger = logging.getLogger(__name__)


class SyncInferenceEngine(InferenceEngine):
    """Record every tick, predict a chunk inline when the previous one has executed, play it back.

    ``step`` takes batched observations (``[B, ...]``, any number of environments) and
    returns ``[B, A]``; ``get_action`` serves one robot from its observation frame. Each
    chunk runs ``preprocessor -> predict_action_chunk -> postprocessor`` once, on the
    policy's declared observation windows.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        fps: float | None = None,
        observation_keys: list[str] | None = None,
        task: str = "",
        device: str | None = None,
        dataset_features: dict | None = None,
        robot_type: str = "",
        robot_wrapper: InferenceRobot | None = None,
    ) -> None:
        """Own the policy and its processors.

        Args:
            policy: The policy; its ``chunk_inference_spec`` and config say what to sample.
            preprocessor: Its saved preprocessor; runs once per chunk.
            postprocessor: Its saved postprocessor; runs on whole chunks.
            fps: Ticks per unit of the times passed to ``step``: the control rate for times in
                seconds. ``None`` counts times in ticks (e.g. simulation steps).
            observation_keys: Keys of the observations passed to ``step``, before any rename;
                defaults to the observation keys of ``dataset_features``, else the policy's
                input features.
            task: The initial instruction (``get_action``).
            device: Where ``get_action`` places observations.
            dataset_features: The robot's dataset features (``get_action`` and text queries).
            robot_type: The robot type passed to the policy (``get_action``).
            robot_wrapper: The robot, whose capture times sample past observations.
        """
        super().__init__(task=task)
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features or {}
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._robot = robot_wrapper
        self._spec = policy.chunk_inference_spec()
        if observation_keys is None:
            observation_keys = [key for key in self._dataset_features if key.startswith(OBS_STR)] or list(
                policy.config.input_features or {}
            )
        names = recorded_names(
            observation_keys,
            (
                step.rename_map
                for step in preprocessor.steps
                if isinstance(step, RenameObservationsProcessorStep)
            ),
        )
        self._recorded_names = names
        action_feature = (getattr(policy.config, "output_features", None) or {}).get(ACTION)
        self._action_dim = action_feature.shape[0] if action_feature is not None else None
        self._model_action_dim: int | None = None
        self._batch_size = 1
        interval = 1 / fps if fps else 1.0
        self._record = ExecutionRecord(
            rename_history(policy_history(policy.config), names),
            interval=interval,
            action_padding_value=self._padding_action,
        )
        # Inference is inline and on the caller's clock: no age or deadline applies.
        self._now = 0.0
        self._runtime = ChunkRuntime(
            mode=ExecutionMode.CHUNK,
            action_interval=interval,
            refill_seconds=0.0,
            max_observation_age_s=math.inf,
            action_timeout_s=math.inf,
            startup_timeout_s=math.inf,
            complete_execution=True,
            clock=lambda: self._now,
        )
        self._runtime.activate()
        self.prediction: PolicyPrediction | None = None
        if self._record.active:
            logger.info("Sampling the policy's past inputs by capture time at %.1f Hz", 1 / interval)
        logger.info("SyncInferenceEngine initialized (device=%s)", self._device)

    def _padding_action(self) -> Tensor:
        if self._action_dim is None:
            raise ValueError("Command history padding needs the policy's action feature.")
        return torch.zeros(self._batch_size, self._action_dim)

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Start a new episode: reset the policy, its processors, the record and the chunk."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._record.clear()
        self._runtime.invalidate()
        self.prediction = None
        # The policy was just reset, so a pending task change has nothing stale to flush.
        self._discard_task_change()

    @property
    def queued_actions(self) -> int:
        """Actions of the current chunk not yet played back."""
        return self._runtime.queue.qsize()

    def drop_queued_actions(self) -> None:
        """Discard the remaining actions, e.g. after an instruction change.

        A policy fed back its complete chunks cannot continue after an interrupted
        one, so for it this starts a new episode.
        """
        if self._spec.complete_execution and self.queued_actions:
            self.reset()
        self._runtime.invalidate()

    def step(
        self, observation: dict[str, Any], time: float, recorded: Mapping[str, Any] | None = None
    ) -> Tensor:
        """Record this tick and return its postprocessed action ``[B, A]``.

        When this tick starts a new chunk, :attr:`prediction` holds what the policy predicted
        besides it (``None`` otherwise).

        Args:
            observation: The batch the policy preprocessor receives this tick.
            time: The tick's capture time, increasing within an episode.
            recorded: The values to record for this tick (e.g. compact camera frames);
                defaults to the observation's tensors.
        """
        tensors = {key: value for key, value in observation.items() if isinstance(value, Tensor)}
        self._batch_size = next(iter(tensors.values())).shape[0] if tensors else 1
        self._now = time
        self._record.observe(time, tensors if recorded is None else recorded)
        if self._runtime.should_request():
            self._predict(observation, time)
        queued = self._runtime.pop()
        if queued is None:
            raise RuntimeError("The policy returned no action to execute.")
        action, provenance = queued
        self._record.dispatched(action)
        self.prediction = self._runtime.take_prediction()
        return action

    def _history_tensor(self, name: str, value: Any) -> Tensor:
        """A recorded value as a batched tensor; frames recorded as arrays are converted."""
        if isinstance(value, Tensor):
            return value
        return prepare_observation_for_inference({name: value.copy()}, self._device)[name]

    def _predict(self, observation: dict[str, Any], time: float) -> None:
        # One snapshot labels the chunk: an instruction changed during inference is the next chunk's.
        task, task_version = self.task, self.task_version
        request = self._runtime.begin(ObservationSnapshot({}, time, task, task_version))
        if request is None:
            raise RuntimeError("The synchronous runtime refused a chunk request.")
        history = self._record.sample(time)
        batch = apply_history(dict(observation), history, self._history_tensor)
        with chunk_inference_context(ExecutionMode.CHUNK):
            prediction = predict_chunk(
                self._policy,
                self._postprocessor,
                self._preprocessor(batch),
                spec=self._spec,
                mode=ExecutionMode.CHUNK,
                canonical_action_dim=self._action_dim or 0,
                model_action_dim=self._model_action_dim,
                device=self._device,
            )
        self._model_action_dim = prediction.model_actions.shape[2]
        # The runtime queues time first: [T, B, A].
        actions = prediction.canonical_actions.transpose(0, 1).to("cpu")
        chunk = ActionChunk(
            model_actions=None,
            canonical_actions=actions,
            provenance=ActionProvenance(time, task, task_version),
            execution_steps=len(actions),
            prediction=rename_prediction(prediction.prediction, self._recorded_names)
            if prediction.prediction
            else None,
        )
        if not self._runtime.accept(request, chunk, task_version=task_version):
            raise RuntimeError(self._runtime.failure or "The synchronous runtime rejected a chunk.")

    @staticmethod
    def observation_keys(observation: Mapping[str, Any]) -> list[str]:
        """The observation keys of a batch, for ``observation_keys``."""
        return [key for key in observation if key.startswith(OBS_PREFIX)]

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | EnvTransition | None:
        """Record ``obs_frame`` and return this tick's action tensor.

        When this tick starts a chunk that came with a prediction, returns it with the action
        as a transition.
        """
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
        task, task_changed = self._take_task()
        if task_changed:
            # The queued actions were computed under the previous instruction; drop them so the
            # new one takes effect on this tick.
            logger.info("Task changed to '%s': dropping precomputed actions", task)
            self.drop_queued_actions()
        captured = getattr(self._robot, "observation_time", None)
        recorded = dict(observation)
        measure = self.latency.interval_s > 0 and not self.queued_actions
        started = time.monotonic() if measure else 0.0
        with autocast_ctx:
            batch = prepare_observation_for_inference(observation, self._device, task, self._robot_type)
            action = self.step(
                batch, captured if captured is not None else time.monotonic(), recorded=recorded
            )
        if measure:
            completed = time.monotonic()
            self.latency.record(completed - started, completed)
        # ``task`` is the pre-inference snapshot: a /subtask landing mid-inference must
        # not relabel this action.
        self._set_dispatched_task(task)
        # Policy's own dimension order — the order ``send_next_action`` labels it with.
        action_tensor = action.squeeze(0).cpu()
        if self.prediction is None:
            return action_tensor
        return create_transition(action=action_tensor, prediction=self.prediction)

    def dispatch_allowed(self) -> bool:
        """Permit dispatch unless an external planner has yet to issue its first instruction.

        Inline inference has no worker to pause, so the planner hold is applied here: the
        control loop clears interpolation and holds the robot until the first reply.
        """
        return not self.hold_for_planner()

    # ------------------------------------------------------------------
    # Text queries
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when an external text backend is attached or the policy has a text head."""
        return super().supports_text_queries or self._policy.supports_text_generation()

    @property
    def control_thread_owns_policy(self) -> bool:
        """Inference runs inline on the control thread, so queries are served there too."""
        return True

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Run the policy's text head on the current observation."""
        obs_frame = build_dataset_frame(self._dataset_features, obs_processed, prefix=OBS_STR)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        # Live task, read without consuming the task-changed edge (the action path needs it).
        task = self.task
        # Generation blocks the control thread, so any precomputed chunk is stale by its
        # full duration afterwards. Drop it and resume from fresh inference, matching the
        # hold-and-invalidate rule of the asynchronous backends.
        self.drop_queued_actions()
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(obs_frame, self._device, task, self._robot_type)
            observation = self._mark_query(observation, query)
            # Reusing the action path's preprocessor is safe only while its steps are
            # stateless per call; the one that is not (an enabled RelativeActionsProcessorStep)
            # is rejected for this backend at context-build time.
            observation = self._preprocessor(observation)
            # No str() coercion: _service_query validates the return value.
            return self._policy.generate_text(observation)
