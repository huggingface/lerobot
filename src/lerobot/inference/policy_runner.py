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

"""Exclusive worker boundary for snapshot preparation, chunk prediction and text generation."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy, validate_trained_rtc_horizon
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.utils import OBS_STATE, QUERY_KIND, QUERY_TEXT

from .contracts import (
    DEFAULT_MAX_TEXT_INPUT_CHARS,
    DEFAULT_MAX_TEXT_OUTPUT_CHARS,
    ActionChunk,
    ActionProvenance,
    ExecutionMode,
    FeatureSpec,
    ObservationSnapshot,
    PolicyCapabilities,
    QueryKind,
)
from .history import HistoryInputs, apply_history, policy_history
from .prediction import chunk_inference_context, predict_chunk


class PolicyRunner:
    """Default current-observation runner; its processor pair is session owned.

    Features have already been mapped into checkpoint names and component order.
    Semantic conventions come from verified checkpoint metadata or explicit
    deployment configuration; dimensions alone are never claimed as compatibility.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        action_interval: float,
        features: tuple[FeatureSpec, ...],
        action_feature: FeatureSpec,
        modes: tuple[ExecutionMode, ...] = (ExecutionMode.CHUNK,),
        language_processors: tuple[PolicyProcessorPipeline, PolicyProcessorPipeline] | None = None,
        robot_type: str = "",
        max_text_input: int = DEFAULT_MAX_TEXT_INPUT_CHARS,
        max_text_output: int = DEFAULT_MAX_TEXT_OUTPUT_CHARS,
        language_enabled: bool | None = None,
    ) -> None:
        """Validate a deployment and own its canonical processor pairs."""
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.robot_type = robot_type
        self.max_text_input = max_text_input
        self.max_text_output = max_text_output
        self._model_action_dim: int | None = None
        if max_text_input <= 0 or max_text_output <= 0:
            raise ValueError("Text input and output bounds must be positive.")
        supports_language = policy.supports_text_generation()
        if language_enabled is True and not supports_language:
            raise ValueError("Language is enabled but this policy has no text capability.")
        spec = self._policy_spec = policy.chunk_inference_spec()
        modes = tuple(ExecutionMode(mode) for mode in modes)
        if any(mode not in spec.modes for mode in modes):
            raise ValueError(f"Unsupported execution mode; policy declares {spec.modes!r}.")
        rtc_config = getattr(policy.config, "rtc_config", None)
        rtc_modes = tuple(mode for mode in modes if mode != ExecutionMode.CHUNK)
        if rtc_modes:
            if rtc_config is None or not rtc_config.enabled:
                raise ValueError("RTC serving requires deployment-owned enabled RTC configuration.")
            resolved_rtc = ExecutionMode(f"rtc_{rtc_config.mode}")
            if any(mode != resolved_rtc for mode in rtc_modes):
                raise ValueError("Requested RTC mode differs from the deployment's effective RTC mode.")
            if not 0 < rtc_config.execution_horizon <= spec.prediction_steps:
                raise ValueError("RTC execution_horizon must fit the policy prediction horizon.")
            if ExecutionMode.RTC_TRAINED in rtc_modes:
                validate_trained_rtc_horizon(
                    rtc_config.execution_horizon, spec.prediction_steps, spec.training_max_delay
                )
            if not policy.supports_rtc():
                raise ValueError("RTC serving requires a policy that declares RTC support.")
        self.capabilities = PolicyCapabilities(
            modes=modes,
            prediction_steps=spec.prediction_steps,
            execution_steps=spec.execution_steps,
            action_interval=action_interval,
            features=tuple(features),
            action_feature=action_feature,
            language=supports_language and language_enabled is not False,
            training_max_delay=spec.training_max_delay,
            rtc_horizon=rtc_config.execution_horizon if rtc_modes and rtc_config is not None else 0,
            retains_session_state=spec.retains_session_state,
            complete_execution=spec.complete_execution,
            history=policy_history(policy.config),
        )
        self._validate_feature_contract()
        self._validate_processor_pair(preprocessor, postprocessor)
        self._relative_step = next(
            (
                step
                for step in preprocessor.steps
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        self._normalizer_step = next(
            (step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)), None
        )
        if self._relative_step is not None:
            action_names = self.capabilities.action_feature.names
            if action_names:
                if (
                    self._relative_step.action_names
                    and tuple(self._relative_step.action_names) != action_names
                ):
                    raise ValueError(
                        "Relative-action processor names differ from the canonical action order."
                    )
                self._relative_step.action_names = list(action_names)
            elif self._relative_step.exclude_joints:
                raise ValueError("Relative joint exclusions require ordered canonical action names.")
        self.language_processors = language_processors
        if self.capabilities.language and language_processors is None:
            # Copy the PAIR at once: AbsoluteActions must still point to its own
            # RelativeActions step, not the action pipeline's mutable anchor.
            self.language_processors = deepcopy((preprocessor, postprocessor))
        if self.language_processors is not None:
            self._validate_processor_pair(*self.language_processors)
            if any(
                step is other for step in preprocessor.steps for other in self.language_processors[0].steps
            ):
                raise ValueError("Language and action processors must have isolated step instances.")

    @staticmethod
    def _validate_processor_pair(
        preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
    ) -> None:
        for step in preprocessor.steps:
            if isinstance(step, RenameObservationsProcessorStep) and step.rename_map:
                raise ValueError(
                    "Canonical features are already mapped; the runner rename_map must be empty."
                )
        relative = any(isinstance(s, RelativeActionsProcessorStep) and s.enabled for s in preprocessor.steps)
        for step in postprocessor.steps:
            if isinstance(step, AbsoluteActionsProcessorStep) and step.enabled and not relative:
                raise ValueError("Relative action postprocessor must be paired with its preprocessor.")

    def _validate_feature_contract(self) -> None:
        self.policy.validate_chunk_input_features(self.capabilities.features)
        output = self.policy.config.action_feature
        action = self.capabilities.action_feature
        if output is None or action.name != "action" or action.shape != tuple(output.shape):
            raise ValueError("Canonical action feature differs from policy output_features.")
        if action.kind != "tensor" or len(action.shape) != 1 or action.dtype != "float32":
            raise ValueError("The default runner requires a one-dimensional float32 action representation.")
        checkpoint_names = getattr(self.policy.config, "action_feature_names", None)
        if checkpoint_names and tuple(checkpoint_names) != action.names:
            raise ValueError("Canonical action component order differs from checkpoint action_feature_names.")
        state = next((feature for feature in self.capabilities.features if feature.name == OBS_STATE), None)
        if (
            state is not None
            and state.names
            and action.names
            and set(state.names) == set(action.names)
            and state.names != action.names
        ):
            raise ValueError("State and action component order must be aligned.")

    def _batch(self, observation: ObservationSnapshot) -> dict[str, Any]:
        if set(observation.features) != {feature.name for feature in self.capabilities.features}:
            raise ValueError("Observation feature names differ from the serving contract.")
        if len(observation.task) > self.max_text_input:
            raise ValueError("Instruction exceeds the configured text-input limit.")
        batch: dict[str, Any] = {}
        for feature in self.capabilities.features:
            array = observation.features[feature.name]
            if array.shape != feature.shape or array.dtype != np.dtype(feature.dtype):
                raise ValueError(
                    f"Observation shape/dtype differs from the serving contract: {feature.name}."
                )
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError(f"Observation contains non-finite values: {feature.name}.")
            batch[feature.name] = self._tensor(feature, array)
        batch["task"] = [observation.task]
        batch["robot_type"] = self.robot_type
        return batch

    @staticmethod
    def _tensor(feature: FeatureSpec, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.array(array, copy=True))
        if feature.kind == "rgb":
            tensor = tensor.permute(2, 0, 1).contiguous().to(torch.float32).div_(255)
        return tensor.unsqueeze(0)

    def _history_batch(self, observation: ObservationSnapshot, batch: dict[str, Any]) -> dict[str, Any]:
        """Validate the request's history against the declaration and lay it out for the policy."""
        declared = self.capabilities.history
        history = observation.history or HistoryInputs()
        features = {feature.name: feature for feature in self.capabilities.features}
        action = self.capabilities.action_feature

        def check(feature: FeatureSpec, values: tuple[Any, ...], count: int, what: str) -> None:
            if len(values) != count:
                raise ValueError(f"{what} for {feature.name} has {len(values)} values, expected {count}.")
            for value in values:
                if (
                    not isinstance(value, np.ndarray)
                    or value.shape != feature.shape
                    or value.dtype != np.dtype(feature.dtype)
                ):
                    raise ValueError(f"{what} for {feature.name} differs from the serving contract.")
                if value.dtype.kind == "f" and not np.isfinite(value).all():
                    raise ValueError(f"{what} for {feature.name} contains non-finite values.")

        if set(history.observations) != {window.feature for window in declared.observations}:
            raise ValueError("History windows differ from the declared features.")
        for window in declared.observations:
            count = len(window.offsets) - 1
            check(features[window.feature], history.observations[window.feature], count, "History")
            if len(history.observation_padding.get(window.feature, ())) != count:
                raise ValueError(f"History padding for {window.feature} differs from its window.")
        check(action, history.actions, len(declared.actions), "Command history")
        if len(history.action_padding) != len(declared.actions):
            raise ValueError("Command history padding differs from its window.")
        return apply_history(
            batch, history, lambda name, value: self._tensor(features.get(name, action), value)
        )

    def predict(
        self,
        observation: ObservationSnapshot,
        *,
        mode: ExecutionMode = ExecutionMode.CHUNK,
        inference_delay: int = 0,
        model_continuation: torch.Tensor | None = None,
        canonical_continuation: torch.Tensor | None = None,
        provenance: ActionProvenance | None = None,
    ) -> ActionChunk:
        """Preprocess a fresh snapshot and return a validated canonical action chunk."""
        mode = ExecutionMode(mode)
        if mode not in self.capabilities.modes:
            raise ValueError(f"Execution mode {mode.value!r} was not enabled by this deployment.")
        started = time.perf_counter()
        with chunk_inference_context(mode):
            batch = self._history_batch(observation, self._batch(observation))
            prepared = self.preprocessor(batch)
            preprocessed_at = time.perf_counter()
            prediction = predict_chunk(
                self.policy,
                self.postprocessor,
                prepared,
                spec=self._policy_spec,
                mode=mode,
                canonical_action_dim=self.capabilities.action_feature.shape[0],
                model_action_dim=self._model_action_dim,
                device=self.policy.config.device or "cpu",
                rtc_horizon=self.capabilities.rtc_horizon,
                inference_delay=inference_delay,
                model_continuation=model_continuation,
                canonical_continuation=canonical_continuation,
                relative=self._relative_step is not None,
                normalizer_step=self._normalizer_step,
            )
            # One robot per session: drop the batch axis.
            model_actions, canonical_actions = prediction.model_actions[0], prediction.canonical_actions[0]
            self._model_action_dim = model_actions.shape[1]
            self.capabilities = replace(self.capabilities, model_action_dim=self._model_action_dim)
            canonical = canonical_actions.to(device="cpu", dtype=torch.float32).clone()
            model = (
                model_actions.to(device="cpu", dtype=torch.float32).clone()
                if mode != ExecutionMode.CHUNK
                else None
            )
        finished = time.perf_counter()
        return ActionChunk(
            model_actions=model,
            canonical_actions=canonical,
            provenance=provenance
            or ActionProvenance(
                capture_time=observation.capture_time,
                task=observation.task,
                task_version=observation.task_version,
                observation_id=observation.observation_id,
            ),
            execution_steps=len(canonical),
            prediction=prediction.prediction,
            server_durations={
                "preprocessing": preprocessed_at - started,
                "policy": prediction.predicted_at - preprocessed_at,
                "postprocessing": finished - prediction.predicted_at,
            },
        )

    def query(self, observation: ObservationSnapshot, *, kind: str, text: str) -> str:
        """Use isolated processors for one bounded language request."""
        if not self.capabilities.language or self.language_processors is None:
            raise ValueError("This policy does not support text queries.")
        try:
            query_kind = QueryKind(kind)
        except (ValueError, TypeError) as exc:
            raise ValueError("Unsupported language query kind.") from exc
        if not text.strip() or len(text) > self.max_text_input:
            raise ValueError("Query text is empty or exceeds the configured input limit.")
        batch = self._batch(observation)
        batch[QUERY_KIND] = query_kind.value
        batch[QUERY_TEXT] = text
        with torch.inference_mode():
            prepared = self.language_processors[0](batch)
            answer = self.policy.generate_text(prepared)
        if not isinstance(answer, str) or not answer.strip() or len(answer) > self.max_text_output:
            raise ValueError("Text generation returned an empty, invalid, or oversized answer.")
        return answer.strip()

    def reset(self, *, full: bool = True) -> None:
        """Full run reset clears state; motion invalidation preserves planner memory.

        A complete-execution policy cannot continue after an interrupted chunk, so any
        invalidation resets its episode context.
        """
        if not full and not self._policy_spec.complete_execution:
            return
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        if self.language_processors is not None:
            self.language_processors[0].reset()
            self.language_processors[1].reset()
