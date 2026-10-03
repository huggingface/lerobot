# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""The exclusive worker's canonical policy/processor execution boundary.

No transport callbacks or robot-control code belong here. A runner is used by one
worker at a time, for both local and remote executors. Custom runners may adapt a
policy's observation preparation or action representation at this boundary.
"""

from __future__ import annotations

import inspect
import time
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import validate_trained_rtc_horizon
from lerobot.policies.rtc.relative import reanchor_relative_rtc_prefix
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.utils.constants import OBS_STATE, QUERY_KIND, QUERY_TEXT

from .contracts import (
    ActionChunk,
    ActionProvenance,
    ExecutionMode,
    FeatureSpec,
    ObservationSnapshot,
    PolicyCapabilities,
    QueryKind,
)


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
        max_text_input: int = 4096,
        max_text_output: int = 8192,
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
        spec = policy.chunk_inference_spec()
        if not spec.current_observation_only:
            raise ValueError("The default runner does not implement temporal observation sampling.")
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
            try:
                inspect.signature(policy.predict_action_chunk).bind(
                    {}, inference_delay=0, prev_chunk_left_over=None
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("Declared RTC support requires the RTC chunk call interface.") from exc
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
        # Chunk callers never use select_action's private playback queue. A prior
        # sync binding must not hold an old anchor across complete chunk calls.
        if self._relative_step is not None:
            self._relative_step.bind_action_queue(None)
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
        for step in postprocessor.steps:
            if (
                isinstance(step, AbsoluteActionsProcessorStep)
                and step.enabled
                and not any(step.relative_step is candidate for candidate in preprocessor.steps)
            ):
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
            tensor = torch.from_numpy(array.copy())
            if feature.kind == "rgb":
                tensor = tensor.permute(2, 0, 1).contiguous().to(torch.float32).div_(255)
            batch[feature.name] = tensor.unsqueeze(0)
        batch["task"] = [observation.task]
        batch["robot_type"] = self.robot_type
        return batch

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
        # Guided RTC locally enables autograd for its prefix correction. no_grad
        # permits that override; inference_mode would suppress its gradient graph.
        with torch.inference_mode(mode is not ExecutionMode.RTC_GUIDED), torch.no_grad():
            prepared = self.preprocessor(self._batch(observation))
            preprocessed_at = time.perf_counter()
            kwargs: dict[str, Any] = {}
            if mode != ExecutionMode.CHUNK:
                prefix = self._prepare_continuation(model_continuation, canonical_continuation)
                if inference_delay < 0:
                    raise ValueError("RTC delay must be nonnegative.")
                if prefix is None and inference_delay:
                    raise ValueError("RTC delay requires a real continuation prefix.")
                if mode == ExecutionMode.RTC_TRAINED:
                    available = 0 if model_continuation is None else len(model_continuation)
                    if inference_delay > min(
                        self.capabilities.training_max_delay, available, self.capabilities.rtc_horizon
                    ):
                        raise ValueError(
                            "Trained RTC delay exceeds checkpoint or available continuation limits."
                        )
                kwargs = {"inference_delay": inference_delay, "prev_chunk_left_over": prefix}
            predicted = self.policy.predict_action_chunk(prepared, **kwargs)
            predicted_at = time.perf_counter()
            self._validate_model_actions(predicted)
            # A postprocessor may modify its argument. Preserve model coordinates
            # before invoking the paired canonical postprocessor.
            original = predicted.detach().clone()
            canonical = self.postprocessor(predicted)
            self._validate_actions(canonical)
            steps = (
                self.capabilities.execution_steps
                if mode == ExecutionMode.CHUNK
                else self.capabilities.prediction_steps
            )
            canonical = canonical[0, :steps].detach().to(device="cpu", dtype=torch.float32).clone()
            model = (
                original[0, :steps].detach().to(device="cpu", dtype=torch.float32).clone()
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
            execution_steps=steps,
            server_durations={
                "preprocessing": preprocessed_at - started,
                "policy": predicted_at - preprocessed_at,
                "postprocessing": finished - predicted_at,
            },
        )

    def _validate_actions(self, actions: torch.Tensor) -> None:
        expected = (1, self.capabilities.prediction_steps, self.capabilities.action_feature.shape[0])
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != expected:
            raise ValueError(f"Policy/processor must return action shape {expected}.")
        if not actions.is_floating_point() or not torch.isfinite(actions).all():
            raise ValueError("Policy/processor returned non-finite or non-floating actions.")

    def _validate_model_actions(self, actions: torch.Tensor) -> None:
        """Keep model coordinates intact; only the paired processor defines canonical actions."""
        if (
            not isinstance(actions, torch.Tensor)
            or actions.ndim != 3
            or actions.shape[:2] != (1, self.capabilities.prediction_steps)
            or actions.shape[2] <= 0
            or not actions.is_floating_point()
            or not torch.isfinite(actions).all()
        ):
            raise ValueError(
                "Policy must return finite floating actions with its declared batch and horizon."
            )
        width = actions.shape[2]
        if self._model_action_dim is not None and width != self._model_action_dim:
            raise ValueError("Policy model action width changed after deployment warmup.")
        if (
            self._relative_step is not None
            and any(mode != ExecutionMode.CHUNK for mode in self.capabilities.modes)
            and width != self.capabilities.action_feature.shape[0]
        ):
            raise ValueError("Relative RTC with different model/canonical widths requires a custom runner.")
        self._model_action_dim = width
        self.capabilities = replace(self.capabilities, model_action_dim=width)

    def _prepare_continuation(
        self, model: torch.Tensor | None, canonical: torch.Tensor | None
    ) -> torch.Tensor | None:
        if model is None or model.numel() == 0:
            return None
        width = self._model_action_dim or self.capabilities.action_feature.shape[0]
        if (
            model.ndim != 2
            or model.shape[1] != width
            or not model.is_floating_point()
            or not torch.isfinite(model).all()
        ):
            raise ValueError("Invalid model-space continuation.")
        device = self.policy.config.device or "cpu"
        if self._relative_step is not None:
            if canonical is None or canonical.shape != model.shape or not torch.isfinite(canonical).all():
                raise ValueError("Relative RTC requires matching finite canonical continuation.")
            state = self._relative_step.get_cached_state()
            if state is None:
                raise ValueError("Relative RTC preprocessor did not cache the current raw state.")
            model = reanchor_relative_rtc_prefix(
                canonical, state, self._relative_step, self._normalizer_step, device
            )
        horizon = self.capabilities.rtc_horizon
        model = model.to(device)
        if len(model) < horizon:
            model = torch.cat([model, model[-1:].expand(horizon - len(model), -1)])
        return model[:horizon]

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
        """Full run reset clears state; motion invalidation preserves planner memory."""
        if not full:
            self.policy.drop_queued_actions()
            return
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        if self.language_processors is not None:
            self.language_processors[0].reset()
            self.language_processors[1].reset()
