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

"""Exclusive control ownership for VLA execution and bounded VLM interventions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import torch

from lerobot.utils.action_interpolator import ActionInterpolator

from ..end_effector import EndEffectorKinematics
from ..hybrid import HybridConfig, PlannerDecision
from .base import InferenceEngine, QueryAnswer, QueryKind


class HybridInferenceEngine(InferenceEngine):
    """Wrap an inference engine; only the control thread applies planner decisions.

    The external-query worker can publish a proposal, never command the robot or
    resume inference. Every review pauses the delegate, flushes queued actions,
    holds a measured pose, and waits for settling before taking fresh images.
    """

    interpolates_actions = True

    def __init__(self, delegate, config: HybridConfig, keys: list[str], multiplier: int):
        super().__init__(task=delegate.task)
        if set(keys) != set(config.limits):
            raise ValueError("Hybrid limits must cover exactly the robot's ordered action keys")
        self.delegate = delegate
        self.config = config
        self.keys = keys
        self.kinematics = {name: EndEffectorKinematics(ee) for name, ee in config.end_effectors.items()}
        self.control_interpolator = ActionInterpolator(multiplier=multiplier)
        self._mode = "idle"
        self._obs: dict | None = None
        self._observed_at = 0.0
        self._hold: dict[str, float] | None = None
        self._target: dict[str, float] | None = None
        self._pending_decision = None
        self._manual_task: str | None = None
        self._due = 0.0
        self._review_started = 0.0
        self._motion_started = 0.0
        self._motion_duration = 0.0
        self._consecutive = 0
        self._ee_targets: dict[str, dict[str, list[float]]] = {}
        self.terminal = False

    @property
    def control_thread_owns_policy(self):
        # The wrapper only services external requests; the delegate still owns its policy.
        return True

    @property
    def ready(self):
        return self.delegate.ready

    @property
    def failed(self):
        return self.delegate.failed

    @property
    def failure_traceback(self):
        return self.delegate.failure_traceback

    def start(self):
        self.delegate.start()
        self.delegate.pause()

    def stop(self):
        self.stop_autosteer()
        self.drop_pending_query()
        self.delegate.stop()

    def pause(self):
        with self._query_lock:
            self._query_epoch += 1
            self._pending_decision = None
            self._mode = "idle"
            self.delegate.pause()
            self.delegate.discard_actions()
            self.control_interpolator.reset()

    def resume(self):
        # Called only when the rollout starts, never during setup.
        with self._query_lock:
            if self._autosteer_goal is None:
                self.start_autosteer(self.task, self.config.policy_window_s)
            self._mode = "review_due"

    def reset(self):
        self.pause()
        self.delegate.reset()
        self._hold = None
        self._obs = None
        self._manual_task = None
        self._consecutive = 0
        self._ee_targets = {}
        self.terminal = False

    def start_autosteer(self, goal, interval_s):
        with self._query_lock:
            super().start_autosteer(goal, interval_s)
            self._pending_decision = None
            self._manual_task = None
            self._mode = "review_due"
            self.terminal = False

    def stop_autosteer(self):
        with self._query_lock:
            goal = super().stop_autosteer()
            self._pending_decision = None
            self._manual_task = None
            self._mode = "hold"
            self.delegate.pause()
            self.delegate.discard_actions()
            self.control_interpolator.reset()
            return goal

    def set_task(self, task):
        # Operator takeover is queued for the control thread, including identical text.
        with self._query_lock:
            changed = super().set_task(task)
            self._query_epoch += 1
            self._pending_decision = None
            self._manual_task = task
            return changed

    def notify_observation(self, obs):
        with self._query_lock:
            self._obs = obs
            self._observed_at = time.perf_counter()
            if self._mode == "policy":
                self.delegate.notify_observation(obs)

    def _pose(self):
        if self._obs is None or time.perf_counter() - self._observed_at > self.config.max_observation_age_s:
            raise ValueError("Hybrid observation is stale or missing")
        pose = {key: float(self._obs[key]) for key in self.keys}
        for key, value in pose.items():
            limit = self.config.limits[key]
            # Permit only measurement-level excursions, never proposed out-of-range targets.
            if not limit.minimum - limit.tolerance <= value <= limit.maximum + limit.tolerance:
                raise ValueError(f"Hybrid measured position is non-finite or out of range: {key}")
            pose[key] = max(limit.minimum, min(limit.maximum, value))
        return pose

    def _begin_review(self):
        self.delegate.pause()
        self.delegate.discard_actions()
        self.control_interpolator.reset()
        self._hold = self._pose()
        self._mode = "review"
        self._review_started = time.perf_counter()
        self._pending_decision = None
        self._autosteer_due_at = self._review_started + self.config.settle_s

    def _check_hold(self):
        pose = self._pose()
        if self._hold is None:
            raise ValueError("Hybrid has no held pose")
        for key in self.keys:
            if abs(pose[key] - self._hold[key]) > self.config.limits[key].tolerance:
                raise ValueError(f"Robot moved during planner review: {key}; proposal discarded")

    def _start_policy(self, instruction, *, manual=False):
        self.delegate.pause()
        self.delegate.set_task(instruction)
        self.delegate.discard_actions()
        self.control_interpolator.reset()
        self.delegate.notify_observation(self._obs)
        self.delegate.resume()
        InferenceEngine.set_task(self, instruction)
        self._mode = "policy"
        self._due = float("inf") if manual else time.perf_counter() + self.config.policy_window_s
        self._consecutive = 0

    def _finish(self, *, error=None, completed=False, message="Planner requested operator input"):
        self.delegate.pause()
        self.delegate.discard_actions()
        self.control_interpolator.reset()
        self._mode = "hold"
        self.terminal = True
        goal = self._autosteer_goal or self.task
        self._autosteer_goal = None
        self._query_epoch += 1
        self._pending_decision = None
        # A timed-out HTTP worker may still be running. Keep the single-request
        # slot occupied until it returns, including across a subsequent /start.
        in_flight = self._query_in_flight
        self._publish_answer(
            QueryAnswer(
                question=goal,
                answer=message if error is None else None,
                error=error,
                kind=QueryKind.NEXT_SUBTASK,
                completed=completed,
            )
        )
        self._query_in_flight = in_flight

    def _accept_decision(self):
        decision, observation, epoch, error = self._pending_decision
        self._pending_decision = None
        if epoch != self._query_epoch:
            return
        if error is not None:
            self._finish(error=error)
            return
        self._check_hold()
        if time.perf_counter() - self._review_started > self.config.review_timeout_s:
            raise ValueError("Planner review deadline exceeded; proposal discarded")
        if decision.mode == "policy":
            self._start_policy(decision.instruction)
        elif decision.mode in {"intervention", "end_effector"}:
            if self._consecutive >= self.config.max_consecutive_interventions:
                raise ValueError("Consecutive intervention limit reached")
            target = decision.resolve_motion(self.config, self._hold, self.kinematics)
            # IK is bounded CPU work; reject if feedback aged or a deadline elapsed meanwhile.
            self._check_hold()
            if time.perf_counter() - self._review_started > self.config.review_timeout_s:
                raise ValueError("Planner/IK deadline exceeded")
            self._target = target
            self._ee_targets = decision.ee_targets
            self.control_interpolator.reset()
            self._motion_started = time.perf_counter()
            self._motion_duration = decision.duration_s
            self._mode = "intervention"
            self._consecutive += 1
        else:
            self._finish(completed=decision.mode == "done", message=f"{decision.mode}: {decision.reason}")
        description = json.dumps(asdict(decision))
        self.external_history.append((observation, description))
        if not self.terminal:
            self._publish_answer(
                QueryAnswer(question=self._autosteer_goal, answer=description, kind=QueryKind.NEXT_SUBTASK)
            )

    def get_action(self, obs_frame):
        with self._query_lock:
            try:
                pose = self._pose()
                if self._manual_task is not None:
                    task, self._manual_task = self._manual_task, None
                    self._hold = pose
                    self._start_policy(task, manual=True)
                if self._mode == "review_due" or (
                    self._mode == "policy" and time.perf_counter() >= self._due
                ):
                    self._begin_review()
                if self._pending_decision is not None:
                    self._accept_decision()
                if (
                    self._mode == "review"
                    and time.perf_counter() - self._review_started > self.config.review_timeout_s
                ):
                    self._finish(error="Planner review deadline exceeded; held for operator")
                if self._mode == "policy":
                    if self.control_interpolator.needs_new_action():
                        action = self.delegate.get_action(obs_frame)
                        if action is not None:
                            self.control_interpolator.add(action.cpu())
                    action = self.control_interpolator.get()
                    if action is not None:
                        self._set_dispatched_task(self.delegate.dispatched_task)
                        self._hold = dict(zip(self.keys, action.tolist(), strict=True))
                        return action
                if self._mode == "intervention":
                    elapsed = time.perf_counter() - self._motion_started
                    fraction = min(1.0, elapsed / self._motion_duration)
                    target = {
                        k: self._hold[k] + fraction * (self._target[k] - self._hold[k]) for k in self.keys
                    }
                    self._set_dispatched_task("VLM intervention")
                    if elapsed >= self._motion_duration:
                        reached = all(
                            abs(pose[k] - self._target[k]) <= self.config.limits[k].tolerance
                            for k in self.keys
                        )
                        reached = reached and all(
                            self.kinematics[name].reached(goal, pose)
                            for name, goal in self._ee_targets.items()
                        )
                        if reached:
                            self._begin_review()
                            target = self._hold
                        elif elapsed > self._motion_duration + self.config.settle_s:
                            raise ValueError("Intervention did not reach its target in time")
                    return self._emit_position(target, "VLM intervention")
                if self._hold is None:
                    self._hold = pose
                return self._emit_position(self._hold, "VLM hold")
            except (ValueError, KeyError, TypeError) as exc:
                self._finish(error=f"{type(exc).__name__}: {exc}")
                return None

    def _emit_position(self, target, task):
        if self.control_interpolator.needs_new_action():
            self.control_interpolator.add(torch.tensor([target[k] for k in self.keys], dtype=torch.float32))
        self._set_dispatched_task(task)
        return self.control_interpolator.get()

    def _poll_autosteer(self, obs_processed):
        if self._mode == "review" and not self.terminal:
            super()._poll_autosteer(obs_processed)

    def _resolve_query(self, query, obs_processed, generate, *, epoch=None):
        if query.kind is QueryKind.VQA:
            return super()._resolve_query(query, obs_processed, generate, epoch=epoch)
        decision, error = None, None
        try:
            decision = generate(obs_processed, query)
            if not isinstance(decision, PlannerDecision):
                raise TypeError("Hybrid planner must return a PlannerDecision")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        with self._query_lock:
            if epoch == self._query_epoch and self._autosteer_goal == query.text:
                self._pending_decision = (decision, obs_processed, epoch, error)
                self._autosteer_due_at = float("inf")
            self._query_in_flight = False
