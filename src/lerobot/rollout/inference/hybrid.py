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
import math
import time
from dataclasses import asdict

import torch

from lerobot.utils.action_interpolator import ActionInterpolator

from ..end_effector import EndEffectorKinematics
from ..hybrid import HybridConfig, PlannerDecision
from .base import ActionProposal, InferenceEngine, QueryAnswer, QueryKind


class ReviewDriftError(ValueError):
    """Fresh feedback invalidates a review snapshot, but can seed another review."""


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
        if config.review_policy_chunks and not delegate.supports_action_proposals:
            raise ValueError(
                "review_policy_chunks requires an inference backend with isolated proposals (RTC)"
            )
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
        self._motion_keys: set[str] = set()
        self.terminal = False
        self._proposal: ActionProposal | None = None
        self._approved_actions: torch.Tensor | None = None
        self._approved_index = 0
        self._proposal_started = 0.0
        self._execution_start: dict | None = None
        self._last_execution: dict | None = None
        self._executed_steps = 0
        self._executed_seconds = 0.0
        self._vlm_feedback = None

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
            self._clear_proposal()
            self.delegate.pause()
            self.delegate.discard_actions()
            self.control_interpolator.reset()

    def resume(self):
        # Called only when the rollout starts, never during setup.
        with self._query_lock:
            self._reset_execution_feedback()
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
        self._reset_execution_feedback()

    def _reset_execution_feedback(self):
        self._execution_start = None
        self._last_execution = None
        self._executed_steps = 0
        self._executed_seconds = 0.0
        self._vlm_feedback = None

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
            self._clear_proposal()
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

    def _begin_review(self, *, retain_endpoint=False):
        self.delegate.pause()
        self.delegate.discard_actions()
        self.control_interpolator.reset()
        measured = self._pose()
        # After an approved prefix, keep its final commanded endpoint during
        # settling. Replacing it with lagging feedback cancels the remaining
        # tracking motion and repeatedly re-anchors each chunk short of target.
        if not retain_endpoint or self._hold is None:
            self._hold = measured
        self._clear_proposal()
        self._mode = "settling" if self.config.review_policy_chunks else "review"
        self._review_started = time.perf_counter()
        self._pending_decision = None
        self._autosteer_due_at = self._review_started + self.config.settle_s

    def _clear_proposal(self):
        self._proposal = None
        self._approved_actions = None
        self._approved_index = 0

    def _poll_proposal(self):
        if self._mode == "settling" and time.perf_counter() >= self._autosteer_due_at:
            observation = dict(self._obs)
            if self._last_execution is not None:
                observation["_hybrid_execution"] = self._last_execution | {
                    "measured_after_settling": self._pose(),
                    "cumulative_policy_steps": self._executed_steps,
                    "cumulative_policy_seconds": self._executed_seconds,
                    "time_basis": "Policy action time only; excludes inference, API reviews and settling",
                }
            self.delegate.request_action_proposal(observation, self.task)
            self._proposal_started = time.perf_counter()
            self._mode = "proposing"
        if self._mode != "proposing":
            return
        if time.perf_counter() - self._proposal_started > self.config.proposal_timeout_s:
            raise ValueError("Policy proposal inference deadline exceeded")
        proposal = self.delegate.take_action_proposal()
        if proposal is None:
            return
        actions = proposal.actions
        if (
            actions.ndim != 2
            or not 1 <= actions.shape[0] <= 1024
            or actions.shape[1] != len(self.keys)
            or not torch.isfinite(actions).all()
            or not math.isfinite(proposal.fps)
            or proposal.fps <= 0
            or proposal.task != self.task
        ):
            raise ValueError("Invalid policy proposal shape, values, FPS or task")
        self._check_hold(proposal.observation)
        self._proposal = proposal
        self._mode = "review"
        self._review_started = time.perf_counter()
        self._autosteer_due_at = self._review_started

    def pump_query(self, obs_processed=None):
        if self.config.vlm_only and obs_processed is not None:
            obs_processed = dict(obs_processed)
            if self._vlm_feedback is not None:
                obs_processed["_vlm_feedback"] = self._vlm_feedback
            if self._last_execution is not None:
                obs_processed["_hybrid_execution"] = self._last_execution | {
                    "measured_after_settling": self._pose(),
                }
        if self.config.review_policy_chunks and self._proposal is not None and obs_processed is not None:
            proposal = self._proposal
            # Images, joints and prediction come from the SAME inference snapshot.
            obs_processed = dict(proposal.observation)
            obs_processed["_hybrid_proposal"] = {
                "id": proposal.proposal_id,
                "task": proposal.task,
                "fps": proposal.fps,
                "action_keys": self.keys,
                "actions": proposal.actions.tolist(),
                "execute_steps": min(self.config.proposal_execution_steps, len(proposal.actions)),
            }
        return super().pump_query(obs_processed)

    def _check_hold(self, observation):
        pose = self._pose()
        if self._hold is None:
            raise ValueError("Hybrid has no held pose")
        for key in self.keys:
            reference = float(observation[key])
            delta = abs(pose[key] - reference)
            tolerance = self.config.limits[key].tolerance
            if not math.isfinite(reference):
                raise ValueError(f"Non-finite planner observation: {key}")
            if delta > tolerance:
                raise ReviewDriftError(
                    f"Robot moved since planner observation: {key}, delta={delta:.5f}, "
                    f"tolerance={tolerance:.5f}; proposal discarded"
                )

    def _retry_drifted_review(self, error):
        # Never apply a stale approval/correction. Invalidate its epoch and ask
        # again from new images/joints, retaining the last authorized hold target.
        # A query worker may still be exiting; do not free its single-flight slot.
        self._query_epoch += 1
        self._begin_review(retain_endpoint=True)
        in_flight = self._query_in_flight
        self._publish_answer(
            QueryAnswer(
                question=self._autosteer_goal or self.task,
                answer=f"Review refreshed — {error}. Holding; requesting fresh prediction and review.",
                kind=QueryKind.NEXT_SUBTASK,
                status_only=True,
            )
        )
        self._query_in_flight = in_flight

    def _start_policy(self, instruction, *, manual=False):
        if instruction != self.task or manual:
            self._reset_execution_feedback()
        if self.config.review_policy_chunks and not manual:
            InferenceEngine.set_task(self, instruction)
            self.delegate.set_task(instruction)
            self._begin_review()
            return
        self._clear_proposal()
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
        self._clear_proposal()
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
            if self.config.vlm_only and error.startswith("AgentToolError:"):
                self._agent_feedback("rejected", error, None, observation)
                return
            # Authentication, transport and hardware failures are not valid actions.
            self._finish(error=error)
            return
        self._check_hold(observation)
        if time.perf_counter() - self._review_started > self.config.review_timeout_s:
            raise ValueError("Planner review deadline exceeded; proposal discarded")
        if self.config.vlm_only and decision.mode in {"accept", "policy"}:
            raise ValueError("VLM-only control cannot delegate to a policy")
        if decision.mode == "observe" and self.config.vlm_only:
            self._vlm_feedback = {"status": "observed", "note": decision.reason}
            self._begin_review(retain_endpoint=True)
        elif decision.mode == "accept":
            if not self.config.review_policy_chunks or self._proposal is None:
                raise ValueError("No unexecuted proposal to accept")
            if observation.get("_hybrid_proposal", {}).get("id") != self._proposal.proposal_id:
                raise ValueError("Reviewed proposal identity mismatch")
            count = min(self.config.proposal_execution_steps, len(self._proposal.actions))
            self._approved_actions = self._proposal.actions[:count].clone()
            self._approved_index = 0
            self._execution_start = self._pose()
            self.control_interpolator.reset()
            self.control_interpolator.add(torch.tensor([self._hold[key] for key in self.keys]))
            self.control_interpolator.get()  # Seed interpolation from the held pose, without dispatch.
            self._mode = "approved"
            self._consecutive = 0
        elif decision.mode == "policy":
            self._start_policy(decision.instruction)
        elif decision.mode in {"intervention", "end_effector"}:
            self._last_execution = None
            if (
                self.config.review_policy_chunks
                and decision.execution_status != "failed"
                and decision.intent_status != "misaligned"
            ):
                raise ValueError("Correction requires failed execution or misaligned intent")
            if not self.config.vlm_only and self._consecutive >= self.config.max_consecutive_interventions:
                raise ValueError("Consecutive intervention limit reached")
            # IK and correction deltas start from fresh measured joints, not the
            # hold command (a loaded arm can settle at a small tracking offset).
            motion_start = self._pose()
            try:
                resolved = decision.resolve_motion(self.config, motion_start, self.kinematics)
            except ValueError as exc:
                if not self.config.vlm_only:
                    raise
                self._agent_feedback("rejected", str(exc), decision, observation)
                return
            self._execution_start = motion_start
            motion_keys = set(decision.targets)
            for name in decision.ee_targets:
                motion_keys.update(self.config.end_effectors[name].action_keys)
            target = dict(self._hold)
            target.update({key: resolved[key] for key in motion_keys})
            # IK is bounded CPU work; reject if feedback aged or a deadline elapsed meanwhile.
            self._check_hold(observation)
            if time.perf_counter() - self._review_started > self.config.review_timeout_s:
                raise ValueError("Planner/IK deadline exceeded")
            self._target = target
            self._motion_keys = motion_keys
            # Keep uncommanded axes at their existing hold setpoints; re-anchoring
            # those setpoints to each sagged measurement would cause repeated drift.
            self._hold.update({key: motion_start[key] for key in motion_keys})
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
                QueryAnswer(
                    question=self._autosteer_goal,
                    answer=f"VLM {decision.mode}: {decision.reason}" if self.config.vlm_only else description,
                    kind=QueryKind.NEXT_SUBTASK,
                    status_only=self.config.vlm_only,
                )
            )

    def _agent_feedback(self, status, message, decision, observation):
        self._vlm_feedback = {"status": status, "message": message, "executed": False}
        self.external_history.append((observation, json.dumps(asdict(decision)) if decision else message))
        self._begin_review(retain_endpoint=True)
        self._publish_answer(
            QueryAnswer(
                question=self._autosteer_goal,
                answer=f"VLM {status}: {message}; holding and requesting a revised action.",
                kind=QueryKind.NEXT_SUBTASK,
                status_only=True,
            )
        )

    def get_action(self, obs_frame):
        with self._query_lock:
            try:
                pose = self._pose()
                if self._manual_task is not None:
                    task, self._manual_task = self._manual_task, None
                    self._hold = pose
                    if self.config.vlm_only:
                        self.start_autosteer(task, self.config.policy_window_s)
                        self._reset_execution_feedback()
                        self._begin_review()
                    else:
                        self._start_policy(task, manual=True)
                if self._mode == "review_due" or (
                    self._mode == "policy" and time.perf_counter() >= self._due
                ):
                    self._begin_review()
                if self.config.review_policy_chunks:
                    self._poll_proposal()
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
                if self._mode == "approved":
                    if self.control_interpolator.needs_new_action():
                        if self._approved_index >= len(self._approved_actions):
                            count = len(self._approved_actions)
                            self._executed_steps += count
                            self._executed_seconds += count / self._proposal.fps
                            self._last_execution = {
                                "task": self._proposal.task,
                                "proposal_id": self._proposal.proposal_id,
                                "executed_steps": count,
                                "measured_start": self._execution_start,
                                "commanded_endpoint": dict(self._hold),
                            }
                            self._begin_review(retain_endpoint=True)
                        else:
                            self.control_interpolator.add(self._approved_actions[self._approved_index])
                            self._approved_index += 1
                    if self._mode == "approved":
                        action = self.control_interpolator.get()
                        self._set_dispatched_task(self._proposal.task)
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
                            for k in self._motion_keys
                        )
                        reached = reached and all(
                            self.kinematics[name].reached(goal, pose)
                            for name, goal in self._ee_targets.items()
                        )
                        expired = elapsed > self._motion_duration + self.config.settle_s
                        if self.config.vlm_only and (reached or expired):
                            self._last_execution = {
                                "measured_start": self._execution_start,
                                "commanded_endpoint": dict(self._target),
                                "reached": reached,
                            }
                            self._vlm_feedback = {
                                "status": "reached" if reached else "not_reached",
                                "residual_robot_units": {
                                    k: self._target[k] - pose[k] for k in self._motion_keys
                                },
                                "note": "Measured arrival, not object/task success. Reobserve before next move.",
                            }
                            # Stop pursuing an obstructed endpoint; let the next request
                            # reason from measured feedback rather than force contact.
                            self._hold = dict(self._target) if reached else pose
                            self._begin_review(retain_endpoint=True)
                            target = self._hold
                        elif reached:
                            self._begin_review()
                            target = self._hold
                        elif expired:
                            raise ValueError("Intervention did not reach its target in time")
                    return self._emit_position(target, "VLM intervention")
                if self._hold is None:
                    self._hold = pose
                return self._emit_position(self._hold, "VLM hold")
            except ReviewDriftError as exc:
                self._retry_drifted_review(exc)
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
