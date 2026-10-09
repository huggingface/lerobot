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

"""Inference lifecycle, task ownership and serialized language-query interface."""

from __future__ import annotations

import abc
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Protocol

import torch

from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT

from .contracts import QueryKind as QueryKind

logger = logging.getLogger(__name__)


class InferenceRobot(Protocol):
    """Hardware metadata and waiting capability, without a dependency on rollout."""

    @property
    def action_features(self) -> dict[str, Any]:
        """Return the robot's named actuator features."""
        ...

    @property
    def robot_type(self) -> str:
        """Return the robot type used to condition the policy."""
        ...

    @property
    def observation_time(self) -> float | None:
        """Return the monotonic sample bound of the latest observation."""
        ...

    @property
    def supports_hold(self) -> bool:
        """Return whether local waiting behavior is configured."""
        ...

    @property
    def supports_position_hold(self) -> bool:
        """Return whether the robot declares compatible position control."""
        ...

    def configure_position_hold(self) -> None:
        """Validate and enable the robot's local waiting behavior."""
        ...

    def configure_hold(self) -> None:
        """Enable the driver's declared command or position hold behavior."""
        ...


@dataclass(frozen=True)
class PolicyQuery:
    """A queued request for the policy's text head."""

    kind: QueryKind
    text: str
    task_version: int = 0
    intent_generation: int = 0


@dataclass(frozen=True)
class QueryAnswer:
    """Result of a policy text query.

    Exactly one of ``answer`` / ``error`` is set.  A ``NEXT_SUBTASK`` success carries a
    subtask the engine has *already applied*, so the receiver only announces it.
    """

    question: str
    answer: str | None = None
    error: str | None = None
    kind: QueryKind = QueryKind.VQA

    @property
    def ok(self) -> bool:
        """True when the policy produced an answer."""
        return self.error is None


class InferenceEngine(abc.ABC):
    """Rollout backend with thread-safe task/query submission and exclusive policy ownership.

    Async backends receive captures through ``notify_observation``; sync uses
    ``get_action(obs_frame)``. Serve queries on the policy owner and deliver answers
    through ``pump_query`` on the control thread. Subclasses must call ``super``.

    ``get_action`` alone does not authorize motor dispatch. Strategies must use
    ``send_next_action`` or preserve its per-tick permission, interpolation
    invalidation, local hold/acknowledgment and applied-command notification.
    """

    def __init__(self, task: str = "") -> None:
        """Initialize task ownership and serialized language-query bookkeeping."""
        self._task = task
        self._task_changed = False
        self._task_version = 0
        self._dispatched_task = task
        self._task_lock = Lock()

        # Text-query channel.  Its own lock, never held across a text generation.
        self._query_lock = Lock()
        self._pending_query: PolicyQuery | None = None
        # Set from the claim (``_take_query``) until the answer is published or the turn
        # is discarded, so the autosteer poll cannot queue a duplicate turn meanwhile.
        self._query_in_flight = False
        # Kept separately from the busy flag: a controller can report cancellation
        # before a blocked worker finishes, without allowing another query to run.
        self._claimed_query: PolicyQuery | None = None
        # Answers awaiting delivery; a queue so an undelivered one is never overwritten.
        self._ready_answers: deque[QueryAnswer] = deque()
        self._answer_observer: Callable[[QueryAnswer], None] | None = None

        # Autosteer sequencer state (same lock: it writes _pending_query).
        self._autosteer_goal: str | None = None
        self._autosteer_interval_s: float = 0.0
        self._autosteer_due_at: float = 0.0
        self._autosteer_generation = 0
        self._autosteer_waiting_for_motion = False

    # ------------------------------------------------------------------
    # Task (language instruction)
    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        """The language instruction currently conditioning inference."""
        with self._task_lock:
            return self._task

    def set_task(self, task: str) -> bool:
        """Set the instruction used from the next inference onwards.

        Callable from any thread.  Returns ``True`` when the value actually changed.
        """
        with self._task_lock:
            if task == self._task:
                return False
            previous, self._task = self._task, task
            self._task_changed = True
            self._task_version += 1
        logger.info("Task changed: '%s' -> '%s'", previous, task)
        return True

    @property
    def task_version(self) -> int:
        """Monotonic instruction identity, including changes back to the same text."""
        with self._task_lock:
            return self._task_version

    @property
    def query_intent_generation(self) -> int:
        """Return the generation used to discard superseded query results."""
        with self._query_lock:
            return self._autosteer_generation

    def _take_task(self) -> tuple[str, bool]:
        """Read the task and consume the "changed" edge.  Call from the inference thread."""
        with self._task_lock:
            changed, self._task_changed = self._task_changed, False
            return self._task, changed

    @property
    def dispatched_task(self) -> str:
        """Task attached to the last returned action, used for recording labels.

        Read on the control thread after ``get_action``; it can lag the requested
        task while eligible old actions finish. Reset restores the requested task.
        """
        with self._task_lock:
            return self._dispatched_task

    def _set_dispatched_task(self, task: str) -> None:
        """Record the task of the action a ``get_action`` call is returning."""
        with self._task_lock:
            self._dispatched_task = task

    def _discard_task_change(self) -> None:
        """Clear the task-change edge and restore the requested task as the recording label."""
        with self._task_lock:
            self._task_changed = False
            self._dispatched_task = self._task
        with self._query_lock:
            self._autosteer_waiting_for_motion = False

    # ------------------------------------------------------------------
    # Text queries (VQA)
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """Whether this backend accepts policy text queries; false by default."""
        return False

    def set_answer_observer(self, observer: Callable[[QueryAnswer], None] | None) -> None:
        """Register the callback :meth:`pump_query` hands ready answers to."""
        with self._query_lock:
            self._answer_observer = observer

    @property
    def has_pending_query(self) -> bool:
        """True while a query is queued and not yet served."""
        with self._query_lock:
            return self._pending_query is not None

    @property
    def autosteer_goal(self) -> str | None:
        """The high-level goal currently driving the task, if any."""
        with self._query_lock:
            return self._autosteer_goal

    def ask(self, question: str) -> bool:
        """Queue one operator VQA from any thread; reject invalid input or a busy channel."""
        if self.text_input_error(question) is not None:
            return False
        return self._queue_query(
            PolicyQuery(kind=QueryKind.VQA, text=question, task_version=self.task_version)
        )

    def text_input_error(self, text: str, *, instruction: bool = False) -> str | None:
        """Validate without mutation: local queries allow 1–4096 characters.

        Local instructions are unrestricted; remote overrides use deployment limits.
        """
        if not instruction:
            if not text.strip():
                return "Enter a non-empty question or goal."
            if len(text) > 4096:
                return f"Text contains {len(text)} characters; the limit is 4096. Shorten it and try again."
        return None

    def start_autosteer(self, goal: str, interval_s: float) -> None:
        """Request subtasks for a fixed goal; callable from any thread.

        The interval starts when a subtask is applied. At least one action must
        dispatch before another turn; planner progress belongs to the policy.
        """
        if error := self.text_input_error(goal):
            raise ValueError(error)
        with self._query_lock:
            self._autosteer_generation += 1
            if self._pending_query is not None and self._pending_query.kind is QueryKind.NEXT_SUBTASK:
                self._pending_query = None
            self._autosteer_goal = goal
            self._autosteer_waiting_for_motion = False
            self._autosteer_interval_s = max(0.0, interval_s)
            # Due immediately: first subtask requested on the next control tick.
            self._autosteer_due_at = time.perf_counter()
        logger.info("Autosteer started for goal '%s' (every %.1fs)", goal, interval_s)

    def stop_autosteer(self) -> str | None:
        """Stop the sequencer, returning the goal it was driving (or ``None``)."""
        with self._query_lock:
            self._autosteer_generation += 1
            goal, self._autosteer_goal = self._autosteer_goal, None
            self._autosteer_waiting_for_motion = False
            if self._pending_query is not None and self._pending_query.kind is QueryKind.NEXT_SUBTASK:
                self._pending_query = None
        if goal is not None:
            logger.info("Autosteer stopped (goal was '%s')", goal)
        return goal

    def drop_pending_query(self) -> PolicyQuery | None:
        """Discard an unclaimed query at run end; return it and queue one VQA cancellation notice."""
        with self._query_lock:
            dropped, self._pending_query = self._pending_query, None
            if dropped is not None and dropped.kind is QueryKind.VQA:
                self._ready_answers.append(
                    QueryAnswer(
                        question=dropped.text,
                        error="cancelled: the run changed before it could be answered",
                        kind=dropped.kind,
                    )
                )
        return dropped

    def report_cancelled_query(self) -> None:
        """Queue one cancellation notice for a claimed VQA whose run context ended.

        Callable from the controller or policy worker after invalidation. The worker
        retains the busy slot until it finishes; observers still run only from the
        control-thread pump. Obsolete autosteer turns produce no announcement.
        """
        with self._query_lock:
            query = self._claimed_query
            if query is not None and query.kind is QueryKind.VQA:
                self._claimed_query = None
                self._ready_answers.append(
                    QueryAnswer(
                        question=query.text,
                        error="cancelled: the instruction or run changed while it was being answered",
                        kind=query.kind,
                    )
                )

    def _discard_invalid_query(self) -> None:
        """Finish an obsolete worker turn, reporting only an unanswered operator VQA."""
        self.report_cancelled_query()
        with self._query_lock:
            self._query_in_flight = False
            self._claimed_query = None

    @property
    @abc.abstractmethod
    def control_thread_owns_policy(self) -> bool:
        """Whether the control thread is the one allowed to touch the policy.

        True (inline backends): :meth:`pump_query` serves pending queries itself.  False
        (async backends): their inference thread must call :meth:`_service_query`, and
        :meth:`pump_query` only advances the sequencer and delivers finished answers.
        """

    def pump_query(self, obs_processed: dict | None = None) -> bool:
        """Advance the text-query channel by one tick.  Control thread only.

        Polls the autosteer sequencer, serves a pending query when
        :attr:`control_thread_owns_policy` (async backends answer on their own thread),
        then delivers ready answers, so observers always fire on this thread.  Called at
        the end of a tick rather than from :meth:`get_action`: a text generation far
        outlasts a control tick.  Returns ``True`` when a query was served inline.  With
        ``obs_processed=None`` (the controller's idle poll) a pending query stays queued
        and the sequencer does not advance.
        """
        self._poll_autosteer(obs_processed)
        served = False
        if self.control_thread_owns_policy:
            served = self._service_query(obs_processed)
        self._deliver_answer()
        return served

    def _queue_query(self, query: PolicyQuery) -> bool:
        with self._query_lock:
            if self._pending_query is not None or self._query_in_flight or len(self._ready_answers) >= 16:
                return False
            if query.kind is QueryKind.VQA:
                query = replace(
                    query, task_version=self.task_version, intent_generation=self._autosteer_generation
                )
            self._pending_query = query
        return True

    def _poll_autosteer(self, obs_processed: dict | None) -> None:
        """Queue the next-subtask query if the sequencer is due (control loop only)."""
        if obs_processed is None:
            return
        with self._query_lock:
            if self._autosteer_goal is None or self._autosteer_waiting_for_motion:
                return
            if time.perf_counter() < self._autosteer_due_at:
                return
            if self._pending_query is not None or self._query_in_flight:
                # A /vqa (or our own previous query) is still queued or being generated.
                # The deadline stays in the past, so the next tick retries this turn.
                return
            self._pending_query = PolicyQuery(
                kind=QueryKind.NEXT_SUBTASK,
                text=self._autosteer_goal,
                task_version=self.task_version,
                intent_generation=self._autosteer_generation,
            )

    def _take_query(self) -> PolicyQuery | None:
        """Claim the pending query.  Call from the policy-owning thread."""
        with self._query_lock:
            query, self._pending_query = self._pending_query, None
            if query is not None:
                self._query_in_flight = True
                self._claimed_query = query
            return query

    def _service_query(self, obs_processed: dict | None) -> bool:
        """Serve a pending query.  Call ONLY from the policy-owning thread.

        Failures become error answers instead of exceptions, so a bad query never takes
        down the calling thread.  Returns ``True`` when a query was claimed and served.
        """
        if obs_processed is None:
            return False
        query = self._take_query()
        if query is None:
            return False
        try:
            text = self._generate_text(obs_processed, query)
            if not self._query_context_valid(query):
                self._discard_invalid_query()
                return True
            if not isinstance(text, str) or not text.strip():
                # Fail here so garbage becomes an error answer instead of steering the
                # robot and labeling recorded frames.
                raise TypeError(
                    f"generate_text() must return a non-empty str, got {text!r} ({type(text).__name__})"
                )
        except Exception as e:
            if not self._query_context_valid(query):
                self._discard_invalid_query()
                return True
            logger.exception("Policy text query failed (%s) for %r", query.kind.value, query.text)
            if query.kind is QueryKind.NEXT_SUBTASK and not self._fail_subtask(query):
                return True  # the sequencer this turn belonged to is gone; discard
            self._publish_answer(
                QueryAnswer(question=query.text, error=f"{type(e).__name__}: {e}", kind=query.kind)
            )
            return True
        if query.kind is QueryKind.NEXT_SUBTASK and not self._apply_subtask(query, text):
            return True  # sequencer stopped meanwhile; the turn was discarded
        # Published after being applied, so an announcing observer never gets ahead of
        # the task it describes.
        self._publish_answer(QueryAnswer(question=query.text, answer=text, kind=query.kind))
        return True

    def _query_context_valid(self, query: PolicyQuery) -> bool:
        """Whether a completed query may be applied or published by this backend.

        Async engines additionally validate execution generation, instruction, and
        operator intent. Obsolete success and error payloads are discarded; VQA
        cancellation is reported once without exposing the old answer.
        """
        return True

    def _fail_subtask(self, query: PolicyQuery) -> bool:
        """Stop the sequencer after a failed turn — unless it stopped or retargeted meanwhile.

        A sequencer that cannot get its next subtask must stop rather than fail every
        interval — but only if it is still the one that requested this turn.  Returns
        ``True`` when the failure answer should be published.
        """
        with self._query_lock:
            live = (
                self._autosteer_goal == query.text
                and self._autosteer_generation == query.intent_generation
                and self.task_version == query.task_version
            )
            if live:
                self._autosteer_goal = None
            else:
                self._query_in_flight = False  # no answer will be published
                self._claimed_query = None
        if live:
            logger.info("Autosteer stopped (goal was '%s') — planning failed", query.text)
        else:
            logger.info(
                "Discarding failed autosteer turn for %r — the sequencer stopped or was "
                "retargeted while it was being generated",
                query.text,
            )
        return live

    def _apply_subtask(self, query: PolicyQuery, subtask: str) -> bool:
        """Apply a generated subtask, unless the sequencer stopped meanwhile.

        The generation ran lock-free for seconds, so check and apply happen atomically
        under ``_query_lock`` (``_task_lock`` nests inside it, never the reverse) or a
        stale plan could overwrite a newer instruction.  Returns ``True`` when applied.
        """
        with self._query_lock:
            live = (
                self._autosteer_goal == query.text
                and self._autosteer_generation == query.intent_generation
                and self.task_version == query.task_version
            )
            if live:
                self.set_task(subtask)
                # Armed only now, so the interval measures motion between subtasks.
                self._autosteer_due_at = time.perf_counter() + self._autosteer_interval_s
                # Even interval=0 must let a fresh action reach the robot before
                # another planned language hold can consume the run.
                self._autosteer_waiting_for_motion = True
            else:
                self._query_in_flight = False  # no answer will be published
                self._claimed_query = None
        if not live:
            logger.info(
                "Discarding autosteer subtask %r — the sequencer stopped while it was being generated",
                subtask,
            )
        return live

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Generate text on the policy-owning thread using the backend's processor path."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support text queries — no /vqa or /autosteer on this backend."
        )

    @staticmethod
    def _mark_query(batch: dict, query: PolicyQuery) -> dict:
        """Add complementary query kind/text after observation preparation, before preprocessing."""
        batch[QUERY_KIND] = query.kind.value
        batch[QUERY_TEXT] = query.text
        return batch

    def drop_ready_subtask_answers(self) -> None:
        """Discard undelivered subtask announcements at segment end; preserve VQA answers."""
        with self._query_lock:
            kept = [a for a in self._ready_answers if a.kind is not QueryKind.NEXT_SUBTASK]
            dropped = len(self._ready_answers) - len(kept)
            self._ready_answers = deque(kept)
        if dropped:
            logger.debug("Dropped %d undelivered autosteer answer(s) at segment end", dropped)

    def _publish_answer(self, answer: QueryAnswer) -> None:
        with self._query_lock:
            cancelled = self._query_in_flight and self._claimed_query is None
            self._query_in_flight = False
            self._claimed_query = None
            if not cancelled:
                self._ready_answers.append(answer)

    def _deliver_answer(self) -> None:
        with self._query_lock:
            answers = list(self._ready_answers)
            self._ready_answers.clear()
            observer = self._answer_observer
        if observer is None:
            return
        for answer in answers:
            try:
                observer(answer)
            except Exception:  # a broken observer must not kill the control loop
                logger.exception("Error in inference-engine answer observer")

    @abc.abstractmethod
    def start(self) -> None:
        """Initialise the backend."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Tear the backend down."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear episode-scoped state."""

    @abc.abstractmethod
    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return the next action tensor, or ``None`` if unavailable."""

    def notify_observation(self, obs: dict) -> None:  # noqa: B027
        """Publish the latest processed observation.  Default: no-op."""

    def pause(self) -> None:  # noqa: B027
        """Pause background inference.  Default: no-op."""

    def resume(self) -> None:  # noqa: B027
        """Resume background inference.  Default: no-op."""

    def dispatch_allowed(self) -> bool:
        """Control-thread gate, checked on every motor tick, including interpolation."""
        return True

    def acknowledge_hold(self) -> None:  # noqa: B027
        """Confirm that the control thread invalidated interpolation and applied a local hold."""

    def record_dispatch(self, canonical: dict, command: dict, measured: dict) -> None:
        """Observe an actual robot command and allow the next due autosteer turn."""
        with self._query_lock:
            self._autosteer_waiting_for_motion = False

    def begin_control_tick(self) -> None:  # noqa: B027
        """Advance optional provenance for a motor tick, including held ticks."""

    @property
    def ready(self) -> bool:
        """True once the backend can produce actions (e.g. warmup done)."""
        return True

    @property
    def failed(self) -> bool:
        """True if an unrecoverable error occurred in the backend."""
        return False

    @property
    def failure_traceback(self) -> str | None:
        """Formatted traceback of the unrecoverable error, when ``failed`` is True."""
        return None
