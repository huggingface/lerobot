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
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import partial
from threading import Lock, Thread
from typing import Any, Protocol

import torch

from lerobot.lerobot_types import EnvTransition
from lerobot.utils import QUERY_KIND, QUERY_TEXT

from .contracts import DEFAULT_MAX_TEXT_INPUT_CHARS, QueryKind

logger = logging.getLogger(__name__)

EXTERNAL_HISTORY_DEFAULT = 4


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


@dataclass(frozen=True)
class PolicyQuery:
    """A queued request for the policy's text head or an external text backend."""

    kind: QueryKind
    text: str
    task_version: int = 0
    intent_generation: int = 0
    history: tuple[tuple[dict, str], ...] = field(default=(), repr=False, compare=False)
    """External planner observations paired with the commands the engine accepted after them."""


@dataclass(frozen=True)
class PlannerReply:
    """What an external text backend returns: the answer text plus what the operator should see."""

    text: str
    detail: str = ""
    """The model's own assessment, such as its scene description."""
    completed: bool = False
    """The autosteer goal is complete: ``text`` is a status, not an instruction."""


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
    held: bool = False
    """A NEXT_SUBTASK answer that repeated the current instruction: nothing was sent."""
    completed: bool = False
    """The planner reports the autosteer goal complete: ``answer`` is a status, not an instruction."""
    detail: str = ""
    """What an external planner reported beside its answer, such as its scene assessment."""
    latency_s: float | None = None

    @property
    def ok(self) -> bool:
        """True when the policy produced an answer."""
        return self.error is None


class InferenceEngine(abc.ABC):
    """Action-production interface with shared task and language-query coordination.

    Lifecycle: ``start``/``stop`` own backend resources; ``reset`` clears episode
    state. Async ``pause`` revokes motion, while ``resume`` permits fresh work.
    The default pause/resume hooks are no-ops for inline inference.

    Actions: sync computes in ``get_action(obs_frame)`` on the control thread.
    Async engines receive owned captures through ``notify_observation`` and
    return buffered actions without waiting for their worker. A returned action
    is not dispatch permission: use rollout's ``send_next_action`` to check
    eligibility, invalidate interpolation and acknowledge supported holds.

    Tasks and queries: submission is thread-safe and never calls the policy.
    One owner serializes policy/processor work. ``pump_query`` runs on the control
    thread: it serves sync queries inline, or delivers async worker answers.
    The base also sequences autosteer and prevents superseded answers applying.
    Plain VQA discards buffered continuation everywhere: sync drops queued actions
    inline, async engines hold and invalidate first; subtasks can change the task.

    External text backend: with :attr:`external_text` set, every query is answered by
    that callable on a worker thread instead of the policy, while the robot keeps
    acting. Such requests live in their own slot, invisible to ``has_pending_query``,
    so no backend requests a language hold for them. The one hold they do cause is
    deliberate: an autosteer goal pauses the backend until the planner's first
    accepted reply, because the goal is context for the planner, not an instruction
    for the policy.

    Subclasses must call ``super().__init__`` and implement resource ownership,
    action production and query execution. Async implementations additionally
    enforce their deadlines, context validity and dispatch permission.
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

        # External text backend: an off-policy planner answering on its own worker thread.
        # It never touches the policy, so its requests get their own slot and the backends'
        # language-hold logic (keyed on ``has_pending_query``) never sees them.
        self.external_text: Callable[..., str | PlannerReply] | None = None
        # (observation, accepted command) pairs shown back to the planner. The engine fills
        # them; the planner configuration only sets how many are kept.
        self.external_history: deque[tuple[dict, str]] = deque(maxlen=EXTERNAL_HISTORY_DEFAULT)
        self._external_pending: PolicyQuery | None = None
        self._external_in_flight = False
        self._external_claimed: PolicyQuery | None = None
        # Bumped whenever the run context ends, so a worker result from before is discarded.
        self._external_generation = 0
        # Motion is revoked from autosteer start until the planner's first accepted reply.
        self._planner_hold = False
        self._planner_resume_due = False

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
        """Whether this backend accepts text queries.

        The base knows only the external backend; subclasses OR in their policy's text head.
        """
        return self.external_text is not None

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
        query = PolicyQuery(kind=QueryKind.VQA, text=question)
        if self.external_text is not None:
            return self._queue_external_query(query)
        return self._queue_query(query)

    def text_input_error(self, text: str, *, instruction: bool = False) -> str | None:
        """Validate without mutation: local queries allow 1–4096 characters.

        Local instructions are unrestricted; remote overrides use deployment limits.
        """
        if not instruction:
            if not text.strip():
                return "Enter a non-empty question or goal."
            if len(text) > DEFAULT_MAX_TEXT_INPUT_CHARS:
                return (
                    f"Text contains {len(text)} characters; the limit is {DEFAULT_MAX_TEXT_INPUT_CHARS}. "
                    "Shorten it and try again."
                )
        return None

    def start_autosteer(self, goal: str, interval_s: float) -> None:
        """Request subtasks for a fixed goal; callable from any thread.

        The interval starts when a subtask is applied. At least one action must
        dispatch before another turn; planner progress belongs to the policy.

        With an external text backend the goal is context for the planner, not an
        instruction for the policy: the backend is paused until the planner's first
        accepted reply, so no action predicted for the previous instruction is released
        after it. :meth:`pump_query` resumes the backend on the control thread.
        """
        if error := self.text_input_error(goal):
            raise ValueError(error)
        external = self.external_text is not None
        if external:
            # pause() also ends the external run context (see drop_pending_query), so the
            # hold is armed only after it.
            self.pause()
        with self._query_lock:
            self._autosteer_generation += 1
            if self._pending_query is not None and self._pending_query.kind is QueryKind.NEXT_SUBTASK:
                self._pending_query = None
            if self._external_pending is not None and self._external_pending.kind is QueryKind.NEXT_SUBTASK:
                self._external_pending = None
            self._autosteer_goal = goal
            self._autosteer_waiting_for_motion = False
            self._autosteer_interval_s = max(0.0, interval_s)
            # Due immediately: first subtask requested on the next control tick.
            self._autosteer_due_at = time.perf_counter()
            self.external_history.clear()
            self._planner_hold = external
            self._planner_resume_due = False
        logger.info("Autosteer started for goal '%s' (every %.1fs)", goal, interval_s)

    def stop_autosteer(self) -> str | None:
        """Stop the sequencer, returning the goal it was driving (or ``None``).

        A planner hold still in force is lifted: the operator took over before the
        planner spoke, so the policy moves again under the current instruction.
        """
        with self._query_lock:
            self._autosteer_generation += 1
            goal, self._autosteer_goal = self._autosteer_goal, None
            self._autosteer_waiting_for_motion = False
            if self._pending_query is not None and self._pending_query.kind is QueryKind.NEXT_SUBTASK:
                self._pending_query = None
            if self._external_pending is not None and self._external_pending.kind is QueryKind.NEXT_SUBTASK:
                self._external_pending = None
            self.external_history.clear()
            self._release_planner_hold()
        if goal is not None:
            logger.info("Autosteer stopped (goal was '%s')", goal)
        return goal

    def hold_for_planner(self) -> bool:
        """True while an external planner owns the instruction stream but has not spoken yet.

        :meth:`start_autosteer` pauses asynchronous backends for that interval; the
        synchronous backend consults this from ``dispatch_allowed``.
        """
        with self._query_lock:
            return self._planner_hold

    def drop_pending_query(self) -> PolicyQuery | None:
        """Discard an unclaimed query at run end; return it and queue one VQA cancellation notice.

        This also ends the external backend's run context: its unclaimed request is
        dropped the same way, a worker result still in flight will be discarded, the
        history is cleared and a planner hold is forgotten, since whoever paused or reset
        the backend owns its activation from here on.
        """
        with self._query_lock:
            dropped, self._pending_query = self._pending_query, None
            external, self._external_pending = self._external_pending, None
            for query in (dropped, external):
                if query is not None and query.kind is QueryKind.VQA:
                    self._ready_answers.append(
                        self._cancelled(query, "cancelled: the run changed before it could be answered")
                    )
            self._external_generation += 1
            self.external_history.clear()
            self._planner_hold = False
            self._planner_resume_due = False
        return dropped if dropped is not None else external

    @staticmethod
    def _cancelled(query: PolicyQuery, error: str) -> QueryAnswer:
        return QueryAnswer(question=query.text, error=error, kind=query.kind)

    def report_cancelled_query(self) -> None:
        """Queue one cancellation notice for a claimed VQA whose run context ended.

        Callable from the controller or policy worker after invalidation. The worker
        retains the busy slot until it finishes; observers still run only from the
        control-thread pump. Obsolete autosteer turns produce no announcement.
        """
        error = "cancelled: the instruction or run changed while it was being answered"
        with self._query_lock:
            query = self._claimed_query
            if query is not None and query.kind is QueryKind.VQA:
                self._claimed_query = None
                self._ready_answers.append(self._cancelled(query, error))
            query = self._external_claimed
            if query is not None and query.kind is QueryKind.VQA:
                self._external_claimed = None
                self._ready_answers.append(self._cancelled(query, error))

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
        and the sequencer does not advance.  External requests are handed to their worker
        thread here, and a backend paused for the planner is resumed here once the first
        reply is applied, so activation stays with the control thread.
        """
        self._poll_autosteer(obs_processed)
        served = False
        if self.control_thread_owns_policy:
            served = self._service_query(obs_processed)
        if self.external_text is not None:
            self._service_external_query(obs_processed)
        self._resume_after_planner(obs_processed)
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
            external = self.external_text is not None
            busy = (
                self._external_pending is not None or self._external_in_flight
                if external
                else self._pending_query is not None or self._query_in_flight
            )
            if busy:
                # A /vqa (or our own previous query) is still queued or being generated.
                # The deadline stays in the past, so the next tick retries this turn.
                return
            query = PolicyQuery(
                kind=QueryKind.NEXT_SUBTASK,
                text=self._autosteer_goal,
                task_version=self.task_version,
                intent_generation=self._autosteer_generation,
            )
            if external:
                self._external_pending = query
            else:
                self._pending_query = query

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

    def _autosteer_turn_is_current(self, query: PolicyQuery) -> bool:
        """Check sequencer identity while the caller holds ``_query_lock``."""
        return (
            self._autosteer_goal == query.text
            and self._autosteer_generation == query.intent_generation
            and self.task_version == query.task_version
        )

    def _fail_subtask(self, query: PolicyQuery) -> bool:
        """Stop the sequencer after a failed turn — unless it stopped or retargeted meanwhile.

        A sequencer that cannot get its next subtask must stop rather than fail every
        interval — but only if it is still the one that requested this turn.  Returns
        ``True`` when the failure answer should be published.
        """
        with self._query_lock:
            live = self._autosteer_turn_is_current(query)
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
            live = self._autosteer_turn_is_current(query)
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

    # ------------------------------------------------------------------
    # External text backend (off-policy planner)
    # ------------------------------------------------------------------

    def _queue_external_query(self, query: PolicyQuery) -> bool:
        with self._query_lock:
            if (
                self._external_pending is not None
                or self._external_in_flight
                or len(self._ready_answers) >= 16
            ):
                return False
            self._external_pending = replace(
                query, task_version=self.task_version, intent_generation=self._autosteer_generation
            )
        return True

    def _service_external_query(self, obs_processed: dict | None) -> bool:
        """Hand the pending external request to a worker thread.  Control thread only.

        The observation is copied first, so the backend sees the capture the request was
        made from rather than what the control loop overwrites next tick.  Returns ``True``
        when a request was claimed.
        """
        if obs_processed is None or self.external_text is None:
            return False
        with self._query_lock:
            if self._external_pending is None or self._external_in_flight:
                return False
            query, self._external_pending = self._external_pending, None
            self._external_in_flight = True
            self._external_claimed = query
            query = replace(query, history=tuple(self.external_history))
            generation = self._external_generation
            generate = partial(self.external_text, task=self.task)
        Thread(
            target=self._resolve_external_query,
            args=(query, deepcopy(obs_processed), generate, generation),
            name="external-text-query",
            daemon=True,
        ).start()
        return True

    def _resolve_external_query(
        self,
        query: PolicyQuery,
        obs_processed: dict,
        generate: Callable[[dict, PolicyQuery], str | PlannerReply],
        generation: int,
    ) -> None:
        """Run the external backend and apply its reply.  Worker thread.

        Cancellation and application are atomic under ``_query_lock``: a reply whose run
        context or sequencer ended meanwhile is discarded, so it can neither overwrite a
        newer instruction nor announce a stale plan, and a restart with the identical
        goal counts as a new sequencer.
        """
        started = time.perf_counter()
        try:
            result = generate(obs_processed, query)
            reply = result if isinstance(result, PlannerReply) else PlannerReply(text=result)
            if not isinstance(reply.text, str) or not reply.text.strip():
                # Fail here so garbage becomes an error answer instead of steering the robot.
                raise TypeError(
                    "the external text backend must return a non-empty str, "
                    f"got {reply.text!r} ({type(reply.text).__name__})"
                )
            answer = QueryAnswer(
                question=query.text,
                answer=reply.text.strip(),
                kind=query.kind,
                completed=reply.completed,
                detail=reply.detail,
            )
        except Exception as e:
            logger.exception("External text query failed (%s) for %r", query.kind.value, query.text)
            answer = QueryAnswer(question=query.text, error=f"{type(e).__name__}: {e}", kind=query.kind)
        answer = replace(answer, latency_s=round(time.perf_counter() - started, 3))
        with self._query_lock:
            if generation != self._external_generation:
                self._finish_external_query(None)  # the run ended; its VQA notice was queued then
                return
            if query.kind is not QueryKind.NEXT_SUBTASK:
                self._finish_external_query(answer)
                return
            if not self._autosteer_turn_is_current(query):
                logger.info(
                    "Discarding autosteer reply for %r — the sequencer stopped or was retargeted "
                    "while it was being generated",
                    query.text,
                )
                self._finish_external_query(None)
                return
            subtask = answer.answer if answer.ok and not answer.completed else None
            if subtask is not None and (error := self.text_input_error(subtask, instruction=True)):
                answer = replace(answer, answer=None, error=f"ValueError: unusable subtask: {error}")
                subtask = None
            if not answer.ok:
                self._autosteer_goal = None
                self._release_planner_hold()
                logger.info("Autosteer stopped (goal was '%s') — planning failed", query.text)
            elif subtask is None:
                # Completion is a normal outcome, not an instruction: the sequencer ends and
                # a paused backend stays paused until the operator resets or restarts.
                self._autosteer_goal = None
                logger.info("Autosteer stopped (goal was '%s') — the planner reports it complete", query.text)
            else:
                changed = self.set_task(subtask)
                # Armed only now, so the interval measures motion between subtasks; a fresh
                # action must also reach the robot before another turn.
                self._autosteer_due_at = time.perf_counter() + self._autosteer_interval_s
                self._autosteer_waiting_for_motion = True
                # A repeat enters the history too, so the planner can judge lack of progress.
                self.external_history.append((obs_processed, subtask))
                if not changed:
                    answer = replace(answer, held=True)
                    logger.info("Autosteer holds %r — nothing sent to the policy", subtask)
                self._release_planner_hold()
            self._finish_external_query(answer)

    def _finish_external_query(self, answer: QueryAnswer | None) -> None:
        """Free the external slot and publish ``answer`` when it is still wanted.

        ``None`` means the turn was discarded; a slot whose claim was already reported
        as cancelled publishes nothing either.  Caller holds ``_query_lock``.
        """
        reported = self._external_claimed is None
        self._external_in_flight = False
        self._external_claimed = None
        if answer is not None and not reported:
            self._ready_answers.append(answer)

    def _release_planner_hold(self) -> None:
        """Schedule the backend's resumption on the control thread.  Caller holds ``_query_lock``."""
        if self._planner_hold:
            self._planner_hold = False
            self._planner_resume_due = True

    def _resume_after_planner(self, obs_processed: dict | None) -> None:
        """Resume a backend paused for the planner once its hold was released.  Control thread only.

        The idle poll (``obs_processed=None``) runs between segments, where the backend
        must stay paused; :meth:`drop_pending_query` clears the flag at segment end.
        """
        if obs_processed is None:
            return
        with self._query_lock:
            due, self._planner_resume_due = self._planner_resume_due, False
        if due and not self.failed:
            self.resume()

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
        """Prepare resources on the control thread; async backends launch their owner worker."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Revoke execution and release resources; async worker joins must be bounded."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear episode state; async backends invalidate motion before queuing worker resets."""

    @abc.abstractmethod
    def get_action(self, obs_frame: dict | None) -> torch.Tensor | EnvTransition | None:
        """Return a canonical action on the control thread, or ``None`` when unavailable.

        Sync may block on inference; async only consumes buffered work. Check
        ``dispatch_allowed`` again before sending, including interpolation ticks.
        Like the policy itself: when the policy returned a prediction besides the action,
        the action comes back with it as an ``EnvTransition`` (see
        ``PreTrainedPolicy.select_action``).
        """

    def notify_observation(self, obs: dict) -> None:  # noqa: B027
        """Publish a control-thread capture; async engines copy it before returning. Default: no-op."""

    def pause(self) -> None:  # noqa: B027
        """Revoke async motion without waiting for the worker; caller clears interpolation. Default: no-op."""

    def resume(self) -> None:  # noqa: B027
        """Permit fresh async work in a healthy session; never clear a terminal fault. Default: no-op."""

    def dispatch_allowed(self) -> bool:
        """Check eligibility/deadlines without waiting for inference on every motor tick.

        False requires clearing interpolation and applying/acknowledging a
        supported local hold. The synchronous default permits dispatch.
        """
        return True

    def acknowledge_hold(self) -> None:  # noqa: B027
        """Confirm successful control-thread hold application after clearing interpolation.

        Async workers may then proceed or await a fresh capture. This never clears
        a fault or proves physical stillness. Default: no-op.
        """

    def record_dispatch(self) -> None:
        """Notify a successful policy-command send so the next autosteer turn may proceed."""
        with self._query_lock:
            self._autosteer_waiting_for_motion = False

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
