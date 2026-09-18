# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Nonblocking supervision of the existing language runtime.

Only the runtime thread touches robot I/O. Inference and remote reasoning consume
snapshots and submit revision-tagged results. All motion passes the same executor.
"""

from __future__ import annotations

import copy
import math
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from lerobot.utils.cycle_timer import CycleTimer


class Worker:
    """One bounded daemon worker; a stalled network request cannot stall robot control."""

    def __init__(self, fn: Callable):
        self.fn = fn
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.busy = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, item):
        if self.busy or self.closed.is_set():
            return False
        self.busy = True
        self.requests.put_nowait(item)
        return True

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def _run(self):
        while not self.closed.is_set():
            try:
                item = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                value, error = self.fn(item), None
            except Exception as exc:  # Worker errors are surfaced to the runtime, never hidden.
                value, error = None, f"{type(exc).__name__}: {exc}"
            if not self.closed.is_set():
                self.results.put((item, value, error))

    def close(self):
        self.closed.set()
        self.thread.join(timeout=0.2)


@dataclass
class ControlConfig:
    fps: float = 30.0
    inference_max_age_s: float = 2.0
    decision_max_age_s: float = 3.0
    supervisor_interval_s: float = 1.0
    joint_speed_deg_s: float = 15.0
    gripper_speed_deg_s: float = 90.0
    max_motion_s: float = 5.0
    max_episode_s: float = 120.0
    max_episodes: int = 50
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        for name in (
            "fps",
            "inference_max_age_s",
            "decision_max_age_s",
            "supervisor_interval_s",
            "joint_speed_deg_s",
            "gripper_speed_deg_s",
            "max_motion_s",
            "max_episode_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        for name, bounds in self.joint_limits.items():
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid joint limits for {name}")


@dataclass
class ControlState:
    task: str = ""
    language_context: dict = field(default_factory=dict)
    action_queue: deque = field(default_factory=deque)
    mode: str = "paused"
    revision: int = 0
    actions_dispatched: int = 0
    lock: object = field(default_factory=threading.RLock)

    def set_context(self, key, value):
        self.language_context[key] = value
        self.revision += 1


class HybridRuntime:
    """VLA by default, with asynchronous tool-based corrections and explicit handback."""

    def __init__(
        self,
        *,
        predict,
        observe,
        execute,
        config=None,
        supervisor=None,
        record=None,
        finish=None,
        event=None,
        solve_ik=None,
    ):
        self.config = config or ControlConfig()
        self.state = ControlState()
        self.observation_provider = observe
        self.action_executor = execute
        self._stop = False
        self._tick_index = 0
        self.state.mode = "paused"
        self.source = "policy"
        self._predict = Worker(predict)
        self._supervise = Worker(supervisor) if supervisor else None
        self._record = record
        self._finish = finish
        self._event = event or (lambda *_args: None)
        self._solve_ik = solve_ik
        self._commands = queue.Queue(maxsize=64)
        self._snapshot = None
        self._proposal = None
        self._seen_revision = self.state.revision
        self._last_supervision = -math.inf
        self._episode_start = None
        self._episodes = 0
        self._last_error = None
        self._last_result = None
        self._motion_id = None
        self._motion_instruction = None
        self._last_applied = None
        self._shutdown_done = False
        self._queue_deadline = None
        self._pending_events = deque(maxlen=32)

    def instruction(self):
        return (
            self._motion_instruction
            if self.source == "tool" and self._motion_instruction
            else self.state.language_context.get("subtask", self.state.task)
        )

    def _joint_speed(self, key):
        return (
            self.config.gripper_speed_deg_s if key.endswith("gripper.pos") else self.config.joint_speed_deg_s
        )

    def _log_event(self, kind, **data):
        event = {"kind": kind, "time": time.monotonic(), "revision": self.state.revision, **data}
        self._pending_events.append(event)
        self._event(event)

    def set_task(self, task):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("An instruction is required")
        with self.state.lock:
            self.state.task = task
            self.state.set_context("subtask", task)
            self._invalidate()
            self.source = "policy"
            self._log_event("instruction", text=task)

    def _invalidate(self):
        self.state.revision += 1
        self.state.action_queue.clear()
        self._queue_deadline = None
        self._proposal = None
        self._seen_revision = self.state.revision
        if self._motion_id:
            self._log_event("motion_cancelled", execution_id=self._motion_id)
            self._motion_id = None

    def snapshot(self):
        with self.state.lock:
            return copy.deepcopy(self._snapshot)

    def status(self):
        with self.state.lock:
            return {
                "mode": self.state.mode,
                "source": self.source,
                "task": self.state.task,
                "instruction": self.instruction(),
                "revision": self.state.revision,
                "episodes": self._episodes,
                "execution_id": self._motion_id,
                "last_error": self._last_error,
                "last_result": copy.deepcopy(self._last_result),
            }

    def submit(self, name, arguments, *, revision=None, observed_at=None, actor="agent"):
        """Queue a command; caller may wait on the returned event without owning robot I/O."""
        command = {
            "id": uuid.uuid4().hex,
            "name": name,
            "arguments": copy.deepcopy(arguments),
            "revision": revision,
            "observed_at": observed_at,
            "actor": actor,
            "done": threading.Event(),
            "result": None,
        }
        if self._stop:
            raise RuntimeError("Runtime is stopped")
        self._commands.put_nowait(command)
        return command

    def _apply_command(self, command):
        name, args = command["name"], command["arguments"]
        # A pause is always allowed. Late motion/steering must never override newer operator input.
        if name != "pause":
            if command["revision"] is not None and command["revision"] != self.state.revision:
                raise ValueError("Stale decision: instruction or controller changed")
            if (
                command["observed_at"] is not None
                and time.monotonic() - command["observed_at"] > self.config.decision_max_age_s
            ):
                raise ValueError("Stale decision: obtain a fresh observation")
        self._log_event("tool_call", name=name, arguments=args, actor=command["actor"], call_id=command["id"])
        if name == "pause":
            self.state.mode = "paused"
            self._invalidate()
        elif name in ("set_task", "steer", "resume_policy"):
            if self._episodes >= self.config.max_episodes:
                raise ValueError("Collection target reached; start a new session")
            if name == "set_task":
                self.set_task(args["instruction"])
            elif name == "steer":
                if not self.state.task or not args["instruction"].strip():
                    raise ValueError("Set a task and provide a nonempty steering instruction")
                self.state.set_context("subtask", args["instruction"])
                self._invalidate()
                self.source = "policy"
            else:
                if not self.state.task:
                    raise ValueError("Set an instruction first")
                self._invalidate()
                self.source = "policy"
            self.state.mode = "action"
        elif name in ("move_joints", "offset_joints", "set_gripper", "move_ee"):
            if self._snapshot is None:
                raise ValueError("No observation available")
            if self._episodes >= self.config.max_episodes:
                raise ValueError("Collection target reached")
            if not self.state.task:
                raise ValueError("Set the episode instruction before moving")
            joints = self._snapshot["joints"]
            if name == "move_ee":
                if self._solve_ik is None:
                    raise ValueError("No calibrated IK configuration installed")
                targets = self._solve_ik(args, joints)
            elif name == "set_gripper":
                targets = {args["joint"]: args["position_deg"]}
                if not args["joint"].endswith("gripper.pos"):
                    raise ValueError("set_gripper requires a gripper joint")
            else:
                targets = args["targets"]
                if name == "offset_joints":
                    targets = {key: joints[key] + value for key, value in targets.items()}
            duration = float(args["duration_s"])
            if not math.isfinite(duration) or not 0 < duration <= self.config.max_motion_s:
                raise ValueError("Motion duration is outside configured bounds")
            if not targets:
                raise ValueError("Motion must specify at least one joint")
            for key, value in targets.items():
                if key not in joints or key not in self.config.joint_limits:
                    raise ValueError(f"Unknown joint or missing configured limit: {key}")
                low, high = self.config.joint_limits[key]
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"Target outside joint limits: {key}")
                if abs(value - joints[key]) / duration > self._joint_speed(key):
                    raise ValueError(f"Motion exceeds configured joint speed: {key}")
            self._invalidate()
            steps = max(1, math.ceil(duration * self.config.fps))
            for step in range(1, steps + 1):
                self.state.action_queue.append(
                    {
                        **joints,
                        **{
                            key: joints[key] + (value - joints[key]) * step / steps
                            for key, value in targets.items()
                        },
                    }
                )
            self._motion_instruction = args.get("instruction") or self.state.language_context.get(
                "subtask", self.state.task
            )
            self.source = "tool"
            self.state.mode = "action"
            self._motion_id = command["id"]
        elif name == "finish_episode":
            outcome = args["outcome"]
            if outcome not in ("success", "failure", "unknown"):
                raise ValueError("outcome must be success, failure, or unknown")
            evidence = args.get("evidence", "").strip()
            if outcome != "unknown" and not evidence:
                raise ValueError("A labeled outcome requires evidence")
            self._finish_episode(outcome, evidence, command["actor"])
        else:
            raise ValueError(f"Unknown control tool: {name}")
        return {"call_id": command["id"], **self.status()}

    def _finish_episode(self, outcome, evidence, actor):
        self.state.mode = "paused"
        self._invalidate()
        if self._episode_start is None:
            raise ValueError("No episode has started")
        result = {
            "outcome": outcome,
            "evidence": evidence,
            "label_source": actor,
            "duration_s": time.monotonic() - self._episode_start,
            "task": self.state.task,
        }
        if self._finish:
            self._finish(result)
        self._episodes += 1
        self._episode_start = None
        self._last_result = result
        self._log_event("episode_finished", **result)

    def step_once(self):
        with self.state.lock:
            self._tick_index += 1
            if self._stop:
                return
            if self._seen_revision != self.state.revision:
                self._invalidate()
            try:
                raw = self.observation_provider()
                if raw is None:
                    raise RuntimeError("Observation unavailable")
                joints = {key: float(value) for key, value in raw.items() if key.endswith(".pos")}
                if not joints or not all(math.isfinite(value) for value in joints.values()):
                    raise ValueError("Robot observation has missing or nonfinite joints")
                self._snapshot = {
                    "id": self._tick_index,
                    "observed_at": time.monotonic(),
                    "revision": self.state.revision,
                    "task": self.state.task,
                    "joints": joints,
                    "raw": raw,
                    "instruction": self.instruction(),
                    "status": self.status(),
                    "events": list(self._pending_events),
                    "proposal": self._proposal,
                }
            except Exception as exc:
                self._last_error = f"Observation failed: {exc}"
                self.state.mode = "paused"
                self._invalidate()
                self._snapshot = None
                self._log_event("error", message=self._last_error)
                return
            for _ in range(64):
                try:
                    command = self._commands.get_nowait()
                except queue.Empty:
                    break
                try:
                    command["result"] = self._apply_command(command)
                except Exception as exc:
                    command["result"] = {"error": str(exc), "call_id": command["id"]}
                finally:
                    command["done"].set()
            self._snapshot.update(
                revision=self.state.revision,
                task=self.state.task,
                instruction=self.instruction(),
                status=self.status(),
            )
            if self._supervise:
                result = self._supervise.poll()
                if result:
                    snapshot, calls, error = result
                    if error:
                        self._log_event("supervisor_error", message=error)
                    else:
                        # Commands are applied on the next tick, with freshness checked at application time.
                        for call in calls or []:
                            self.submit(
                                call["name"],
                                call["arguments"],
                                revision=snapshot["revision"],
                                observed_at=snapshot["observed_at"],
                                actor="supervisor",
                            )
                now = time.monotonic()
                if (
                    self.state.mode == "action"
                    and now - self._last_supervision >= self.config.supervisor_interval_s
                ):
                    snapshot = self.snapshot()
                    snapshot.update(revision=self.state.revision, task=self.state.task, status=self.status())
                    if self._supervise.submit(snapshot):
                        self._last_supervision = now
            result = self._predict.poll()
            if result:
                request, actions, error = result
                if error and request["revision"] == self.state.revision:
                    self._last_error = error
                    self._log_event("inference_error", message=error)
                    self.state.mode = "paused"
                    self._invalidate()
                elif (
                    not error
                    and request["revision"] == self.state.revision
                    and self.state.mode == "action"
                    and self.source == "policy"
                    and time.monotonic() - request["observed_at"] <= self.config.inference_max_age_s
                ):
                    self._proposal = {
                        "observation_id": request["id"],
                        "revision": request["revision"],
                        "actions": actions,
                    }
                    self._log_event("proposal", **self._proposal)
                    self.state.action_queue.extend(actions)
                    self._queue_deadline = request["observed_at"] + self.config.inference_max_age_s
                else:
                    self._log_event("proposal_discarded", observation_id=request["id"])
            if (
                self._episode_start is not None
                and time.monotonic() - self._episode_start >= self.config.max_episode_s
            ):
                self._finish_episode("unknown", "Episode deadline reached", "runtime")
                return
            if self.state.mode != "action":
                self._record_hold()
                return
            if self._episode_start is None:
                self._episode_start = time.monotonic()
                self._log_event("episode_started", task=self.state.task)
            if time.monotonic() - self._episode_start >= self.config.max_episode_s:
                self._finish_episode("unknown", "Episode deadline reached", "runtime")
                return
            if (
                self.source == "policy"
                and self._queue_deadline is not None
                and time.monotonic() > self._queue_deadline
            ):
                self.state.action_queue.clear()
                self._queue_deadline = None
            if self.source == "policy" and not self.state.action_queue:
                request = self.snapshot()
                request.update(revision=self.state.revision, task=self.instruction())
                self._predict.submit(request)
            if not self.state.action_queue:
                self._record_hold()
                return
            action = self.state.action_queue.popleft()
            try:
                if set(action) != set(joints) or not all(math.isfinite(value) for value in action.values()):
                    raise ValueError("Action must contain exactly the observed joints with finite values")
                bounded = {}
                for key, value in action.items():
                    if key not in self.config.joint_limits:
                        raise ValueError(f"No configured joint limit for {key}")
                    low, high = self.config.joint_limits[key]
                    step = self._joint_speed(key) / self.config.fps
                    bounded[key] = min(
                        high, max(low, min(joints[key] + step, max(joints[key] - step, value)))
                    )
                applied = self.action_executor(bounded)
                if not isinstance(applied, dict):
                    raise ValueError("Executor must return the action actually sent to the robot")
                if set(applied) != set(joints) or not all(math.isfinite(value) for value in applied.values()):
                    raise ValueError("Executor returned incomplete or nonfinite applied actions")
                applied = {key: float(value) for key, value in applied.items()}
                self._last_applied = applied
                if self._record:
                    self._record(self._snapshot, action, applied, self.source, self.instruction())
            except Exception as exc:
                self._last_error = f"Execution/recording failed: {exc}"
                self.state.mode = "paused"
                self._invalidate()
                self._log_event("error", message=self._last_error)
                return
            self.state.actions_dispatched += 1
            if self.source == "tool" and not self.state.action_queue:
                self._log_event(
                    "motion_finished",
                    execution_id=self._motion_id,
                    outcome="commands_dispatched",
                    followup_observation_required=True,
                )
                self._motion_id = None
                self.state.mode = "paused"  # Handback is an explicit resume_policy call.

    def _record_hold(self):
        if self._record and self._episode_start is not None and self._last_applied is not None:
            try:
                self._record(
                    self._snapshot, self._last_applied, self._last_applied, "hold", self.instruction()
                )
            except Exception as exc:
                self._last_error = f"Recording failed: {exc}"
                self.state.mode = "paused"
                self._invalidate()

    def stop(self):
        self._stop = True

    def run(self, *, max_ticks=None):
        timer = CycleTimer(self.config.fps)
        try:
            while not self._stop:
                timer.tick(new_cycle=True)
                self.step_once()
                if max_ticks is not None and self._tick_index >= max_ticks:
                    break
                timer.wait()
        finally:
            self.stop()
            self._on_shutdown()

    def _on_shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True
        try:
            with self.state.lock:
                self.state.mode = "paused"
                self._invalidate()
                if self._episode_start is not None:
                    self._finish_episode("unknown", "Session stopped", "runtime")
        finally:
            self._predict.close()
            if self._supervise:
                self._supervise.close()
            while not self._commands.empty():
                command = self._commands.get_nowait()
                command["result"] = {"error": "Runtime stopped"}
                command["done"].set()
