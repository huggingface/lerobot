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

"""Single-flight native-tool agent, with hardware access confined to rollout's thread."""

from __future__ import annotations

import json
import time
import traceback
from collections import deque
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from threading import Event, RLock, Thread
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

from lerobot.utils.import_utils import _inspect_robots_agent_available, require_package

from ..inference.base import InferenceEngine
from .adapter import RobotAdapter
from .configuration import AgentSettings

if TYPE_CHECKING or _inspect_robots_agent_available:
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene
    from inspect_robots_agent.policy import LLMAgentPolicy


class AgentInferenceEngine(InferenceEngine):
    """One measured observation → one native tool motion → fresh observation.

    HTTP, conversation mutation and IK only run on the worker. A generation check
    gates every result, including done/give_up. The worker never touches hardware.
    """

    def __init__(
        self,
        config: AgentSettings,
        *,
        keys: list[str],
        features: dict,
        robot_type: str,
        fps: float,
        task: str,
        transport=None,
        env=None,
    ):
        super().__init__(task)
        require_package("inspect-robots-agent", extra="agent", import_name="inspect_robots_agent")
        self.config = config
        self.adapter = RobotAdapter(config, keys, features, robot_type, fps)
        # Omit unset values: upstream distinguishes omitted effort from explicit
        # None (which disables thinking), and chooses a protocol per provider.
        provider_options = {
            key: value for key in ("wire", "effort") if (value := getattr(config, key)) is not None
        }
        self.agent = LLMAgentPolicy(
            model=config.model,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            **provider_options,
            service_tier=config.service_tier,
            max_llm_calls=config.max_llm_calls,
            max_retries=config.max_retries,
            max_speed_frac=config.max_speed_frac,
            images=config.images,
            depth="off",
            transcript_echo=config.transcript_echo,
            prior_learnings=config.prior_learnings,
            wire_capture=False,
            pre_check=self.adapter.pre_check,
            transport=transport,
            env=env,
        )
        self.agent.bind(self.adapter.info)  # validates tool schema before robot.connect()
        self._lock = RLock()
        self._wake, self._closed = Event(), Event()
        self._thread: Thread | None = None
        self._paused = True
        self._generation = self._session = 0
        self._queue: deque[np.ndarray] = deque()
        self._request: tuple[int, int, dict, str, int, list[dict], list[dict]] | None = None
        self._inflight = False
        self._latest: dict | None = None
        self._observed_at = 0.0
        self._held: np.ndarray | None = None
        self._due = 0.0
        self._steps = 0
        self._feedback: list[dict] = []
        self._approvals: list[dict] = []
        self._completion: str | None = None
        self._failure: str | None = None
        self._run_id = uuid4().hex
        self._directory = Path(config.log_dir).expanduser() / self._run_id
        self._directory.mkdir(parents=True, exist_ok=False)
        self._write(
            "config.json",
            {
                "config": asdict(config),
                "robot_type": robot_type,
                "action_keys": keys,
                "fps": fps,
                "task": task,
                "agent_version": version("inspect-robots-agent"),
                "core_version": version("inspect-robots"),
                "embodiment_notes": self.adapter.info.docs,
            },
        )

    @property
    def control_thread_owns_policy(self) -> bool:
        return False

    @property
    def completion(self) -> str | None:
        with self._lock:
            return self._completion

    @property
    def return_home_on_completion(self) -> bool:
        return self.config.return_home_on_done

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._failure is not None

    @property
    def failure_traceback(self) -> str | None:
        with self._lock:
            return self._failure

    def start(self) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = Thread(target=self._run, name="lerobot-agent", daemon=True)
                self._thread.start()

    def stop(self) -> None:
        self.pause()
        self._closed.set()
        self._wake.set()
        if self._thread is not None:
            # An HTTP request may still be finishing. It cannot publish or command hardware.
            self._thread.join(timeout=1)

    def _invalidate(self, *, new_session: bool) -> None:
        self._generation += 1
        self._queue.clear()
        self._request = None
        self._completion = None
        self._due = 0.0
        if new_session:
            self._session += 1
            self._steps = 0
            self._held = None
            self._feedback.clear()
            self._approvals.clear()

    def reset(self) -> None:
        with self._lock:
            self._invalidate(new_session=True)
            self._paused = True
            self._latest = None
        self._discard_task_change()

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._generation += 1
            self._queue.clear()
            self._request = None

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def set_task(self, task: str) -> bool:
        with self._lock:
            changed = super().set_task(task)
            # Even reasserting the same goal is an explicit new attempt.
            self._invalidate(new_session=True)
            return changed

    def add_feedback(self, text: str) -> bool:
        with self._lock:
            if self._paused or self._closed.is_set() or not text.strip():
                return False
            self._invalidate(new_session=False)
            self._feedback.append({"t": self._steps, "text": text, "source": "operator"})
            self._approvals.append(
                {
                    "t": self._steps,
                    "detail": "Operator feedback interrupted the preceding motion/reply. Only measured state shows what actually executed.",
                }
            )
            return True

    def notify_observation(self, obs: dict) -> None:
        with self._lock:
            self._latest, self._observed_at = obs, time.monotonic()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        with self._lock:
            if self._paused or self._closed.is_set() or self._completion:
                return None
            if self._failure:
                raise RuntimeError(f"Agent failed: {self._failure}")
            now = time.monotonic()
            if self._latest is None or now - self._observed_at > self.config.observation_timeout_s:
                raise RuntimeError("Agent requires fresh measured robot feedback")
            measured = self.adapter.vector(self._latest)
            if self._held is None:
                self._held = np.clip(measured, self.adapter.low, self.adapter.high)
            if np.any(np.abs(measured - self._held) > self.adapter.tolerance):
                raise RuntimeError("Agent joint tracking tolerance exceeded")
            if self._queue:
                self._held = self._queue.popleft()
                self._steps += 1
                if not self._queue:
                    self._due = now + self.config.settle_s
            elif not self._inflight and self._request is None and now >= self._due:
                raw = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self._latest.items()}
                self._request = (
                    self._generation,
                    self._session,
                    raw,
                    self.task,
                    self._steps,
                    self._feedback[:],
                    self._approvals[:],
                )
                self._feedback.clear()
                self._approvals.clear()
                self._wake.set()
            self._set_dispatched_task(self.task)
            # Hold the last command while the provider thinks; no API call runs on this thread.
            return torch.from_numpy(self._held.copy())

    def _write(self, name: str, value) -> None:
        path = self._directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v))
            + "\n"
        )
        temporary.replace(path)

    def _run(self) -> None:
        session = -1
        record = None
        turn = 0
        try:
            while not self._closed.is_set():
                self._wake.wait(timeout=0.1)
                self._wake.clear()
                with self._lock:
                    request, self._request = self._request, None
                    if request is None or self._paused:
                        continue
                    self._inflight = True
                generation, requested_session, raw, task, steps, feedback, approvals = request
                try:
                    if requested_session != session:
                        if record is not None:
                            if not record.terminated:
                                record.status = "cancelled"
                                record.truncated = True
                            self.agent.on_trial_end(record, str(self._directory), self._run_id)
                            self._write(f"trial-{session}.json", asdict(record))
                        session = requested_session
                        record = TrialRecord(scene_id=f"trial-{session}", epoch=0, seed=None)
                        self.agent.reset(Scene(id=record.scene_id, instruction=task))
                        self.adapter.reset(raw)
                        self.agent.on_trial_start(record.scene_id, 0, str(self._directory), self._run_id)
                    assert record is not None
                    observation = self.adapter.observation(raw, task, steps, feedback, approvals)
                    turn += 1
                    for name, frame in observation.images.items():
                        # Use an index rather than a user-supplied camera name as a path component.
                        index = list(observation.images).index(name)
                        Image.fromarray(frame).save(self._directory / f"turn-{turn}-camera-{index}.png")
                    self._write(
                        f"turn-{turn}-observation.json",
                        {
                            "state": observation.state,
                            "cameras": list(observation.images),
                            "task": task,
                            "extra": observation.extra,
                        },
                    )
                    chunk = self.agent.act(observation)
                    terminal = next((a.meta for a in chunk.actions if a.meta.get("request_stop")), None)
                    actions = (
                        [] if terminal else self.adapter.translate(np.array([a.data for a in chunk.actions]))
                    )
                    with self._lock:
                        accepted = (
                            generation == self._generation and not self._paused and not self._closed.is_set()
                        )
                        if accepted:
                            if terminal:
                                self._completion = (
                                    f"{terminal.get('stop_reason')}: {terminal.get('stop_detail', '')}"
                                )
                                record.terminated = True
                                record.termination_reason = self._completion
                            elif actions:
                                # All waypoints were validated before any are dispatched. Reject a
                                # newly stale start rather than jump from the currently held target.
                                if self._held is not None and np.any(
                                    np.abs(actions[0] - self._held) > self.adapter.step + 1e-7
                                ):
                                    accepted = False
                                    self._approvals.append(
                                        {
                                            "t": self._steps,
                                            "detail": "Motion discarded: measured start differs from held target. Reobserve and propose a smaller motion.",
                                        }
                                    )
                                else:
                                    self._queue.extend(actions)
                    self._write(
                        f"turn-{turn}-result.json",
                        {
                            "accepted": accepted,
                            "terminal": terminal,
                            "native_waypoints": [a.tolist() for a in actions],
                        },
                    )
                    self._write(f"trial-{session}-transcript.json", self.agent.transcript())
                except Exception:
                    failure = traceback.format_exc()
                    with self._lock:
                        if generation == self._generation and not self._paused and not self._closed.is_set():
                            self._failure = failure
                    self._write(f"turn-{turn}-error.json", {"traceback": failure})
                finally:
                    with self._lock:
                        self._inflight = False
        finally:
            if record is not None:
                if not record.terminated:
                    record.status = "error" if self._failure else "cancelled"
                    record.truncated = True
                    record.error = self._failure
                self.agent.on_trial_end(record, str(self._directory), self._run_id)
                self._write(f"trial-{session}.json", asdict(record))
