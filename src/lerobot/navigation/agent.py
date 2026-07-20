#!/usr/bin/env python

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

"""Deterministic agent wrapper + language-parser interface.

Ported from the dyna360 research stack. The high-level agent is a thin
deterministic wrapper, not LLM-driven: explore-vs-go control lives here
in plain Python. A language model (when wired up) only parses a
natural-language command into a typed :class:`Task`; the deterministic
wrapper then executes it. Swapping the parser (regex vs a real LLM) must
not change the spatial behaviour.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lerobot.navigation.skills import SpatialSkills

LOG = logging.getLogger(__name__)


# ============== task data structures ====================================== #


@dataclass(frozen=True)
class Task:
    """Parsed command, ready for the deterministic wrapper to execute.

    ``go to X`` yields ``Task(targets=['X'])``; ``go to X then Y`` yields
    ``Task(targets=['X', 'Y'])``, executed sequentially.
    """

    targets: list[str]
    raw: str = ""


@dataclass(frozen=True)
class TargetResult:
    """Outcome of executing the policy for a single target."""

    target: str
    reached: bool
    final_xyz: tuple[float, float, float] | None
    n_explore_iters: int
    confidence: float
    reason: str
    """'ok' | 'no_path' | 'budget_exhausted' | 'no_frontier' | 'parse_empty'."""


@dataclass(frozen=True)
class AgentResult:
    """Outcome of executing a full Task (one or more sequential targets)."""

    task: Task
    target_results: list[TargetResult] = field(default_factory=list)

    @property
    def fully_successful(self) -> bool:
        return bool(self.target_results) and all(r.reached for r in self.target_results)


# ============== language parser ========================================== #


@runtime_checkable
class TaskParser(Protocol):
    """Anything that turns a free-text command into a :class:`Task`."""

    def parse(self, command: str) -> Task: ...


class HardcodedTaskParser:
    """Regex-only parser — fast, dependency-free, good enough to validate
    the deterministic policy without loading a language model.

    Handles ``go to (the) X`` / ``find (the) X`` → single target, ``go to
    X then Y`` → multi-step, and falls back to "the whole command is the
    target" if no pattern matches.
    """

    _SINGLE_PATTERNS = (
        re.compile(
            r"^\s*(?:go to|navigate to|find|locate|look for)\s+(?:the\s+)?(.+?)\s*$",
            re.IGNORECASE,
        ),
    )
    _SPLIT_PATTERN = re.compile(r"\s+(?:then|and then)\s+|\s*,\s*", re.IGNORECASE)

    def parse(self, command: str) -> Task:
        raw = command.strip()
        if not raw:
            return Task(targets=[], raw=raw)

        parts = self._SPLIT_PATTERN.split(raw)
        targets: list[str] = []
        for part in parts:
            t = self._extract_target(part)
            if t:
                targets.append(t)
        return Task(targets=targets, raw=raw)

    def _extract_target(self, text: str) -> str:
        text = text.strip().rstrip(".?!")
        for p in self._SINGLE_PATTERNS:
            m = p.match(text)
            if m:
                return m.group(1).strip()
        prefix = re.match(r"^\s*(?:the\s+)?(.+)$", text, re.IGNORECASE)
        if prefix:
            return prefix.group(1).strip()
        return text


# ============== deterministic agent ====================================== #


@dataclass(frozen=True)
class AgentConfig:
    """Agent policy knobs."""

    max_explore_iters: int = 5
    """How many ``explore → relocate`` loops before giving up on a target."""

    explore_step_uses_goto: bool = True
    """Drive to the explore frontier via closed-loop ``goto``. False
    teleports instead (fast offline eval)."""


class DeterministicAgent:
    """Executes a :class:`Task` via a fixed policy.

    For each target: locate; if found, goto and done; else explore(query),
    goto the frontier, and relocate — up to ``max_explore_iters``, then give
    up. The control flow is plain Python; no LLM in the loop.
    """

    def __init__(self, skills: SpatialSkills, cfg: AgentConfig | None = None) -> None:
        self.skills = skills
        self.cfg = cfg or AgentConfig()

    def execute(self, task: Task) -> AgentResult:
        out: list[TargetResult] = []
        for target in task.targets:
            out.append(self._execute_target(target))
            if not out[-1].reached:
                # Don't auto-skip after a failed multi-step leg; bail so the
                # caller sees the failure clearly.
                break
        return AgentResult(task=task, target_results=out)

    def execute_command(self, command: str, parser: TaskParser) -> AgentResult:
        """Parse a free-text command, then execute."""
        task = parser.parse(command)
        if not task.targets:
            return AgentResult(
                task=task,
                target_results=[
                    TargetResult(
                        target="",
                        reached=False,
                        final_xyz=None,
                        n_explore_iters=0,
                        confidence=-1.0,
                        reason="parse_empty",
                    )
                ],
            )
        return self.execute(task)

    # ----- single-target inner loop ----------------------------------------

    def _execute_target(self, target: str) -> TargetResult:
        last_conf = -1.0
        for it in range(self.cfg.max_explore_iters + 1):
            loc = self.skills.locate(target)
            last_conf = loc.confidence
            if loc.found and loc.xyz is not None:
                LOG.info(
                    "agent: locate(%r) found at %s (conf %.3f); goto",
                    target,
                    loc.xyz,
                    loc.confidence,
                )
                gr = self.skills.goto(loc.xyz)
                return TargetResult(
                    target=target,
                    reached=gr.reached,
                    final_xyz=gr.final_xyz,
                    n_explore_iters=it,
                    confidence=loc.confidence,
                    reason="ok" if gr.reached else gr.reason,
                )

            if it >= self.cfg.max_explore_iters:
                LOG.info(
                    "agent: locate(%r) NOT_FOUND (conf %.3f) and explore budget exhausted",
                    target,
                    loc.confidence,
                )
                return TargetResult(
                    target=target,
                    reached=False,
                    final_xyz=None,
                    n_explore_iters=it,
                    confidence=loc.confidence,
                    reason="budget_exhausted",
                )

            # NOT_FOUND → explore once, then loop and re-locate.
            LOG.info(
                "agent: locate(%r) NOT_FOUND (conf %.3f) → explore iter %d",
                target,
                loc.confidence,
                it + 1,
            )
            ex = self.skills.explore(query=target)
            if not ex.found_frontier or ex.target_xyz is None:
                return TargetResult(
                    target=target,
                    reached=False,
                    final_xyz=None,
                    n_explore_iters=it,
                    confidence=loc.confidence,
                    reason="no_frontier",
                )
            if self.cfg.explore_step_uses_goto:
                self.skills.goto(ex.target_xyz)
            else:
                # Teleport for offline-eval speed.
                self.skills.base.move(0.0, 0.0, dt=0.0)
                pose = self.skills.base.pose()
                pose[0, 3] = ex.target_xyz[0]
                pose[2, 3] = ex.target_xyz[2]
                if hasattr(self.skills.base, "_pose"):
                    self.skills.base._pose = pose  # noqa: SLF001

        return TargetResult(
            target=target,
            reached=False,
            final_xyz=None,
            n_explore_iters=self.cfg.max_explore_iters,
            confidence=last_conf,
            reason="budget_exhausted",
        )
