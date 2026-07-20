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

"""Unit tests for the C1 deterministic agent + language parser."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lerobot.navigation.agent import (
    AgentConfig,
    DeterministicAgent,
    HardcodedTaskParser,
    Task,
)
from lerobot.navigation.skills import ExploreResult, GotoResult, LocateResult

# ----- fakes -------------------------------------------------------------


@dataclass
class FakeSkills:
    """Programmable :class:`SpatialSkills` stand-in. Each method consults a
    pre-recorded script and bumps a call counter so tests can assert the
    deterministic policy made the right sequence of calls."""

    locate_script: list[LocateResult] = field(default_factory=list)
    explore_script: list[ExploreResult] = field(default_factory=list)
    goto_script: list[GotoResult] = field(default_factory=list)

    locate_calls: list[str] = field(default_factory=list)
    goto_calls: list[tuple[float, float, float]] = field(default_factory=list)
    explore_calls: list[str | None] = field(default_factory=list)

    def locate(self, text: str) -> LocateResult:
        self.locate_calls.append(text)
        if not self.locate_script:
            return LocateResult(False, None, -1.0, 0, text)
        return self.locate_script.pop(0)

    def explore(self, query: str | None = None) -> ExploreResult:
        self.explore_calls.append(query)
        if not self.explore_script:
            return ExploreResult(None, False, 0.0, "no frontier")
        return self.explore_script.pop(0)

    def goto(self, xyz: tuple[float, float, float], **_: object) -> GotoResult:
        self.goto_calls.append(xyz)
        if not self.goto_script:
            return GotoResult(True, xyz, 0.0, 0, "ok", [])
        return self.goto_script.pop(0)

    @property
    def base(self):
        # Minimal stub: agent's teleport branch isn't exercised by these tests.
        class _Base:
            def move(self, *a, **k):
                pass

            def pose(self):
                return np.eye(4)

        return _Base()


# ----- HardcodedTaskParser ------------------------------------------------


def test_parser_simple_go_to():
    t = HardcodedTaskParser().parse("go to the mug")
    assert t.targets == ["mug"]


def test_parser_strips_punctuation_and_articles():
    t = HardcodedTaskParser().parse("Find the red lamp.")
    assert t.targets == ["red lamp"]


def test_parser_multi_step():
    t = HardcodedTaskParser().parse("go to the mug then the chair")
    assert t.targets == ["mug", "chair"]


def test_parser_no_verb_treats_command_as_target():
    """``parser.parse('couch')`` should still produce a usable Task."""
    t = HardcodedTaskParser().parse("couch")
    assert t.targets == ["couch"]


def test_parser_empty_string_returns_empty_task():
    t = HardcodedTaskParser().parse("   ")
    assert t.targets == []


def test_parser_split_by_comma():
    t = HardcodedTaskParser().parse("go to mug, chair")
    assert t.targets == ["mug", "chair"]


# ----- DeterministicAgent policy -----------------------------------------


def _ok_locate(xyz=(1.0, 0.0, 1.0), conf=0.9) -> LocateResult:
    return LocateResult(True, xyz, conf, 10, "x")


def _miss_locate(conf=0.05) -> LocateResult:
    return LocateResult(False, None, conf, 0, "x")


def _ok_goto(xyz=(1.0, 0.0, 1.0)) -> GotoResult:
    return GotoResult(True, xyz, 0.0, 5, "ok", [])


def _failed_goto(xyz=(1.0, 0.0, 1.0)) -> GotoResult:
    return GotoResult(False, (0.0, 0.0, 0.0), 1.4, 0, "no path", [])


def _explore_to(xyz=(2.0, 0.0, 2.0)) -> ExploreResult:
    return ExploreResult(xyz, True, 2.8, "ok")


def test_agent_hit_then_goto():
    """Found on first call → no explore, single goto."""
    skills = FakeSkills(
        locate_script=[_ok_locate()],
        goto_script=[_ok_goto()],
    )
    agent = DeterministicAgent(skills, AgentConfig(max_explore_iters=3))
    res = agent.execute(Task(targets=["mug"]))
    assert res.fully_successful
    assert skills.locate_calls == ["mug"]
    assert skills.goto_calls == [(1.0, 0.0, 1.0)]
    assert skills.explore_calls == []
    assert res.target_results[0].n_explore_iters == 0


def test_agent_explore_then_relocate_then_goto():
    """First locate misses → explore → goto-to-frontier → re-locate finds → final goto."""
    skills = FakeSkills(
        locate_script=[_miss_locate(), _ok_locate()],
        explore_script=[_explore_to((3.0, 0.0, 0.0))],
        goto_script=[_ok_goto((3.0, 0.0, 0.0)), _ok_goto((1.0, 0.0, 1.0))],
    )
    agent = DeterministicAgent(skills, AgentConfig(max_explore_iters=3))
    res = agent.execute(Task(targets=["mug"]))
    assert res.fully_successful
    assert skills.locate_calls == ["mug", "mug"]
    assert skills.explore_calls == ["mug"]
    assert skills.goto_calls == [(3.0, 0.0, 0.0), (1.0, 0.0, 1.0)]
    assert res.target_results[0].n_explore_iters == 1


def test_agent_budget_exhaustion():
    """All N+1 locate calls miss → return budget_exhausted."""
    skills = FakeSkills(
        locate_script=[_miss_locate() for _ in range(5)],
        explore_script=[_explore_to() for _ in range(4)],
        goto_script=[_ok_goto((2.0, 0.0, 2.0)) for _ in range(4)],
    )
    agent = DeterministicAgent(skills, AgentConfig(max_explore_iters=3))
    res = agent.execute(Task(targets=["mug"]))
    assert res.fully_successful is False
    r = res.target_results[0]
    assert r.reason == "budget_exhausted"
    assert r.n_explore_iters == 3
    # 4 locate calls: initial + 3 retries.
    assert len(skills.locate_calls) == 4
    assert len(skills.explore_calls) == 3


def test_agent_no_frontier_short_circuits():
    """If explore can't find a frontier, give up immediately — no point looping."""
    skills = FakeSkills(
        locate_script=[_miss_locate()],
        explore_script=[ExploreResult(None, False, 0.0, "no frontier")],
    )
    agent = DeterministicAgent(skills, AgentConfig(max_explore_iters=3))
    res = agent.execute(Task(targets=["mug"]))
    r = res.target_results[0]
    assert r.reached is False
    assert r.reason == "no_frontier"
    assert len(skills.locate_calls) == 1
    assert len(skills.explore_calls) == 1


def test_agent_failed_goto_does_not_loop_back():
    """If locate finds the target but goto fails (e.g. no path), report the
    failure cleanly rather than retrying."""
    skills = FakeSkills(
        locate_script=[_ok_locate()],
        goto_script=[_failed_goto()],
    )
    agent = DeterministicAgent(skills)
    res = agent.execute(Task(targets=["mug"]))
    r = res.target_results[0]
    assert r.reached is False
    assert r.reason == "no path"


def test_agent_multi_target_bails_on_first_failure():
    """The spec says sequential targets stop at the first failure so the
    caller sees the failure clearly."""
    skills = FakeSkills(
        locate_script=[_miss_locate()],
        explore_script=[ExploreResult(None, False, 0.0, "no frontier")],
    )
    agent = DeterministicAgent(skills, AgentConfig(max_explore_iters=0))
    res = agent.execute(Task(targets=["mug", "chair"]))
    assert len(res.target_results) == 1  # bailed before chair
    assert res.target_results[0].target == "mug"


def test_agent_swap_parser_does_not_change_policy():
    """Acceptance from the spec: 'swapping Qwen for a hardcoded target string
    yields the same spatial behaviour'. Same skills script, same scripted
    locate/goto, regardless of how the command was parsed."""
    parser = HardcodedTaskParser()
    for command in ("mug", "go to the mug", "find the mug"):
        skills = FakeSkills(locate_script=[_ok_locate()], goto_script=[_ok_goto()])
        agent = DeterministicAgent(skills)
        res = agent.execute_command(command, parser)
        assert res.fully_successful
        assert skills.goto_calls == [(1.0, 0.0, 1.0)]


def test_agent_empty_command_reports_parse_failure():
    skills = FakeSkills()
    agent = DeterministicAgent(skills)
    res = agent.execute_command("", HardcodedTaskParser())
    assert res.fully_successful is False
    assert res.target_results[0].reason == "parse_empty"
    assert skills.locate_calls == []
