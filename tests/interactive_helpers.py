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

"""Small engine and rollout contexts for interactive control tests."""

from __future__ import annotations

import time
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

from lerobot.inference import (  # noqa: E402
    InferenceEngine,
    PolicyQuery,
    QueryKind,
    SyncInferenceConfig,
)
from lerobot.rollout import (  # noqa: E402
    BaseStrategyConfig,
    LinkedEvent,
    RolloutStrategy,
)


class _FakeEngine(InferenceEngine):
    """Real task-holder and query-channel semantics with mocked lifecycle methods."""

    failed = False
    failure_traceback = None
    # Plain attributes shadowing the base properties, so tests can flip them.
    supports_text_queries = True
    # Answer inline on the caller's thread like the sync backend, exercising pump_query.
    control_thread_owns_policy = True

    # Declared here to satisfy the ABC; the instances below shadow them.
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...
    def get_action(self, obs_frame=None): ...

    def __init__(self, task: str = "pick up the cube") -> None:
        super().__init__(task=task)
        self.start = MagicMock()
        self.stop = MagicMock()
        self.reset = MagicMock()
        self.pause = MagicMock()
        self.resume = MagicMock()
        self.notify_observation = MagicMock()
        self.get_action = MagicMock(return_value=None)
        self.text_error: Exception | None = None
        self.seen_query_obs: list = []
        self.seen_queries: list = []
        self.subtasks_planned = 0

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        self.seen_queries.append((query.kind, query.text))
        self.seen_query_obs.append(obs_processed)
        if self.text_error is not None:
            raise self.text_error
        if query.kind is QueryKind.NEXT_SUBTASK:
            # A stand-in decomposer: one subtask per query, advancing.
            self.subtasks_planned += 1
            return f"subtask {self.subtasks_planned} of {query.text}"
        return f"answer: {query.text}"


def _loop_ctx(engine, robot=None, stop_event=None, **cfg_overrides):
    """The rollout context a real control loop needs, with identity processors."""

    def identity(x):
        return x

    cfg = {
        "play_sounds": False,
        "fps": 100.0,
        "duration": 0.0,
        "use_torch_compile": False,
        "interpolation_multiplier": 1,
        "display_data": False,
        "autosteer_interval_s": 0.0,
        "inference": SyncInferenceConfig(),
    }
    cfg.update(cfg_overrides)
    return SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(**cfg),
            shutdown_event=stop_event if stop_event is not None else LinkedEvent(Event()),
            cadence_report=None,
        ),
        policy=SimpleNamespace(inference=engine),
        hardware=SimpleNamespace(
            robot_wrapper=robot if robot is not None else MagicMock(),
            teleop=None,
            initial_position={"joint.pos": 0.0},
        ),
        processors=SimpleNamespace(
            teleop_action_processor=identity,
            robot_action_processor=identity,
            robot_observation_processor=identity,
        ),
        data=SimpleNamespace(dataset=None, dataset_features={}, hw_features={}, ordered_action_keys=[]),
    )


def _make_ctx(run_behavior=None):
    """A mock strategy plus the fake context the controller needs."""
    parent = Event()
    engine = _FakeEngine()
    ctx = _loop_ctx(engine, stop_event=LinkedEvent(parent))

    # Autospec so a strategy signature drift fails here, not on real hardware during /reset.
    strategy = create_autospec(RolloutStrategy, instance=True)
    strategy.config = BaseStrategyConfig()  # the controller enforces supports_interactive on it
    run_started = Event()

    def default_run(c):
        run_started.set()
        while not c.runtime.shutdown_event.is_set():
            time.sleep(0.005)

    strategy.run.side_effect = run_behavior or default_run
    return ctx, strategy, engine, parent, run_started
