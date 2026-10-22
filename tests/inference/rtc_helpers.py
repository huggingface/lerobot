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

"""Gated policies and RTC setup for inference and simulation tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Semaphore
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.inference import RTCInferenceEngine, dispatch_tick
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.feature_utils import hw_to_dataset_features
from tests.inference.execution_helpers import Clock
from tests.utils import wait_for

if TYPE_CHECKING:
    from lerobot.robots.sim import SimRobot


class Pipeline:
    steps: tuple = ()

    def __init__(self, width: int | None = None):
        self.width = width

    def __call__(self, value):
        return value if self.width is None else value[..., : self.width]

    def reset(self):
        pass


class ControlledPolicy:
    chunk_inference_spec = PreTrainedPolicy.chunk_inference_spec
    drop_queued_actions = PreTrainedPolicy.drop_queued_actions
    _action_queue_attrs = PreTrainedPolicy._action_queue_attrs

    def __init__(self, width: int = 2):
        self.width = width
        self.config = SimpleNamespace(
            n_obs_steps=1,
            chunk_size=20,
            n_action_steps=20,
            rtc_training_max_delay=4,
            action_feature=PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        )
        self.release = Semaphore(0)
        self.calls: list[tuple[torch.Tensor | None, int]] = []
        self.action_observations: list[torch.Tensor] = []
        self.text_observations: list[torch.Tensor] = []

    def predict_action_chunk(self, batch, *, inference_delay=0, prev_chunk_left_over=None):
        self.calls.append((prev_chunk_left_over, inference_delay))
        self.action_observations.append(batch["observation.state"].clone())
        assert self.release.acquire(timeout=3), "test must release the model call"
        return torch.ones(1, 20, self.width)

    def generate_text(self, batch):
        self.text_observations.append(batch["observation.state"].clone())
        return "answer"

    def reset(self):
        pass

    def supports_rtc(self):
        return True

    def supports_text_generation(self):
        return True


def make_engine(policy=None, robot=None, postprocessor=None, **kwargs):
    policy = policy or ControlledPolicy()
    robot = robot or SimpleNamespace(robot_type="mock", action_features={}, supports_hold=True)
    return RTCInferenceEngine(
        policy=policy,
        preprocessor=Pipeline(),
        postprocessor=postprocessor or Pipeline(width=2),
        robot_wrapper=robot,
        rtc_config=kwargs.pop("rtc_config", RTCConfig(enabled=True, execution_horizon=4)),
        dataset_features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
        },
        task="pick",
        fps=30,
        device="cpu",
        **kwargs,
    )


def observe(engine, value=0.0):
    engine.notify_observation({"j1.pos": value, "j2.pos": value})


def stop(engine):
    engine._policy.release.release(20)
    engine.stop()


class GatedSimPolicy(ControlledPolicy):
    """Not a trained model: identifiable safe targets make chunk alignment observable."""

    def __init__(self, base: torch.Tensor) -> None:
        super().__init__(width=len(base))
        self.base = base
        self.config.chunk_size = self.config.n_action_steps = 8
        self.config.action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(len(base),))
        self.chunks: list[torch.Tensor] = []

    def predict_action_chunk(
        self,
        batch: dict[str, Any],
        *,
        inference_delay: int = 0,
        prev_chunk_left_over: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions = self.base.repeat(self.config.chunk_size, 1)
        actions[:, 0] += 0.001 * (10 * len(self.calls) + torch.arange(len(actions)) + 1)
        self.chunks.append(actions)
        self.action_observations.append(batch["observation.state"].clone())
        self.calls.append((prev_chunk_left_over, inference_delay))
        assert self.release.acquire(timeout=10), "test must release the pending prediction"
        return actions.unsqueeze(0)


class _ClockedRobot(ThreadSafeRobot):
    """Use one test clock for observation age, inference latency, and deadlines."""

    def __init__(self, robot: SimRobot, clock: Clock) -> None:
        super().__init__(robot)
        self.clock = clock

    @property
    def observation_time(self) -> float:
        return self.clock()


class SimRTC:
    """Drive real dispatch and simulation with controlled predictions and client time."""

    def __init__(self, robot: SimRobot, mode: str, grace_s: float) -> None:
        self.robot = robot
        self.clock = Clock()
        self.wrapper = _ClockedRobot(robot, self.clock)
        observation = self.wrapper.get_observation()
        names = list(robot.action_features)
        base = [observation[name] if robot.descriptor.control == "position" else 0.0 for name in names]
        self.policy = GatedSimPolicy(torch.tensor(base, dtype=torch.float32))
        features = {
            **hw_to_dataset_features(robot.observation_features, "observation", use_video=False),
            **hw_to_dataset_features(robot.action_features, "action"),
        }
        self.engine = RTCInferenceEngine(
            policy=self.policy,
            preprocessor=Pipeline(),
            postprocessor=Pipeline(),
            robot_wrapper=self.wrapper,
            rtc_config=RTCConfig(enabled=True, mode=mode, execution_horizon=4),
            dataset_features=features,
            task=robot.task_description,
            fps=robot.descriptor.fps,
            device="cpu",
            rtc_queue_threshold=7,
            action_starvation_grace_s=grace_s,
        )
        self.engine._runtime.clock = self.clock
        self.interpolator = ActionInterpolator(multiplier=1)

    def capture(self) -> None:
        self.engine.notify_observation(self.wrapper.get_observation())

    def wait_for_call(self, count: int) -> None:
        assert wait_for(lambda: len(self.policy.calls) >= count), self.engine.failure_traceback

    def complete(self, count: int) -> None:
        self.wait_for_call(count)
        self.policy.release.release()

        def completed() -> bool:
            # Counters are published before tensors are merged; wait for the whole transaction.
            with self.engine._runtime.lock:
                return self.engine._runtime._completed >= count

        assert wait_for(completed), self.engine.failure_traceback

    def tick(self) -> torch.Tensor | None:
        # A held client tick advances client time, not simulator physics.
        self.clock.now += 1 / self.robot.descriptor.fps
        self.capture()

        def send(target: torch.Tensor) -> None:
            self.wrapper.send_action(dict(zip(self.robot.action_features, target.tolist(), strict=True)))

        return dispatch_tick(self.engine, self.interpolator, send=send, hold=self.wrapper.hold)


@contextmanager
def sim_rtc(robot: SimRobot, *, mode: str = "guided", grace_s: float = 1.0) -> Iterator[SimRTC]:
    """Own only the RTC worker; the caller owns the robot and its graphics context."""
    rig = SimRTC(robot, mode, grace_s)
    try:
        rig.engine.start()
        rig.engine.resume()
        rig.capture()
        yield rig
    finally:
        stop(rig.engine)
        thread = rig.engine._rtc_thread
        assert thread is None or not thread.is_alive()


def assert_delayed_chunk_overlap(rig: SimRTC, delay_steps: int) -> None:
    """Finish prediction two after exactly N old actions have advanced the native world."""
    rig.complete(1)
    torch.testing.assert_close(rig.tick(), rig.policy.chunks[0][0])
    rig.wait_for_call(2)
    prefix, _ = rig.policy.calls[1]
    torch.testing.assert_close(prefix, rig.policy.chunks[0][1:5])
    for index in range(delay_steps):
        torch.testing.assert_close(rig.tick(), rig.policy.chunks[0][index + 1])
    assert rig.engine._runtime._completed == 1, "Physics must advance while inference is blocked"
    assert rig.robot.episode_status.step == delay_steps + 1
    rig.complete(2)
    assert rig.engine._runtime.last_accept["trimmed_actions"] == delay_steps
    assert rig.engine.action_queue.qsize() == 8 - delay_steps
    torch.testing.assert_close(rig.tick(), rig.policy.chunks[1][delay_steps])
    assert rig.robot.episode_status.step == delay_steps + 2
    assert not rig.engine.failed
