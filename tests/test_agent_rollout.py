# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real upstream tool execution with scripted HTTP; no robot or API key needed."""

import json
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("inspect_robots_agent")
pytest.importorskip("datasets")
import httpx

from lerobot.rollout.agent.adapter import RobotAdapter
from lerobot.rollout.agent.configuration import CartesianArm, PositionAxis
from lerobot.rollout.agent.engine import AgentInferenceEngine
from lerobot.rollout.inference.factory import AgentInferenceConfig


def response(name, arguments):
    return httpx.Response(
        200,
        json={
            "id": "resp_test",
            "status": "completed",
            "output": [
                {
                    "id": "fc_test",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_test",
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            ],
        },
    )


def config(tmp_path):
    return AgentInferenceConfig(
        model="gpt-6.1-sol",
        base_url="https://api.openai.com/v1",
        effort="low",
        service_tier="fast",
        log_dir=str(tmp_path),
        settle_s=0,
        transcript_echo=False,
        robot_notes="Test robot. j.pos increases to the left. cam is RGB; uncalibrated.",
        axes={"j.pos": PositionAxis(-2, 2, 1, "rad", "base yaw, positive counterclockwise", 0.2)},
    )


def raw(position=0):
    return {"j.pos": position, "cam": np.zeros((8, 8, 3), dtype=np.uint8)}


def engine(tmp_path, handler, cfg=None):
    return AgentInferenceEngine(
        cfg or config(tmp_path),
        keys=["j.pos"],
        features={"j.pos": float, "cam": (8, 8, 3)},
        robot_type="mock",
        fps=10,
        task="Pick up the red cube",
        transport=httpx.MockTransport(handler),
        env={"OPENAI_API_KEY": "test-not-a-real-key"},
    )


def wait_for(predicate):
    deadline = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < deadline, "worker did not reach expected state"
        time.sleep(0.005)


def tick(e, position=0):
    e.notify_observation(raw(position))
    return e.get_action(None)


def test_native_tools_move_then_done_and_record_harness(tmp_path):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response("move_joints", {"targets": {"j.pos": 0.1}, "note": "Approach the cube."})
        return response("done", {"summary": "Cube placed in bin", "hindsight": "Check the wrist view."})

    e = engine(tmp_path, handler)
    e.start()
    try:
        assert tick(e) is None  # initialised but paused until /start
        assert not requests
        e.resume()
        assert tick(e).item() == 0
        wait_for(lambda: e._queue or e.failed)
        assert not e.failed, e.failure_traceback
        while e._queue:
            commanded = tick(e).item()
        assert commanded == pytest.approx(0.1)
        tick(e, 0.1)
        wait_for(lambda: e.completion or e.failed)
        assert e.completion.startswith("done:")
        assert not e.failed
        assert tick(e, 0.1) is None
        assert requests[0]["model"] == "gpt-6.1-sol"  # no erroneous openai/ prefix at explicit URL
        assert requests[0]["service_tier"] == "fast"
        assert requests[0]["reasoning"]["effort"] == "low"
        assert any(t["name"] == "move_joints" for t in requests[0]["tools"])
        assert "Approach the cube" in json.dumps(requests[1])
    finally:
        e.stop()
    directory = next(tmp_path.iterdir())
    assert list(directory.glob("turn-*-camera-0.png"))
    assert list(directory.glob("trial-*-transcript.json"))
    # transcript files have a different suffix and may sort before the metadata file
    records = [
        json.loads(p.read_text())
        for p in directory.glob("trial-*.json")
        if not p.name.endswith("transcript.json")
    ]
    assert records[0]["terminated"]
    assert "test-not-a-real-key" not in (directory / "config.json").read_text()


@pytest.mark.parametrize("cancel", ["reset", "feedback", "task", "pause", "stop"])
def test_late_completion_is_discarded_single_flight(tmp_path, cancel):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def handler(request):
        calls.append(request)
        entered.set()
        assert release.wait(4)
        return response("done", {"summary": "old completion"})

    e = engine(tmp_path, handler)
    e.start()
    e.resume()
    try:
        tick(e)
        assert entered.wait(2)
        for _ in range(5):
            tick(e)
        assert len(calls) == 1
        if cancel == "feedback":
            assert e.add_feedback("Try the blue cube")
        elif cancel == "task":
            e.set_task("Pick the blue cube")
        else:
            getattr(e, cancel)()
        release.set()
        wait_for(lambda: not e._inflight)
        assert e.completion is None
        assert not e._queue
        assert not e.failed
    finally:
        release.set()
        e.stop()


def test_feedback_reaches_native_conversation(tmp_path):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response("move_joints", {"targets": {"j.pos": 0.1}, "note": "Inspect approach."})

    e = engine(tmp_path, handler)
    e.start()
    e.resume()
    try:
        tick(e)
        wait_for(lambda: bool(e._queue))
        assert e.add_feedback("The blue cube is the target now")
        tick(e)
        wait_for(lambda: len(requests) == 2)
        assert "The blue cube is the target now" in json.dumps(requests[1])
        assert "interrupted" in json.dumps(requests[1])
    finally:
        e.stop()


def test_stale_feedback_never_emits_a_command(tmp_path):
    e = engine(tmp_path, lambda _: response("done", {"summary": "ok"}))
    e.resume()
    e.notify_observation(raw())
    e._observed_at -= 2
    with pytest.raises(RuntimeError, match="fresh measured"):
        e.get_action(None)
    e.stop()


def test_tracking_error_stops_dispatch(tmp_path):
    e = engine(tmp_path, lambda _: response("done", {"summary": "ok"}))
    e.resume()
    tick(e)
    with pytest.raises(RuntimeError, match="tracking tolerance"):
        tick(e, 0.5)
    e.stop()


@pytest.mark.parametrize("bad", [float("nan"), 3.0, -3.0])
def test_bad_measured_state_is_rejected(tmp_path, bad):
    e = engine(tmp_path, lambda _: response("done", {"summary": "ok"}))
    e.resume()
    with pytest.raises(ValueError, match="Measured robot"):
        tick(e, bad)
    e.stop()


def test_axis_contract_uses_native_degrees_and_gripper_units(tmp_path):
    cfg = replace(
        config(tmp_path),
        axes={
            "joint.pos": PositionAxis(-90, 90, 20, "degrees", "elbow", 5),
            "gripper.pos": PositionAxis(0, 100, 50, "percent", "0 closed; 100 open", 10),
        },
    )
    adapter = RobotAdapter(
        cfg, list(cfg.axes), {"joint.pos": float, "gripper.pos": float, "cam": (8, 8, 3)}, "so101", 20
    )
    assert "degrees" in adapter.info.docs
    assert "0 closed; 100 open" in adapter.info.docs
    assert adapter.info.action_space.semantics.max_step == (1, 2.5)
    adapter.pose = {"joint.pos": 0, "gripper.pos": 0}
    np.testing.assert_allclose(adapter.translate(np.array([[1, 2.5], [2, 5]])), [[1, 2.5], [2, 5]])
    assert "speed" in adapter.pre_check(np.array([[4, 0]]))


def test_no_silent_drop_of_mobile_base(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="EVERY"):
        RobotAdapter(
            cfg, ["j.pos", "base.vel"], {"j.pos": float, "base.vel": float, "cam": (8, 8, 3)}, "mobile", 10
        )


def cartesian_adapter(tmp_path):
    pytest.importorskip("mujoco")
    body = '<site name="tip" size="0.001"/>'
    for i, (kind, axis) in reversed(
        list(
            enumerate(
                [
                    ("slide", "1 0 0"),
                    ("slide", "0 1 0"),
                    ("slide", "0 0 1"),
                    ("hinge", "0 0 1"),
                    ("hinge", "0 1 0"),
                    ("hinge", "1 0 0"),
                ]
            )
        )
    ):
        body = f'<body><joint name="q{i}" type="{kind}" axis="{axis}" range="-1 1"/><geom type="sphere" size="0.01" mass="0.1"/>{body}</body>'
    model = tmp_path / "cartesian.xml"
    model.write_text(f'<mujoco><compiler angle="radian"/><worldbody>{body}</worldbody></mujoco>')
    keys = [f"q{i}.pos" for i in range(6)]
    axes = {k: PositionAxis(-1, 1, 1, "m" if i < 3 else "rad", "test axis", 0.1) for i, k in enumerate(keys)}
    arm = CartesianArm(
        str(model),
        "tip",
        {k: f"q{i}" for i, k in enumerate(keys)},
        "Fixed base; X forward, Y left, Z up",
        [-0.5] * 3,
        [0.5] * 3,
        angle_low=[-0.5] * 3,
        angle_high=[0.5] * 3,
        linear_speed=0.1,
        angular_speed=0.1,
    )
    cfg = replace(config(tmp_path), axes=axes, cartesian={"arm": arm})
    adapter = RobotAdapter(cfg, keys, {**dict.fromkeys(keys, float), "cam": (8, 8, 3)}, "cartesian-test", 10)
    pose = dict.fromkeys(keys, 0.0)
    pose["q3.pos"] = 0.1
    adapter.reset(pose)
    adapter.observation({**pose, "cam": raw()["cam"]}, "move", 0, [], [])
    return adapter, pose


def test_cartesian_waypoints_preserve_line_and_negative_pitch_convention(tmp_path):
    adapter, pose = cartesian_adapter(tmp_path)
    points = np.array([[i * 0.005, 0, 0, 0, i * 0.005, 0] for i in range(1, 5)])
    joints = adapter.translate(points)
    kin = adapter.arms["arm"]
    for expected, q in zip(points, joints, strict=True):
        actual = kin.observe(dict(zip(adapter.keys, q, strict=True)))
        np.testing.assert_allclose(actual, expected, atol=0.0003)
    assert joints[-1][4] < 0  # positive tool pitch rotates about negative base Y
    assert np.linalg.norm(joints[-1] - adapter.vector(pose)) > 0.01


def test_ik_rejects_path_before_any_dispatch(tmp_path):
    adapter, _ = cartesian_adapter(tmp_path)
    assert "speed" in adapter.pre_check(np.array([[0.4, 0, 0, 0, 0, 0]]))


def test_config_allows_no_vla_and_rejects_double_interpolation(tmp_path, monkeypatch):
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.rollout.configs import RolloutConfig

    monkeypatch.setattr("sys.argv", ["lerobot-rollout"])
    cfg = RolloutConfig(robot=SO101FollowerConfig(port="mock"), inference=config(tmp_path), task="pick cube")
    assert cfg.policy is None
    assert cfg.device == "cpu"
    with pytest.raises(ValueError, match="already interpolate"):
        replace(cfg, interpolation_multiplier=2)


def test_agent_context_builds_before_hardware_without_vla_loading(tmp_path, monkeypatch):
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.rollout import context
    from lerobot.rollout.configs import RolloutConfig

    monkeypatch.setattr("sys.argv", ["lerobot-rollout"])
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")

    class Robot:
        observation_features = {"j.pos": float, "cam": (8, 8, 3)}
        action_features = {"j.pos": float}
        name = robot_type = "mock"
        is_connected = False

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

        def get_observation(self):
            return raw()

    robot = Robot()
    monkeypatch.setattr(context, "make_robot_from_config", lambda _: robot)

    def forbidden(*args, **kwargs):
        raise AssertionError("VLA loader called")

    monkeypatch.setattr(context, "_load_pretrained_policy", forbidden)
    cfg = RolloutConfig(robot=SO101FollowerConfig(port="mock"), inference=config(tmp_path), task="pick cube")
    ctx = context.build_rollout_context(cfg, threading.Event())
    assert ctx.policy.policy is None
    assert robot.is_connected
    assert ctx.data.ordered_action_keys == ["j.pos"]
    ctx.policy.inference.stop()
    robot.disconnect()
    bad = replace(cfg, inference=replace(cfg.inference, axes={}))
    with pytest.raises(ValueError, match="EVERY"):
        context.build_rollout_context(bad, threading.Event())
    assert not robot.is_connected


def test_agent_config_draccus_roundtrip(tmp_path):
    import draccus

    from lerobot.rollout.inference.factory import InferenceEngineConfig

    cfg = config(tmp_path)
    encoded = draccus.encode(cfg)
    encoded["type"] = cfg.type
    assert draccus.decode(InferenceEngineConfig, encoded) == cfg


def test_checked_in_examples_decode_agent_contracts():
    from pathlib import Path

    import draccus
    import yaml

    from lerobot.rollout.inference.factory import InferenceEngineConfig

    for name, dimensions in (("so_follower", 6), ("yam", 14)):
        value = yaml.safe_load(Path(f"examples/agent_rollout/{name}.yaml").read_text())
        cfg = draccus.decode(InferenceEngineConfig, value["inference"])
        assert isinstance(cfg, AgentInferenceConfig)
        assert len(cfg.axes) == dimensions
        adapter = RobotAdapter(
            cfg, list(cfg.axes), {**dict.fromkeys(cfg.axes, float), "cam": (8, 8, 3)}, name, value["fps"]
        )
        assert adapter.info.action_space.shape == (dimensions,)


def test_encoder_residual_is_reported_but_never_commanded_outside_bounds(tmp_path):
    cfg = config(tmp_path)
    cfg.axes["j.pos"] = PositionAxis(0, 2, 1, "rad", "joint near its lower bound", 0.2)
    e = engine(tmp_path, lambda _: response("done", {"summary": "ok"}), cfg)
    e.resume()
    assert tick(e, -0.0001907377).item() == 0.0
    obs = e.adapter.observation(raw(-0.0001907377), "task", 0, [], [])
    assert obs.state["command_state"][0] < 0
    with pytest.raises(ValueError, match="outside"):
        tick(e, -0.01)
    e.stop()
