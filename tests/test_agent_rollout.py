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
        api_key_env="OPENAI_API_KEY",
        wire="responses",
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


def engine(tmp_path, handler, cfg=None, env=None):
    return AgentInferenceEngine(
        cfg or config(tmp_path),
        keys=["j.pos"],
        features={"j.pos": float, "cam": (8, 8, 3)},
        robot_type="mock",
        fps=10,
        task="Pick up the red cube",
        transport=httpx.MockTransport(handler),
        env=env if env is not None else {"OPENAI_API_KEY": "test-not-a-real-key"},
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
        wait_for(lambda: not e._inflight)
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
    pytest.importorskip("placo")
    body = '<link name="base"/>'
    origins = ["0 0 0.2", "0 0 0", "0.25 0 0", "0.25 0 0", "0 0 0", "0 0 0"]
    directions = ["0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"]
    for i, (origin, axis) in enumerate(zip(origins, directions, strict=True)):
        parent = "base" if i == 0 else f"link{i - 1}"
        body += f'''<link name="link{i}"/><joint name="q{i}" type="revolute">
          <parent link="{parent}"/><child link="link{i}"/><origin xyz="{origin}"/>
          <axis xyz="{axis}"/><limit lower="-2" upper="2" effort="10" velocity="2"/></joint>'''
    body += """<link name="tip"/><joint name="tool" type="fixed">
      <parent link="link5"/><child link="tip"/><origin xyz="0.1 0 0"/></joint>
      <link name="finger"/><joint name="finger" type="prismatic">
      <parent link="tip"/><child link="finger"/><axis xyz="0 1 0"/>
      <limit lower="0" upper="0.1" effort="1" velocity="1"/></joint>"""
    model = tmp_path / "arm.urdf"
    model.write_text(f'<robot name="test">{body}</robot>')
    keys = [f"q{i}.pos" for i in range(6)]
    axes = {k: PositionAxis(-2, 2, 2, "rad", "test axis", 0.1) for k in keys}
    arm = CartesianArm(
        str(model),
        "tip",
        {k: f"q{i}" for i, k in enumerate(keys)},
        "Fixed base; X forward, Y left, Z up",
        [-1.0] * 3,
        [1.0] * 3,
        angle_low=[-0.5] * 3,
        angle_high=[0.5] * 3,
        linear_speed=0.1,
        angular_speed=0.1,
    )
    cfg = replace(config(tmp_path), axes=axes, cartesian={"arm": arm})
    adapter = RobotAdapter(cfg, keys, {**dict.fromkeys(keys, float), "cam": (8, 8, 3)}, "cartesian-test", 10)
    pose = dict.fromkeys(keys, 0.0)
    pose["q1.pos"] = 0.3
    pose["q2.pos"] = -0.6
    pose["q4.pos"] = 0.3
    adapter.reset(pose)
    adapter.observation({**pose, "cam": raw()["cam"]}, "move", 0, [], [])
    return adapter, pose


def test_cartesian_waypoints_preserve_line_and_negative_pitch_convention(tmp_path):
    adapter, pose = cartesian_adapter(tmp_path)
    from scipy.spatial.transform import Rotation

    kin = adapter.arms["arm"]
    initial = np.array(kin.observe(pose))
    points = np.array([initial + [0, i * 0.002, 0, 0, i * 0.005, 0] for i in range(1, 5)])
    joints = adapter.translate(points)
    for expected, q in zip(points, joints, strict=True):
        actual = kin.observe(dict(zip(adapter.keys, q, strict=True)))
        np.testing.assert_allclose(actual, expected, atol=0.0003)
    # Independent rotation check, not just round-tripping the adapter's convention.
    transform = kin.kinematics.forward_kinematics(np.rad2deg(joints[-1]))
    expected_rotation = Rotation.from_rotvec([0, -0.02, 0]).as_matrix()
    np.testing.assert_allclose(transform[:3, :3], expected_rotation, atol=0.002)
    assert kin.kinematics.robot.get_joint("finger") == pytest.approx(0)
    assert np.linalg.norm(joints[-1] - adapter.vector(pose)) > 0.01


def test_ik_rejects_path_before_any_dispatch(tmp_path):
    adapter, _ = cartesian_adapter(tmp_path)
    assert adapter.pre_check(np.array([[0.99, 0, 0.2, 0, 0, 0]]))


@pytest.mark.parametrize("unit,scale,value", [("rad", 1.0, np.pi / 2), ("degrees", np.pi / 180, 90.0)])
def test_urdf_fk_explicit_order_units_sign_and_offset(tmp_path, unit, scale, value):
    from lerobot.rollout.agent.kinematics import CartesianKinematics

    adapter, _ = cartesian_adapter(tmp_path)
    arm = adapter.arms["arm"].config
    # Reverse mapping order and use an inverted encoder with a nonzero zero offset.
    keys = list(reversed(arm.joints))
    arm = replace(
        arm,
        joints={k: arm.joints[k] for k in keys},
        joint_scale=dict.fromkeys(keys, -scale),
        joint_offset=dict.fromkeys(keys, 0.1),
    )
    axes = {k: PositionAxis(-2 / scale, 2 / scale, 2 / scale, unit, "joint", 0.1 / scale) for k in keys}
    kin = CartesianKinematics(arm, axes)
    pose = dict.fromkeys(keys, 0.1 / scale)
    pose["q0.pos"] -= value
    kin.reset(pose)
    # Analytic FK: a 0.6 m horizontal arm rotated +90 degrees about its base Z.
    np.testing.assert_allclose(kin.observe(pose)[:3], [0, 0.6, 0.2], atol=1e-8)
    target = np.array(kin.observe(pose))
    result = kin.solve(target, pose)
    np.testing.assert_allclose([result[k] for k in keys], [pose[k] for k in keys], atol=1e-6)


def test_urdf_limits_and_missing_joint_mapping_are_rejected(tmp_path):
    from lerobot.rollout.agent.kinematics import CartesianKinematics

    adapter, pose = cartesian_adapter(tmp_path)
    arm = adapter.arms["arm"].config
    axes = adapter.config.axes
    missing = replace(arm, joints={k: v for k, v in arm.joints.items() if k != "q2.pos"})
    with pytest.raises(ValueError, match="exactly the moving joints"):
        CartesianKinematics(missing, axes)
    nonoverlap = {**axes, "q0.pos": PositionAxis(3, 4, 1, "rad", "joint", 0.1)}
    with pytest.raises(ValueError, match="do not overlap"):
        CartesianKinematics(arm, nonoverlap)
    kin = adapter.arms["arm"]
    with pytest.raises(ValueError, match="outside model joint limits"):
        kin.observe({**pose, "q0.pos": 2.1})


def test_ik_cannot_return_out_of_bounds_or_wrong_pose(tmp_path, monkeypatch):
    adapter, pose = cartesian_adapter(tmp_path)
    kin = adapter.arms["arm"]
    target = np.array(kin.observe(pose))
    monkeypatch.setattr(kin.kinematics, "inverse_kinematics", lambda *a, **k: np.full(6, 200.0))
    with pytest.raises(ValueError, match="joint limits"):
        kin.solve(target, pose)
    monkeypatch.setattr(kin.kinematics, "inverse_kinematics", lambda *a, **k: np.zeros(6))
    with pytest.raises(ValueError, match="unreachable"):
        kin.solve(target, pose)


@pytest.mark.parametrize("joint_type", ["prismatic", "continuous", "floating", "planar"])
def test_cartesian_rejects_non_revolute_tool_chain(tmp_path, joint_type):
    from pathlib import Path

    from lerobot.rollout.agent.kinematics import CartesianKinematics

    adapter, _ = cartesian_adapter(tmp_path)
    arm = adapter.arms["arm"].config
    path = Path(arm.urdf_path)
    path.write_text(path.read_text().replace('type="revolute"', f'type="{joint_type}"', 1))
    with pytest.raises(ValueError, match="limited revolute"):
        CartesianKinematics(arm, adapter.config.axes)


def test_bimanual_ik_preserves_native_gripper_ranges(tmp_path):
    template, initial = cartesian_adapter(tmp_path)
    arms, axes, pose = {}, {}, {}
    for side, scale, aperture in (("left", 1.0, 1.0), ("right", np.pi / 180, 100.0)):
        joints = {f"{side}_{k}": v for k, v in template.arms["arm"].config.joints.items()}
        grip = f"{side}_gripper.pos"
        arms[side] = replace(
            template.arms["arm"].config, joints=joints, gripper=grip, joint_scale=dict.fromkeys(joints, scale)
        )
        for key in joints:
            axes[key] = PositionAxis(
                -2 / scale, 2 / scale, 2 / scale, "rad" if scale == 1 else "degrees", "joint", 0.1 / scale
            )
            pose[key] = initial[key.removeprefix(f"{side}_")] / scale
        axes[grip] = PositionAxis(
            0, aperture, aperture, "fraction" if aperture == 1 else "percent", "0 closed", aperture / 10
        )
        pose[grip] = aperture / 2
    cfg = replace(config(tmp_path), axes=axes, cartesian=arms)
    adapter = RobotAdapter(cfg, list(axes), {**dict.fromkeys(axes, float), "cam": (8, 8, 3)}, "bimanual", 10)
    adapter.reset(pose)
    obs = adapter.observation({**pose, "cam": raw()["cam"]}, "hold", 0, [], [])
    target = obs.state["command_state"].copy()
    target[6], target[13] = 0.55, 45.0
    command = dict(zip(adapter.keys, adapter.translate(target[None, :])[0], strict=True))
    assert command["left_gripper.pos"] == pytest.approx(0.55)
    assert command["right_gripper.pos"] == pytest.approx(45.0)
    for key in pose:
        if "gripper" not in key:
            assert command[key] == pytest.approx(pose[key], abs=1e-5)


@pytest.mark.parametrize(
    "model,url,key_env,wire,path",
    [
        ("gpt-6.1-sol", "https://api.openai.com/v1", "OPENAI_API_KEY", "responses", "/v1/responses"),
        ("claude-test", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "messages", "/v1/messages"),
        (
            "gemini-test",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "GEMINI_API_KEY",
            "chat",
            "/v1beta/openai/chat/completions",
        ),
        (
            "org/vision-model:provider",
            "https://router.huggingface.co/v1",
            "HF_TOKEN",
            "chat",
            "/v1/chat/completions",
        ),
        ("local-vision-model", "http://localhost:8000/v1", None, None, "/v1/chat/completions"),
    ],
)
def test_provider_images_tools_and_conversation(tmp_path, model, url, key_env, wire, path):
    from inspect_robots.scene import Scene

    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == path
        assert body["model"] == model
        assert "reasoning" not in body and "reasoning_effort" not in body
        assert "service_tier" not in body
        assert "data:image/png;base64," in json.dumps(body) or '"type": "base64"' in json.dumps(body)
        assert "move_joints" in json.dumps(body["tools"])
        if key_env:
            header = "x-api-key" if wire == "messages" else "authorization"
            assert "provider-test-key" in request.headers[header]
        name = "move_joints" if len(requests) == 1 else "done"
        args = (
            {"targets": {"j.pos": 0.1}, "note": "Approach."}
            if name == "move_joints"
            else {"summary": "Finished."}
        )
        if wire == "responses":
            return response(name, args)
        if wire == "messages":
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "tool_use", "id": f"call_{len(requests)}", "name": name, "input": args}
                    ],
                    "stop_reason": "tool_use",
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{len(requests)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                    "extra_content": {"google": {"thought_signature": "test-signature"}},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    cfg = replace(
        config(tmp_path),
        model=model,
        base_url=url,
        api_key_env=key_env,
        wire=wire,
        effort=None,
        service_tier=None,
    )
    e = engine(tmp_path, handler, cfg, env={key_env: "provider-test-key"} if key_env else {})
    try:
        e.agent.reset(Scene(id="test", instruction="Pick cube"))
        first = e.agent.act(e.adapter.observation(raw(), "Pick cube", 0, [], []))
        assert first.actions and not first.actions[0].meta.get("request_stop")
        second = e.agent.act(e.adapter.observation(raw(0.1), "Pick cube", len(first), [], []))
        assert second.actions[0].meta.get("request_stop")
        assert "Approach." in json.dumps(requests[1])
        if model == "gemini-test":
            assert "test-signature" in json.dumps(requests[1])
    finally:
        e.stop()


def test_provider_defaults_do_not_select_openai():
    cfg = AgentInferenceConfig()
    assert (cfg.model, cfg.base_url, cfg.api_key_env, cfg.wire, cfg.effort, cfg.service_tier) == (None,) * 6


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
