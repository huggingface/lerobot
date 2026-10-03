"""Real checkpoint/processor loading and identity, without network or model downloads."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lerobot.inference.contracts import FeatureSpec
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import RenderRuntimeMessagesStep
from lerobot.remote_inference.configs import ExecutionConfig, LanguageConfig, ModelConfig, ServerConfig
from lerobot.remote_inference.protocol import ErrorCode, MessageType
from lerobot.remote_inference.server import SessionWorker
from lerobot.scripts import lerobot_policy_server
from lerobot.scripts.lerobot_policy_server import artifact_identity, load_deployment
from lerobot.transport.zenoh import ZenohConfig
from lerobot.utils.constants import MESSAGES_RENDERED, QUERY_KIND
from tests.inference.test_policy_runner import ConformingPolicy, observation, processors, tiny_config
from tests.remote_inference.test_session import action_request, admit


def test_server_loads_saved_act_and_processors_warms_and_resets(tmp_path):
    config = tiny_config()
    policy = ACTPolicy(config)
    policy.save_pretrained(tmp_path)
    pre, post = processors(config)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    server = ServerConfig(
        deployment="test",
        model=ModelConfig(str(tmp_path)),
        execution=ExecutionConfig(action_fps=30, warmup_calls=1),
        zenoh=ZenohConfig(listen_endpoints=["tcp/127.0.0.1:7447"]),
        semantics="radians-v1",
        features=[FeatureSpec(key, (3,), "float32", semantics="radians-v1") for key in config.input_features],
        action_feature=FeatureSpec("action", (3,), "float32", semantics="radians-v1"),
    )
    runner, identity = load_deployment(server)
    assert identity.startswith("sha256:") and len(identity) == 71
    assert not runner.capabilities.language
    assert not runner.policy._action_queue
    with torch.inference_mode():
        expected = post(policy.predict_action_chunk(pre(runner._batch(observation()))))[0, :3]
    torch.testing.assert_close(runner.predict(observation()).canonical_actions, expected)
    changed, changed_identity = load_deployment(replace(server, semantics="degrees-v1"))
    assert changed_identity != identity
    assert changed.capabilities.execution_steps == 3
    _, debug_identity = load_deployment(replace(server, log_level="DEBUG"))
    assert debug_identity == identity, "console verbosity must not invalidate a pinned artifact"


def test_content_identity_changes_with_processor_statistics_and_effective_settings(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "policy_preprocessor.json").write_text('{"mean": 0}')
    first = artifact_identity({"checkpoint": tmp_path}, {"mode": "chunk"})
    (tmp_path / "policy_preprocessor.json").write_text('{"mean": 1}')
    assert artifact_identity({"checkpoint": tmp_path}, {"mode": "chunk"}) != first
    second = artifact_identity({"checkpoint": tmp_path}, {"mode": "chunk"})
    assert artifact_identity({"checkpoint": tmp_path}, {"mode": "rtc_guided"}) != second


def _language_deployment(tmp_path, monkeypatch, *, recipe_kind="none", use_renderer=True):
    config = tiny_config()
    config.save_pretrained(tmp_path)
    policy = ConformingPolicy(config)
    calls = []

    def generate_text(batch):
        calls.append(batch.get(MESSAGES_RENDERED, batch.get(QUERY_KIND)))
        return "a cube"

    monkeypatch.setattr(policy, "generate_text", generate_text)
    monkeypatch.setattr(
        lerobot_policy_server,
        "get_policy_class",
        lambda _: SimpleNamespace(from_pretrained=lambda *args, **kwargs: policy),
    )
    pre, post = processors(config)
    recipe = None
    if recipe_kind != "none":
        pytest.importorskip("datasets")
        from lerobot.datasets.recipe import MessageTurn, TrainingRecipe

        target = "${subtask}" if recipe_kind == "supported" else "${memory}"
        recipe = TrainingRecipe(
            messages=[
                MessageTurn(role="user", content="Goal: ${task}", stream="high_level"),
                MessageTurn(role="assistant", content=target, stream="high_level", target=True),
            ]
        )
    if use_renderer:
        pre.steps.insert(0, RenderRuntimeMessagesStep(recipe))
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    server = ServerConfig(
        deployment="test",
        model=ModelConfig(str(tmp_path)),
        execution=ExecutionConfig(action_fps=30, warmup_calls=1),
        language=LanguageConfig(enabled=True),
        zenoh=ZenohConfig(listen_endpoints=["tcp/127.0.0.1:7447"]),
        semantics="radians-v1",
        features=[FeatureSpec(key, (3,), "float32", semantics="radians-v1") for key in config.input_features],
        action_feature=FeatureSpec("action", (3,), "float32", semantics="radians-v1"),
    )
    return server, policy, calls


@pytest.mark.parametrize("recipe_kind", ["none", "missing_target"])
def test_language_warmup_skips_only_unsupported_saved_subtask_prompt(
    tmp_path, monkeypatch, caplog, recipe_kind
):
    server, policy, calls = _language_deployment(tmp_path, monkeypatch, recipe_kind=recipe_kind)
    runner, _ = load_deployment(server)
    assert runner.capabilities.language
    assert len(calls) == 1
    assert calls[0] == [[{"role": "user", "content": "Describe the scene."}]]
    assert policy.resets == 2  # constructor and reset after warmup
    assert "Skipping next-subtask warmup" in caplog.text
    assert "Actions and VQA remain available" in caplog.text
    assert "do not enable autosteer" in caplog.text
    assert runner.query(observation(), kind="vqa", text="What is visible?") == "a cube"
    assert runner.predict(observation()).canonical_actions.shape == (3, 3)


def test_language_warmup_runs_supported_saved_subtask_prompt(tmp_path, monkeypatch, caplog):
    server, policy, calls = _language_deployment(tmp_path, monkeypatch, recipe_kind="supported")
    load_deployment(server)
    assert calls == [
        [[{"role": "user", "content": "Describe the scene."}]],
        [[{"role": "user", "content": "Goal: Describe the next task."}]],
    ]
    assert policy.resets == 2
    assert "Skipping next-subtask warmup" not in caplog.text


def test_malformed_saved_recipe_is_not_treated_as_missing_subtask_support(tmp_path, monkeypatch, caplog):
    server, _, calls = _language_deployment(tmp_path, monkeypatch, recipe_kind="supported")
    path = tmp_path / "policy_preprocessor.json"
    saved = json.loads(path.read_text())
    saved["steps"][0]["config"]["recipe"]["messages"][1]["content"] = "${unknown_binding}"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="unknown binding"):
        load_deployment(server)
    assert calls == []
    assert "Skipping next-subtask warmup" not in caplog.text


@pytest.mark.parametrize("failure_kind", ["vqa", "next_subtask", "custom_processor"])
def test_language_warmup_model_errors_still_fail_startup_and_reset(
    tmp_path, monkeypatch, caplog, failure_kind
):
    server, policy, _ = _language_deployment(
        tmp_path,
        monkeypatch,
        recipe_kind="supported" if failure_kind == "next_subtask" else "none",
        use_renderer=failure_kind != "custom_processor",
    )
    calls = 0

    def fail_generation(batch):
        nonlocal calls
        calls += 1
        if failure_kind == "vqa" or calls == 2:
            raise ValueError("genuine model failure")
        return "a cube"

    monkeypatch.setattr(policy, "generate_text", fail_generation)
    with pytest.raises(ValueError, match="genuine model failure"):
        load_deployment(server)
    assert policy.resets == 2
    assert "Skipping next-subtask warmup" not in caplog.text


def test_unsupported_runtime_subtask_returns_bounded_error_and_preserves_session(tmp_path, monkeypatch):
    server, _, _ = _language_deployment(tmp_path, monkeypatch)
    runner, identity = load_deployment(server)
    worker = SessionWorker(runner, deployment="test", artifact_identity=identity, semantics="radians-v1")
    try:
        session = admit(worker)
        request = replace(action_request(worker, session), message_type=MessageType.LANGUAGE_REQUEST)
        request.body.update(kind="next_subtask", text="Pick up the cube", intent_generation=0)
        response = worker.submit(request).result(2)
        assert response.message_type is MessageType.ERROR
        assert response.body["code"] == ErrorCode.EXECUTION
        assert "requires a checkpoint recipe" in response.body["message"]
        assert worker.submit(action_request(worker, session)).result(2).message_type is MessageType.ACTION
    finally:
        worker.close()
