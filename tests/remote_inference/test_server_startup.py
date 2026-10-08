"""Deployment warmup and server lifecycle without model downloads."""

import signal
import subprocess
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lerobot.inference import FeatureSpec
from lerobot.processor import RenderRuntimeMessagesStep
from lerobot.remote_inference import deployment as serving
from lerobot.remote_inference.configs import ExecutionConfig, LanguageConfig, ModelConfig, ServerConfig
from lerobot.remote_inference.deployment import artifact_identity, load_deployment
from lerobot.remote_inference.protocol import ErrorCode, MessageType
from lerobot.remote_inference.server import SessionWorker
from lerobot.transport.zenoh import ZenohConfig
from lerobot.utils.constants import MESSAGES_RENDERED, QUERY_KIND
from tests.inference.test_policy_runner import ConformingPolicy, observation, processors, tiny_config
from tests.remote_inference.test_session import action_request, admit


def test_content_identity_changes_with_processor_statistics_and_effective_settings(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "policy_preprocessor.json").write_text('{"mean": 0}')
    first = artifact_identity({"checkpoint": tmp_path}, {"mode": "chunk"})
    (tmp_path / "policy_preprocessor.json").write_text('{"mean": 1}')
    assert artifact_identity({"checkpoint": tmp_path}, {"mode": "chunk"}) != first
    second = artifact_identity({"checkpoint": tmp_path}, {"mode": "chunk"})
    assert artifact_identity({"checkpoint": tmp_path}, {"mode": "rtc_guided"}) != second


def _language_deployment(tmp_path, monkeypatch, *, recipe_kind="none"):
    config = tiny_config()
    config.save_pretrained(tmp_path)
    policy = ConformingPolicy(config)
    calls = []

    def generate_text(batch):
        calls.append(batch.get(MESSAGES_RENDERED, batch.get(QUERY_KIND)))
        return "a cube"

    monkeypatch.setattr(policy, "generate_text", generate_text)
    monkeypatch.setattr(
        serving,
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


def test_language_warmup_model_errors_fail_startup_and_reset(tmp_path, monkeypatch):
    server, policy, _ = _language_deployment(tmp_path, monkeypatch)

    def fail_generation(batch):
        raise ValueError("genuine model failure")

    monkeypatch.setattr(policy, "generate_text", fail_generation)
    with pytest.raises(ValueError, match="genuine model failure"):
        load_deployment(server)
    assert policy.resets == 2


def test_unsupported_runtime_subtask_returns_bounded_error_and_preserves_session(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level("INFO", logger="lerobot.remote_inference.server")
    server, _, _ = _language_deployment(tmp_path, monkeypatch)
    runner, identity = load_deployment(server)
    worker = SessionWorker(runner, deployment="test", artifact_identity=identity, semantics="radians-v1")
    try:
        session = admit(worker)
        vqa = replace(action_request(worker, session), message_type=MessageType.LANGUAGE_REQUEST)
        vqa.body.update(kind="vqa", text="Private scene question", intent_generation=0)
        assert worker.submit(vqa).result(2).message_type is MessageType.LANGUAGE_RESULT
        request = replace(action_request(worker, session), message_type=MessageType.LANGUAGE_REQUEST)
        request.body.update(kind="next_subtask", text="Pick up the cube", intent_generation=0)
        response = worker.submit(request).result(2)
        assert response.message_type is MessageType.ERROR
        assert response.body["code"] == ErrorCode.EXECUTION
        assert "requires a checkpoint recipe" in response.body["message"]
        assert worker.submit(action_request(worker, session)).result(2).message_type is MessageType.ACTION
        assert "Language request accepted: deployment=test kind=vqa" in caplog.text
        assert "Language request completed: deployment=test kind=vqa elapsed=" in caplog.text
        assert "Language request accepted: deployment=test kind=next_subtask" in caplog.text
        assert "Language request failed: deployment=test kind=next_subtask elapsed=" in caplog.text
        assert "Private scene question" not in caplog.text
    finally:
        worker.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process signals")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_server_signal_logs_and_closes_real_transport(tmp_path, signum):
    pytest.importorskip("zenoh")
    script = """
import signal
from lerobot.remote_inference import ModelConfig, ServerConfig
from lerobot.scripts import lerobot_policy_server as entry
from lerobot.transport.zenoh import ZenohConfig
from tests.inference.test_policy_runner import ConformingPolicy, runner_for, tiny_config

runner = runner_for(ConformingPolicy(tiny_config()))
entry.load_deployment = lambda cfg: (runner, "test-artifact")
cfg = ServerConfig(
    deployment="signal-test", model=ModelConfig("unused"), semantics="radians-v1",
    features=list(runner.capabilities.features), action_feature=runner.capabilities.action_feature,
    zenoh=ZenohConfig(listen_endpoints=["tcp/127.0.0.1:0"]),
)
previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
entry.serve(cfg)
assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())
"""
    log = tmp_path / "server.log"
    with log.open("w") as output:
        process = subprocess.Popen([sys.executable, "-c", script], stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while "Policy server ready:" not in log.read_text():
                assert process.poll() is None and time.monotonic() < deadline, log.read_text()
                time.sleep(0.02)
            process.send_signal(signum)
            assert process.wait(timeout=10) == 0, log.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    text = log.read_text()
    assert text.count("Policy server shutdown requested:") == 1
    assert f"reason={signal.Signals(signum).name}" in text
    assert "Policy server stopping:" in text
    assert "Policy server transport closed:" in text
    assert "Policy worker stopped:" in text
