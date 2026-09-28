"""Real checkpoint/processor loading and identity, without network or model downloads."""

from dataclasses import replace

import torch

from lerobot.inference.contracts import FeatureSpec
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.remote_inference.configs import ExecutionConfig, ModelConfig, ServerConfig
from lerobot.scripts.lerobot_policy_server import artifact_identity, load_deployment
from tests.inference.test_policy_runner import observation, processors, tiny_config


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


def test_content_identity_changes_with_processor_statistics_and_effective_settings(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "policy_preprocessor.json").write_text('{"mean": 0}')
    first = artifact_identity({"checkpoint": tmp_path}, {"mode": "chunk"})
    (tmp_path / "policy_preprocessor.json").write_text('{"mean": 1}')
    assert artifact_identity({"checkpoint": tmp_path}, {"mode": "chunk"}) != first
    second = artifact_identity({"checkpoint": tmp_path}, {"mode": "chunk"})
    assert artifact_identity({"checkpoint": tmp_path}, {"mode": "rtc_guided"}) != second
