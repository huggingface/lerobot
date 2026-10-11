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
"""VQ-BeT inference regressions using the real model, without downloaded weights."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

CAMERA = f"{OBS_IMAGES}.camera"


@pytest.fixture
def vqbet_policy(request):
    config = VQBeTConfig(
        device="cpu",
        input_features={
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        pretrained_backbone_weights=None,
        crop_shape=None,
        n_obs_steps=getattr(request, "param", 3),
        action_chunk_size=3,
        n_action_pred_token=2,
        spatial_softmax_num_keypoints=8,
        gpt_input_dim=32,
        gpt_output_dim=32,
        gpt_hidden_dim=32,
        gpt_n_layer=1,
        gpt_n_head=2,
        vqvae_n_embed=4,
        vqvae_embedding_dim=8,
        vqvae_enc_hidden_dim=16,
    )
    return VQBeTPolicy(config).eval()


def make_observation(batch_size, step=0):
    return {
        CAMERA: torch.full((batch_size, 3, 64, 64), step / 10),
        OBS_STATE: torch.full((batch_size, 6), float(step)),
    }


@torch.no_grad()
def model_actions(policy, batch):
    # VQ-BeT samples discrete codes even in eval mode. Reuse the same RNG state
    # when comparing the public inference paths with the real model's output.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return policy.vqbet(batch, rollout=True)


@torch.no_grad()
def predict_actions(policy, batch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return policy.predict_action_chunk(batch)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("vqbet_policy", [1, 3, 5], indirect=True)
def test_vqbet_predict_action_chunk_without_select_action(vqbet_policy, batch_size):
    policy = vqbet_policy
    n_obs_steps = policy.config.n_obs_steps
    policy.reset()
    for step in (1, 2):
        batch = make_observation(batch_size, step)
        # Offline batches can contain actions and unrelated metadata. Neither
        # should be queued or removed from the caller's input during inference.
        batch[ACTION] = torch.randn(batch_size, policy.config.action_chunk_size, 6)
        batch["index"] = torch.arange(batch_size)
        originals = dict(batch)
        copies = {key: value.clone() for key, value in batch.items()}
        expected_batch = {
            OBS_STATE: torch.stack([batch[OBS_STATE]] * n_obs_steps, dim=1),
            OBS_IMAGES: torch.stack([batch[CAMERA]] * n_obs_steps, dim=1).unsqueeze(2),
        }
        expected = model_actions(policy, expected_batch)
        actions = predict_actions(policy, batch)
        assert actions.shape == (batch_size, policy.config.action_chunk_size, 6)
        assert torch.isfinite(actions).all()
        torch.testing.assert_close(actions, expected)
        assert batch.keys() == originals.keys()
        for key in batch:
            assert batch[key] is originals[key]
            torch.testing.assert_close(batch[key], copies[key])
        assert all(not queue for queue in policy._queues.values())

    policy.reset()
    torch.testing.assert_close(predict_actions(policy, batch), expected)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_vqbet_predict_action_chunk_preserves_temporal_batch(vqbet_policy, batch_size):
    policy = vqbet_policy
    frames = [make_observation(batch_size, step) for step in range(policy.config.n_obs_steps)]
    batch = {key: torch.stack([frame[key] for frame in frames], dim=1) for key in frames[0]}
    originals = {key: value.clone() for key, value in batch.items()}
    expected_batch = {OBS_STATE: batch[OBS_STATE], OBS_IMAGES: batch[CAMERA].unsqueeze(2)}
    expected = model_actions(policy, expected_batch)
    torch.testing.assert_close(predict_actions(policy, batch), expected)
    assert batch.keys() == originals.keys()
    for key in batch:
        torch.testing.assert_close(batch[key], originals[key])
    assert all(not queue for queue in policy._queues.values())


@pytest.mark.parametrize("batch_size", [1, 2])
def test_vqbet_select_action_preserves_history_and_action_queue(vqbet_policy, batch_size):
    policy = vqbet_policy
    model_batches = []

    def record_batch(_module, args):
        model_batches.append({key: value.clone() for key, value in args[0].items()})

    handle = policy.vqbet.register_forward_pre_hook(record_batch)
    try:
        for step in range(policy.config.action_chunk_size + 1):
            batch = make_observation(batch_size, step)
            if step % policy.config.action_chunk_size == 0:
                expected_batch = {
                    OBS_STATE: torch.stack(
                        [
                            make_observation(batch_size, max(0, i))[OBS_STATE]
                            for i in range(step - policy.config.n_obs_steps + 1, step + 1)
                        ],
                        dim=1,
                    ),
                    OBS_IMAGES: torch.stack(
                        [
                            make_observation(batch_size, max(0, i))[CAMERA]
                            for i in range(step - policy.config.n_obs_steps + 1, step + 1)
                        ],
                        dim=1,
                    ).unsqueeze(2),
                }
                expected_chunk = model_actions(policy, expected_batch)
                model_batches.clear()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                with pytest.warns(UserWarning, match="pretrained Residual VQ"):
                    action = policy.select_action(batch)
            torch.testing.assert_close(action, expected_chunk[:, step % policy.config.action_chunk_size])
            assert len(model_batches) == 1
            for key, value in expected_batch.items():
                torch.testing.assert_close(model_batches[0][key], value)
            # One observation is appended per environment step, including steps
            # served from cached actions; chunk prediction must not append again.
            expected_steps = [max(0, i) for i in range(step - policy.config.n_obs_steps + 1, step + 1)]
            assert [frame[0, 0].item() for frame in policy._queues[OBS_STATE]] == expected_steps
            assert (
                len(policy._queues[ACTION])
                == policy.config.action_chunk_size - 1 - step % policy.config.action_chunk_size
            )
            assert OBS_IMAGES not in batch

        # Direct chunk prediction also reuses existing synchronous history without
        # changing either observation history or cached actions.
        queues = {key: list(queue) for key, queue in policy._queues.items()}
        torch.testing.assert_close(predict_actions(policy, batch), expected_chunk)
        for key, values in queues.items():
            assert len(policy._queues[key]) == len(values)
            assert all(
                actual is previous for actual, previous in zip(policy._queues[key], values, strict=True)
            )
        policy.reset()
        assert all(not queue for queue in policy._queues.values())
        assert predict_actions(policy, batch).shape == (batch_size, policy.config.action_chunk_size, 6)
    finally:
        handle.remove()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("actions_per_chunk", [1, 2, 5])
def test_vqbet_policy_server_truncates_time_axis(vqbet_policy, batch_size, actions_per_chunk):
    pytest.importorskip("grpc", reason="grpcio is required (install lerobot[grpcio-dep])")
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    policy = vqbet_policy
    server = PolicyServer(PolicyServerConfig())
    server.policy = policy
    server.actions_per_chunk = actions_per_chunk
    for step in (1, 2):
        batch = make_observation(batch_size, step)
        expected = predict_actions(policy, batch)[:, :actions_per_chunk]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            chunk = server._get_action_chunk(batch)
        assert chunk.shape == (batch_size, min(actions_per_chunk, policy.config.action_chunk_size), 6)
        torch.testing.assert_close(chunk, expected)
        assert all(not queue for queue in policy._queues.values())
    policy.reset()
    assert server._get_action_chunk(batch).shape == chunk.shape
