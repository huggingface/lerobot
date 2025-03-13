#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import inspect
from copy import deepcopy
from pathlib import Path

import einops
import pytest
import torch
from safetensors.torch import load_file

from lerobot import available_policies
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle, dataset_to_policy_features
from lerobot.common.envs.factory import make_env, make_env_config
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.common.policies.factory import (
    get_policy_class,
    make_policy,
    make_policy_config,
)
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import seeded_context
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from tests.artifacts.policies.save_policy_to_safetensors import get_policy_stats
from tests.utils import DEVICE, require_cpu, require_env, require_x86_64_kernel


@pytest.fixture
def dummy_dataset_metadata(lerobot_dataset_metadata_factory, info_factory, tmp_path):
    # Create only one camera input which is squared to fit all current policy constraints
    # e.g. vqbet and tdmpc works with one camera only, and tdmpc requires it to be squared
    camera_features = {
        "observation.images.laptop": {
            "shape": (84, 84, 3),
            "names": ["height", "width", "channels"],
            "info": None,
        },
    }
    motor_features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        },
    }
    info = info_factory(
        total_episodes=1, total_frames=1, camera_features=camera_features, motor_features=motor_features
    )
    ds_meta = lerobot_dataset_metadata_factory(root=tmp_path / "init", info=info)
    return ds_meta


@pytest.mark.parametrize("policy_name", available_policies)
def test_get_policy_and_config_classes(policy_name: str):
    """Check that the correct policy and config classes are returned."""
    policy_cls = get_policy_class(policy_name)
    policy_cfg = make_policy_config(policy_name)
    assert policy_cls.name == policy_name
    assert issubclass(
        policy_cfg.__class__, inspect.signature(policy_cls.__init__).parameters["config"].annotation
    )


@pytest.mark.parametrize(
    "ds_repo_id,env_name,env_kwargs,policy_name,policy_kwargs",
    [
        ("lerobot/xarm_lift_medium", "xarm", {}, "tdmpc", {"use_mpc": True}),
        ("lerobot/pusht", "pusht", {}, "diffusion", {}),
        ("lerobot/pusht", "pusht", {}, "vqbet", {}),
        ("lerobot/pusht", "pusht", {}, "act", {}),
        ("lerobot/aloha_sim_insertion_human", "aloha", {"task": "AlohaInsertion-v0"}, "act", {}),
        (
            "lerobot/aloha_sim_insertion_scripted",
            "aloha",
            {"task": "AlohaInsertion-v0"},
            "act",
            {},
        ),
        (
            "lerobot/aloha_sim_insertion_human",
            "aloha",
            {"task": "AlohaInsertion-v0"},
            "diffusion",
            {},
        ),
        (
            "lerobot/aloha_sim_transfer_cube_human",
            "aloha",
            {"task": "AlohaTransferCube-v0"},
            "act",
            {},
        ),
        (
            "lerobot/aloha_sim_transfer_cube_scripted",
            "aloha",
            {"task": "AlohaTransferCube-v0"},
            "act",
            {},
        ),
    ],
)
@require_env
def test_policy(ds_repo_id, env_name, env_kwargs, policy_name, policy_kwargs):
    """
    Tests:
        - Making the policy object.
        - Checking that the policy follows the correct protocol and subclasses nn.Module
            and PyTorchModelHubMixin.
        - Updating the policy.
        - Using the policy to select actions at inference time.
        - Test the action can be applied to the policy

    Note: We test various combinations of policy and dataset. The combinations are by no means exhaustive,
          and for now we add tests as we see fit.
    """

    train_cfg = TrainPipelineConfig(
        # TODO(rcadene, aliberts): remove dataset download
        dataset=DatasetConfig(repo_id=ds_repo_id, episodes=[0]),
        policy=make_policy_config(policy_name, **policy_kwargs),
        env=make_env_config(env_name, **env_kwargs),
    )

    # Check that we can make the policy object.
    dataset = make_dataset(train_cfg)
    policy = make_policy(train_cfg.policy, ds_meta=dataset.meta)
    assert isinstance(policy, PreTrainedPolicy)

    # Check that we run select_actions and get the appropriate output.
    env = make_env(train_cfg.env, n_envs=2)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=2,
        shuffle=True,
        pin_memory=DEVICE != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    batch = next(dl_iter)

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(DEVICE, non_blocking=True)

    # Test updating the policy (and test that it does not mutate the batch)
    batch_ = deepcopy(batch)
    policy.forward(batch)
    assert set(batch) == set(batch_), "Batch keys are not the same after a forward pass."
    assert all(
        torch.equal(batch[k], batch_[k]) if isinstance(batch[k], torch.Tensor) else batch[k] == batch_[k]
        for k in batch
    ), "Batch values are not the same after a forward pass."

    # reset the policy and environment
    policy.reset()
    observation, _ = env.reset(seed=train_cfg.seed)

    # apply transform to normalize the observations
    observation = preprocess_observation(observation)

    # send observation to device/gpu
    observation = {key: observation[key].to(DEVICE, non_blocking=True) for key in observation}

    # get the next action for the environment (also check that the observation batch is not modified)
    observation_ = deepcopy(observation)
    with torch.inference_mode():
        action = policy.select_action(observation).cpu().numpy()
    assert set(observation) == set(observation_), (
        "Observation batch keys are not the same after a forward pass."
    )
    assert all(torch.equal(observation[k], observation_[k]) for k in observation), (
        "Observation batch values are not the same after a forward pass."
    )

    # Test step through policy
    env.step(action)


# TODO(rcadene, aliberts): This test is quite end-to-end. Move this test in test_optimizer?
def test_act_backbone_lr():
    """
    Test that the ACT policy can be instantiated with a different learning rate for the backbone.
    """

    cfg = TrainPipelineConfig(
        # TODO(rcadene, aliberts): remove dataset download
        dataset=DatasetConfig(repo_id="lerobot/aloha_sim_insertion_scripted", episodes=[0]),
        policy=make_policy_config("act", optimizer_lr=0.01, optimizer_lr_backbone=0.001),
    )
    cfg.validate()  # Needed for auto-setting some parameters

    assert cfg.policy.optimizer_lr == 0.01
    assert cfg.policy.optimizer_lr_backbone == 0.001

    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    optimizer, _ = make_optimizer_and_scheduler(cfg, policy)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == cfg.policy.optimizer_lr
    assert optimizer.param_groups[1]["lr"] == cfg.policy.optimizer_lr_backbone
    assert len(optimizer.param_groups[0]["params"]) == 133
    assert len(optimizer.param_groups[1]["params"]) == 20


@pytest.mark.parametrize("policy_name", available_policies)
def test_policy_defaults(dummy_dataset_metadata, policy_name: str):
    """Check that the policy can be instantiated with defaults."""
    policy_cls = get_policy_class(policy_name)
    policy_cfg = make_policy_config(policy_name)
    features = dataset_to_policy_features(dummy_dataset_metadata.features)
    policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    policy_cfg.input_features = {
        key: ft for key, ft in features.items() if key not in policy_cfg.output_features
    }
    policy_cls(policy_cfg)


@pytest.mark.parametrize("policy_name", available_policies)
def test_save_and_load_pretrained(dummy_dataset_metadata, tmp_path, policy_name: str):
    policy_cls = get_policy_class(policy_name)
    policy_cfg = make_policy_config(policy_name)
    features = dataset_to_policy_features(dummy_dataset_metadata.features)
    policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    policy_cfg.input_features = {
        key: ft for key, ft in features.items() if key not in policy_cfg.output_features
    }
    policy = policy_cls(policy_cfg)
    policy.to(policy_cfg.device)
    save_dir = tmp_path / f"test_save_and_load_pretrained_{policy_cls.__name__}"
    policy.save_pretrained(save_dir)
    loaded_policy = policy_cls.from_pretrained(save_dir, config=policy_cfg)
    torch.testing.assert_close(list(policy.parameters()), list(loaded_policy.parameters()), rtol=0, atol=0)


@pytest.mark.parametrize("insert_temporal_dim", [False, True])
def test_normalize(insert_temporal_dim):
    """
    Test that normalize/unnormalize can run without exceptions when properly set up, and that they raise
    an exception when the forward pass is called without the stats having been provided.

    TODO(rcadene, alexander-soare): This should also test that the normalization / unnormalization works as
    expected.
    """

    input_features = {
        "observation.image": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 96, 96),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(10,),
        ),
    }
    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(5,),
        ),
    }

    norm_map = {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }

    dataset_stats = {
        "observation.image": {
            "mean": torch.randn(3, 1, 1),
            "std": torch.randn(3, 1, 1),
            "min": torch.randn(3, 1, 1),
            "max": torch.randn(3, 1, 1),
        },
        "observation.state": {
            "mean": torch.randn(10),
            "std": torch.randn(10),
            "min": torch.randn(10),
            "max": torch.randn(10),
        },
        "action": {
            "mean": torch.randn(5),
            "std": torch.randn(5),
            "min": torch.randn(5),
            "max": torch.randn(5),
        },
    }

    bsize = 2
    input_batch = {
        "observation.image": torch.randn(bsize, 3, 96, 96),
        "observation.state": torch.randn(bsize, 10),
    }
    output_batch = {
        "action": torch.randn(bsize, 5),
    }

    if insert_temporal_dim:
        tdim = 4

        for key in input_batch:
            # [2,3,96,96] -> [2,tdim,3,96,96]
            input_batch[key] = torch.stack([input_batch[key]] * tdim, dim=1)

        for key in output_batch:
            output_batch[key] = torch.stack([output_batch[key]] * tdim, dim=1)

    # test without stats
    normalize = Normalize(input_features, norm_map, stats=None)
    with pytest.raises(AssertionError):
        normalize(input_batch)

    # test with stats
    normalize = Normalize(input_features, norm_map, stats=dataset_stats)
    normalize(input_batch)

    # test loading pretrained models
    new_normalize = Normalize(input_features, norm_map, stats=None)
    new_normalize.load_state_dict(normalize.state_dict())
    new_normalize(input_batch)

    # test without stats
    unnormalize = Unnormalize(output_features, norm_map, stats=None)
    with pytest.raises(AssertionError):
        unnormalize(output_batch)

    # test with stats
    unnormalize = Unnormalize(output_features, norm_map, stats=dataset_stats)
    unnormalize(output_batch)

    # test loading pretrained models
    new_unnormalize = Unnormalize(output_features, norm_map, stats=None)
    new_unnormalize.load_state_dict(unnormalize.state_dict())
    unnormalize(output_batch)


@pytest.mark.parametrize(
    "ds_repo_id, policy_name, policy_kwargs, file_name_extra",
    [
        # TODO(alexander-soare): `policy.use_mpc=false` was previously the default in the config yaml but it
        # was changed to true. For some reason, tests would pass locally, but not in CI. So here we override
        # to test with `policy.use_mpc=false`.
        ("lerobot/xarm_lift_medium", "tdmpc", {"use_mpc": False}, "use_policy"),
        # ("lerobot/xarm_lift_medium", "tdmpc", {"use_mpc": True}, "use_mpc"),
        # TODO(rcadene): the diffusion model was normalizing the image in mean=0.5 std=0.5 which is a hack supposed to
        # to normalize the image at all. In our current codebase we dont normalize at all. But there is still a minor difference
        # that fails the test. However, by testing to normalize the image with 0.5 0.5 in the current codebase, the test pass.
        # Thus, we deactivate this test for now.
        (
            "lerobot/pusht",
            "diffusion",
            {
                "n_action_steps": 8,
                "num_inference_steps": 10,
                "down_dims": [128, 256, 512],
            },
            "",
        ),
        ("lerobot/aloha_sim_insertion_human", "act", {"n_action_steps": 10}, ""),
        (
            "lerobot/aloha_sim_insertion_human",
            "act",
            {"n_action_steps": 1000, "chunk_size": 1000},
            "1000_steps",
        ),
    ],
)
# As artifacts have been generated on an x86_64 kernel, this test won't
# pass if it's run on another platform due to floating point errors
@require_x86_64_kernel
@require_cpu
def test_backward_compatibility(ds_repo_id: str, policy_name: str, policy_kwargs: dict, file_name_extra: str):
    """
    NOTE: If this test does not pass, and you have intentionally changed something in the policy:
        1. Inspect the differences in policy outputs and make sure you can account for them. Your PR should
           include a report on what changed and how that affected the outputs.
        2. Go to the `if __name__ == "__main__"` block of `tests/scripts/save_policy_to_safetensors.py` and
           add the policies you want to update the test artifacts for.
        3. Run `python tests/scripts/save_policy_to_safetensors.py`. The test artifact
           should be updated.
        4. Check that this test now passes.
        5. Remember to restore `tests/scripts/save_policy_to_safetensors.py` to its original state.
        6. Remember to stage and commit the resulting changes to `tests/artifacts`.
    """
    ds_name = ds_repo_id.split("/")[-1]
    artifact_dir = Path("tests/artifacts/policies") / f"{ds_name}_{policy_name}_{file_name_extra}"
    saved_output_dict = load_file(artifact_dir / "output_dict.safetensors")
    saved_grad_stats = load_file(artifact_dir / "grad_stats.safetensors")
    saved_param_stats = load_file(artifact_dir / "param_stats.safetensors")
    saved_actions = load_file(artifact_dir / "actions.safetensors")

    output_dict, grad_stats, param_stats, actions = get_policy_stats(ds_repo_id, policy_name, policy_kwargs)

    for key in saved_output_dict:
        torch.testing.assert_close(output_dict[key], saved_output_dict[key])
    for key in saved_grad_stats:
        torch.testing.assert_close(grad_stats[key], saved_grad_stats[key])
    for key in saved_param_stats:
        torch.testing.assert_close(param_stats[key], saved_param_stats[key])
    for key in saved_actions:
        rtol, atol = (2e-3, 5e-6) if policy_name == "diffusion" else (None, None)  # HACK
        torch.testing.assert_close(actions[key], saved_actions[key], rtol=rtol, atol=atol)


def test_act_temporal_ensembler():
    """Check that the online method in ACTTemporalEnsembler matches a simple offline calculation."""
    temporal_ensemble_coeff = 0.01
    chunk_size = 100
    episode_length = 101
    ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff, chunk_size)
    # An batch of arbitrary sequences of 1D actions we wish to compute the average over. We'll keep the
    # "action space" in [-1, 1]. Apart from that, there is no real reason for the numbers chosen.
    with seeded_context(0):
        # Dimension is (batch, episode_length, chunk_size, action_dim(=1))
        # Stepping through the episode_length dim is like running inference at each rollout step and getting
        # a different action chunk.
        batch_seq = torch.stack(
            [
                torch.rand(episode_length, chunk_size) * 0.05 - 0.6,
                torch.rand(episode_length, chunk_size) * 0.02 - 0.01,
                torch.rand(episode_length, chunk_size) * 0.2 + 0.3,
            ],
            dim=0,
        ).unsqueeze(-1)  # unsqueeze for action dim
    batch_size = batch_seq.shape[0]
    # Exponential weighting (normalized). Unsqueeze once to match the position of the `episode_length`
    # dimension of `batch_seq`.
    weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size)).unsqueeze(-1)

    # Simulate stepping through a rollout and computing a batch of actions with model on each step.
    for i in range(episode_length):
        # Mock a batch of actions.
        actions = torch.zeros(size=(batch_size, chunk_size, 1)) + batch_seq[:, i]
        online_avg = ensembler.update(actions)
        # Simple offline calculation: avg = Σ(aᵢ*wᵢ) / Σ(wᵢ).
        # Note: The complicated bit here is the slicing. Think about the (episode_length, chunk_size) grid.
        # What we want to do is take diagonal slices across it starting from the left.
        #  eg: chunk_size=4, episode_length=6
        #  ┌───────┐
        #  │0 1 2 3│
        #  │1 2 3 4│
        #  │2 3 4 5│
        #  │3 4 5 6│
        #  │4 5 6 7│
        #  │5 6 7 8│
        #  └───────┘
        chunk_indices = torch.arange(min(i, chunk_size - 1), -1, -1)
        episode_step_indices = torch.arange(i + 1)[-len(chunk_indices) :]
        seq_slice = batch_seq[:, episode_step_indices, chunk_indices]
        offline_avg = (
            einops.reduce(seq_slice * weights[: i + 1], "b s 1 -> b 1", "sum") / weights[: i + 1].sum()
        )
        # Sanity check. The average should be between the extrema.
        assert torch.all(einops.reduce(seq_slice, "b s 1 -> b 1", "min") <= offline_avg)
        assert torch.all(offline_avg <= einops.reduce(seq_slice, "b s 1 -> b 1", "max"))
        # Selected atol=1e-4 keeping in mind actions in [-1, 1] and excepting 0.01% error.
        torch.testing.assert_close(online_avg, offline_avg, rtol=1e-4, atol=1e-4)
