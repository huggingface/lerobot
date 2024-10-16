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
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_file

from lerobot import available_policies
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.common.policies.factory import (
    _policy_cfg_from_hydra_cfg,
    get_policy_and_config_classes,
    make_policy,
)
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.utils.utils import init_hydra_config, seeded_context
from lerobot.scripts.train import make_optimizer_and_scheduler
from tests.scripts.save_policy_to_safetensors import get_policy_stats
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE, require_cpu, require_env, require_x86_64_kernel


@pytest.mark.parametrize("policy_name", available_policies)
def test_get_policy_and_config_classes(policy_name: str):
    """Check that the correct policy and config classes are returned."""
    policy_cls, config_cls = get_policy_and_config_classes(policy_name)
    assert policy_cls.name == policy_name
    assert issubclass(config_cls, inspect.signature(policy_cls.__init__).parameters["config"].annotation)


# TODO(aliberts): refactor using lerobot/__init__.py variables
@pytest.mark.parametrize(
    "env_name,policy_name,extra_overrides",
    [
        ("xarm", "tdmpc", ["policy.use_mpc=true", "dataset_repo_id=lerobot/xarm_lift_medium"]),
        ("pusht", "diffusion", []),
        ("pusht", "vqbet", []),
        ("aloha", "act", ["env.task=AlohaInsertion-v0", "dataset_repo_id=lerobot/aloha_sim_insertion_human"]),
        (
            "aloha",
            "act",
            ["env.task=AlohaInsertion-v0", "dataset_repo_id=lerobot/aloha_sim_insertion_scripted"],
        ),
        (
            "aloha",
            "act",
            ["env.task=AlohaTransferCube-v0", "dataset_repo_id=lerobot/aloha_sim_transfer_cube_human"],
        ),
        (
            "aloha",
            "act",
            ["env.task=AlohaTransferCube-v0", "dataset_repo_id=lerobot/aloha_sim_transfer_cube_scripted"],
        ),
        # Note: these parameters also need custom logic in the test function for overriding the Hydra config.
        (
            "aloha",
            "diffusion",
            ["env.task=AlohaInsertion-v0", "dataset_repo_id=lerobot/aloha_sim_insertion_human"],
        ),
        # Note: these parameters also need custom logic in the test function for overriding the Hydra config.
        ("pusht", "act", ["env.task=PushT-v0", "dataset_repo_id=lerobot/pusht"]),
        ("dora_aloha_real", "act_real", []),
        ("dora_aloha_real", "act_real_no_state", []),
    ],
)
@require_env
def test_policy(env_name, policy_name, extra_overrides):
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
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            f"device={DEVICE}",
        ]
        + extra_overrides,
    )

    # Additional config override logic.
    if env_name == "aloha" and policy_name == "diffusion":
        for keys in [
            ("training", "delta_timestamps"),
            ("policy", "input_shapes"),
            ("policy", "input_normalization_modes"),
        ]:
            dct = dict(cfg[keys[0]][keys[1]])
            dct["observation.images.top"] = dct["observation.image"]
            del dct["observation.image"]
            cfg[keys[0]][keys[1]] = dct
        cfg.override_dataset_stats = None

    # Additional config override logic.
    if env_name == "pusht" and policy_name == "act":
        for keys in [
            ("policy", "input_shapes"),
            ("policy", "input_normalization_modes"),
        ]:
            dct = dict(cfg[keys[0]][keys[1]])
            dct["observation.image"] = dct["observation.images.top"]
            del dct["observation.images.top"]
            cfg[keys[0]][keys[1]] = dct
        cfg.override_dataset_stats = None

    # Check that we can make the policy object.
    dataset = make_dataset(cfg)
    policy = make_policy(hydra_cfg=cfg, dataset_stats=dataset.stats)
    # Check that the policy follows the required protocol.
    assert isinstance(
        policy, Policy
    ), f"The policy does not follow the required protocol. Please see {Policy.__module__}.{Policy.__name__}."
    assert isinstance(policy, torch.nn.Module)
    assert isinstance(policy, PyTorchModelHubMixin)

    # Check that we run select_actions and get the appropriate output.
    env = make_env(cfg, n_envs=2)

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
        batch[key] = batch[key].to(DEVICE, non_blocking=True)

    # Test updating the policy (and test that it does not mutate the batch)
    batch_ = deepcopy(batch)
    policy.forward(batch)
    assert set(batch) == set(batch_), "Batch keys are not the same after a forward pass."
    assert all(
        torch.equal(batch[k], batch_[k]) for k in batch
    ), "Batch values are not the same after a forward pass."

    # reset the policy and environment
    policy.reset()
    observation, _ = env.reset(seed=cfg.seed)

    # apply transform to normalize the observations
    observation = preprocess_observation(observation)

    # send observation to device/gpu
    observation = {key: observation[key].to(DEVICE, non_blocking=True) for key in observation}

    # get the next action for the environment (also check that the observation batch is not modified)
    observation_ = deepcopy(observation)
    with torch.inference_mode():
        action = policy.select_action(observation).cpu().numpy()
    assert set(observation) == set(
        observation_
    ), "Observation batch keys are not the same after a forward pass."
    assert all(
        torch.equal(observation[k], observation_[k]) for k in observation
    ), "Observation batch values are not the same after a forward pass."

    # Test step through policy
    env.step(action)


def test_act_backbone_lr():
    """
    Test that the ACT policy can be instantiated with a different learning rate for the backbone.
    """
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            "env=aloha",
            "policy=act",
            f"device={DEVICE}",
            "training.lr_backbone=0.001",
            "training.lr=0.01",
        ],
    )
    assert cfg.training.lr == 0.01
    assert cfg.training.lr_backbone == 0.001

    dataset = make_dataset(cfg)
    policy = make_policy(hydra_cfg=cfg, dataset_stats=dataset.stats)
    optimizer, _ = make_optimizer_and_scheduler(cfg, policy)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == cfg.training.lr
    assert optimizer.param_groups[1]["lr"] == cfg.training.lr_backbone
    assert len(optimizer.param_groups[0]["params"]) == 133
    assert len(optimizer.param_groups[1]["params"]) == 20


@pytest.mark.parametrize("policy_name", available_policies)
def test_policy_defaults(policy_name: str):
    """Check that the policy can be instantiated with defaults."""
    policy_cls, _ = get_policy_and_config_classes(policy_name)
    policy_cls()


@pytest.mark.parametrize(
    "env_name,policy_name",
    [
        ("xarm", "tdmpc"),
        ("pusht", "diffusion"),
        ("aloha", "act"),
    ],
)
def test_yaml_matches_dataclass(env_name: str, policy_name: str):
    """Check that dataclass configs match their respective yaml configs."""
    hydra_cfg = init_hydra_config(DEFAULT_CONFIG_PATH, overrides=[f"env={env_name}", f"policy={policy_name}"])
    _, policy_cfg_cls = get_policy_and_config_classes(policy_name)
    policy_cfg_from_hydra = _policy_cfg_from_hydra_cfg(policy_cfg_cls, hydra_cfg)
    policy_cfg_from_dataclass = policy_cfg_cls()
    assert policy_cfg_from_hydra == policy_cfg_from_dataclass


@pytest.mark.parametrize("policy_name", available_policies)
def test_save_and_load_pretrained(policy_name: str):
    policy_cls, _ = get_policy_and_config_classes(policy_name)
    policy: Policy = policy_cls()
    save_dir = "/tmp/test_save_and_load_pretrained_{policy_cls.__name__}"
    policy.save_pretrained(save_dir)
    policy_ = policy_cls.from_pretrained(save_dir)
    assert all(torch.equal(p, p_) for p, p_ in zip(policy.parameters(), policy_.parameters(), strict=True))


@pytest.mark.parametrize("insert_temporal_dim", [False, True])
def test_normalize(insert_temporal_dim):
    """
    Test that normalize/unnormalize can run without exceptions when properly set up, and that they raise
    an exception when the forward pass is called without the stats having been provided.

    TODO(rcadene, alexander-soare): This should also test that the normalization / unnormalization works as
    expected.
    """

    input_shapes = {
        "observation.image": [3, 96, 96],
        "observation.state": [10],
    }
    output_shapes = {
        "action": [5],
    }

    normalize_input_modes = {
        "observation.image": "mean_std",
        "observation.state": "min_max",
    }
    unnormalize_output_modes = {
        "action": "min_max",
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
    normalize = Normalize(input_shapes, normalize_input_modes, stats=None)
    with pytest.raises(AssertionError):
        normalize(input_batch)

    # test with stats
    normalize = Normalize(input_shapes, normalize_input_modes, stats=dataset_stats)
    normalize(input_batch)

    # test loading pretrained models
    new_normalize = Normalize(input_shapes, normalize_input_modes, stats=None)
    new_normalize.load_state_dict(normalize.state_dict())
    new_normalize(input_batch)

    # test without stats
    unnormalize = Unnormalize(output_shapes, unnormalize_output_modes, stats=None)
    with pytest.raises(AssertionError):
        unnormalize(output_batch)

    # test with stats
    unnormalize = Unnormalize(output_shapes, unnormalize_output_modes, stats=dataset_stats)
    unnormalize(output_batch)

    # test loading pretrained models
    new_unnormalize = Unnormalize(output_shapes, unnormalize_output_modes, stats=None)
    new_unnormalize.load_state_dict(unnormalize.state_dict())
    unnormalize(output_batch)


@pytest.mark.parametrize(
    "env_name, policy_name, extra_overrides, file_name_extra",
    [
        # TODO(alexander-soare): `policy.use_mpc=false` was previously the default in the config yaml but it
        # was changed to true. For some reason, tests would pass locally, but not in CI. So here we override
        # to test with `policy.use_mpc=false`.
        ("xarm", "tdmpc", ["policy.use_mpc=false"], "use_policy"),
        # ("xarm", "tdmpc", ["policy.use_mpc=true"], "use_mpc"),
        (
            "pusht",
            "diffusion",
            ["policy.n_action_steps=8", "policy.num_inference_steps=10", "policy.down_dims=[128, 256, 512]"],
            "",
        ),
        ("aloha", "act", ["policy.n_action_steps=10"], ""),
        ("aloha", "act", ["policy.n_action_steps=1000", "policy.chunk_size=1000"], "_1000_steps"),
        ("dora_aloha_real", "act_aloha_real", ["policy.n_action_steps=10"], ""),
    ],
)
# As artifacts have been generated on an x86_64 kernel, this test won't
# pass if it's run on another platform due to floating point errors
@require_x86_64_kernel
@require_cpu
def test_backward_compatibility(env_name, policy_name, extra_overrides, file_name_extra):
    """
    NOTE: If this test does not pass, and you have intentionally changed something in the policy:
        1. Inspect the differences in policy outputs and make sure you can account for them. Your PR should
           include a report on what changed and how that affected the outputs.
        2. Go to the `if __name__ == "__main__"` block of `tests/scripts/save_policy_to_safetensors.py` and
           add the policies you want to update the test artifacts for.
        3. Run `DATA_DIR=tests/data python tests/scripts/save_policy_to_safetensors.py`. The test artifact
           should be updated.
        4. Check that this test now passes.
        5. Remember to restore `tests/scripts/save_policy_to_safetensors.py` to its original state.
        6. Remember to stage and commit the resulting changes to `tests/data`.
    """
    env_policy_dir = (
        Path("tests/data/save_policy_to_safetensors") / f"{env_name}_{policy_name}{file_name_extra}"
    )
    saved_output_dict = load_file(env_policy_dir / "output_dict.safetensors")
    saved_grad_stats = load_file(env_policy_dir / "grad_stats.safetensors")
    saved_param_stats = load_file(env_policy_dir / "param_stats.safetensors")
    saved_actions = load_file(env_policy_dir / "actions.safetensors")

    output_dict, grad_stats, param_stats, actions = get_policy_stats(env_name, policy_name, extra_overrides)

    for key in saved_output_dict:
        assert torch.isclose(output_dict[key], saved_output_dict[key], rtol=0.1, atol=1e-7).all()
    for key in saved_grad_stats:
        assert torch.isclose(grad_stats[key], saved_grad_stats[key], rtol=0.1, atol=1e-7).all()
    for key in saved_param_stats:
        assert torch.isclose(param_stats[key], saved_param_stats[key], rtol=50, atol=1e-7).all()
    for key in saved_actions:
        assert torch.isclose(actions[key], saved_actions[key], rtol=0.1, atol=1e-7).all()


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
        assert torch.allclose(online_avg, offline_avg, atol=1e-4)


if __name__ == "__main__":
    test_act_temporal_ensembler()
