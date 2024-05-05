import inspect
from pathlib import Path

import pytest
import torch
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_file

from lerobot import available_policies
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import get_policy_and_config_classes, make_policy
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.utils.utils import init_hydra_config
from tests.scripts.save_policy_to_safetensor import get_policy_stats
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE, require_env, require_x86_64_kernel


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
        num_workers=4,
        batch_size=2,
        shuffle=True,
        pin_memory=DEVICE != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    batch = next(dl_iter)

    for key in batch:
        batch[key] = batch[key].to(DEVICE, non_blocking=True)

    # Test updating the policy
    policy.forward(batch)

    # reset the policy and environment
    policy.reset()
    observation, _ = env.reset(seed=cfg.seed)

    # apply transform to normalize the observations
    observation = preprocess_observation(observation)

    # send observation to device/gpu
    observation = {key: observation[key].to(DEVICE, non_blocking=True) for key in observation}

    # get the next action for the environment
    with torch.inference_mode():
        action = policy.select_action(observation).cpu().numpy()

    # Test step through policy
    env.step(action)


@pytest.mark.parametrize("policy_name", available_policies)
def test_policy_defaults(policy_name: str):
    """Check that the policy can be instantiated with defaults."""
    policy_cls, _ = get_policy_and_config_classes(policy_name)
    policy_cls()


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
    "env_name, policy_name, extra_overrides",
    [
        # ("xarm", "tdmpc", ["policy.n_action_repeats=2"]),
        (
            "pusht",
            "diffusion",
            ["policy.n_action_steps=8", "policy.num_inference_steps=10", "policy.down_dims=[128, 256, 512]"],
        ),
        ("aloha", "act", ["policy.n_action_steps=10"]),
    ],
)
# As artifacts have been generated on an x86_64 kernel, this test won't
# pass if it's run on another platform due to floating point errors
@require_x86_64_kernel
def test_backward_compatibility(env_name, policy_name, extra_overrides):
    """
    NOTE: If this test does not pass, and you have intentionally changed something in the policy:
        1. Inspect the differences in policy outputs and make sure you can account for them. Your PR should
           include a report on what changed and how that affected the outputs.
        2. Go to the `if __name__ == "__main__"` block of `test/scripts/save_policy_to_safetensors.py` and
           add the policies you want to update the test artifacts for.
        3. Run `python test/scripts/save_policy_to_safetensors.py`. The test artifact should be updated.
        4. Check that this test now passes.
        5. Remember to restore `test/scripts/save_policy_to_safetensors.py` to its original state.
        6. Remember to stage and commit the resulting changes to `tests/data`.
    """
    env_policy_dir = Path("tests/data/save_policy_to_safetensors") / f"{env_name}_{policy_name}"
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
