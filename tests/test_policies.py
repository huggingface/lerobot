import pytest
import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import postprocess_action, preprocess_observation
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.utils.utils import init_hydra_config
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE, require_env


# TODO(aliberts): refactor using lerobot/__init__.py variables
@pytest.mark.parametrize(
    "env_name,policy_name,extra_overrides",
    [
        # ("xarm", "tdmpc", ["policy.mpc=true"]),
        # ("pusht", "tdmpc", ["policy.mpc=false"]),
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
        - Checking that the policy follows the correct protocol.
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
    policy = make_policy(cfg, dataset_stats=dataset.stats)
    # Check that the policy follows the required protocol.
    assert isinstance(
        policy, Policy
    ), f"The policy does not follow the required protocol. Please see {Policy.__module__}.{Policy.__name__}."

    # Check that we run select_actions and get the appropriate output.
    env = make_env(cfg, num_parallel_envs=2)

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
    policy.forward(batch, step=0)

    # reset the policy and environment
    policy.reset()
    observation, _ = env.reset(seed=cfg.seed)

    # apply transform to normalize the observations
    observation = preprocess_observation(observation)

    # send observation to device/gpu
    observation = {key: observation[key].to(DEVICE, non_blocking=True) for key in observation}

    # get the next action for the environment
    with torch.inference_mode():
        action = policy.select_action(observation, step=0)

    # convert action to cpu numpy array
    action = postprocess_action(action)

    # Test step through policy
    env.step(action)

    # Test load state_dict
    if policy_name != "tdmpc":
        # TODO(rcadene, alexander-soare): make it work for tdmpc
        new_policy = make_policy(cfg)
        new_policy.load_state_dict(policy.state_dict())


@pytest.mark.parametrize("policy_cls", [DiffusionPolicy, ACTPolicy])
def test_policy_defaults(policy_cls):
    kwargs = {}
    # TODO(alexander-soare): Remove this kwargs hack when we move the scheduler out of DP.
    if policy_cls is DiffusionPolicy:
        kwargs = {"lr_scheduler_num_training_steps": 1}
    policy_cls(**kwargs)


@pytest.mark.parametrize(
    "insert_temporal_dim",
    [
        False,
        True,
    ],
)
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
