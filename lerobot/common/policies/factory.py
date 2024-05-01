import inspect

from omegaconf import DictConfig, OmegaConf

from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.utils.utils import get_safe_torch_device


def _policy_cfg_from_hydra_cfg(policy_cfg_class, hydra_cfg):
    expected_kwargs = set(inspect.signature(policy_cfg_class).parameters)
    assert set(hydra_cfg.policy).issuperset(
        expected_kwargs
    ), f"Hydra config is missing arguments: {set(expected_kwargs).difference(hydra_cfg.policy)}"
    policy_cfg = policy_cfg_class(
        **{
            k: v
            for k, v in OmegaConf.to_container(hydra_cfg.policy, resolve=True).items()
            if k in expected_kwargs
        }
    )
    return policy_cfg


def get_policy_and_config_classes(name: str) -> tuple[Policy, object]:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        raise NotImplementedError("Coming soon!")
    elif name == "diffusion":
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy, DiffusionConfig
    elif name == "act":
        from lerobot.common.policies.act.configuration_act import ACTConfig
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        return ACTPolicy, ACTConfig
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy(
    hydra_cfg: DictConfig | None = None, pretrained_policy_name_or_path: str | None = None, dataset_stats=None
) -> Policy:
    """Make an instance of a policy class.

    Args:
        hydra_cfg: A parsed Hydra configuration (see scripts). Note that this argument is mutually exclusive
            with `pretrained_policy_name_or_path`.
        pretrained_policy_name_or_path: Either the repo ID of a model hosted on the Hub or a path to a
            directory containing weights saved using `Policy.save_pretrained`. Note that this argument is
            mutually exclusive with `hydra_cfg`.
        dataset_stats: Dataset statistics to use for (un)normalization of inputs/outputs in the policy. Must
            be provided when initializing a new policy (therefore it must be provided with `hydra_cfg`).
    """
    if not ((hydra_cfg is None) ^ (pretrained_policy_name_or_path is None)):
        raise ValueError(
            "Either `hydra_cfg` or `pretrained_policy_name_or_path` should be provided. Not both."
        )
    if (hydra_cfg is None) ^ (dataset_stats is None):
        raise ValueError("If `hydra_cfg` is/isn't provided, `dataset_stats` should/shouldn't be provided.")
    policy_cls, policy_cfg_class = get_policy_and_config_classes(hydra_cfg.policy.name)

    if hydra_cfg is not None:
        policy_cfg = _policy_cfg_from_hydra_cfg(policy_cfg_class, hydra_cfg)
        policy = policy_cls(policy_cfg, dataset_stats)
        policy.to(get_safe_torch_device(hydra_cfg.device))
    elif pretrained_policy_name_or_path is not None:
        policy = policy_cls.from_pretrained(pretrained_policy_name_or_path)

    return policy
