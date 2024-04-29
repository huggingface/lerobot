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


def make_policy(hydra_cfg: DictConfig, dataset_stats=None):
    if hydra_cfg.policy.name == "tdmpc":
        from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig
        from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        cfg_cls = TDMPCConfig
        policy_cls = TDMPCPolicy
    elif hydra_cfg.policy.name == "diffusion":
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        cfg_cls = DiffusionConfig
        policy_cls = DiffusionPolicy
    elif hydra_cfg.policy.name == "act":
        from lerobot.common.policies.act.configuration_act import ActionChunkingTransformerConfig
        from lerobot.common.policies.act.modeling_act import ActionChunkingTransformerPolicy

        cfg_cls = ActionChunkingTransformerConfig
        policy_cls = ActionChunkingTransformerPolicy
    else:
        raise ValueError(hydra_cfg.policy.name)

    policy_cfg = _policy_cfg_from_hydra_cfg(cfg_cls, hydra_cfg)
    policy: Policy = policy_cls(policy_cfg, dataset_stats)
    policy.to(get_safe_torch_device(hydra_cfg.device))

    if hydra_cfg.policy.pretrained_model_path:
        policy.load(hydra_cfg.policy.pretrained_model_path)

    return policy
