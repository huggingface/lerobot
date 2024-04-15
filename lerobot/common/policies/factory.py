import inspect

from omegaconf import DictConfig, OmegaConf

from lerobot.common.utils import get_safe_torch_device


def _policy_cfg_from_hydra_cfg(policy_cfg_class, hydra_cfg):
    expected_kwargs = set(inspect.signature(policy_cfg_class).parameters)
    assert set(hydra_cfg.policy).issuperset(
        expected_kwargs
    ), f"Hydra config is missing arguments: {set(hydra_cfg.policy).difference(expected_kwargs)}"
    policy_cfg = policy_cfg_class(
        **{
            k: v
            for k, v in OmegaConf.to_container(hydra_cfg.policy, resolve=True).items()
            if k in expected_kwargs
        }
    )
    return policy_cfg


def make_policy(hydra_cfg: DictConfig):
    if hydra_cfg.policy.name == "tdmpc":
        from lerobot.common.policies.tdmpc.policy import TDMPCPolicy

        policy = TDMPCPolicy(
            hydra_cfg.policy,
            n_obs_steps=hydra_cfg.n_obs_steps,
            n_action_steps=hydra_cfg.n_action_steps,
            device=hydra_cfg.device,
        )
    elif hydra_cfg.policy.name == "diffusion":
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        policy_cfg = _policy_cfg_from_hydra_cfg(DiffusionConfig, hydra_cfg)
        policy = DiffusionPolicy(policy_cfg)
        policy.to(get_safe_torch_device(hydra_cfg.device))
    elif hydra_cfg.policy.name == "act":
        from lerobot.common.policies.act.configuration_act import ActionChunkingTransformerConfig
        from lerobot.common.policies.act.modeling_act import ActionChunkingTransformerPolicy

        policy_cfg = _policy_cfg_from_hydra_cfg(ActionChunkingTransformerConfig, hydra_cfg)
        policy = ActionChunkingTransformerPolicy(policy_cfg)
        policy.to(get_safe_torch_device(hydra_cfg.device))
    else:
        raise ValueError(hydra_cfg.policy.name)

    if hydra_cfg.policy.pretrained_model_path:
        # TODO(rcadene): hack for old pretrained models from fowm
        if hydra_cfg.policy.name == "tdmpc" and "fowm" in hydra_cfg.policy.pretrained_model_path:
            if "offline" in hydra_cfg.policy.pretrained_model_path:
                policy.step[0] = 25000
            elif "final" in hydra_cfg.policy.pretrained_model_path:
                policy.step[0] = 100000
            else:
                raise NotImplementedError()
        policy.load(hydra_cfg.policy.pretrained_model_path)

    return policy
