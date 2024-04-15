import inspect

from omegaconf import DictConfig, OmegaConf

from lerobot.common.utils import get_safe_torch_device


def make_policy(cfg: DictConfig | None = None, pretrained_policy_name_or_path: str | None = None):
    """
    Args:
        cfg: Hydra configuration.
        pretrained_policy_name_or_path: Hugging Face hub ID (repository name), or path to a local folder with
            the policy weights and configuration.

    TODO(alexander-soare): This function is currently in a transitional state where we are using both Hydra
    configurations and policy dataclass configurations. We will remove the Hydra configuration from this file
    once all models are et up to use dataclass configurations.
    """
    if cfg.policy.name == "tdmpc":
        from lerobot.common.policies.tdmpc.policy import TDMPCPolicy

        policy = TDMPCPolicy(
            cfg.policy, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, device=cfg.device
        )
    elif cfg.policy.name == "diffusion":
        from lerobot.common.policies.diffusion.policy import DiffusionPolicy

        policy = DiffusionPolicy(
            cfg=cfg.policy,
            cfg_device=cfg.device,
            cfg_noise_scheduler=cfg.noise_scheduler,
            cfg_rgb_model=cfg.rgb_model,
            cfg_obs_encoder=cfg.obs_encoder,
            cfg_optimizer=cfg.optimizer,
            cfg_ema=cfg.ema,
            # n_obs_steps=cfg.n_obs_steps,
            # n_action_steps=cfg.n_action_steps,
            **cfg.policy,
        )
    elif cfg.policy.name == "act":
        from lerobot.common.policies.act.configuration_act import ActionChunkingTransformerConfig
        from lerobot.common.policies.act.modeling_act import ActionChunkingTransformerPolicy

        if pretrained_policy_name_or_path is None:
            expected_kwargs = set(inspect.signature(ActionChunkingTransformerConfig).parameters)
            assert set(cfg.policy).issuperset(
                expected_kwargs
            ), f"Hydra config is missing arguments: {set(cfg.policy).difference(expected_kwargs)}"
            policy_cfg = ActionChunkingTransformerConfig(
                **{
                    k: v
                    for k, v in OmegaConf.to_container(cfg.policy, resolve=True).items()
                    if k in expected_kwargs
                }
            )
            policy = ActionChunkingTransformerPolicy(policy_cfg)
        else:
            policy = ActionChunkingTransformerPolicy.from_pretrained(pretrained_policy_name_or_path)
        policy.to(get_safe_torch_device(cfg.device))
    else:
        raise ValueError(cfg.policy.name)

    if cfg.policy.pretrained_model_path and pretrained_policy_name_or_path is None:
        # TODO(rcadene): hack for old pretrained models from fowm
        if cfg.policy.name == "tdmpc" and "fowm" in cfg.policy.pretrained_model_path:
            if "offline" in cfg.policy.pretrained_model_path:
                policy.step[0] = 25000
            elif "final" in cfg.policy.pretrained_model_path:
                policy.step[0] = 100000
            else:
                raise NotImplementedError()
        policy.load(cfg.policy.pretrained_model_path)

    return policy
