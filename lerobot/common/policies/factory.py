import inspect

from omegaconf import OmegaConf

from lerobot.common.utils import get_safe_torch_device


def make_policy(cfg):
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
        from lerobot.common.policies.act.configuration_act import ActConfig
        from lerobot.common.policies.act.modeling_act import ActPolicy

        expected_kwargs = set(inspect.signature(ActConfig).parameters)
        assert set(cfg.policy).issuperset(
            expected_kwargs
        ), f"Hydra config is missing arguments: {set(cfg.policy).difference(expected_kwargs)}"
        policy_cfg = ActConfig(
            **{
                k: v
                for k, v in OmegaConf.to_container(cfg.policy, resolve=True).items()
                if k in expected_kwargs
            }
        )
        policy = ActPolicy(policy_cfg)
        policy.to(get_safe_torch_device(cfg.device))
    else:
        raise ValueError(cfg.policy.name)

    if cfg.policy.pretrained_model_path:
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
