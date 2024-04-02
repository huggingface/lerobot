""" Factory for policies
"""

from lerobot.common.policies.abstract import AbstractPolicy


def make_policy(cfg: dict) -> AbstractPolicy:
    """ Instantiate a policy from the configuration.
        Currently supports TD-MPC, Diffusion, and ACT: select the policy with cfg.policy.name: tdmpc, diffusion, act.
    
    Args:
        cfg: The configuration (DictConfig)

    """
    policy_kwargs = {}
    if cfg.policy.name != "diffusion" and cfg.rollout_batch_size > 1:
        raise NotImplementedError("Only diffusion policy supports rollout_batch_size > 1 for the time being.")

    if cfg.policy.name == "tdmpc":
        from lerobot.common.policies.tdmpc.policy import TDMPCPolicy

        policy_cls = TDMPCPolicy
        policy_kwargs = {"cfg": cfg.policy, "device": cfg.device}
    elif cfg.policy.name == "diffusion":
        from lerobot.common.policies.diffusion.policy import DiffusionPolicy

        policy_cls = DiffusionPolicy
        policy_kwargs = {
            "cfg": cfg.policy,
            "cfg_device": cfg.device,
            "cfg_noise_scheduler": cfg.noise_scheduler,
            "cfg_rgb_model": cfg.rgb_model,
            "cfg_obs_encoder": cfg.obs_encoder,
            "cfg_optimizer": cfg.optimizer,
            "cfg_ema": cfg.ema,
            "n_action_steps": cfg.n_action_steps + cfg.n_latency_steps,
            **cfg.policy,
        }
    elif cfg.policy.name == "act":
        from lerobot.common.policies.act.policy import ActionChunkingTransformerPolicy

        policy_cls = ActionChunkingTransformerPolicy
        policy_kwargs = {"cfg": cfg.policy, "device": cfg.device, "n_action_steps": cfg.n_action_steps + cfg.n_latency_steps}
    else:
        raise ValueError(cfg.policy.name)

    if cfg.policy.pretrained_model_path:
        # policy.load(cfg.policy.pretrained_model_path, device=cfg.device)
        policy = policy_cls.from_pretrained(cfg.policy.pretrained_model_path, map_location=cfg.device, **policy_kwargs)
    
        # TODO(rcadene): hack for old pretrained models from fowm
        if cfg.policy.name == "tdmpc" and "fowm" in cfg.policy.pretrained_model_path:
            if "offline" in cfg.pretrained_model_path:
                policy.step[0] = 25000
            elif "final" in cfg.pretrained_model_path:
                policy.step[0] = 100000
            else:
                raise NotImplementedError()

    return policy
