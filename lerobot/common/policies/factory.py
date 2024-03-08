def make_policy(cfg):
    if cfg.policy.name == "tdmpc":
        from lerobot.common.policies.tdmpc.policy import TDMPC

        policy = TDMPC(cfg.policy, cfg.device)
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
            n_action_steps=cfg.n_action_steps + cfg.n_latency_steps,
            **cfg.policy,
        )
    elif cfg.policy.name == "act":
        from lerobot.common.policies.act.policy import ActionChunkingTransformerPolicy

        policy = ActionChunkingTransformerPolicy(
            cfg.policy, cfg.device, n_action_steps=cfg.n_action_steps + cfg.n_latency_steps
        )
    else:
        raise ValueError(cfg.policy.name)

    if cfg.policy.pretrained_model_path:
        # TODO(rcadene): hack for old pretrained models from fowm
        if cfg.policy.name == "tdmpc" and "fowm" in cfg.policy.pretrained_model_path:
            if "offline" in cfg.pretrained_model_path:
                policy.step[0] = 25000
            elif "final" in cfg.pretrained_model_path:
                policy.step[0] = 100000
            else:
                raise NotImplementedError()
        policy.load(cfg.policy.pretrained_model_path)

    return policy
