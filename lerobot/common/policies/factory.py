def make_policy(cfg):
    if cfg.policy.name == "tdmpc":
        from lerobot.common.policies.tdmpc import TDMPC

        policy = TDMPC(cfg.policy)
    elif cfg.policy.name == "diffusion":
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.model.vision.model_getter import get_resnet
        from diffusion_policy.model.vision.multi_image_obs_encoder import (
            MultiImageObsEncoder,
        )

        from lerobot.common.policies.diffusion import DiffusionPolicy

        noise_scheduler = DDPMScheduler(**cfg.noise_scheduler)

        rgb_model = get_resnet(**cfg.rgb_model)

        obs_encoder = MultiImageObsEncoder(
            rgb_model=rgb_model,
            **cfg.obs_encoder,
        )

        policy = DiffusionPolicy(
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            n_action_steps=cfg.n_action_steps + cfg.n_latency_steps,
            **cfg.policy,
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
