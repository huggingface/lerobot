from lerobot.common.policies.tdmpc import TDMPC


def make_policy(cfg):
    if cfg.policy == "tdmpc":
        policy = TDMPC(cfg)
    else:
        raise ValueError(cfg.policy)

    if cfg.pretrained_model_path:
        # TODO(rcadene): hack for old pretrained models from fowm
        if cfg.policy == "tdmpc" and "fowm" in cfg.pretrained_model_path:
            if "offline" in cfg.pretrained_model_path:
                policy.step[0] = 25000
            elif "final" in cfg.pretrained_model_path:
                policy.step[0] = 100000
            else:
                raise NotImplementedError()
        policy.load(cfg.pretrained_model_path)

    return policy
