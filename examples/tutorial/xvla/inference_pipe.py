from lerobot.policies.factory import make_policy, make_pre_post_processors
# from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env_config
from lerobot.policies.xvla.utils import Rotate6D_to_AxisAngle
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
import torch
import numpy as np

observation_height: int = 360
observation_width: int = 360
# create an observation dict
OBS = {
    f"{OBS_IMAGES}.image1": torch.randn(1, 3, observation_height, observation_width),
    f"{OBS_IMAGES}.image2": torch.randn(1, 3, observation_height, observation_width),
    OBS_STATE: torch.randn(1, 9),                  # ONLY if OBS_STATE is already a string
    "task": "put the object in the box",
}
def fake_rgb(H, W):
    img = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8).numpy()
    return img

OBS[f"{OBS_IMAGES}.image1"] = fake_rgb(observation_height, observation_width)
OBS[f"{OBS_IMAGES}.image2"] = fake_rgb(observation_height, observation_width)

# observation = preprocessor(OBS)
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("2toINF/X-VLA-WidowX", num_views=2, trust_remote_code=True)
inputs = processor([OBS[f"{OBS_IMAGES}.image1"], OBS[f"{OBS_IMAGES}.image2"]], OBS["task"])
breakpoint()

cfg = PreTrainedConfig.from_pretrained("/raid/jade/models/xvla-libero-og_migrated")
cfg.pretrained_path = "/raid/jade/models/xvla-libero-og_migrated"
env_cfg = make_env_config("libero", task="libero_spatial")
policy = make_policy(
    cfg=cfg,
    env_cfg=env_cfg,
)

policy.eval()

preprocessor_overrides = {
        "device_processor": {"device": str(cfg.device)},
}

preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
)

observation = preprocessor(OBS)
action = policy.select_action(observation)

target_eef = action[:, :3].to("cpu").numpy()
target_axis = Rotate6D_to_AxisAngle(action[:, 3:9].to("cpu").numpy())
target_act = action[:, 9:10].to("cpu").numpy()
final_action = np.concatenate([target_eef, target_axis, target_act], axis=-1)
