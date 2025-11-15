from lerobot.policies.factory import make_policy, make_pre_post_processors
# from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env_config
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from xvla.models.modeling_xvla import XVLA
import torch
import numpy as np
import random
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
observation_height: int = 224
observation_width: int = 224 # todo: jadechoghari, image size is different for the two models
# create an observation dict
OBS = {
    f"{OBS_IMAGES}.image": torch.randn(1, 3, observation_height, observation_width),
    f"{OBS_IMAGES}.image2": torch.randn(1, 3, observation_height, observation_width),
    OBS_STATE: torch.randn(1, 20),                  # ONLY if OBS_STATE is already a string
    "task": "put the object in the box",
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
def fake_rgb(H, W):
    arr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1)      # CHW
    t = t.unsqueeze(0).float()
    # normalize pixel to imagenet
    return t


OBS[f"{OBS_IMAGES}.image"] = fake_rgb(observation_height, observation_width)
OBS[f"{OBS_IMAGES}.image2"] = fake_rgb(observation_height, observation_width)

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
inputs = policy._build_model_inputs(observation)


#### now the og model ###########################################################
from xvla.models.processing_xvla import XVLAProcessor

processor = XVLAProcessor.from_pretrained("/raid/jade/models/xvla-libero", num_views=2)
inputs_1 = processor([OBS[f"{OBS_IMAGES}.image"], OBS[f"{OBS_IMAGES}.image2"]], OBS["task"])
domain_id = torch.tensor([int(3)], dtype=torch.long)
inputs.update({
    "proprio": OBS[OBS_STATE].to("cuda"),
    "domain_id": domain_id.to("cuda"),
})

for k in inputs.keys() & inputs_1.keys():   # intersection of keys
    a = inputs[k]
    b = inputs_1[k].to("cuda")

    print(f"\n🔎 Key: {k}")

    # Check shape
    print("  shape:", a.shape, b.shape)

    # Check if close
    if torch.allclose(a, b, atol=1e-5, rtol=1e-5):
        print("  ✔️ tensors are equal (allclose)")
    else:
        diff = torch.abs(a - b)
        print("  ❌ tensors differ")
        print("  max diff:", diff.max().item())
        print("  mean diff:", diff.mean().item())


model = XVLA.from_pretrained("/raid/jade/models/xvla-libero")
model.eval()
model.to("cuda")

action = model.generate_actions(**inputs, steps=10).squeeze(0).float().cpu().numpy()
# (Pdb) inputs['input_ids'].shape
# torch.Size([1, 64])
# (Pdb) inputs_1['input_ids'].shape
# torch.Size([1, 50])
# (Pdb) [0, 0, :, :4, 0]
action_1 = policy.model.generate_actions(**inputs, steps=10).squeeze(0).float().cpu().numpy()

#np all close
print(np.allclose(action, action_1, atol=1e-4, rtol=1e-4))
print("max diff:", np.max(np.abs(action - action_1)))
print("mean diff:", np.mean(np.abs(action - action_1)))
breakpoint()