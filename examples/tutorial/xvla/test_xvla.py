import os

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_policy_config

cfg = make_policy_config("xvla")

dataset_id = "lerobot/svla_so101_pickplace"
# This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
dataset_metadata = LeRobotDatasetMetadata(dataset_id)
policy = make_policy(cfg=cfg, ds_meta=dataset_metadata)

for name, param in policy.state_dict().items():
    print(name, param.shape)


# now let's load in safetensors
import safetensors.torch
from huggingface_hub import snapshot_download

cache_dir = snapshot_download(
    repo_id="2toINF/X-VLA-Libero", repo_type="model", cache_dir="/fsx/jade_choghari/.cache/huggingface/model"
)
state_dict = safetensors.torch.load_file(os.path.join(cache_dir, "model.safetensors"))
# policy.load_state_dict(state_dict)
# 3. Add "model." prefix to every key
new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
keys_to_skip = [
    "model.transformer.action_encoder.fc.weight",
    "model.transformer.action_encoder.fc.bias",
]

new_state_dict = {k: v for k, v in new_state_dict.items() if k not in keys_to_skip}
# 4. Load into your model
missing, unexpected = policy.load_state_dict(new_state_dict, strict=False)

print("missing keys:", missing)

print()
print("unexpected keys:", unexpected)


import random

import numpy as np
import torch
from xvla.models.modeling_xvla import XVLA

# from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env_config
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
observation_height: int = 224
observation_width: int = 224  # todo: jadechoghari, image size is different for the two models
# create an observation dict
OBS = {
    f"{OBS_IMAGES}.image": torch.randn(1, 3, observation_height, observation_width),
    f"{OBS_IMAGES}.image2": torch.randn(1, 3, observation_height, observation_width),
    OBS_STATE: torch.randn(1, 20),  # ONLY if OBS_STATE is already a string
    "task": "put the object in the box",
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def fake_rgb(H, W):
    arr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
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
domain_id = torch.tensor([3], dtype=torch.long)
inputs.update(
    {
        "proprio": OBS[OBS_STATE].to("cuda"),
        "domain_id": domain_id.to("cuda"),
    }
)


for k in inputs.keys() & inputs_1.keys():  # intersection of keys
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

# np all close
print(np.allclose(action, action_1, atol=1e-2, rtol=1e-2))
print("max diff:", np.max(np.abs(action - action_1)))
print("mean diff:", np.mean(np.abs(action - action_1)))


import random

import numpy as np
import torch
from PIL import Image
from xvla.models.configuration_xvla import XVLAConfig
from xvla.models.modeling_xvla import XVLA
from xvla.models.processor_xvla import XVLAProcessor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env_config
from lerobot.policies.factory import make_policy

cfg = XVLAConfig.from_pretrained("/raid/jade/models/xvla-libero")
model = XVLA.from_pretrained("/raid/jade/models/xvla-libero")
model.eval()
model.to("cuda")
processor = XVLAProcessor.from_pretrained("/raid/jade/models/xvla-libero")
# /raid/jade/models/xvla-libero
# seet seed
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def make_random_pil_images(num_images=3, H=480, W=640):
    images = []
    for _ in range(num_images):
        # Random RGB image
        arr = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        images.append(img)
    return images


# Example:
images = make_random_pil_images()
language_instruction = "This is a random image"
# Multimodal preprocessing by processor
inputs = processor(images, language_instruction)
if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
    raise ValueError("Processor did not return the expected keys.")

proprio = torch.randn(1, 20)
domain_id = torch.tensor([0], dtype=torch.long)

# Align to model's device/dtype
device = model.device
dtype = next(model.parameters()).dtype


def to_model(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    # cast floats to model dtype, keep integral/bool as-is
    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)


inputs = {k: to_model(v) for k, v in inputs.items()}
inputs.update(
    {
        "proprio": to_model(proprio),
        "domain_id": domain_id.to(device),
    }
)

# Inference
action = model.generate_actions(**inputs, steps=10).squeeze(0).float().cpu().numpy()


#### now for lerobot model #####################################################

cfg = PreTrainedConfig.from_pretrained("/raid/jade/models/xvla-libero-og_migrated")
env_cfg = make_env_config("libero", task="libero_spatial")
cfg.pretrained_path = "/raid/jade/models/xvla-libero-og_migrated"
policy = make_policy(cfg=cfg, env_cfg=env_cfg)
policy.eval()
policy.to("cuda")

action_1 = policy.model.generate_actions(**inputs, steps=10).squeeze(0).float().cpu().numpy()
