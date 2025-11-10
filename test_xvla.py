from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_policy_config
import os
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

cache_dir = snapshot_download(repo_id="2toINF/X-VLA-Libero", repo_type="model", cache_dir="/fsx/jade_choghari/.cache/huggingface/model")
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

