import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
# import make_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.configs.policies import PreTrainedConfig

cfg = PreTrainedConfig.from_pretrained(
    pretrained_name_or_path="/fsx/jade_choghari/outputs/pi0_training_new/checkpoints/last/pretrained_model",
)
cfg.dtype = "bfloat16"

pre_processor, post_processor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path="/fsx/jade_choghari/outputs/pi0_training_new/checkpoints/last/pretrained_model",
)


dataset = LeRobotDataset(repo_id="local", root="/fsx/jade_choghari/outputs/pgen_annotations1")
# rename map --rename_map='{
#         "observation.images.side": "observation.images.base_0_rgb",
#         "observation.images.up": "observation.images.left_wrist_0_rgb"
#         }'
rename_map = {
    "observation.images.side": "observation.images.base_0_rgb",
    "observation.images.up": "observation.images.left_wrist_0_rgb"
}
policy = make_policy(
    cfg=cfg,
    ds_meta=dataset.meta,
    rename_map=rename_map,
)

dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=4,
        shuffle=True,
)

batch = next(iter(dataloader))

batch = pre_processor(batch)

# Test training forward pass
policy.train()
loss, loss_dict = policy.forward(batch)
print(f"Training loss: {loss_dict}")

# Test inference
policy.eval()
with torch.no_grad():
    actions = policy.predict_action_chunk(batch)
    print(f"Predicted actions shape: {actions.shape}")