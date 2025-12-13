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
policy.train()
# run inference
# action = policy.select_action(batch)
loss, loss_dict = policy.forward(batch)
# import requests
# from PIL import Image
# from transformers import AutoProcessor
# model = policy.model.paligemma_with_expert.paligemma
# model = model.to(device="cuda", dtype=torch.bfloat16)
# model.eval()
# prompt = "Describe this image."
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
# image = Image.open(requests.get(url, stream=True).raw)
# processor = AutoProcessor.from_pretrained(
#     "google/paligemma-3b-pt-224",
# )
# inputs = processor(image, prompt, return_tensors="pt").to(model.device)
# print("generating...")
# output = model.generate(
#     **inputs,
#     max_new_tokens=50,
#     use_cache=True,  # default dynamic cache
# )
# print(processor.decode(output[0], skip_special_tokens=True))


# # other model
# from transformers import PaliGemmaForConditionalGeneration
# model = PaliGemmaForConditionalGeneration.from_pretrained(
#     "google/paligemma2-3b-pt-224",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# model.eval()
# print("generating...")
# output = model.generate(
#     **inputs,
#     max_new_tokens=100,
#     use_cache=True,  # default dynamic cache
# )
# print("Model 2 output:")
# print(processor.decode(output[0], skip_special_tokens=True))
