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
    pretrained_name_or_path="/fsx/jade_choghari/models/pi05-base",
)
cfg.dtype = "bfloat16"

pre_processor, post_processor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path="/fsx/jade_choghari/models/pi05-base",
)

delta_timestamps = {'action': [0.0, 0.03333333333333333, 0.06666666666666667, 0.1, 0.13333333333333333, 0.16666666666666666, 0.2, 0.23333333333333334, 0.26666666666666666, 0.3, 0.3333333333333333, 0.36666666666666664, 0.4, 0.43333333333333335, 0.4666666666666667, 0.5, 0.5333333333333333, 0.5666666666666667, 0.6, 0.6333333333333333, 0.6666666666666666, 0.7, 0.7333333333333333, 0.7666666666666667, 0.8, 0.8333333333333334, 0.8666666666666667, 0.9, 0.9333333333333333, 0.9666666666666667, 1.0, 1.0333333333333334, 1.0666666666666667, 1.1, 1.1333333333333333, 1.1666666666666667, 1.2, 1.2333333333333334, 1.2666666666666666, 1.3, 1.3333333333333333, 1.3666666666666667, 1.4, 1.4333333333333333, 1.4666666666666666, 1.5, 1.5333333333333334, 1.5666666666666667, 1.6, 1.6333333333333333]}

dataset = LeRobotDataset(repo_id="local", root="/fsx/jade_choghari/outputs/pgen_annotations1", delta_timestamps=delta_timestamps)

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
breakpoint()
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