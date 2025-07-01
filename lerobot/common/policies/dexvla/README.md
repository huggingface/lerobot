<h1 align="center">
DexVLA: Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning</h1>

This policy is Community Contributed. For more information about DexVLA, you can also refer to [this](https://github.com/juruobenruo/DexVLA).
This is [project website](https://dex-vla.github.io/).

## Dataset
### Data format
DexVLA takes RGB images, language instructions and states. For our setting, we use three camera views, namely a top camera and two wrist cameras.

⭐A major difference between DexVLA and other VLAs is: DexVLA takes in raw language, and outputs sub-step reasoning based on current observations.
So you have to <font color='red'>add sub-step reasoning in your data for training</font>.

Specifically, your data should include a key ``reasoning`` which is a list of sub-step reasoning corresponding to each observation.
For example, if the episode is 10 steps. The length of this list should be 10 as well. And it may looks like:
~~~python
reasoning = [
    "This is step 1.",
    "This is step 1.",
    "This is step 2.",
    "This is step 2.",
    ...
    "This is step 4.",
]
~~~

Besides, your data should include another key ``action_is_pad`` which is a bool mask indicating whether this action chunk is padded.
Suppose the size of the action chunk is 5, and the length of the episode is 10. So the action chunk for the last 4 actions must be padded to make sure the length of action chunk is 5.
And the mask looks like:
~~~python
The 6th chunk: [false, false, false, false, true]
The 7th chunk: [false, false, false, true,  true]
The 8th chunk: [false, false, true,  true,  true]
The 9th chunk: [false, true,  true,  true,  true]
~~~

### Training Data for DexVLA
The pretraining dataset comprises approximately 100 hours of collected data by ourselves. The dataset mainly including four embodiments which are: moblie Agilex Aloha, single Franka Emika and single UR5e.
We haven't use any public dataset such as Open-X or DROID.

## 🤗Download Pretrained Weights
### Download official Qwen2_VL weights
We construct the VLM backbone by integrating Qwen2-VL-2B, a powerful and efficient model, into our framework.
The Qwen2-VL 2B serves as the core of our architecture, providing robust capabilities
for vision-language tasks. We use off-the-shelf Qwen2-VL model proposed
in [Qwen2-VL](https://arxiv.org/pdf/2409.12191) without any post training on VLM itself. You can download the official weights from this link:

| Model               | Link                                                           |
|---------------------|----------------------------------------------------------------|
| Qwen2-VL (~2B)      | [huggingface](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |

**❗❗** After downloading the standard weights, you have to replace the official "config.json"
with our ["config.json"](https://github.com/juruobenruo/DexVLA/blob/main/docs/config.json) designed for VLA.
### Download our pretrained ScaleDP-H weights(Stage 1)
We released our pretrained weights of ScaleDP-H which is trained after Stage1. Now you can download the weights and directly finetuning your data on Stage 2.

| Model             | Link                                                           |
|-------------------|----------------------------------------------------------------|
| ScaleDP-H (~1B)   | [huggingface](https://huggingface.co/lesjie/scale_dp_h)  |
| ScaleDP-L (~400M) | [huggingface](https://huggingface.co/lesjie/scale_dp_l)  |

**❗❗**After downloading the weights, you have to transform it into ``safetensors`` format, you can simply run this code:
~~~python
import torch
from safetensors.torch import save_file
path = "/path/to/open_scale_dp_l_backbone.ckpt"
checkpoint = torch.load(path, map_location=torch.device('cpu'))['nets']['nets']

# Save the weights in safetensors format
safetensors_path = "/path/to/open_scale_dp_l_backbone.safetensors"
save_file(checkpoint, safetensors_path)
print(f"Converted {path} to {safetensors_path}")
pass

~~~

## 🦾Train
We have already provided pretrained weights of ScaleDP which is stage 1. Belows are mainly about training process of Stage2 and Stage3.

### Training Stage 2
~~~shell
python lerobot/scripts/train.py \
--policy.type dexvla \
--policy.qwen2_vl_path /path/to/official/Qwen2-VL-2B-Instruct \
--policy.pretrain_scaledp_path /path/to/pretrained/scale_dp_h/open_scale_dp_l_backbone.safetensors \
--policy.policy_head_size 'scaledp_h' \
--policy.training_stage 2 \
--dataset.repo_i lerobot/aloha_mobile_chair \
--policy.using_film true \
--output_dir /path/to/output \
--steps 10000 \
--save_freq 1000 \
--optimizer_lr 2e-5
~~~

### Training Stage 3
Stage3 can be viewed as continual training on specific dexterous tasks like laundry folding which is same as PI0. So stage3 is trained based on stage2.
~~~shell
python lerobot/scripts/train.py \
--policy.type dexvla \
--policy.qwen2_vl_path /path/to/official/Qwen2-VL-2B-Instruct \
--.pretrained_path /path/to/pretrained/stage2/weights \
--policy.policy_head_size 'scaledp_h' \
--policy.training_stage 3 \
--dataset.repo_i lerobot/aloha_mobile_chair \
--batch_size 2 \
--policy.using_film true \
--output_dir /path/to/output \
--steps 10000 \
--save_freq 1000 \
--optimizer_lr 2e-5
~~~

### Training Time
Original DexVLA is trained on 8 x H100 GPUs. And the training time for each stage is listed as follows:

| Stage  | Batch Size(each gpu) | Steps  | Time(hour) |
|--------|----------------------|--------|------------|
| Stage1 | 32                   | 60000  | 30         |
| Stage2 | 12                   | 100000 | 30         |
| Stage3 | 12                   | 60000  | 18         |


## Evaluation
### Evaluation Script
You can evaluate dexvla by following scripts.
~~~shell
python lerobot/scripts/eval.py \
--policy.type dexvla \
--policy.pretrained_path /path/to/pretrained/stage2/or/stage3/weights \
--env.type aloha \
--env.episode_length 5 \
--policy.qwen2_vl_path /path/to/official/Qwen2-VL-2B-Instruct \
--env.task AlohaInsertion-v0 \
--eval.n_episodes 1 \
--eval.batch_size 1
~~~

### Inference Speed
Tested on a single A6000 GPU, the DexVLA could infer 3.4 action chunks in one second. For each action chunk, if we execute 25 actions, the real control frequency can be 85 (3.4*25)Hz.
