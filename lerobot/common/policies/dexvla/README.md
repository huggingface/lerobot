<h1 align="center">
DexVLA: Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning</h1>

### This is the lerobot version of DexVLA. For more information, you can refer to [this](https://github.com/juruobenruo/DexVLA).

## Data Input
DexVLA takes into RGB images, language instructions and states. For our setting, we use three camera views: a top camera, two wrist cameras. 

⭐A major difference between DexVLA with other VLAs is: DexVLA takes raw language in, and outputs sub-step reasoning based on current observations and robot states.
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

Besides, your data should include another key ``action_is_pad`` which is a bool mask indicated whether this action chunk is padded.
For example, suppose action chunk is 5, and the length of episode is 10. So the action chunk for last 4 actions must be padded to make sure the length of action chunk is 5.
And the mask looks like:
~~~python
The 6th chunk: [false, false, false, false, true] 
The 7th chunk: [false, false, false, true,  true]
The 8th chunk: [false, false, true,  true,  true]
The 9th chunk: [false, true,  true,  true,  true]
~~~

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

## 🦾Train
We have already provided pretrained weights of ScaleDP which is stage 1. Belows are mainly about training process of Stage2 and Stage3.

### Training Stage 2
~~~shell
python lerobot/scripts/train.py \
--policy.type dexvla \
--policy.qwen2_vl_path /path/to/official/Qwen2-VL-2B-Instruct \
--policy.pretrain_scaledp_path /path/to/pretrained/scale_dp_h/open_scale_dp_l_backbone.ckpt \
--policy.policy_head_size 'ScaleDP_H' \
--policy.training_stage 2 \
--dataset.repo_i folding_blue_tshirt \
--dataset.local_files_only true \
--batch_size 2 \
--policy.using_film true \
--output_dir /path/to/output \
--steps 10000 \
--save_freq 1000 \
--optimizer_lr 2e-5 \
--policy.device=cuda
~~~

### Training Stage 3
Stage3 can be viewed as continual training on specific dexterous tasks like laundry folding which is same as PI0. So stage3 is trained based on stage2.
~~~shell
python lerobot/scripts/train.py \
--policy.type dexvla \
--policy.qwen2_vl_path /path/to/official/Qwen2-VL-2B-Instruct \
--.pretrained_path /path/to/pretrained/stage2/weights \
--policy.policy_head_size 'ScaleDP_H' \
--policy.training_stage 3 \
--dataset.repo_i folding_blue_tshirt \
--dataset.local_files_only true \
--batch_size 2 \
--policy.using_film true \
--output_dir /path/to/output \
--steps 10000 \
--save_freq 1000 \
--optimizer_lr 2e-5 \
--policy.device=cuda
~~~

## Evaluation
~~~shell
python lerobot/scripts/eval.py \
--policy.type dexvla \
--policy.pretrained_path /path/to/pretrained/stage2/or/stage3/weights \
--env.type aloha \
--env.episode_length 5 \
--policy.qwen2_vl_path /path/to/official/Qwen2-VL-2B-Instruct \
--env.task AlohaInsertion-v0 \
--eval.n_episodes 1 \
--eval.batch_size 1 \
--device cuda 
~~~


