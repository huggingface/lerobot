
#  Heterogenous Pre-trained Transformers
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=flat-square)](https://huggingface.co/liruiw/hpt-base)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Paper](https://badgen.net/badge/icon/arXiv?icon=awesome&label&color=red&style=flat-square)]()
[![Website](https://img.shields.io/badge/Website-hpt-blue?style=flat-square)](https://liruiw.github.io/hpt)
[![Python](https://img.shields.io/badge/Python-%3E=3.8-blue?style=flat-square)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E=2.0-orange?style=flat-square)]()

[Lirui Wang](https://liruiw.github.io/), [Xinlei Chen](https://xinleic.xyz/), [Jialiang Zhao](https://alanz.info/), [Kaiming He](https://people.csail.mit.edu/kaiming/)

Neural Information Processing Systems (Spotlight), 2024
<hr style="border: 2px solid gray;"></hr>


This is a Huggingface LeRobot implementation for pre-training Heterogenous Pre-trained Transformers (HPTs).

## LeRobot Installation

Download our source code:
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

Install 🤗 LeRobot:
```bash
pip install -e .
```

> **NOTE:** Depending on your platform, If you encounter any build errors during this step
you may need to install `cmake` and `build-essential` for building some of our dependencies.
On linux: `sudo apt-get install cmake build-essential`

For simulations, 🤗 LeRobot comes with gymnasium environments that can be installed as extras:
- [aloha](https://github.com/huggingface/gym-aloha)
- [xarm](https://github.com/huggingface/gym-xarm)
- [pusht](https://github.com/huggingface/gym-pusht)

For instance, to install 🤗 LeRobot with aloha and pusht, use:
```bash
pip install -e ".[aloha, pusht]"
```

To use [Weights and Biases](https://docs.wandb.ai/quickstart) for experiment tracking, log in with
```bash
wandb login
```

(note: you will also need to enable WandB in the configuration. See below.)

## Code Modification Walkthrough

Check the following two folders for most of the modifications.
```
├── lerobot
|   ├── configs          # contains hydra yaml files with all options that you can override in the command line
|   |   ├── ...            # various sim environments and their datasets: aloha.yaml, pusht.yaml, xarm.yaml
|   |   └── policy         # including policies config for hpt.yaml
|   ├── common           # contains classes and utilities
|   |   ├── ...       # various datasets of human demonstrations: aloha, pusht, xarm
|   |   ├── ...            # various sim environments: aloha, pusht, xarm
|   |   ├── policies       # including modeling and configuration for hpt
|   ├── ...
```

## Experiment Scripts
0. By default, the HPT model loads the x-large pre-trained trunk. Use these config parameters ``policy.embed_dim=256 policy.num_heads=8 policy.num_blocks=16`` to switch to the hpt-base trunk for example.

1. Run the following scripts for aloha transfer cube experiments.

<details>
  <summary><span style="font-weight: bold;">Metaworld 20 Task Experiments</span></summary>

```
python lerobot/scripts/train.py \
policy=hpt_transformer env=aloha  env.task=AlohaTransferCube-v0 \
dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
wandb.enable=true
```

</details>

2. Run the following scripts for push-T experiments.

<details>
  <summary><span style="font-weight: bold;">Metaworld 20 Task Experiments</span></summary>

```
python lerobot/scripts/train.py \
policy=hpt_pusht  env=pusht  env.task=PushT-v0 \
dataset_repo_id=lerobot/pusht \
wandb.enable=true
```

</details>

3. Run the following scripts for real-world Koch experiments.

<details>
  <summary><span style="font-weight: bold;">Metaworld 20 Task Experiments</span></summary>

```
python lerobot/scripts/train.py policy=hpt_koch_real env=koch_real \
dataset_repo_id=lerobot/koch_pick_place_5_lego  \
wandb.enable=true
```

</details>


### Citation
If you find HPT useful in your research, please consider citing:
```
@inproceedings{wang2024hpt,
author    = {Lirui Wang, Xinlei Chen, Jialiang Zhao, Kaiming He, Russ Tedrake},
title     = {Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers},
booktitle = {Neurips},
year      = {2024}
}
```

## Acknowledgement

Our implementation is built upon the excellent [LeRobot](https://github.com/huggingface/lerobot) codebase.
