<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/lerobot-logo-thumbnail.png">
    <source media="(prefers-color-scheme: light)" srcset="media/lerobot-logo-thumbnail.png">
    <img alt="LeRobot, Hugging Face Robotics Library" src="media/lerobot-logo-thumbnail.png" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly-tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly-tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/huggingface/lerobot/branch/main/graph/badge.svg?token=TODO)](https://codecov.io/gh/huggingface/lerobot)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Examples](https://img.shields.io/badge/Examples-green.svg)](https://github.com/huggingface/lerobot/tree/main/examples)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://dcbadge.vercel.app/api/server/C5P34WJ68S?style=flat)](https://discord.gg/s3KuuzsPFb)

</div>

<h3 align="center">
    <p>State-of-the-art Machine Learning for real-world robotics</p>
</h3>

---


🤗 LeRobot aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier for entry to robotics so that everyone can contribute and benefit from sharing datasets and pretrained models.

🤗 LeRobot contains state-of-the-art approaches that have been shown to transfer to the real-world with a focus on imitation learning and reinforcement learning.

🤗 LeRobot already provides a set of pretrained models, datasets with human collected demonstrations, and simulated environments so that everyone can get started. In the coming weeks, the plan is to add more and more support for real-world robotics on the most affordable and capable robots out there.

🤗 LeRobot hosts pretrained models and datasets on this HuggingFace community page: [huggingface.co/lerobot](https://huggingface.co/lerobot)

#### Examples of pretrained models and environments

<table>
  <tr>
    <td><img src="http://remicadene.com/assets/gif/aloha_act.gif" width="100%" alt="ACT policy on ALOHA env"/></td>
    <td><img src="http://remicadene.com/assets/gif/simxarm_tdmpc.gif" width="100%" alt="TDMPC policy on SimXArm env"/></td>
    <td><img src="http://remicadene.com/assets/gif/pusht_diffusion.gif" width="100%" alt="Diffusion policy on PushT env"/></td>
  </tr>
  <tr>
    <td align="center">ACT policy on ALOHA env</td>
    <td align="center">TDMPC policy on SimXArm env</td>
    <td align="center">Diffusion policy on PushT env</td>
  </tr>
</table>

### Acknowledgment

- ACT policy and ALOHA environment are adapted from [ALOHA](https://tonyzhaozh.github.io/aloha/)
- Diffusion policy and Pusht environment are adapted from [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- TDMPC policy and Simxarm environment are adapted from [FOWM](https://www.yunhaifeng.com/FOWM/)
- Abstractions and utilities for Reinforcement Learning come from [TorchRL](https://github.com/pytorch/rl)

## Installation

Download our source code:
```bash
git clone https://github.com/huggingface/lerobot.git && cd lerobot
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
```

Install 🤗 LeRobot:
```bash
pip install .
```

For simulations, 🤗 LeRobot comes with gymnasium environments that can be installed as extras:
- [aloha](https://github.com/huggingface/gym-aloha)
- [xarm](https://github.com/huggingface/gym-xarm)
- [pusht](https://github.com/huggingface/gym-pusht)

For instance, to install 🤗 LeRobot with aloha and pusht, use:
```bash
pip install ".[aloha, pusht]"
```

To use [Weights and Biases](https://docs.wandb.ai/quickstart) for experiments tracking, log in with
```bash
wandb login
```

## Walkthrough

```
.
├── lerobot
|   ├── configs          # contains hydra yaml files with all options that you can override in the command line
|   |   ├── default.yaml   # selected by default, it loads pusht environment and diffusion policy
|   |   ├── env            # various sim environments and their datasets: aloha.yaml, pusht.yaml, xarm.yaml
|   |   └── policy         # various policies: act.yaml, diffusion.yaml, tdmpc.yaml
|   ├── common           # contains classes and utilities
|   |   ├── datasets       # various datasets of human demonstrations: aloha, pusht, xarm
|   |   ├── envs           # various sim environments: aloha, pusht, xarm
|   |   └── policies       # various policies: act, diffusion, tdmpc
|   └── scripts                  # contains functions to execute via command line
|       ├── visualize_dataset.py  # load a dataset and render its demonstrations
|       ├── eval.py               # load policy and evaluate it on an environment
|       └── train.py              # train a policy via imitation learning and/or reinforcement learning
├── outputs               # contains results of scripts execution: logs, videos, model checkpoints
├── .github
|   └── workflows
|       └── test.yml      # defines install settings for continuous integration and specifies end-to-end tests
└── tests                 # contains pytest utilities for continuous integration

```

### Visualize datasets

Check out [examples](./examples) to see how you can import our dataset class, download the data from the HuggingFace hub and use our rendering utilities.

Or you can achieve the same result by executing our script from the command line:
```bash
python lerobot/scripts/visualize_dataset.py \
env=pusht \
hydra.run.dir=outputs/visualize_dataset/example
# >>> ['outputs/visualize_dataset/example/episode_0.mp4']
```

### Evaluate a pretrained policy

Check out [examples](./examples) to see how you can load a pretrained policy from HuggingFace hub, load up the corresponding environment and model, and run an evaluation.

Or you can achieve the same result by executing our script from the command line:
```bash
python lerobot/scripts/eval.py \
-p lerobot/diffusion_policy_pusht_image \
eval_episodes=10 \
hydra.run.dir=outputs/eval/example_hub
```

After training your own policy, you can also re-evaluate the checkpoints with:

```bash
python lerobot/scripts/eval.py \
-p PATH/TO/TRAIN/OUTPUT/FOLDER \
eval_episodes=10 \
hydra.run.dir=outputs/eval/example_dir
```

See `python lerobot/scripts/eval.py --help` for more instructions.

### Train your own policy

Check out [examples](./examples) to see how you can start training a model on a dataset, which will be automatically downloaded if needed.

In general, you can use our training script to easily train any policy on any environment:
```bash
python lerobot/scripts/train.py \
env=aloha \
task=sim_insertion \
repo_id=lerobot/aloha_sim_insertion_scripted \
policy=act \
hydra.run.dir=outputs/train/aloha_act
```

After training, you may want to revisit model evaluation to change the evaluation settings. In fact, during training every checkpoint is already evaluated but on a low number of episodes for efficiency. Check out [example](./examples) to evaluate any model checkpoint on more episodes to increase statistical significance.

## Contribute

If you would like to contribute to 🤗 LeRobot, please check out our [contribution guide](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md).

### Add a new dataset

```python
# TODO(rcadene, AdilZouitine): rewrite this section
```

To add a dataset to the hub, first login and use a token generated from [huggingface settings](https://huggingface.co/settings/tokens) with write access:
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Then you can upload it to the hub with:
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload $HF_USER/$DATASET data/$DATASET \
--repo-type dataset  \
--revision v1.0
```

You will need to set the corresponding version as a default argument in your dataset class:
```python
  version: str | None = "v1.1",
```
See: [`lerobot/common/datasets/pusht.py`](https://github.com/Cadene/lerobot/blob/main/lerobot/common/datasets/pusht.py)

For instance, for [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht), we used:
```bash
HF_USER=lerobot
DATASET=pusht
```

If you want to improve an existing dataset, you can download it locally with:
```bash
mkdir -p data/$DATASET
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ${HF_USER}/$DATASET \
--repo-type dataset \
--local-dir data/$DATASET \
--local-dir-use-symlinks=False \
--revision v1.0
```

Iterate on your code and dataset with:
```bash
DATA_DIR=data python train.py
```

Upload a new version (v2.0 or v1.1 if the changes are respectively more or less significant):
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload $HF_USER/$DATASET data/$DATASET \
--repo-type dataset \
--revision v1.1 \
--delete "*"
```

Then you will need to set the corresponding version as a default argument in your dataset class:
```python
  version: str | None = "v1.1",
```
See: [`lerobot/common/datasets/pusht.py`](https://github.com/Cadene/lerobot/blob/main/lerobot/common/datasets/pusht.py)


Finally, you might want to mock the dataset if you need to update the unit tests as well:
```bash
python tests/scripts/mock_dataset.py --in-data-dir data/$DATASET --out-data-dir tests/data/$DATASET
```

### Add a pretrained policy

```python
# TODO(rcadene, alexander-soare): rewrite this section
```

Once you have trained a policy you may upload it to the HuggingFace hub.

Firstly, make sure you have a model repository set up on the hub. The hub ID looks like HF_USER/REPO_NAME.

Secondly, assuming you have trained a policy, you need the following (which should all be in any of the subdirectories of `checkpoints` in your training output folder, if you've used the LeRobot training script):

- `config.json`: A serialized version of the policy configuration (following the policy's dataclass config).
- `model.safetensors`: The `torch.nn.Module` parameters saved in [Hugging Face Safetensors](https://huggingface.co/docs/safetensors/index) format.
- `config.yaml`: This is the consolidated Hydra training configuration containing the policy, environment, and dataset configs. The policy configuration should match `config.json` exactly. The environment config is useful for anyone who wants to evaluate your policy. The dataset config just serves as a paper trail for reproducibility.

To upload these to the hub, run the following with a desired revision ID.

```bash
huggingface-cli upload $HUB_ID PATH/TO/OUTPUT/DIR --revision $REVISION_ID
```

If you want this to be the default revision also run the following (don't worry, it won't upload the files again; it will just adjust the file pointers):

```bash
huggingface-cli upload $HUB_ID PATH/TO/OUTPUT/DIR
```

See `eval.py` for an example of how a user may use your policy.


### Improve your code with profiling

An example of a code snippet to profile the evaluation of a policy:
```python
from torch.profiler import profile, record_function, ProfilerActivity

def trace_handler(prof):
    prof.export_chrome_trace(f"tmp/trace_schedule_{prof.step_num}.json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=3,
    ),
    on_trace_ready=trace_handler
) as prof:
    with record_function("eval_policy"):
        for i in range(num_episodes):
            prof.step()
            # insert code to profile, potentially whole body of eval_policy function
```

```bash
python lerobot/scripts/eval.py \
--config outputs/pusht/.hydra/config.yaml \
pretrained_model_path=outputs/pusht/model.pt \
eval_episodes=7
```
