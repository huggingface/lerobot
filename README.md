# Le Robot

#### State-of-the-art machine learning for real-world robotics

Le Robot aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier for entry to robotics so that everyone can contribute and benefit from sharing datasets and pretrained models.

Le Robot contains state-of-the-art approaches that have been shown to transfer to the real-world with a focus on imitation learning and reinforcement learning.

Le Robot already provides a set of pretrained models, datasets with human collected demonstrations, and simulated environments so that everyone can get started. In the coming weeks, the plan is to add more and more supports for real-world robotics on the most affordable and capable robots out there.

Le Robot is built upon [TorchRL](https://github.com/pytorch/rl) which provides abstractions and utilities for Reinforcement Learning.

## Acknowledgment

- Our ACT policy and ALOHA environment are adapted from [ALOHA](https://tonyzhaozh.github.io/aloha/)
- Our Diffusion policy and Pusht environment are adapted from [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- Our TDMPC policy and Simxarm environment are adapted from [FOWM](https://www.yunhaifeng.com/FOWM/)


## Installation

Create a virtual environment with Python 3.10, e.g. using `conda`:
```
conda create -y -n lerobot python=3.10
conda activate lerobot
```

[Install `poetry`](https://python-poetry.org/docs/#installation) (if you don't have it already)
```
curl -sSL https://install.python-poetry.org | python -
```

Install dependencies
```
poetry install
```

If you encounter a disk space error, try to change your tmp dir to a location where you have enough disk space, e.g.
```
mkdir ~/tmp
export TMPDIR='~/tmp'
```

To use [Weights and Biases](https://docs.wandb.ai/quickstart) for experiments tracking, log in with
```
wandb login
```

## Usage

### Train

```
python lerobot/scripts/train.py \
hydra.job.name=pusht \
env=pusht
```

### Visualize offline buffer

```
python lerobot/scripts/visualize_dataset.py \
hydra.run.dir=tmp/$(date +"%Y_%m_%d") \
env=pusht
```

### Eval

Run `python lerobot/scripts/eval.py --help` for instructions.

## TODO

If you are not sure how to contribute or want to know the next features we working on, look on this project page: [LeRobot TODO](https://github.com/users/Cadene/projects/1)

Ask [Remi Cadene](re.cadene@gmail.com) for access if needed.


## Profile

**Example**
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
```

```bash
python lerobot/scripts/eval.py \
    --config /home/rcadene/code/fowm/logs/xarm_lift/all/default/2/.hydra/config.yaml \
    pretrained_model_path=/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/final.pt \
    eval_episodes=7
```

## Contribute

**Style**
```
# install if needed
pre-commit install
# apply style and linter checks before git commit
pre-commit run -a
```

**Adding dependencies (temporary)**

Right now, for the CI to work, whenever a new dependency is added it needs to be also added to the cpu env, eg:

```
# Run in this directory, adds the package to the main env with cuda
poetry add some-package

# Adds the same package to the cpu env
cd .github/poetry/cpu && poetry add some-package
```

**Tests**

Install [git lfs](https://git-lfs.com/) to retrieve test artifacts (if you don't have it already).

On Mac:
```
brew install git-lfs
git lfs install
```

On Ubuntu:
```
sudo apt-get install git-lfs
git lfs install
```

Pull artifacts if they're not in [tests/data](tests/data)
```
git lfs pull
```

When adding a new dataset, mock it with
```
python tests/scripts/mock_dataset.py --in-data-dir data/$DATASET --out-data-dir tests/data/$DATASET
```

Run tests
```
DATA_DIR="tests/data" pytest -sx tests
```

**Datasets**

To add a dataset to the hub, first login and use a token generated from [huggingface settings](https://huggingface.co/settings/tokens) with write access:
```
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Then you can upload it to the hub with:
```
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload $HF_USER/$DATASET data/$DATASET \
--repo-type dataset  \
--revision v1.0
```

You will need to set the corresponding version as a default argument in your dataset class:
```python
  version: str | None = "v1.0",
```
See: [`lerobot/common/datasets/pusht.py`](https://github.com/Cadene/lerobot/blob/main/lerobot/common/datasets/pusht.py)

For instance, for [cadene/pusht](https://huggingface.co/datasets/cadene/pusht), we used:
```
HF_USER=cadene
DATASET=pusht
```

If you want to improve an existing dataset, you can download it locally with:
```
mkdir -p data/$DATASET
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ${HF_USER}/$DATASET \
--repo-type dataset \
--local-dir data/$DATASET \
--local-dir-use-symlinks=False \
--revision v1.0
```

Iterate on your code and dataset with:
```
DATA_DIR=data python train.py
```

Upload a new version (v2.0 or v1.1 if the changes are respectively more or less significant):
```
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
```
python tests/scripts/mock_dataset.py --in-data-dir data/$DATASET --out-data-dir tests/data/$DATASET
```

**Models**

Once you have trained a model you may upload it to the HuggingFace hub.

Firstly, make sure you have a model repository set up on the hub. The hub ID looks like HF_USER/REPO_NAME.

Secondly, assuming you have trained a model, you need:

- `config.yaml` which you can get from the `.hydra` directory of your training output folder.
- `model.pt` which should be one of the saved models in the `models` directory of your training output folder (they won't be named `model.pt` but you will need to choose one).
- `staths.pth` which should point to the same file in the dataset directory (found in `data/{dataset_name}`).

To upload these to the hub, prepare a folder with the following structure (you can use symlinks rather than copying):

```
to_upload
    ├── config.yaml
    ├── model.pt
    └── stats.pth
```

With the folder prepared, run the following with a desired revision ID.

```
huggingface-cli upload $HUB_ID to_upload --revision $REVISION_ID
```

If you want this to be the default revision also run the following (don't worry, it won't upload the files again; it will just adjust the file pointers):

```
huggingface-cli upload $HUB_ID to_upload
```

See `eval.py` for an example of how a user may use your model.
