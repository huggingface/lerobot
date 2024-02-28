# LeRobot

## Installation

Install dependencies using `conda`:

```
conda env create -f environment.yaml
conda activate lerobot
```

Install `torchrl`, `tensordict` and `diffusion_policy` dev builds
```
cd path/to/root
git clone https://github.com/pytorch/tensordict
git clone https://github.com/pytorch/rl
git clone https://github.com/real-stanford/diffusion_policy
cd tensordict
python setup.py develop
cd ../rl
python setup.py develop
cd ../diffusion_policy
python setup.py develop
```

Install additional modules
```
pip install \
    hydra \
    termcolor \
    einops \
    pygame \
    pymunk \
    zarr \
    gym \
    shapely \
    opencv-python \
    scikit-image \
    mpmath==1.3.0 \
```

Fix Hydra
```
pip install hydra-core --upgrade
```

**dev**

```
python setup.py develop
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

### Visualize online buffer / Eval

```
python lerobot/scripts/eval.py \
hydra.run.dir=tmp/$(date +"%Y_%m_%d") \
env=pusht
```


## TODO

- [x] priority update doesnt match FOWM or original paper
- [x] self.step=100000 should be updated at every step to adjust to horizon of planner
- [ ] prefetch replay buffer to speedup training
- [ ] parallelize env to speedup eval
- [ ] clean checkpointing / loading
- [ ] clean logging
- [ ] clean config
- [ ] clean hyperparameter tuning
- [ ] add pusht
- [ ] add aloha
- [ ] add act
- [ ] add diffusion
- [ ] add aloha 2

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
pretrained_model_path=/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/final.pt \
eval_episodes=7
```

## Contribute

**Style**
```
isort lerobot && isort tests && black lerobot && black tests
pylint lerobot && pylint tests  # not enforce for now
```

**Tests**
```
pytest -sx tests
```