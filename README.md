# LeRobot

## Installation

Install dependencies using `conda`:

```
conda env create -f environment.yaml
conda activate lerobot
```

**dev**

```
python setup.py develop
```

## Usage


### Train

```
python lerobot/scripts/train.py \
--config-name=pusht hydra.job.name=pusht
```

### Visualize offline buffer

```
python lerobot/scripts/visualize_dataset.py \
--config-name=pusht hydra.run.dir=tmp/$(date +"%Y_%m_%d")
```

### Visualize online buffer / Eval

```
python lerobot/scripts/eval.py \
--config-name=pusht hydra.run.dir=tmp/$(date +"%Y_%m_%d")
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

**style**
```
isort lerobot
black lerobot
isort test
black test
pylint lerobot
```
