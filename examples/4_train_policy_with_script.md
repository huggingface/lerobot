This tutorial will explain the training script, how to use it, and particularly how to configure everything needed for the training run.
> **Note:** The following assume you're running these commands on a machine equipped with a cuda GPU. If you don't have one (or if you're using a Mac), you can add `--device=cpu` (`--device=mps` respectively). However, be advised that the code executes much slower on cpu.


## The training script

LeRobot offers a training script at [`lerobot/scripts/train.py`](../../lerobot/scripts/train.py). At a high level it does the following:

- Initialize/load a configuration for the following steps using.
- Instantiates a dataset.
- (Optional) Instantiates a simulation environment corresponding to that dataset.
- Instantiates a policy.
- Runs a standard training loop with forward pass, backward pass, optimization step, and occasional logging, evaluation (of the policy on the environment), and checkpointing.

## Overview of the configuration system

In the training script, the main function `train` expects a `TrainPipelineConfig` object:
```python
# train.py
@parser.wrap()
def train(cfg: TrainPipelineConfig):
```

You can inspect the `TrainPipelineConfig` defined in [`lerobot/configs/train.py`](../../lerobot/configs/train.py) (which is heavily commented and meant to be a reference to understand any option)

When running the script, inputs for the command line are parsed thanks to the `@parser.wrap()` decorator and an instance of this class is automatically generated. Under the hood, this is done with [Draccus](https://github.com/dlwh/draccus) which is a tool dedicated for this purpose. If you're familiar with Hydra, Draccus can similarly load configurations from config files (.json, .yaml) and also override their values through command line inputs. Unlike Hydra, these configurations are pre-defined in the code through dataclasses rather than being defined entirely in config files. This allows for more rigorous serialization/deserialization, typing, and to manipulate configuration as objects directly in the code and not as dictionaries or namespaces (which enables nice features in an IDE such as autocomplete, jump-to-def, etc.)

Let's have a look at a simplified example. Amongst other attributes, the training config has the following attributes:
```python
@dataclass
class TrainPipelineConfig:
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
```
in which `DatasetConfig` for example is defined as such:
```python
@dataclass
class DatasetConfig:
    repo_id: str
    episodes: list[int] | None = None
    video_backend: str = "pyav"
```

This creates a hierarchical relationship where, for example assuming we have a `cfg` instance of `TrainPipelineConfig`, we can access the `repo_id` value with `cfg.dataset.repo_id`.
From the command line, we can specify this value with using a very similar syntax `--dataset.repo_id=repo/id`.

By default, every field takes its default value specified in the dataclass. If a field doesn't have a default value, it needs to be specified either from the command line or from a config file – which path is also given in the command line (more in this below). In the example above, the `dataset` field doesn't have a default value which means it must be specified.


## Specifying values from the CLI

Let's say that we want to train [Diffusion Policy](../../lerobot/common/policies/diffusion) on the [pusht](https://huggingface.co/datasets/lerobot/pusht) dataset, using the [gym_pusht](https://github.com/huggingface/gym-pusht) environment for evaluation. The command to do so would look like this:
```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=lerobot/pusht \
    --policy.type=diffusion \
    --env.type=pusht
```

Let's break this down:
- To specify the dataset, we just need to specify its `repo_id` on the hub which is the only required argument in the `DatasetConfig`. The rest of the fields have default values and in this case we are fine with those so we can just add the option `--dataset.repo_id=lerobot/pusht`.
- To specify the policy, we can just select diffusion policy using `--policy` appended with `.type`. Here, `.type` is a special argument which allows us to select config classes inheriting from `draccus.ChoiceRegistry` and that have been decorated with the `register_subclass()` method. To have a better explanation of this feature, have a look at this [Draccus demo](https://github.com/dlwh/draccus?tab=readme-ov-file#more-flexible-configuration-with-choice-types). In our code, we use this mechanism mainly to select policies, environments, robots, and some other components like optimizers. The policies available to select are located in [lerobot/common/policies](../../lerobot/common/policies)
- Similarly, we select the environment with `--env.type=pusht`. The different environment configs are available in [`lerobot/common/envs/configs.py`](../../lerobot/common/envs/configs.py)

Let's see another example. Let's say you've been training [ACT](../../lerobot/common/policies/act) on [lerobot/aloha_sim_insertion_human](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) using the [gym-aloha](https://github.com/huggingface/gym-aloha) environment for evaluation with:
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --output_dir=outputs/train/act_aloha_insertion
```
> Notice we added `--output_dir` to explicitly tell where to write outputs from this run (checkpoints, training state, configs etc.). This is not mandatory and if you don't specify it, a default directory will be created from the current date and time, env.type and policy.type. This will typically look like `outputs/train/2025-01-24/16-10-05_aloha_act`.

We now want to train a different policy for aloha on another task. We'll change the dataset and use [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) instead. Of course, we also need to change the task of the environment as well to match this other task.
Looking at the [`AlohaEnv`](../../lerobot/common/envs/configs.py) config, the task is `"AlohaInsertion-v0"` by default, which corresponds to the task we trained on in the command above. The [gym-aloha](https://github.com/huggingface/gym-aloha?tab=readme-ov-file#description) environment also has the `AlohaTransferCube-v0` task which corresponds to this other task we want to train on. Putting this together, we can train this new policy on this different task using:
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --output_dir=outputs/train/act_aloha_transfer
```

## Loading from a config file

Now, let's assume that we want to reproduce the run just above. That run has produced a `train_config.json` file in its checkpoints, which serializes the `TrainPipelineConfig` instance it used:
```json
{
    "dataset": {
        "repo_id": "lerobot/aloha_sim_transfer_cube_human",
        "episodes": null,
        ...
    },
    "env": {
        "type": "aloha",
        "task": "AlohaTransferCube-v0",
        "fps": 50,
        ...
    },
    "policy": {
        "type": "act",
        "n_obs_steps": 1,
        ...
    },
    ...
}
```

We can then simply load the config values from this file using:
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model/ \
    --output_dir=outputs/train/act_aloha_transfer_2
```
`--config_path` is also a special argument which allows to initialize the config from a local config file. It can point to a directory that contains `train_config.json` or to the config file itself directly.

Similarly to Hydra, we can still override some parameters in the CLI if we want to, e.g.:
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model/ \
    --output_dir=outputs/train/act_aloha_transfer_2
    --policy.n_action_steps=80
```
> Note: While `--output_dir` is not required in general, in this case we need to specify it since it will otherwise take the value from the `train_config.json` (which is `outputs/train/act_aloha_transfer`). In order to prevent accidental deletion of previous run checkpoints, we raise an error if you're trying to write in an existing directory. This is not the case when resuming a run, which is what you'll learn next.

`--config_path` can also accept the repo_id of a repo on the hub that contains a `train_config.json` file, e.g. running:
```bash
python lerobot/scripts/train.py --config_path=lerobot/diffusion_pusht
```
will start a training run with the same configuration used for training [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht)


## Resume training

Being able to resume a training run is important in case it crashed or aborted for any reason. We'll demonstrate how to that here.

Let's reuse the command from the previous run and add a few more options:
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --log_freq=25 \
    --save_freq=100 \
    --output_dir=outputs/train/run_resumption
```

Here we've taken care to set up the log frequency and checkpointing frequency to low numbers so we can showcase resumption. You should be able to see some logging and have a first checkpoint within 1 minute (depending on hardware). Wait for the first checkpoint to happen, you should see a line that looks like this in your terminal:
```
INFO 2025-01-24 16:10:56 ts/train.py:263 Checkpoint policy after step 100
```
Now let's simulate a crash by killing the process (hit `ctrl`+`c`). We can then simply resume this run from the last checkpoint available with:
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/run_resumption/checkpoints/last/pretrained_model/ \
    --resume=true
```
You should see from the logging that your training picks up from where it left off.

Another reason for which you might want to resume a run is simply to extend training and add more training steps. The number of training steps is set by the option `--steps`, which is 100 000 by default.
You could double the number of steps of the previous run with:
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/run_resumption/checkpoints/last/pretrained_model/ \
    --resume=true \
    --steps=200000
```

## Outputs of a run
In the output directory, there will be a folder called `checkpoints` with the following structure:
```bash
outputs/train/run_resumption/checkpoints
├── 000100  # checkpoint_dir for training step 100
│   ├── pretrained_model/
│   │   ├── config.json  # policy config
│   │   ├── model.safetensors  # policy weights
│   │   └── train_config.json  # train config
│   └── training_state/
│       ├── optimizer_param_groups.json  #  optimizer param groups
│       ├── optimizer_state.safetensors  # optimizer state
│       ├── rng_state.safetensors  # rng states
│       ├── scheduler_state.json  # scheduler state
│       └── training_step.json  # training step
├── 000200
└── last -> 000200  # symlink to the last available checkpoint
```

## Fine-tuning a pre-trained policy

In addition to the features currently in Draccus, we've added a special `.path` argument for the policy, which allows to load a policy as you would with `PreTrainedPolicy.from_pretrained()`. In that case, `path` can be a local directory that contains a checkpoint or a repo_id pointing to a pretrained policy on the hub.

For example, we could fine-tune a [policy pre-trained on the aloha transfer task](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human) on the aloha insertion task. We can achieve this with:
```bash
python lerobot/scripts/train.py \
    --policy.path=lerobot/act_aloha_sim_transfer_cube_human \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0
```

When doing so, keep in mind that the features of the fine-tuning dataset would have to match the input/output features of the pretrained policy.

## Typical logs and metrics

When you start the training process, you will first see your full configuration being printed in the terminal. You can check it to make sure that you configured your run correctly. The final configuration will also be saved with the checkpoint.

After that, you will see training log like this one:
```
INFO 2024-08-14 13:35:12 ts/train.py:192 step:0 smpl:64 ep:1 epch:0.00 loss:1.112 grdn:15.387 lr:2.0e-07 updt_s:1.738 data_s:4.774
```
or evaluation log:
```
INFO 2024-08-14 13:38:45 ts/train.py:226 step:100 smpl:6K ep:52 epch:0.25 ∑rwrd:20.693 success:0.0% eval_s:120.266
```

These logs will also be saved in wandb if `wandb.enable` is set to `true`. Here are the meaning of some abbreviations:
- `smpl`: number of samples seen during training.
- `ep`: number of episodes seen during training. An episode contains multiple samples in a complete manipulation task.
- `epch`: number of time all unique samples are seen (epoch).
- `grdn`: gradient norm.
- `∑rwrd`: compute the sum of rewards in every evaluation episode and then take an average of them.
- `success`: average success rate of eval episodes. Reward and success are usually different except for the sparsing reward setting, where reward=1 only when the task is completed successfully.
- `eval_s`: time to evaluate the policy in the environment, in second.
- `updt_s`: time to update the network parameters, in second.
- `data_s`: time to load a batch of data, in second.

Some metrics are useful for initial performance profiling. For example, if you find the current GPU utilization is low via the `nvidia-smi` command and `data_s` sometimes is too high, you may need to modify batch size or number of dataloading workers to accelerate dataloading. We also recommend [pytorch profiler](https://github.com/huggingface/lerobot?tab=readme-ov-file#improve-your-code-with-profiling) for detailed performance probing.

## In short

We'll summarize here the main use cases to remember from this tutorial.

#### Train a policy from scratch – CLI
```bash
python lerobot/scripts/train.py \
    --policy.type=act \  # <- select 'act' policy
    --env.type=pusht \  # <- select 'pusht' environment
    --dataset.repo_id=lerobot/pusht  # <- train on this dataset
```

#### Train a policy from scratch - config file + CLI
```bash
python lerobot/scripts/train.py \
    --config_path=path/to/pretrained_model \  # <- can also be a repo_id
    --policy.n_action_steps=80  # <- you may still override values
```

#### Resume/continue a training run
```bash
python lerobot/scripts/train.py \
    --config_path=checkpoint/pretrained_model/ \
    --resume=true \
    --steps=200000  # <- you can change some training parameters
```

#### Fine-tuning
```bash
python lerobot/scripts/train.py \
    --policy.path=lerobot/act_aloha_sim_transfer_cube_human \  # <- can also be a local path to a checkpoint
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0
```

---

Now that you know the basics of how to train a policy, you might want to know how to apply this knowledge to actual robots, or how to record your own datasets and train policies on your specific task?
If that's the case, head over to the next tutorial [`7_get_started_with_real_robot.md`](./7_get_started_with_real_robot.md).

Or in the meantime, happy training! 🤗
