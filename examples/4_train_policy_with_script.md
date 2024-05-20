This tutorial will explain the training script, how to use it, and particularly the use of Hydra to configure everything needed for the training run.

## The training script

LeRobot offers a training script at [`lerobot/scripts/train.py`](../../lerobot/scripts/train.py). At a high level it does the following:

- Loads a Hydra configuration file for the following steps (more on Hydra in a moment).
- Makes a simulation environment.
- Makes a dataset corresponding to that simulation environment.
- Makes a policy.
- Runs a standard training loop with forward pass, backward pass, optimization step, and occasional logging, evaluation (of the policy on the environment), and checkpointing.

## Our use of Hydra

Explaining the ins and outs of [Hydra](https://hydra.cc/docs/intro/) is beyond the scope of this document, but here we'll share the main points you need to know.

First, `lerobot/configs` has a directory structure like this:

```
.
├── default.yaml
├── env
│   ├── aloha.yaml
│   ├── pusht.yaml
│   └── xarm.yaml
└── policy
    ├── act.yaml
    ├── diffusion.yaml
    └── tdmpc.yaml
```

**_For brevity, in the rest of this document we'll drop the leading `lerobot/configs` path. So `default.yaml` really refers to `lerobot/configs/default.yaml`._**

When you run the training script with

```python
python lerobot/scripts/train.py
```

Hydra takes over via the `@hydra.main` decorator. If you take a look at the `@hydra.main`'s arguments you will see `config_path="../configs", config_name="default"`. This means Hydra looks for `default.yaml` in `../configs` (which resolves to `lerobot/configs`).

Therefore, `default.yaml` is the first configuration file that Hydra considers. At the top of the file, is a `defaults` section which looks likes this:

```yaml
defaults:
  - _self_
  - env: pusht
  - policy: diffusion
```

So, Hydra then grabs `env/pusht.yaml` and `policy/diffusion.yaml` and incorporates their configuration parameters as well (any configuration parameters already present in `default.yaml` are overriden).

Below the `defaults` section, `default.yaml` also contains regular configuration parameters.

If you want to train Diffusion Policy with PushT, you really only need to run:

```bash
python lerobot/scripts/train.py
```

That's because `default.yaml` already defaults to using Diffusion Policy and PushT. To be more explicit, you could also do the following (which would have the same effect):

```bash
python lerobot/scripts/train.py policy=diffusion env=pusht
```

If you want to train ACT with Aloha, you can do:

```bash
python lerobot/scripts/train.py policy=act env=aloha
```

**Notice, how the config overrides are passed** as `param_name=param_value`. This is the format the Hydra excepts for parsing the overrides.

_As an aside: we've set up our configurations so that they reproduce state-of-the-art results from papers in the literature._

## Overriding configuration parameters in the CLI

If you look in `env/aloha.yaml` you will see something like:

```yaml
# lerobot/configs/env/aloha.yaml
env:
  task: AlohaInsertion-v0
```

And if you look in `policy/act.yaml` you will see something like:

```yaml
# lerobot/configs/policy/act.yaml
dataset_repo_id: lerobot/aloha_sim_insertion_human
```

But our Aloha environment actually supports a cube transfer task as well. To train for this task, you _could_ modify the two configuration files respectively.

We need to select the cube transfer task for the ALOHA environment.

```yaml
# lerobot/configs/env/aloha.yaml
env:
   task: AlohaTransferCube-v0
```

We also need to use the cube transfer dataset.

```yaml
# lerobot/configs/policy/act.yaml
dataset_repo_id: lerobot/aloha_sim_transfer_cube_human
```

Now you'd be able to run:

```bash
python lerobot/scripts/train.py policy=act env=aloha
```

and you'd be training and evaluating on the cube transfer task.

OR, your could leave the configuration files in their original state and override the defaults via the command line:

```bash
python lerobot/scripts/train.py \
    policy=act \
    dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
    env=aloha \
    env.task=AlohaTransferCube-v0
```

There's something new here. Notice the `.` delimiter used to traverse the configuration hierarchy.

Putting all that knowledge together, here's the command that was used to train https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human.

```bash
python lerobot/scripts/train.py \
    hydra.run.dir=outputs/train/act_aloha_sim_transfer_cube_human \
    device=cuda
    env=aloha \
    env.task=AlohaTransferCube-v0 \
    dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
    policy=act \
    training.eval_freq=10000 \
    training.log_freq=250 \
    training.offline_steps=100000 \
    training.save_model=true \
    training.save_freq=25000 \
    eval.n_episodes=50 \
    eval.batch_size=50 \
    wandb.enable=false \
```

There's one new thing here: `hydra.run.dir=outputs/train/act_aloha_sim_transfer_cube_human`, which specifies where to save the training output.

---

So far we've seen how to train Diffusion Policy for PushT and ACT for ALOHA. Now, what if we want to train ACT for PushT? Well, there are aspects of the ACT configuration that are specific to the ALOHA environments, and these happen to be incompatible with PushT. Therefore, trying to run the following will almost certainly raise an exception of sorts (eg: feature dimension mismatch):

```bash
python lerobot/scripts/train.py policy=act env=pusht dataset_repo_id=lerobot/pusht
```

Please, head on over to our [advanced tutorial on adapting policy configuration to various environments](./advanced/train_act_pusht/train_act_pusht.md) to learn more.

Or in the meantime, happy coding! 🤗
