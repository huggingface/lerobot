This tutorial explains how to resume a training run that you've started with the training script. If you don't know how our training script and configuration system works, please read [4_train_policy_with_script.md](./4_train_policy_with_script.md) first.

## Basic training resumption

Let's consider the example of training ACT for one of the ALOHA tasks. Here's a command that can acheive that:

```bash
python lerobot/scripts/train.py \
    hydra.run.dir=outputs/train/run_resumption \
    policy=act \
    dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
    env=aloha \
    env.task=AlohaTransferCube-v0 \
    training.log_freq=25 \
    training.save_checkpoint=true \
    training.save_freq=100
```

Here we're using the default dataset and environment for ACT, and we've taken care to set up the log frequency and checkpointing frequency to low numbers so we can test resumption. You should be able to see some logging and have a first checkpoint within 1 minute. Please interrupt the training after the first checkpoint.

To resume, all that we have to do is run the training script, providing the run directory, and the resume option:

```bash
python lerobot/scripts/train.py hydra.run.dir=outputs/train/run_resumption resume=true
```

You should see from the logging that your training picks up from where it left off.

Note that with `resume=true` the default behavior is to use the configuration file from the last checkpoint in the training output directory. So it doesn't matter that we haven't provided all the other configuration parameters from our previous command.

## Overriding prior configuration parameters

Sometimes we may want to resume training but with some configuration parameters overridden. Let's say for example, that we want to resume training but we want more frequent logging:

```bash
python lerobot/scripts/train.py \
    --config-dir outputs/train/run_resumption/checkpoints/last/pretrained_model \
    --config-name config \
    hydra.run.dir=outputs/train/run_resumption \
    training.log_freq=10 \
    resume=true \
    override_config_on_resume=true
```

You should see from the logging that your training picks up from where it left off, and that the logging is more frequent.

Let's break down this command as compared to the more simple training resumption command above.

- We have added `override_config_on_resume=true`. This means that instead of implictly using the configuration file from the checkpoint, we use the one that Hydra resolves based on the commmand.
- We have changed the logging frequency with `training.log_freq=10`.
- We have provided the `--config-dir` and `--config-name` parameters to point Hydra to the checkpoint configuration. This is important because we have `override_config_on_resume=true`, meaning that the checkpoint configuration is ignored in favor of the one provided to via the command line. In order to handle this, we explicitly provide the checkpoint configuration via the command line.

---

Now you should know how to resume your training run in case it gets interrupted or you want to extend a finished training run.

Happy coding! 🤗
