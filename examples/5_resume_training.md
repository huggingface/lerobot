This tutorial explains how to resume a training run that you've started with the training script. If you don't know how our training script and configuration system works, please read [4_train_policy_with_script.md](./4_train_policy_with_script.md) first.

## Basic training resumption

Let's consider the example of training ACT for one of the ALOHA tasks. Here's a command that can achieve that:

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
python lerobot/scripts/train.py \
    hydra.run.dir=outputs/train/run_resumption \
    resume=true
```

You should see from the logging that your training picks up from where it left off.

Note that with `resume=true`, the configuration file from the last checkpoint in the training output directory is loaded. So it doesn't matter that we haven't provided all the other configuration parameters from our previous command (although there may be warnings to notify you that your command has a different configuration than than the checkpoint).

---

Now you should know how to resume your training run in case it gets interrupted or you want to extend a finished training run.

Happy coding! 🤗
