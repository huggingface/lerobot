# Using the [roarm_m3](https://github.com/waveshareteam/roarm_m3) with LeRobot

## Table of Contents

  - [A. Teleoperate](#a-teleoperate)
  - [B. Record a dataset](#b-record-a-dataset)
  - [C. Visualize a dataset](#c-visualize-a-dataset)
  - [D. Replay an episode](#d-replay-an-episode)
  - [E. Train a policy](#e-train-a-policy)
  - [F. Evaluate your policy](#f-evaluate-your-policy)
  - [G. More Information](#g-more-information)

## A. Teleoperate

**Simple teleop**
#### a. Teleop without displaying cameras
You will be able to teleoperate your robot! (it won't connect and display the cameras):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=roarm_m3 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```


#### b. Teleop with displaying cameras
You will be able to display the cameras while you are teleoperating by running the following code. This is useful to prepare your setup before recording your first dataset.
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=roarm_m3 \
  --control.type=teleoperate
```

## B. Record a dataset

Once you're familiar with teleoperation, you can record your first dataset with roarm_m3.

If you want to use the Hugging Face hub features for uploading your dataset and you haven't previously done it, make sure you've logged in using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Store your Hugging Face repository name in a variable to run these commands:
```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

Record 2 episodes and upload your dataset to the hub:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=roarm_m3 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a block and put it in the bin." \
  --control.repo_id=${HF_USER}/roarm_m3_test \
  --control.tags='["roarm_m3","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=true
```

Note: You can resume recording by adding `--control.resume=true`.

## C. Visualize a dataset

If you uploaded your dataset to the hub with `--control.push_to_hub=true`, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset) by copy pasting your repo id given by:
```bash
echo ${HF_USER}/roarm_m3_test
```

If you didn't upload with `--control.push_to_hub=false`, you can also visualize it locally with (a window can be opened in the browser `http://ip:9090` with the visualization tool):
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/roarm_m3_test \
  --host ip \
  --local-files-only 1
```

## D. Replay an episode

Now try to replay episode nth on your bot:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=roarm_m3 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/roarm_m3_test \
  --control.episode=n-1
```

## E. Train a policy

To train a policy to control your robot, use the [`python lerobot/scripts/train.py`](../lerobot/scripts/train.py) script. A few arguments are required. Here is an example command:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/roarm_m3_test \
  --policy.type=act \
  --output_dir=outputs/train/act_roarm_m3_test \
  --job_name=act_roarm_m3_test \
  --device=cuda \
  --wandb.enable=true
```

Let's explain it:
1. We provided the dataset as argument with `--dataset.repo_id=${HF_USER}/roarm_m3_test`.
2. We provided the policy with `policy.type=act`. This loads configurations from [`configuration_act.py`](../lerobot/common/policies/act/configuration_act.py). Importantly, this policy will automatically adapt to the number of motor sates, motor actions and cameras of your robot (e.g. `laptop` and `phone`) which have been saved in your dataset.
4. We provided `device=cuda` since we are training on a Nvidia GPU, but you could use `device=mps` to train on Apple silicon.
5. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.

Training should take several hours. You will find checkpoints in `outputs/train/act_roarm_m3_test/checkpoints`.

## F. Evaluate your policy

You can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) but with a policy checkpoint as input. For instance, run this command to record 10 evaluation episodes:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=roarm_m3 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a block and put it in the bin." \
  --control.repo_id=${HF_USER}/eval_act_roarm_m3_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_roarm_m3_test/checkpoints/last/pretrained_model
```

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:
1. There is an additional `--control.policy.path` argument which indicates the path to your policy checkpoint with  (e.g. `outputs/train/eval_act_roarm_m3_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `${HF_USER}/act_roarm_m3_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `${HF_USER}/eval_act_roarm_m3_test`).

## G. More Information

Follow this [previous tutorial](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#4-train-a-policy-on-your-data) for a more in-depth tutorial on controlling real robots with LeRobot.
