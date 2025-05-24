This tutorial explains how to use [Aloha and Aloha 2 stationary](https://www.trossenrobotics.com/aloha-stationary) with LeRobot.

## Setup

Follow the [documentation from Trossen Robotics](https://docs.trossenrobotics.com/aloha_docs/2.0/getting_started/stationary/hardware_setup.html) for setting up the hardware and plugging the 4 arms and 4 cameras to your computer.


## Install LeRobot

On your computer:

1. [Install Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install):
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

2. Restart shell or `source ~/.bashrc`

3. Create and activate a fresh conda environment for lerobot
```bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
```

4. Clone LeRobot:
```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
```

5. When using `miniconda`, install `ffmpeg` in your environment:
```bash
conda install ffmpeg -c conda-forge
```

6. Install LeRobot with dependencies for the Aloha motors (dynamixel) and cameras (intelrealsense):
```bash
cd ~/lerobot && pip install -e ".[dynamixel, intelrealsense]"
```

## Teleoperate

**/!\ FOR SAFETY, READ THIS /!\**
Teleoperation consists in manually operating the leader arms to move the follower arms. Importantly:
1. Make sure your leader arms are in the same position as the follower arms, so that the follower arms don't move too fast to match the leader arms,
2. Our code assumes that your robot has been assembled following Trossen Robotics instructions. This allows us to skip calibration, as we use the pre-defined calibration files in `.cache/calibration/aloha_default`. If you replace a motor, make sure you follow the exact instructions from Trossen Robotics.

By running the following code, you can start your first **SAFE** teleoperation:

> **NOTE:** To visualize the data, enable `--control.display_data=true`. This streams the data using `rerun`.

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=aloha \
  --robot.max_relative_target=5 \
  --control.type=teleoperate
```

By adding `--robot.max_relative_target=5`, we override the default value for `max_relative_target` defined in [`AlohaRobotConfig`](lerobot/common/robot_devices/robots/configs.py). It is expected to be `5` to limit the magnitude of the movement for more safety, but the teleoperation won't be smooth. When you feel confident, you can disable this limit by adding `--robot.max_relative_target=null` to the command line:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=aloha \
  --robot.max_relative_target=null \
  --control.type=teleoperate
```

## Record a dataset

Once you're familiar with teleoperation, you can record your first dataset with Aloha.

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
  --robot.type=aloha \
  --robot.max_relative_target=null \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/aloha_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=true
```

## Visualize a dataset

If you uploaded your dataset to the hub with `--control.push_to_hub=true`, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset) by copy pasting your repo id given by:
```bash
echo ${HF_USER}/aloha_test
```

If you didn't upload with `--control.push_to_hub=false`, you can also visualize it locally with:
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/aloha_test
```

## Replay an episode

**/!\ FOR SAFETY, READ THIS /!\**
Replay consists in automatically replaying the sequence of actions (i.e. goal positions for your motors) recorded in a given dataset episode. Make sure the current initial position of your robot is similar to the one in your episode, so that your follower arms don't move too fast to go to the first goal positions. For safety, you might want to add `--robot.max_relative_target=5` to your command line as explained above.

Now try to replay the first episode on your robot:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=aloha \
  --robot.max_relative_target=null \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/aloha_test \
  --control.episode=0
```

## Train a policy

To train a policy to control your robot, use the [`python lerobot/scripts/train.py`](../lerobot/scripts/train.py) script. A few arguments are required. Here is an example command:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/aloha_test \
  --policy.type=act \
  --output_dir=outputs/train/act_aloha_test \
  --job_name=act_aloha_test \
  --policy.device=cuda \
  --wandb.enable=true
```

Let's explain it:
1. We provided the dataset as argument with `--dataset.repo_id=${HF_USER}/aloha_test`.
2. We provided the policy with `policy.type=act`. This loads configurations from [`configuration_act.py`](../lerobot/common/policies/act/configuration_act.py). Importantly, this policy will automatically adapt to the number of motor states, motor actions and cameras of your robot (e.g. `laptop` and `phone`) which have been saved in your dataset.
4. We provided `policy.device=cuda` since we are training on a Nvidia GPU, but you could use `policy.device=mps` to train on Apple silicon.
5. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.

For more information on the `train` script see the previous tutorial: [`examples/4_train_policy_with_script.md`](../examples/4_train_policy_with_script.md)

Training should take several hours. You will find checkpoints in `outputs/train/act_aloha_test/checkpoints`.

## Evaluate your policy

You can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) but with a policy checkpoint as input. For instance, run this command to record 10 evaluation episodes:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=aloha \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/eval_act_aloha_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_aloha_test/checkpoints/last/pretrained_model \
  --control.num_image_writer_processes=1
```

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:
1. There is an additional `--control.policy.path` argument which indicates the path to your policy checkpoint with  (e.g. `outputs/train/eval_act_aloha_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `${HF_USER}/act_aloha_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `${HF_USER}/eval_act_aloha_test`).
3. We use `--control.num_image_writer_processes=1` instead of the default value (`0`). On our computer, using a dedicated process to write images from the 4 cameras on disk allows to reach constant 30 fps during inference. Feel free to explore different values for `--control.num_image_writer_processes`.

## More

Follow this [previous tutorial](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#4-train-a-policy-on-your-data) for a more in-depth explanation.

If you have any question or need help, please reach out on Discord in the channel `#aloha-arm`.
