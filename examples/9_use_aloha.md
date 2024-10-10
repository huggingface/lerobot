This tutorial explains how to use [Aloha and Aloha 2 stationary](https://www.trossenrobotics.com/aloha-stationary) with LeRobot.

## Setup

Follow the [documentation from Trossen Robotics](https://docs.trossenrobotics.com/aloha_docs/getting_started/stationary/hardware_setup.html) for setting up the hardware and plugging the 4 arms and 4 cameras to your computer.


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

5. Install LeRobot with dependencies for the Aloha motors (dynamixel) and cameras (intelrealsense):
```bash
cd ~/lerobot && pip install -e ".[dynamixel intelrealsense]"
```

And install extra dependencies for recording datasets on Linux:
```bash
conda install -y -c conda-forge ffmpeg
pip uninstall -y opencv-python
conda install -y -c conda-forge "opencv>=4.10.0"
```

## Teleoperate

Now try out teleoperation by manually operating the leader arms to move the follower arms:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --robot-path lerobot/configs/robot/aloha.yaml
```

You will certainly notice some lag or latency. This is because, we limit the magnitude of the movements by default to ensure safety. When you feel confident, you can disable it by adding `--robot-overrides max_relative_target=null`:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --robot-path lerobot/configs/robot/aloha.yaml \
    --robot-overrides max_relative_target=null
```

## Record a dataset

Once you're familiar with teleoperation, you can try to record your first dataset with Aloha.

If you want to use the Hugging Face hub features for uploading your dataset and you haven't previously done it, make sure you've logged in using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Store your Hugging Face repository name in a variable to run these commands:
```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

Record 2 episodes and push your dataset on the hub:
```bash
python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/aloha.yaml \
    --robot-overrides max_relative_target=null \
    --fps 30 \
    --root data \
    --repo-id ${HF_USER}/aloha_test \
    --tags aloha tutorial \
    --warmup-time-s 5 \
    --episode-time-s 40 \
    --reset-time-s 10 \
    --num-episodes 2 \
    --push-to-hub 1
```

## Visualize a dataset

If you pushed on the hub with `--push-to-hub 1`, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset) by copy pasting your repo id given by:
```bash
echo ${HF_USER}/aloha_test
```

If you didn't push it with `--push-to-hub 0`, you can also visualize it locally with:
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --root data \
  --repo-id ${HF_USER}/aloha_test
```

## Replay an episode

Now try to replay the first episode, but before, make sure the current initial position of your robot is similar to the one in your dataset:
```bash
python lerobot/scripts/control_robot.py replay \
    --robot-path lerobot/configs/robot/aloha.yaml \
    --robot-overrides max_relative_target=null \
    --fps 30 \
    --root data \
    --repo-id ${HF_USER}/aloha_test \
    --episode 0
```

## Train a policy

To train a policy to control your robot, use the [`python lerobot/scripts/train.py`](../lerobot/scripts/train.py) script. A few arguments are required. Here is an example command:
```bash
DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=${HF_USER}/aloha_test \
  policy=act_aloha_real \
  env=aloha_real \
  env.state_dim=18 \
  env.action_dim=18 \
  hydra.run.dir=outputs/train/act_aloha_test \
  hydra.job.name=act_aloha_test \
  device=cuda \
  wandb.enable=true
```

Let's explain it:
1. We provided the dataset as argument with `dataset_repo_id=${HF_USER}/aloha_test`.
2. We provided the policy with `policy=act_aloha_real`. This loads configurations from [`lerobot/configs/policy/act_aloha_real.yaml`](../lerobot/configs/policy/act_aloha_real.yaml). Importantly, this policy uses 4 cameras as input `cam_right_wrist`, `cam_left_wrist`, `cam_high`, and `cam_low`.
3. We provided an environment as argument with `env=aloha_real`. This loads configurations from [`lerobot/configs/env/aloha_real.yaml`](../lerobot/configs/env/aloha_real.yaml). We also overrided the default `state_dim` and `action_dim` defined in this yaml to fit our number of motors. This is because, we include the `shoulder_shadow` and `elbow_shadow` motors for simplicity.
4. We provided `device=cuda` since we are training on a Nvidia GPU.
5. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.
6. We added `DATA_DIR=data` to access your dataset stored in your local `data` directory. If you dont provide `DATA_DIR`, your dataset will be downloaded from Hugging Face hub to your cache folder `$HOME/.cache/hugginface`. In future versions of `lerobot`, both directories will be in sync.

Training should take several hours. You will find checkpoints in `outputs/train/act_aloha_test/checkpoints`.

## Evaluate your policy

You can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) but with a policy checkpoint as input. For instance, run this command to record 10 evaluation episodes:
```bash
python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/aloha.yaml \
  --robot-overrides max_relative_target=null \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/eval_act_aloha_test \
  --tags aloha tutorial eval \
  --warmup-time-s 5 \
  --episode-time-s 40 \
  --reset-time-s 10 \
  --num-episodes 10 \
  -p outputs/train/act_aloha_test/checkpoints/last/pretrained_model
```

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:
1. There is an additional `-p` argument which indicates the path to your policy checkpoint with  (e.g. `-p outputs/train/eval_aloha_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `-p ${HF_USER}/act_aloha_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `--repo-id ${HF_USER}/eval_act_aloha_test`).

## More

Follow [previous tutorial](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#4-train-a-policy-on-your-data) to train a policy on your data and run inference on your robot. You will need to adapt the code for Aloha.

If you need help, please reach out on Discord in the channel `#aloha-arm`.
