This tutorial explains how to use [Moss v1](https://github.com/jess-moss/moss-robot-arms) with LeRobot.

## Source the parts

Follow this [README](https://github.com/jess-moss/moss-robot-arms). It contains the bill of materials, with link to source the parts, as well as the instructions to 3D print the parts, and advices if it's your first time printing or if you don't own a 3D printer already.

**Important**: Before assembling, you will first need to configure your motors. To this end, we provide a nice script, so let's first install LeRobot. After configuration, we will also guide you through assembly.

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

5. Install LeRobot with dependencies for the feetech motors:
```bash
cd ~/lerobot && pip install -e ".[feetech]"
```

For Linux only (not Mac), install extra dependencies for recording datasets:
```bash
conda install -y -c conda-forge ffmpeg
pip uninstall -y opencv-python
conda install -y -c conda-forge "opencv>=4.10.0"
```

## Configure the motors

Follow steps 1 of the [assembly video](https://www.youtube.com/watch?v=DA91NJOtMic) which illustrates the use of our scripts below.

**Find USB ports associated to your arms**
To find the correct ports for each arm, run the utility script twice:
```bash
python lerobot/scripts/find_motors_bus_port.py
```

Example output when identifying the leader arm's port (e.g., `/dev/tty.usbmodem575E0031751` on Mac, or possibly `/dev/ttyACM0` on Linux):
```
Finding all available ports for the MotorBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the usb cable from your DynamixelMotorsBus and press Enter when done.

[...Disconnect leader arm and press Enter...]

The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0031751
Reconnect the usb cable.
```

Example output when identifying the follower arm's port (e.g., `/dev/tty.usbmodem575E0032081`, or possibly `/dev/ttyACM1` on Linux):
```
Finding all available ports for the MotorBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the usb cable from your DynamixelMotorsBus and press Enter when done.

[...Disconnect follower arm and press Enter...]

The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0032081
Reconnect the usb cable.
```

Troubleshooting: On Linux, you might need to give access to the USB ports by running:
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

**Configure your motors**
Plug your first motor and run this script to set its ID to 1. It will also set its present position to 2048, so expect your motor to rotate:
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58760432961 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```

Note: These motors are currently limitated. They can take values between 0 and 4096 only, which corresponds to a full turn. They can't turn more than that. 2048 is at the middle of this range, so we can take -2048 steps (180 degrees anticlockwise) and reach the maximum range, or take +2048 steps (180 degrees clockwise) and reach the maximum range. The configuration step also sets the homing offset to 0, so that if you misassembled the arm, you can always update the homing offset to account for a shift up to ± 2048 steps (± 180 degrees).

Then unplug your motor and plug the second motor and set its ID to 2.
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58760432961 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 2
```

Redo the process for all your motors until ID 6. Do the same for the 6 motors of the leader arm.

**Remove the gears of the 6 leader motors**
Follow step 2 of the [assembly video](https://www.youtube.com/watch?v=DA91NJOtMic). You need to remove the gear for the motors of the leader arm. As a result, you will only use the position encoding of the motor and reduce friction to more easily operate the leader arm.

**Add motor horn to the motors**
Follow step 3 of the [assembly video](https://www.youtube.com/watch?v=DA91NJOtMic). For Moss v1, you need to align the holes on the motor horn to the motor spline to be approximately 3, 6, 9 and 12 o'clock.
Try to avoid rotating the motor while doing so to keep position 2048 set during configuration. It is especially tricky for the leader motors as it is more sensible without the gears, but it's ok if it's a bit rotated.

## Assemble the arms

Follow step 4 of the [assembly video](https://www.youtube.com/watch?v=DA91NJOtMic). The first arm should take a bit more than 1 hour to assemble, but once you get use to it, you can do it under 1 hour for the second arm.

## Calibrate

Next, you'll need to calibrate your Moss v1 robot to ensure that the leader and follower arms have the same position values when they are in the same physical position. This calibration is essential because it allows a neural network trained on one Moss v1 robot to work on another.

**Manual calibration of follower arm**
/!\ Contrarily to step 6 of the [assembly video](https://www.youtube.com/watch?v=DA91NJOtMic) which illustrates the auto calibration, we will actually do manual calibration of follower for now.

You will need to move the follower arm to these positions sequentially:

| 1. Zero position | 2. Rotated position | 3. Rest position |
|---|---|---|
| <img src="../media/moss/follower_zero.webp?raw=true" alt="Moss v1 follower arm zero position" title="Moss v1 follower arm zero position" style="width:100%;"> | <img src="../media/moss/follower_rotated.webp?raw=true" alt="Moss v1 follower arm rotated position" title="Moss v1 follower arm rotated position" style="width:100%;"> | <img src="../media/moss/follower_rest.webp?raw=true" alt="Moss v1 follower arm rest position" title="Moss v1 follower arm rest position" style="width:100%;"> |

Make sure both arms are connected and run this script to launch manual calibration:
```bash
python lerobot/scripts/control_robot.py calibrate \
    --robot-path lerobot/configs/robot/moss.yaml \
    --robot-overrides '~cameras' --arms main_follower
```

**Manual calibration of leader arm**
Follow step 6 of the [assembly video](https://www.youtube.com/watch?v=DA91NJOtMic) which illustrates the manual calibration. You will need to move the leader arm to these positions sequentially:

| 1. Zero position | 2. Rotated position | 3. Rest position |
|---|---|---|
| <img src="../media/moss/leader_zero.webp?raw=true" alt="Moss v1 leader arm zero position" title="Moss v1 leader arm zero position" style="width:100%;"> | <img src="../media/moss/leader_rotated.webp?raw=true" alt="Moss v1 leader arm rotated position" title="Moss v1 leader arm rotated position" style="width:100%;"> | <img src="../media/moss/leader_rest.webp?raw=true" alt="Moss v1 leader arm rest position" title="Moss v1 leader arm rest position" style="width:100%;"> |

Run this script to launch manual calibration:
```bash
python lerobot/scripts/control_robot.py calibrate \
    --robot-path lerobot/configs/robot/moss.yaml \
    --robot-overrides '~cameras' --arms main_leader
```

## Teleoperate

**Simple teleop**
Then you are ready to teleoperate your robot! Run this simple script (it won't connect and display the cameras):
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --robot-path lerobot/configs/robot/moss.yaml \
    --robot-overrides '~cameras' \
    --display-cameras 0
```


**Teleop with displaying cameras**
Follow [this guide to setup your cameras](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#c-add-your-cameras-with-opencvcamera). Then you will be able to display the cameras on your computer while you are teleoperating by running the following code. This is useful to prepare your setup before recording your first dataset.
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --robot-path lerobot/configs/robot/moss.yaml
```

## Record a dataset

Once you're familiar with teleoperation, you can record your first dataset with Moss v1.

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
python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/moss.yaml \
    --fps 30 \
    --root data \
    --repo-id ${HF_USER}/moss_test \
    --tags moss tutorial \
    --warmup-time-s 5 \
    --episode-time-s 40 \
    --reset-time-s 10 \
    --num-episodes 2 \
    --push-to-hub 1
```

## Visualize a dataset

If you uploaded your dataset to the hub with `--push-to-hub 1`, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset) by copy pasting your repo id given by:
```bash
echo ${HF_USER}/moss_test
```

If you didn't upload with `--push-to-hub 0`, you can also visualize it locally with:
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --root data \
  --repo-id ${HF_USER}/moss_test
```

## Replay an episode

Now try to replay the first episode on your robot:
```bash
DATA_DIR=data python lerobot/scripts/control_robot.py replay \
    --robot-path lerobot/configs/robot/moss.yaml \
    --fps 30 \
    --root data \
    --repo-id ${HF_USER}/moss_test \
    --episode 0
```

## Train a policy

To train a policy to control your robot, use the [`python lerobot/scripts/train.py`](../lerobot/scripts/train.py) script. A few arguments are required. Here is an example command:
```bash
DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=${HF_USER}/moss_test \
  policy=act_moss_real \
  env=moss_real \
  hydra.run.dir=outputs/train/act_moss_test \
  hydra.job.name=act_moss_test \
  device=cuda \
  wandb.enable=true
```

Let's explain it:
1. We provided the dataset as argument with `dataset_repo_id=${HF_USER}/moss_test`.
2. We provided the policy with `policy=act_moss_real`. This loads configurations from [`lerobot/configs/policy/act_moss_real.yaml`](../lerobot/configs/policy/act_moss_real.yaml). Importantly, this policy uses 2 cameras as input `laptop`, `phone`.
3. We provided an environment as argument with `env=moss_real`. This loads configurations from [`lerobot/configs/env/moss_real.yaml`](../lerobot/configs/env/moss_real.yaml).
4. We provided `device=cuda` since we are training on a Nvidia GPU, but you can also use `device=mps` if you are using a Mac with Apple silicon, or `device=cpu` otherwise.
5. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.
6. We added `DATA_DIR=data` to access your dataset stored in your local `data` directory. If you dont provide `DATA_DIR`, your dataset will be downloaded from Hugging Face hub to your cache folder `$HOME/.cache/hugginface`. In future versions of `lerobot`, both directories will be in sync.

Training should take several hours. You will find checkpoints in `outputs/train/act_moss_test/checkpoints`.

## Evaluate your policy

You can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) but with a policy checkpoint as input. For instance, run this command to record 10 evaluation episodes:
```bash
python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/moss.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/eval_act_moss_test \
  --tags moss tutorial eval \
  --warmup-time-s 5 \
  --episode-time-s 40 \
  --reset-time-s 10 \
  --num-episodes 10 \
  -p outputs/train/act_moss_test/checkpoints/last/pretrained_model
```

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:
1. There is an additional `-p` argument which indicates the path to your policy checkpoint with  (e.g. `-p outputs/train/eval_moss_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `-p ${HF_USER}/act_moss_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `--repo-id ${HF_USER}/eval_act_moss_test`).

## More

Follow this [previous tutorial](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#4-train-a-policy-on-your-data) for a more in-depth tutorial on controlling real robots with LeRobot.

If you have any question or need help, please reach out on Discord in the channel [`#moss-arm`](https://discord.com/channels/1216765309076115607/1275374638985252925).
