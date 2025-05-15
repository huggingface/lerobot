# LeKiwi

In the steps below we explain how to assemble the LeKiwi mobile robot.

## Source the parts

Follow this [README](https://github.com/SIGRobotics-UIUC/LeKiwi). It contains the bill of materials, with a link to source the parts, as well as the instructions to 3D print the parts,
and advice if it's your first time printing or if you don't own a 3D printer.

### Wired version
If you have the **wired** LeKiwi version you can skip the installation of the Raspberry Pi and setting up SSH. You can also run all commands directly on your PC for both the LeKiwi scripts and the leader arm scripts for teleoperating.

## Install software on Pi
Now we have to setup the remote PC that will run on the LeKiwi Robot. This is normally a Raspberry Pi, but can be any PC that can run on 5V and has enough usb ports (2 or more) for the cameras and motor control board.

### Install OS
For setting up the Raspberry Pi and its SD-card see: [Setup PI](https://www.raspberrypi.com/documentation/computers/getting-started.html). Here is explained how to download the [Imager](https://www.raspberrypi.com/software/) to install Raspberry Pi OS or Ubuntu.

### Setup SSH
After setting up your Pi, you should enable and setup [SSH](https://www.raspberrypi.com/news/coding-on-raspberry-pi-remotely-with-visual-studio-code/) (Secure Shell Protocol) so you can login into the Pi from your laptop without requiring a screen, keyboard and mouse in the Pi. A great tutorial on how to do this can be found [here](https://www.raspberrypi.com/documentation/computers/remote-access.html#ssh). Logging into your Pi can be done in your Command Prompt (cmd) or if you use VSCode you can use [this](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension.

### Install LeRobot on Pi 🤗

On your Raspberry Pi install LeRobot using our [Installation Guide](./installation)

In addition to these instructions, you need to install the Feetech sdk on your Pi:
```bash
pip install -e ".[feetech]"
```

## Install LeRobot locally
If you already have install LeRobot on your laptop/pc you can skip this step, otherwise please follow along as we do the same steps we did on the Pi.

Follow our [Installation Guide](./installation)

Great :hugs:! You are now done installing LeRobot and we can begin assembling the SO100/SO101 arms and the mobile base :robot:.
Every time you now want to use LeRobot you can go to the `~/lerobot` folder where we installed LeRobot and run one of the commands.

# Step-by-Step Assembly Instructions

First we will assemble the two SO100/SO101 arms. One to attach to the mobile base and one for teleoperation. Then we will assemble the mobile base.

## SO100/SO101 Arms
### Configure motors
The instructions for configuring the motors can be found in the SO101 [docs](./so101#configure-the-motors). Besides the ID's for the arm motors we also need to set the motor ID's for the mobile base. These need to be in a specific order to work. Below an image of the motor ID's and motor mounting positions for the mobile base. Note that we only use one Motor Control board on LeKiwi. This means the motor ID's for the wheels are 7, 8 and 9.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/blob/main/lerobot/motor_ids.webp" alt="Motor ID's for mobile robot" title="Motor ID's for mobile robot" width="60%">

### Assemble arms
[Assemble arms instruction](./so101#step-by-step-assembly-instructions)

## Mobile base (LeKiwi)
[Assemble LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi)

TODO(pepijn): From here downwards,
    - Add the specific config for lekiwi on pi and locally and also for the wired version
    - Add calibration instructions
    - Integrate teleop, record etc somewhere here or in main

# Teleoperate

> [!TIP]
> If you're using a Mac, you might need to give Terminal permission to access your keyboard. Go to System Preferences > Security & Privacy > Input Monitoring and check the box for Terminal.

To teleoperate SSH into your Raspberry Pi, and run `conda activate lerobot` and this script:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

Then on your laptop, also run `conda activate lerobot` and this script:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=teleoperate \
  --control.fps=30
```

> **NOTE:** To visualize the data, enable `--control.display_data=true`. This streams the data using `rerun`. For the `--control.type=remote_robot` you will also need to set `--control.viewer_ip` and `--control.viewer_port`

You should see on your laptop something like this: ```[INFO] Connected to remote robot at tcp://172.17.133.91:5555 and video stream at tcp://172.17.133.91:5556.``` Now you can move the leader arm and use the keyboard (w,a,s,d) to drive forward, left, backwards, right. And use (z,x) to turn left or turn right. You can use (r,f) to increase and decrease the speed of the mobile robot. There are three speed modes, see the table below:

| Speed Mode | Linear Speed (m/s) | Rotation Speed (deg/s) |
| ---------- | ------------------ | ---------------------- |
| Fast       | 0.4                | 90                     |
| Medium     | 0.25               | 60                     |
| Slow       | 0.1                | 30                     |


| Key | Action         |
| --- | -------------- |
| W   | Move forward   |
| A   | Move left      |
| S   | Move backward  |
| D   | Move right     |
| Z   | Turn left      |
| X   | Turn right     |
| R   | Increase speed |
| F   | Decrease speed |

> [!TIP]
>  If you use a different keyboard you can change the keys for each command in the [`LeKiwiConfig`](../lerobot/common/robot_devices/robots/configs.py).

### Wired version
If you have the **wired** LeKiwi version please run all commands including both these teleoperation commands on your laptop.

## Troubleshoot communication

If you are having trouble connecting to the Mobile SO100, follow these steps to diagnose and resolve the issue.

### 1. Verify IP Address Configuration
Make sure that the correct ip for the Pi is set in the configuration file. To check the Raspberry Pi's IP address, run (on the Pi command line):
```bash
hostname -I
```

### 2. Check if Pi is reachable from laptop/pc
Try pinging the Raspberry Pi from your laptop:
```bach
ping <your_pi_ip_address>
```

If the ping fails:
- Ensure the Pi is powered on and connected to the same network.
- Check if SSH is enabled on the Pi.

### 3. Try SSH connection
If you can't SSH into the Pi, it might not be properly connected. Use:
```bash
ssh <your_pi_user_name>@<your_pi_ip_address>
```
If you get a connection error:
- Ensure SSH is enabled on the Pi by running:
  ```bash
  sudo raspi-config
  ```
  Then navigate to: **Interfacing Options -> SSH** and enable it.

### 4. Same config file
Make sure the configuration file on both your laptop/pc and the Raspberry Pi is the same.

# Record a dataset
Once you're familiar with teleoperation, you can record your first dataset with LeKiwi.

To start the program on LeKiwi, SSH into your Raspberry Pi, and run `conda activate lerobot` and this script:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

If you want to use the Hugging Face hub features for uploading your dataset and you haven't previously done it, make sure you've logged in using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Store your Hugging Face repository name in a variable to run these commands:
```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```
On your laptop then run this command to record 2 episodes and upload your dataset to the hub:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/lekiwi_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=true
```

Note: You can resume recording by adding `--control.resume=true`.

### Wired version
If you have the **wired** LeKiwi version please run all commands including both these record dataset commands on your laptop.

# Visualize a dataset

If you uploaded your dataset to the hub with `--control.push_to_hub=true`, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset) by copy pasting your repo id given by:
```bash
echo ${HF_USER}/lekiwi_test
```

If you didn't upload with `--control.push_to_hub=false`, you can also visualize it locally with (a window can be opened in the browser `http://127.0.0.1:9090` with the visualization tool):
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/lekiwi_test \
  --local-files-only 1
```

# Replay an episode
Now try to replay the first episode on your robot:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/lekiwi_test \
  --control.episode=0
```

## Train a policy

To train a policy to control your robot, use the [`python lerobot/scripts/train.py`](../lerobot/scripts/train.py) script. A few arguments are required. Here is an example command:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/lekiwi_test \
  --policy.type=act \
  --output_dir=outputs/train/act_lekiwi_test \
  --job_name=act_lekiwi_test \
  --policy.device=cuda \
  --wandb.enable=true
```

Let's explain it:
1. We provided the dataset as argument with `--dataset.repo_id=${HF_USER}/lekiwi_test`.
2. We provided the policy with `policy.type=act`. This loads configurations from [`configuration_act.py`](../lerobot/common/policies/act/configuration_act.py). Importantly, this policy will automatically adapt to the number of motor states, motor actions and cameras of your robot (e.g. `laptop` and `phone`) which have been saved in your dataset.
4. We provided `policy.device=cuda` since we are training on a Nvidia GPU, but you could use `policy.device=mps` to train on Apple silicon.
5. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.

Training should take several hours. You will find checkpoints in `outputs/train/act_lekiwi_test/checkpoints`.

## Evaluate your policy

You can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) but with a policy checkpoint as input. For instance, run this command to record 10 evaluation episodes:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Drive to the red block and pick it up" \
  --control.repo_id=${HF_USER}/eval_act_lekiwi_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_lekiwi_test/checkpoints/last/pretrained_model
```

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:
1. There is an additional `--control.policy.path` argument which indicates the path to your policy checkpoint with  (e.g. `outputs/train/eval_act_lekiwi_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `${HF_USER}/act_lekiwi_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `${HF_USER}/eval_act_lekiwi_test`).
