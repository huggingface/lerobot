# Using the [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) Robot with LeRobot

## Table of Contents

  - [A. Source the parts](#a-source-the-parts)
  - [B. Install software Pi](#b-install-software-on-pi)
  - [C. Setup LeRobot laptop/pc](#c-install-lerobot-on-laptop)
  - [D. Assemble the arms](#d-assembly)
  - [E. Calibrate](#e-calibration)
  - [F. Teleoperate](#f-teleoperate)
  - [G. Record a dataset](#g-record-a-dataset)
  - [H. Visualize a dataset](#h-visualize-a-dataset)
  - [I. Replay an episode](#i-replay-an-episode)
  - [J. Train a policy](#j-train-a-policy)
  - [K. Evaluate your policy](#k-evaluate-your-policy)

> [!TIP]
>  If you have any questions or need help, please reach out on [Discord](https://discord.com/invite/s3KuuzsPFb) in the channel [`#mobile-so-100-arm`](https://discord.com/channels/1216765309076115607/1318390825528332371).

## A. Source the parts

Follow this [README](https://github.com/SIGRobotics-UIUC/LeKiwi). It contains the bill of materials, with a link to source the parts, as well as the instructions to 3D print the parts, and advice if it's your first time printing or if you don't own a 3D printer.

Before assembling, you will first need to configure your motors. To this end, we provide a nice script, so let's first install LeRobot. After configuration, we will also guide you through assembly.

### Wired version
If you have the **wired** LeKiwi version you can skip the installation of the Raspberry Pi and setting up SSH. You can also run all commands directly on your PC for both the LeKiwi scripts and the leader arm scripts for teleoperating.

## B. Install software on Pi
Now we have to setup the remote PC that will run on the LeKiwi Robot. This is normally a Raspberry Pi, but can be any PC that can run on 5V and has enough usb ports (2 or more) for the cameras and motor control board.

### Install OS
For setting up the Raspberry Pi and its SD-card see: [Setup PI](https://www.raspberrypi.com/documentation/computers/getting-started.html). Here is explained how to download the [Imager](https://www.raspberrypi.com/software/) to install Raspberry Pi OS or Ubuntu.

### Setup SSH
After setting up your Pi, you should enable and setup [SSH](https://www.raspberrypi.com/news/coding-on-raspberry-pi-remotely-with-visual-studio-code/) (Secure Shell Protocol) so you can login into the Pi from your laptop without requiring a screen, keyboard and mouse in the Pi. A great tutorial on how to do this can be found [here](https://www.raspberrypi.com/documentation/computers/remote-access.html#ssh). Logging into your Pi can be done in your Command Prompt (cmd) or if you use VSCode you can use [this](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension.

### Install LeRobot

On your Raspberry Pi:

#### 1. [Install Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install):

#### 2. Restart shell
Copy paste in your shell: `source ~/.bashrc` or for Mac: `source ~/.bash_profile` or `source ~/.zshrc` if you're using zshell

#### 3. Create and activate a fresh conda environment for lerobot

<details>
<summary><strong>Video install instructions</strong></summary>

<video src="https://github.com/user-attachments/assets/17172d3b-3b64-4b80-9cf1-b2b7c5cbd236"></video>

</details>

```bash
conda create -y -n lerobot python=3.10
```

Then activate your conda environment (do this each time you open a shell to use lerobot!):
```bash
conda activate lerobot
```

#### 4. Clone LeRobot:
```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
```

#### 5. Install LeRobot with dependencies for the feetech motors:
```bash
cd ~/lerobot && pip install -e ".[feetech]"
```

## C. Install LeRobot on laptop
If you already have install LeRobot on your laptop you can skip this step, otherwise please follow along as we do the same steps we did on the Pi.

> [!TIP]
> We use the Command Prompt (cmd) quite a lot. If you are not comfortable using the cmd or want to brush up using the command line you can have a look here: [Command line crash course](https://developer.mozilla.org/en-US/docs/Learn_web_development/Getting_started/Environment_setup/Command_line)

On your computer:

#### 1. [Install Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install):

#### 2. Restart shell
Copy paste in your shell: `source ~/.bashrc` or for Mac: `source ~/.bash_profile` or `source ~/.zshrc` if you're using zshell

#### 3. Create and activate a fresh conda environment for lerobot

<details>
<summary><strong>Video install instructions</strong></summary>

<video src="https://github.com/user-attachments/assets/17172d3b-3b64-4b80-9cf1-b2b7c5cbd236"></video>

</details>

```bash
conda create -y -n lerobot python=3.10
```

Then activate your conda environment (do this each time you open a shell to use lerobot!):
```bash
conda activate lerobot
```

#### 4. Clone LeRobot:
```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
```

#### 5. Install LeRobot with dependencies for the feetech motors:
```bash
cd ~/lerobot && pip install -e ".[feetech]"
```

*EXTRA: For Linux only (not Mac)*: install extra dependencies for recording datasets:
```bash
conda install -y -c conda-forge ffmpeg
pip uninstall -y opencv-python
conda install -y -c conda-forge "opencv>=4.10.0"
```
Great :hugs:! You are now done installing LeRobot and we can begin assembling the SO100 arms and Mobile base :robot:.
Every time you now want to use LeRobot you can go to the `~/lerobot` folder where we installed LeRobot and run one of the commands.

# D. Assembly

First we will assemble the two SO100 arms. One to attach to the mobile base and one for teleoperation. Then we will assemble the mobile base.

## SO100 Arms
### Configure motors
The instructions for configuring the motors can be found [Here](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md#c-configure-the-motors) in step C of the SO100 tutorial. Besides the ID's for the arm motors we also need to set the motor ID's for the mobile base. These needs to be in a specific order to work. Below an image of the motor ID's and motor mounting positions for the mobile base. Note that we only use one Motor Control board on LeKiwi. This means the motor ID's for the wheels are 7, 8 and 9.

<img src="../media/lekiwi/motor_ids.webp?raw=true" alt="Motor ID's for mobile robot" title="Motor ID's for mobile robot" width="60%">

### Assemble arms
[Assemble arms instruction](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md#d-assemble-the-arms)

## Mobile base (LeKiwi)
[Assemble LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi)

### Update config
Both config files on the LeKiwi LeRobot and on the laptop should be the same. First we should find the Ip address of the Raspberry Pi of the mobile manipulator. This is the same Ip address used in SSH. We also need the usb port of the control board of the leader arm on the laptop and the port of the control board on LeKiwi. We can find these ports with the following script.

#### a. Run the script to find port

<details>
<summary><strong>Video finding port</strong></summary>
  <video src="https://github.com/user-attachments/assets/4a21a14d-2046-4805-93c4-ee97a30ba33f"></video>
  <video src="https://github.com/user-attachments/assets/1cc3aecf-c16d-4ff9-aec7-8c175afbbce2"></video>
</details>

To find the port for each bus servo adapter, run the utility script:
```bash
python lerobot/scripts/find_motors_bus_port.py
```

#### b. Example outputs

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

#### c. Troubleshooting
On Linux, you might need to give access to the USB ports by running:
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

#### d. Update config file

IMPORTANTLY: Now that you have your ports of leader and follower arm and ip address of the mobile-so100, update the **ip** in Network configuration, **port** in leader_arms and **port** in lekiwi. In the [`LeKiwiRobotConfig`](../lerobot/common/robot_devices/robots/configs.py) file. Where you will find something like:
```python
@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Network Configuration
    ip: str = "172.17.133.91"
    port: int = 5555
    video_port: int = 5556

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "mobile": OpenCVCameraConfig(camera_index="/dev/video0", fps=30, width=640, height=480),
            "mobile2": OpenCVCameraConfig(camera_index="/dev/video2", fps=30, width=640, height=480),
        }
    )

    calibration_dir: str = ".cache/calibration/lekiwi"

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0077581",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                    "left_wheel": (7, "sts3215"),
                    "back_wheel": (8, "sts3215"),
                    "right_wheel": (9, "sts3215"),
                },
            ),
        }
    )

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    mock: bool = False
```

## Wired version

For the wired LeKiwi version your configured IP address should refer to your own laptop (127.0.0.1), because leader arm and LeKiwi are in this case connected to own laptop. Below and example configuration for this wired setup:
```python
@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Network Configuration
    ip: str = "127.0.0.1"
    port: int = 5555
    video_port: int = 5556

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                camera_index=0, fps=30, width=640, height=480, rotation=90
            ),
            "wrist": OpenCVCameraConfig(
                camera_index=1, fps=30, width=640, height=480, rotation=180
            ),
        }
    )

    calibration_dir: str = ".cache/calibration/lekiwi"

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0077581",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431061",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                    "left_wheel": (7, "sts3215"),
                    "back_wheel": (8, "sts3215"),
                    "right_wheel": (9, "sts3215"),
                },
            ),
        }
    )

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    mock: bool = False
```

# E. Calibration
Now we have to calibrate the leader arm and the follower arm. The wheel motors don't have to be calibrated.


### Calibrate follower arm (on mobile base)
> [!IMPORTANT]
> Contrarily to step 6 of the [assembly video](https://youtu.be/FioA2oeFZ5I?t=724) which illustrates the auto calibration, we will actually do manual calibration of follower for now.

You will need to move the follower arm to these positions sequentially:

| 1. Zero position                                                                                                                                                  | 2. Rotated position                                                                                                                                                        | 3. Rest position                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="../media/lekiwi/mobile_calib_zero.webp?raw=true" alt="SO-100 follower arm zero position" title="SO-100 follower arm zero position" style="width:100%;"> | <img src="../media/lekiwi/mobile_calib_rotated.webp?raw=true" alt="SO-100 follower arm rotated position" title="SO-100 follower arm rotated position" style="width:100%;"> | <img src="../media/lekiwi/mobile_calib_rest.webp?raw=true" alt="SO-100 follower arm rest position" title="SO-100 follower arm rest position" style="width:100%;"> |

Make sure the arm is connected to the Raspberry Pi and run this script (on the Raspberry Pi) to launch manual calibration:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'
```

### Wired version
If you have the **wired** LeKiwi version please run all commands including this calibration command on your laptop.

### Calibrate leader arm
Then to calibrate the leader arm (which is attached to the laptop/pc). You will need to move the leader arm to these positions sequentially:

| 1. Zero position                                                                                                                                       | 2. Rotated position                                                                                                                                             | 3. Rest position                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <img src="../media/so100/leader_zero.webp?raw=true" alt="SO-100 leader arm zero position" title="SO-100 leader arm zero position" style="width:100%;"> | <img src="../media/so100/leader_rotated.webp?raw=true" alt="SO-100 leader arm rotated position" title="SO-100 leader arm rotated position" style="width:100%;"> | <img src="../media/so100/leader_rest.webp?raw=true" alt="SO-100 leader arm rest position" title="SO-100 leader arm rest position" style="width:100%;"> |

Run this script (on your laptop/pc) to launch manual calibration:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'
```

# F. Teleoperate
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
>  If you use a different keyboard you can change the keys for each command in the [`LeKiwiRobotConfig`](../lerobot/common/robot_devices/robots/configs.py).

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

# G. Record a dataset
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

# H. Visualize a dataset

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

# I. Replay an episode
Now try to replay the first episode on your robot:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/lekiwi_test \
  --control.episode=0
```

## J. Train a policy

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
2. We provided the policy with `policy.type=act`. This loads configurations from [`configuration_act.py`](../lerobot/common/policies/act/configuration_act.py). Importantly, this policy will automatically adapt to the number of motor sates, motor actions and cameras of your robot (e.g. `laptop` and `phone`) which have been saved in your dataset.
4. We provided `policy.device=cuda` since we are training on a Nvidia GPU, but you could use `policy.device=mps` to train on Apple silicon.
5. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.

Training should take several hours. You will find checkpoints in `outputs/train/act_lekiwi_test/checkpoints`.

## K. Evaluate your policy

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
