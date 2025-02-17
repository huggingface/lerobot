# Using the Mobile SO100: [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) with LeRobot

## Table of Contents

  - [A. Source the parts](#a-source-the-parts)
  - [B. Install LeRobot](#b-install-lerobot)
  - [C. Configure the motors](#c-configure-the-motors)
  - [D. Assemble the arms](#d-assemble-the-arms)
  - [E. Calibrate](#e-calibrate)
  - [F. Teleoperate](#f-teleoperate)
  - [G. Record a dataset](#g-record-a-dataset)
  - [H. Visualize a dataset](#h-visualize-a-dataset)
  - [I. Replay an episode](#i-replay-an-episode)
  - [J. Train a policy](#j-train-a-policy)
  - [K. Evaluate your policy](#k-evaluate-your-policy)

> [!TIP]
>  If you have any questions or need help, please reach out on Discord in the channel [`#lerobot`](https://discord.com/invite/s3KuuzsPFb).

## A. Source the parts

Follow this [README](https://github.com/SIGRobotics-UIUC/LeKiwi). It contains the bill of materials, with a link to source the parts, as well as the instructions to 3D print the parts, and advice if it's your first time printing or if you don't own a 3D printer.

Before assembling, you will first need to configure your motors. To this end, we provide a nice script, so let's first install LeRobot. After configuration, we will also guide you through assembly.

## B. Install software on Pi
Now we have to setup the remote PC that will run on the MobileSO100. This is normally a Raspberry Pi, but can be any PC that can run on 5V and has enough usb ports (2 or more) for the cameras and motor control board.

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
The instructions for configuring the motors can be found [Here](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md#c-configure-the-motors) in step C of the SO100 tutorial.

### Assemble arms
[Assemble arms instruction](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md#d-assemble-the-arms)

## Mobile base (LeKiwi)
[Assemble LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi)

<img src="../media/tutorial/mobile_motor_ids.webp?raw=true" alt="Motor ID's for mobile robot" title="Motor ID's for mobile robot" width="60%">
TODO(pepijn): add instructions + image in LeKiwi repo on setting motorid's and on assembling base and on attaching arms

### Update config
Both config files on the MobileSO100 LeRobot and on the laptop should be the same. First we should find the ip address of the Raspberry Pi of the mobile manipulator. This is the same Ip address used in ssh. We also need the usb port of the leader arm on the laptop and the port of the follower arm on the mobile manipulator. We can find these ports with the following script.

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

IMPORTANTLY: Now that you have your ports of leader and follower arm and ip adress of the mobile-so100, update the **ip** in Network configuration, **port** in leader_arms and **port** in mobile_so100. In the [`SO100RobotConfig`](../lerobot/common/robot_devices/robots/configs.py) file. Where you will find something like:
```python
@RobotConfig.register_subclass("mobile_so100")
@dataclass
class MobileSO100RobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Network Configuration
    ip: str = "192.168.0.193"
    port: int = 5555
    video_port: int = 5556

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "mobile": OpenCVCameraConfig(
                camera_index="/dev/video0",
                fps=30,
                width=640,
                height=480
            ),
            "mobile2": OpenCVCameraConfig(
                camera_index="/dev/video2",
                fps=30,
                width=640,
                height=480
            ),
        }
    )

    calibration_dir: str = ".cache/calibration/so100"

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

    mobile_so100: dict[str, MotorsBusConfig] = field(
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
                    "wheel_1": (7, "sts3215"),
                    "wheel_2": (8, "sts3215"),
                    "wheel_3": (9, "sts3215"),
                },
            ),
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

TODO(pepijn): Add images from mobile base
| 1. Zero position | 2. Rotated position | 3. Rest position |
|---|---|---|
| <img src="../media/so100/follower_zero.webp?raw=true" alt="SO-100 follower arm zero position" title="SO-100 follower arm zero position" style="width:100%;"> | <img src="../media/so100/follower_rotated.webp?raw=true" alt="SO-100 follower arm rotated position" title="SO-100 follower arm rotated position" style="width:100%;"> | <img src="../media/so100/follower_rest.webp?raw=true" alt="SO-100 follower arm rest position" title="SO-100 follower arm rest position" style="width:100%;"> |

Make sure the arm is connected to the Raspberry Pi and run this script (on the Raspberry Pi) to launch manual calibration:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=mobile_so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'
```

### Calibrate leader arm
Then to calibrate the leader arm (which is attached to the laptop/pc). You will need to move the leader arm to these positions sequentially:

| 1. Zero position | 2. Rotated position | 3. Rest position |
|---|---|---|
| <img src="../media/so100/leader_zero.webp?raw=true" alt="SO-100 leader arm zero position" title="SO-100 leader arm zero position" style="width:100%;"> | <img src="../media/so100/leader_rotated.webp?raw=true" alt="SO-100 leader arm rotated position" title="SO-100 leader arm rotated position" style="width:100%;"> | <img src="../media/so100/leader_rest.webp?raw=true" alt="SO-100 leader arm rest position" title="SO-100 leader arm rest position" style="width:100%;"> |

Run this script (on your laptop/pc) to launch manual calibration:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=mobile_so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'
```

# F. Teleoperate
TODO: ssh into pi, start script to run run_mobile_so100.py. Then run teleoperate command on laptop.

## Troubleshoot communication
TODO: Add now to troubleshoot connection. Ping ip address, ssh into pi (to check ip), check if config file is the same. Make sure ip address is correctly set.

# G. Record a dataset

# H. Visualize a dataset

# I. Replay an episode

# J. Train a policy

# K. Evaluate your policy

# L. Automatically Run Pi
How to automatically run the software of pi on startup, run script on startup pi
