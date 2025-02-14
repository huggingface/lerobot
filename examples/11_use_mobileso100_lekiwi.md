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

First we will assemble the two SO100 arms. One to attach to the mobile base and one for teleoperation. Then we will assemble the mobile base

## SO100 Arms
### Configure motors
The instructions for configuring the motors can be found [Here](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md#c-configure-the-motors) in step C of the SO100 tutorial.


### Assemble arms
[Assemble arms instruction](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md#d-assemble-the-arms)

## Mobile base (LeKiwi)
[Assemble LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi)
TODO: add instructions in LeKiwi repo on setting motorid's and on assembling base and on attaching arms
TODO(pepijn): Add explaination about which motor id on base should be which etc...

# E. Calibration

## Calibrate arms
TODO: set same config on Pi and laptop, explain config setup and explain calibration process

## Setup communication and camera's

### Setuup communication
TODO: set ip address laptop, make sure they are on same network (pi and mobilerobot) + bonus add small part on how to switch networks on pi

### Setup camera's
TODO: set same config on Pi and laptop, attach camera + explain how to find camera's that are available on linux (Pi)

# F. Teleoperate

# G. Record a dataset

# H. Visualize a dataset

# I. Replay an episode

# J. Train a policy

# K. Evaluate your policy

# L. Automatically Run Pi
How to automatically run the software of pi on startup
