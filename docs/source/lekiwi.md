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

First we will assemble the two SO100/SO101 arms. One to attach to the mobile base and one for teleoperation. Then we will assemble the mobile base. The instructions for assembling can be found on these two pages:

- [Assemble SO101](./so101#step-by-step-assembly-instructions)
- [Assemble LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/Assembly.md)

### Configure motors
The instructions for configuring the motors can be found in the SO101 [docs](./so101#configure-the-motors). Besides the ID's for the arm motors we also need to set the motor ID's for the mobile base. These need to be in a specific order to work. Below an image of the motor ID's and motor mounting positions for the mobile base. Note that we only use one Motor Control board on LeKiwi. This means the motor ID's for the wheels are 7, 8 and 9.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/blob/main/lerobot/motor_ids.webp" alt="Motor ID's for mobile robot" title="Motor ID's for mobile robot" width="60%">

### Troubleshoot communication

If you are having trouble connecting to the Mobile SO100, follow these steps to diagnose and resolve the issue.

#### 1. Verify IP Address Configuration
Make sure that the correct ip for the Pi is used in the commands or in your code. To check the Raspberry Pi's IP address, run (on the Pi command line):
```bash
hostname -I
```

#### 2. Check if Pi is reachable from laptop/pc
Try pinging the Raspberry Pi from your laptop:
```bach
ping <your_pi_ip_address>
```

If the ping fails:
- Ensure the Pi is powered on and connected to the same network.
- Check if SSH is enabled on the Pi.

#### 3. Try SSH connection
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

### Calibration

Now we have to calibrate the leader arm and the follower arm. The wheel motors don't have to be calibrated.
The calibration process is very important because it allows a neural network trained on one robot to work on another.

### Calibrate follower arm (on mobile base)

Make sure the arm is connected to the Raspberry Pi and run this script or API example (on the Raspberry Pi via ssh) to launch calibration of the follower arm:
<hfoptions id="calibrate_follower">
<hfoption id="Command">
```bash
python -m lerobot.calibrate \
    --robot.type=lekiwi \
    --robot.port=/dev/ttyACM0 \ # <- The port of your robot
    --robot.id=my_awesome_kiwi # <- Give the robot a unique name
```
</hfoption>
<hfoption id="API example">
```python
from lerobot.common.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig

config = LeKiwiClientConfig(
    remote_ip="192.168.0.23",
    id="my_awesome_kiwi",
)

lekiwi = LeKiwiClient(config)
lekiwi.connect(calibrate=False)
lekiwi.calibrate()
lekiwi.disconnect()
```
</hfoption>
</hfoptions>

We unified the calibration method for most robots, thus the calibration steps for this SO100 arm are the same as the steps for the Koch and SO101. First we have to move the robot to position where each joint is in the middle of its range, then we press `Enter`. Secondly we move all joints thru their full range of motion. A video of this same process for the SO101 as reference can be found [here](http://localhost:5173/so101#calibration-video)

### Wired version
If you have the **wired** LeKiwi version please run all commands including this calibration command on your laptop.

### Calibrate leader arm
Then to calibrate the leader arm (which is attached to the laptop/pc). Run the following command of API example on your laptop:
<hfoptions id="calibrate_leader">
<hfoption id="Command">
```bash
python -m lerobot.calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \ # <- The port of your robot
    --teleop.id=my_awesome_leader_arm # <- Give the robot a unique name
```
</hfoption>
<hfoption id="API example">
```python
from lerobot.common.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader

config = SO100LeaderConfig(
    port="/dev/tty.usbmodem58760431551",
    id="my_awesome_leader_arm",
)

leader = SO100Leader(config)
leader.connect(calibrate=False)
leader.calibrate()
leader.disconnect()
```
</hfoption>
</hfoptions>

Congrats 🎉, your robot is all set to learn a task on its own. Start training it by following this tutorial: [Getting started with real-world robots](./getting_started_real_world_robot)

> [!TIP]
>  If you have any questions or need help, please reach out on [Discord](https://discord.com/invite/s3KuuzsPFb).
