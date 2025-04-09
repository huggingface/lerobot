This tutorial explains how to use [Stretch 3](https://hello-robot.com/stretch-3-product) with LeRobot.

## Setup

Familiarize yourself with Stretch by following its [tutorials](https://docs.hello-robot.com/0.3/getting_started/hello_robot/) (recommended).

To use LeRobot on Stretch, 3 options are available:
- [tethered setup](https://docs.hello-robot.com/0.3/getting_started/connecting_to_stretch/#tethered-setup)
- [untethered setup](https://docs.hello-robot.com/0.3/getting_started/connecting_to_stretch/#untethered-setup)
- ssh directly into Stretch (you will first need to install and configure openssh-server on stretch using one of the two above setups)


## Install LeRobot

On Stretch's CLI, follow these steps:

1. [Install Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install):
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

2. Comment out these lines in `~/.profile` (this can mess up paths used by conda and ~/.local/bin should already be in your PATH)
```
# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
```

3. Restart shell or `source ~/.bashrc`

4. Create and activate a fresh conda environment for lerobot
```bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
```

5. Clone LeRobot:
```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
```

6. When using `miniconda`, install `ffmpeg` in your environment:
```bash
conda install ffmpeg -c conda-forge
```

7. Install LeRobot with stretch dependencies:
```bash
cd ~/lerobot && pip install -e ".[stretch]"
```

> **Note:** If you get this message, you can ignore it: `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.`

8. Run a [system check](https://docs.hello-robot.com/0.3/getting_started/stretch_hardware_overview/#system-check) to make sure your robot is ready:
```bash
stretch_system_check.py
```

> **Note:** You may need to free the "robot process" after booting Stretch by running `stretch_free_robot_process.py`. For more info this Stretch's [doc](https://docs.hello-robot.com/0.3/getting_started/stretch_hardware_overview/#turning-off-gamepad-teleoperation).

You should get something like this:
```bash
For use with S T R E T C H (R) from Hello Robot Inc.
---------------------------------------------------------------------

Model = Stretch 3
Tool = DexWrist 3 w/ Gripper
Serial Number = stretch-se3-3054

---- Checking Hardware ----
[Pass] Comms are ready
[Pass] Actuators are ready
[Warn] Sensors not ready (IMU AZ = -10.19 out of range -10.1 to -9.5)
[Pass] Battery voltage is 13.6 V

---- Checking Software ----
[Pass] Ubuntu 22.04 is ready
[Pass] All APT pkgs are setup correctly
[Pass] Firmware is up-to-date
[Pass] Python pkgs are up-to-date
[Pass] ROS2 Humble is ready
```

## Teleoperate, record a dataset and run a policy

**Calibrate (Optional)**
Before operating Stretch, you need to [home](https://docs.hello-robot.com/0.3/getting_started/stretch_hardware_overview/#homing) it first. Be mindful about giving Stretch some space as this procedure will move the robot's arm and gripper. Now run this command:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=stretch \
    --control.type=calibrate
```
This is equivalent to running `stretch_robot_home.py`

> **Note:** If you run any of the LeRobot scripts below and Stretch is not properly homed, it will automatically home/calibrate first.

**Teleoperate**
Before trying teleoperation, you need activate the gamepad controller by pressing the middle button. For more info, see Stretch's [doc](https://docs.hello-robot.com/0.3/getting_started/hello_robot/#gamepad-teleoperation).

Now try out teleoperation (see above documentation to learn about the gamepad controls):

> **NOTE:** To visualize the data, enable `--control.display_data=true`. This streams the data using `rerun`.
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=stretch \
    --control.type=teleoperate
```
This is essentially the same as running `stretch_gamepad_teleop.py`

**Record a dataset**
Once you're familiar with the gamepad controls and after a bit of practice, you can try to record your first dataset with Stretch.

If you want to use the Hugging Face hub features for uploading your dataset and you haven't previously done it, make sure you've logged in using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Store your Hugging Face repository name in a variable to run these commands:
```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

Record one episode:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=stretch \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/stretch_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=true
```

> **Note:** If you're using ssh to connect to Stretch and run this script, you won't be able to visualize its cameras feed (though they will still be recording). To see the cameras stream, use [tethered](https://docs.hello-robot.com/0.3/getting_started/connecting_to_stretch/#tethered-setup) or [untethered setup](https://docs.hello-robot.com/0.3/getting_started/connecting_to_stretch/#untethered-setup).

**Replay an episode**
Now try to replay this episode (make sure the robot's initial position is the same):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=stretch \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/stretch_test \
  --control.episode=0
```

Follow [previous tutorial](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#4-train-a-policy-on-your-data) to train a policy on your data and run inference on your robot. You will need to adapt the code for Stretch.

> TODO(rcadene, aliberts): Add already setup environment and policy yaml configuration files

If you need help, please reach out on Discord in the channel `#stretch3-mobile-arm`.
