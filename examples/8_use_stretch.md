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

6. Install LeRobot
```bash
cd ~/lerobot && pip install -e ".[stretch]"

conda install -y -c conda-forge ffmpeg
pip uninstall -y opencv-python
conda install -y -c conda-forge "opencv>=4.10.0"
```

## Teleoperate, record a dataset and run a policy

> **Note:** As indicated in Stretch's [doc](https://docs.hello-robot.com/0.3/getting_started/stretch_hardware_overview/#turning-off-gamepad-teleoperation), you may need to free the "robot process" after booting Stretch by running `stretch_free_robot_process.py`

Before operating Stretch, you need to [home](https://docs.hello-robot.com/0.3/getting_started/stretch_hardware_overview/#homing) it first. In the scripts used below, if the robot is not already homed it will be automatically homed first. Be mindful about giving Stretch some space as this procedure will move the robot's arm and gripper. If you want to "manually" home Stretch first, you can simply run this command:
```bash
python lerobot/scripts/control_robot.py calibrate \
    --robot-path lerobot/configs/robot/stretch.yaml
```
This is equivalent to running `stretch_robot_home.py`

Try out teleoperation (you can learn about the controls in Stretch's [documentation](https://docs.hello-robot.com/0.3/getting_started/hello_robot/#gamepad-teleoperation)):
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --robot-path lerobot/configs/robot/stretch.yaml
```
This is essentially the same as running `stretch_gamepad_teleop.py`


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
python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/stretch.yaml \
    --fps 30 \
    --root data \
    --repo-id ${HF_USER}/stretch_test \
    --tags stretch tutorial \
    --warmup-time-s 3 \
    --episode-time-s 40 \
    --reset-time-s 10 \
    --num-episodes 1 \
    --push-to-hub 0
```

Note that if you're using ssh to connect to Stretch and run this script, you won't be able to visualize its cameras feed (though they'll still be recording).

Now try to replay this episode (make sure the robot's initial position is the same):
```bash
python lerobot/scripts/control_robot.py replay \
    --robot-path lerobot/configs/robot/stretch.yaml \
    --fps 30 \
    --root data \
    --repo-id ${HF_USER}/stretch_test \
    --episode 0
```
