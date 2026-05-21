# LeRobot Setup Guide

A setup guide for running LeRobot with HIL-SERL (Human-in-the-Loop Sample-Efficient Reinforcement Learning) and the `gym-hil` simulation environment.

## Prerequisites

- Ubuntu 22.04
- `sudo` privileges for installing system packages
- Git

## Installation

### 1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3.sh
chmod +x miniconda3.sh
./miniconda3.sh -b
conda config --set auto_activate_base false
```

### 2. Create and Activate the Conda Environment

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
```

### 3. Install FFmpeg

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

### 4. Install System Dependencies

```bash
sudo apt-get install cmake build-essential python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev
```

### 5. Clone the Repositories

```bash
git clone https://github.com/VAlikV/lerobot.git
git clone https://github.com/syedjameel/gym-hil.git
```

### 6. Install LeRobot

```bash
cd lerobot
pip install -e .
pip install -e ".[hilserl]"
```

### 7. Patch `transformers`

Open the `modeling_utils.py` file:

```bash
gedit /home/<user>/miniconda3/envs/lerobot/lib/python3.12/site-packages/transformers/modeling_utils.py
```

> Replace `<user>` with your actual username.

Add the following line inside the `__init__()` method of the `PreTrainedModel` class:

```python
self.post_init()
```

### 8. Install Additional Dependencies

```bash
pip install matplotlib
```

## Running the Simulation

### 1. Install `gym-hil`

```bash
cd gym-hil
pip install -e .
```

### 2. Launch the Simulation Environment

```bash
cd ../lerobot
python -m lerobot.rl.gym_manipulator --config_path src/lerobot/rl/panda_sim_usb_env.json
```

Use this step to record your offline demonstrations before proceeding to training.

## Training

After recording the offline demonstrations, training requires **two terminals running in parallel**.

### Terminal 1 — Learner

```bash
conda activate lerobot
cd lerobot
python -m lerobot.rl.learner --config_path src/lerobot/rl/panda_sim_usb_train.json
```

### Terminal 2 — Actor

```bash
conda activate lerobot
cd lerobot
python -m lerobot.rl.actor --config_path src/lerobot/rl/panda_sim_usb_train.json
```

## Troubleshooting

- **`transformers` patch not applied**: If you encounter initialization errors related to `PreTrainedModel`, double-check that `self.post_init()` was added to the correct `__init__()` method in `modeling_utils.py`.
- **FFmpeg version conflicts**: Ensure that the conda-forge FFmpeg (7.1.1) is being used inside the activated environment, not the system FFmpeg.
- **Path issues**: All `python -m` commands should be run from the root of the `lerobot` repository.

## References

- [LeRobot (fork)](https://github.com/VAlikV/lerobot)
- [gym-hil](https://github.com/syedjameel/gym-hil)