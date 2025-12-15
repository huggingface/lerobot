#!/bin/bash

# Setup conda


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $HOME/miniconda3 && rm /tmp/miniconda.sh && $HOME/miniconda3/bin/conda init bash && source ~/.bashrc
source ~/.bashrc && conda --version
eval "$($HOME/miniconda3/bin/conda shell.bash hook)" && conda --version
eval "$($HOME/miniconda3/bin/conda shell.bash hook)" && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


eval "$($HOME/miniconda3/bin/conda shell.bash hook)" && conda create -y -n lerobot-arena python=3.11

# create conda env
conda activate lerobot-arena


# ACTIVATE THE ENV AND THEN:

# install deps
conda install -y -c conda-forge ffmpeg=7.1.1
pip install uv # to get things done fast

# install isaac sim 5.1.0
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

# install isaac lab 2.3.0
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.0
./isaaclab.sh -i
cd ..

# install isaaclab arena
git clone https://github.com/isaac-sim/IsaacLab-Arena.git
# NOTE: we do not need the submodules
cd IsaacLab-Arena
uv pip install -e .
uv pip install qpsolvers==4.8.1
uv pip install onnxruntime lightwheel-sdk vuer[all]
cd ..


# install lerobot from source
git clone https://github.com/liorbenhorin/lerobot.git
cd lerobot
git checkout dev-arena
uv pip install -e .
uv pip install -e ".[smolvla]"
uv pip install -e ".[pi]"

# fix numpy 
uv pip install numpy==1.26.0 lxml==4.9.4 packaging==23.2
