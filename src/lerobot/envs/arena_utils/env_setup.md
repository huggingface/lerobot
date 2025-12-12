# create conda env
conda create -y -n lerobot-arena python=3.11

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
git clone git@github.com:huggingface/collab-lerobot.git
cd collab-lerobot
git checkout dev-arena
uv pip install -e .

# fix numpy 
uv pip install numpy==1.26.0 lxml==4.9.4 packaging==23.2


