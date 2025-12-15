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
git clone https://github.com/liorbenhorin/lerobot.git
cd lerobot
git checkout dev-arena
uv pip install -e .
uv pip install -e ".[smolvla]"
uv pip install -e ".[pi]"

# fix numpy 
uv pip install numpy==1.26.0 lxml==4.9.4 packaging==23.2


# Converting HDF5 to LeRobot Dataset

```bash
mkdir -p /tmp/data
export DATASET_DIR=/tmp/data
hf download \
   nvidia/Arena-GR1-Manipulation-Task \
   arena_gr1_manipulation_dataset_generated.hdf5 \
   --repo-type dataset \
   --local-dir $DATASET_DIR

rm -r /home/ksachdev/.cache/huggingface/lerobot/nvkartik/arena_gr1_microwave_partial

# inspect HDF5 dataset
python src/lerobot/envs/arena_utils/inspect_hdf5.py --file_path $DATASET_DIR/arena_gr1_manipulation_dataset_generated.hdf5

export DATASET_DIR=/home/ksachdev/repos/data
python src/lerobot/envs/arena_utils/convert_hdf5_lerobot.py \
    --hdf5-path $DATASET_DIR/arena_gr1_manipulation_dataset_generated.hdf5 \
    --repo-id nvkartik/Arena-GR1-Manipulation-Task \
    --task-name "Open microwave door" \
    --push-to-hub
```

# Train Policy

```bash
# smolvla
# pi05
rm -r outputs/train/my_smolvla 
lerobot-train \
--policy.type=pi05 \
--dataset.repo_id nvkartik/Arena-GR1-Manipulation-Task   \
--rename_map '{"observation.images.robot_pov_cam":"observation.images.camera1"}' \
--policy.empty_cameras=2 \
--policy.max_state_dim=64 \
--policy.max_action_dim=64 \
--batch_size=16   \
--steps=20000   \
--output_dir=outputs/train/pi05   \
--job_name=my_smolvla_training   \
--policy.device=cuda   \
--wandb.enable=true   \
--policy.push_to_hub=false

# Workshop
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id sreetz-nv/so101_teleop_ring_stack   \
--batch_size=16   \
--steps=20000   \
--output_dir=outputs/train/smolvla_sim2real   \
--job_name=my_smolvla_training   \
--policy.device=cuda   \
--wandb.enable=true   \
--policy.push_to_hub=false

--policy.empty_cameras=2 \
--policy.max_state_dim=64 \
--policy.max_action_dim=64 \
```
