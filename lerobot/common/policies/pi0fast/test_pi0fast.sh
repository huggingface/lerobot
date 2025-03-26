cd ~/lerobot
source ~/miniconda3/bin/activate
conda activate lerobot

export MUJOCO_GL=egl

ENV=aloha
TASK=AlohaTransferCube-v0
REPO_ID=lerobot/aloha_sim_transfer_cube_human
OUT_DIR=~/logs/lerobot/tmp/act_aloha_transfer

EVAL_FREQ=5000
OFFLINE_STEP=30000
SAVE_FREQ=100000

POLICY_PATH=~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base_pytorch/
POLICY=pi0fast

python lerobot/scripts/train.py \
    --policy.path=$POLICY_PATH \
    --dataset.repo_id=$REPO_ID \
    --env.type=$ENV \
    --env.task=$TASK \
    --output_dir=$OUT_DIR \
    --eval_freq=$EVAL_FREQ \
    --steps=$OFFLINE_STEP \
    --save_freq=$SAVE_FREQ