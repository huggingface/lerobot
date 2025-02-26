

cd ~/lerobot
source ~/.bashrc
source activate lerobot_master

rm core-*

export TRANSFORMERS_CACHE=$WORK/.cache/huggingface/transformers
export HF_HOME=$WORK/.cache/huggingface
export DATA_DIR=$WORK/.cache/huggingface/datasets
export HF_LEROBOT_HOME=$WORK/.cache/huggingface/lerobot

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export WANDB_CACHE_DIR=/lustre/fsn1/projects/rech/dyf/ugz83ue/wandb
export WANDB_MODE=offline

## then later
## wandb sync wandb/offline-run-*

POLICY=act

ENV=aloha
TASK=AlohaTransferCube-v0
REPO_ID=lerobot/aloha_sim_transfer_cube_human
DATASET_NAME=aloha_sim_transfer_cube_human


TASK_NAME=lerobot_${DATASET_NAME}_${POLICY}_gpus${GPUS}

TRAIN_DIR=$WORK/logs/lerobot/$TASK_NAME
echo $TRAIN_DIR

USE_AMP=false

PORT=29502

EVAL_BATCH_SIZE=10
# EVAL_FREQ=5000 #51000 #10000 51000
SAVE_FREQ=200


EVAL_FREQ=1000

GPUS=2
OFFLINE_STEPS=100000 #25000 17000 12500 50000
BATCH_SIZE=8


export MUJOCO_GL=egl

python -m accelerate.commands.launch --num_processes=$GPUS --mixed_precision=fp16 --main_process_port=$PORT lerobot/scripts/train.py \
     --policy.type=$POLICY  \
     --dataset.repo_id=$REPO_ID \
     --env.type=$ENV \
     --env.task=$TASK \
     --output_dir=$TRAIN_DIR \
     --batch_size=$BATCH_SIZE \
     --steps=$OFFLINE_STEPS \
     --eval_freq=$EVAL_FREQ --save_freq=$SAVE_FREQ --eval.batch_size=$EVAL_BATCH_SIZE --eval.n_episodes=$EVAL_BATCH_SIZE  \
     --use_amp=$USE_AMP
