#!/bin/bash

#SBATCH --job-name=lerobot_aloha_transfer_cube_vla_1gpus_noaccelerate
#SBATCH --nodes=1
#SBATCH --partition=hard
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
###SBATCH --nodelist=zz
#SBATCH --exclude=modjo
#SBATCH --output=/data/mshukor/logs/slurm/lerobot_aloha_transfer_cube_vla_1gpus_noaccelerate.out
#SBATCH --cpus-per-task=20
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr
 
cd /home/mshukor/lerobot
source ~/.bashrc
source activate lerobot
export LC_ALL=C

export http_proxy=http://"192.168.0.100":"3128" 
export https_proxy=http://"192.168.0.100":"3128"


ENV=aloha
ENV_TASK=AlohaTransferCube-v0
dataset_repo_id=lerobot/aloha_sim_transfer_cube_human


# policy=act
# LR=1e-5
# LR_SCHEDULER=
# USE_AMP=false
# ASYNC_ENV=false

policy=vla
LR=1e-5
LR_SCHEDULER=
USE_AMP=true
ASYNC_ENV=false


# TASK_NAME=lerobot_${ENV}_transfer_cube_${policy}_2gpus
# TASK_NAME=lerobot_${ENV}_transfer_cube_${policy}_2gpus_noamp
# TASK_NAME=lerobot_${ENV}_transfer_cube_${policy}_1gpus_noaccelerate
TASK_NAME=lerobot_${ENV}_transfer_cube_${policy}_1gpus_noaccelerate_useamp


# #### Aloha
GPUS=1
EVAL_FREQ=10000 #51000 #10000 51000
OFFLINE_STEPS=100000 #25000 17000 12500 50000
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=50

# GPUS=2
# EVAL_FREQ=10000 #51000 #10000 51000
# OFFLINE_STEPS=100000 #25000 17000 12500 50000
# TRAIN_BATCH_SIZE=4
# EVAL_BATCH_SIZE=50



# python -m accelerate.commands.launch --num_processes=$GPUS --mixed_precision=fp16 lerobot/scripts/train.py \
#  hydra.job.name=base_distributed_aloha_transfer_cube \
#  hydra.run.dir=/data/mshukor/logs/lerobot/${TASK_NAME} \
#  dataset_repo_id=$dataset_repo_id \
#  policy=$policy \
#  env=$ENV env.task=$ENV_TASK \
#  training.offline_steps=$OFFLINE_STEPS training.batch_size=$TRAIN_BATCH_SIZE \
#  training.eval_freq=$EVAL_FREQ eval.n_episodes=50 eval.use_async_envs=$ASYNC_ENV eval.batch_size=$EVAL_BATCH_SIZE \
#  training.lr_scheduler=$LR_SCHEDULER training.lr=$LR \
#  wandb.enable=true 



MUJOCO_GL=egl python lerobot/scripts/train.py \
 hydra.job.name=base_distributed_aloha_transfer_cube \
 hydra.run.dir=/data/mshukor/logs/lerobot/${TASK_NAME} \
 dataset_repo_id=$dataset_repo_id \
 policy=$policy \
 env=$ENV env.task=$ENV_TASK \
 training.offline_steps=$OFFLINE_STEPS training.batch_size=$TRAIN_BATCH_SIZE \
 training.eval_freq=$EVAL_FREQ eval.n_episodes=50 eval.use_async_envs=$ASYNC_ENV eval.batch_size=$EVAL_BATCH_SIZE \
 training.lr=$LR \
 wandb.enable=true use_amp=$USE_AMP


