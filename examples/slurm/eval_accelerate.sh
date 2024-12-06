cd /home/mshukor/lerobot
export LC_ALL=C

export http_proxy=http://"192.168.0.100":"3128" 
export https_proxy=http://"192.168.0.100":"3128"

ENV=aloha
ENV_TASK=AlohaTransferCube-v0
dataset_repo_id=lerobot/aloha_sim_transfer_cube_human
policy=act

# ENV=pusht
# ENV_TASK=PushT-v0
# dataset_repo_id=lerobot/pusht
# policy=diffusion

GPU=1
USE_AMP=false

# TASK_NAME=lerobot_base_distributed_${ENV}_transfer_cube_${policy}_2gpus
# TASK_NAME=lerobot_base_distributed_${ENV}_transfer_cube_${policy}_2gpus_fixlrsched
# TASK_NAME=lerobot_base_distributed_${ENV}_transfer_cube_${policy}_1gpus
# TASK_NAME=lerobot_base_distributed_${ENV}_transfer_cube_${policy}_3gpus
# TASK_NAME=lerobot_base_distributed_${ENV}_transfer_cube_${policy}_4gpus
TASK_NAME=lerobot_base_distributed_${ENV}_transfer_cube_${policy}_1gpus_noaccelerate
# TASK_NAME=lerobot_base_distributed_${ENV}_transfer_cube_${policy}_1gpus_noaccelerate_useamp

echo /data/mshukor/logs/lerobot/${TASK_NAME} 

# python -m accelerate.commands.launch --mixed-precision=fp16 --num_processes=$GPU --main_process_port=29501  lerobot/scripts/eval.py \
# eval.n_episodes=20 eval.use_async_envs=true eval.batch_size=20 \
# --out-dir outputs/accelerate_eval/fp16 -p /data/mshukor/logs/lerobot/${TASK_NAME}/checkpoints/075000/pretrained_model 



### Pusht
GPUS=1
EVAL_FREQ=1 #51000 #10000 51000
OFFLINE_STEPS=200000 #25000 17000 12500 50000
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=50
LR_SCHEDULER=
LR=1e-4

python lerobot/scripts/train.py \
 hydra.job.name=base_distributed_aloha_transfer_cube \
 hydra.run.dir=/data/mshukor/logs/lerobot/${TASK_NAME} \
 dataset_repo_id=$dataset_repo_id \
 policy=$policy \
 env=$ENV env.task=$ENV_TASK \
 training.offline_steps=$OFFLINE_STEPS training.batch_size=$TRAIN_BATCH_SIZE \
 training.eval_freq=$EVAL_FREQ eval.n_episodes=50 eval.use_async_envs=true eval.batch_size=50 \
 training.lr_scheduler=$LR_SCHEDULER training.lr=$LR \
 wandb.enable=true use_amp=$USE_AMP device=cpu





### 
# 1gpus_noaccelerate_useamp: s: 45 s: 35 (with amp) s: 45 (accelerate fp16)
# _2gpus: s: 15 s: 25 (with amp) s: 15 (accelerate fp16)