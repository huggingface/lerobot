#!/bin/bash

#SBATCH --job-name=lerobot_smolvla_test_task1_main_pretrained
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lustre/fswork/projects/rech/dyf/ugz83ue/logs/slurm/lerobot_smolvla_test_task1_main_pretrained.out
###SBATCH --nodelist=jean-zay-a101
#SBATCH --cpus-per-task=20
###SBATCH --exclusive
#SBATCH --time=40:00:00
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


##SBATCH --partition=gpu_p2
##SBATCH --qos=qos_gpu-t3
###SBATCH -C v100-32g
##SBATCH -A dyf@v100

##SBATCH --partition=gpu_p5
##SBATCH -C a100
###SBATCH -A dyf@a100
##SBATCH -A lqm@a100
##SBATCH --qos=qos_gpu_a100-dev 
##SBATCH --qos=qos_gpu_a100-t3 

#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH -A lqm@h100
##SBATCH --qos=qos_gpu_h100-dev 
#SBATCH --qos=qos_gpu_h100-t4 


cd ~/lerobot_pi
source ~/.bashrc
source activate lerobot_main
export LC_ALL=C

rm core-*

export TRANSFORMERS_CACHE=$WORK/.cache/huggingface/transformers
export HF_HOME=$WORK/.cache/huggingface
export DATA_DIR=$WORK/.cache/huggingface/datasets
export HF_LEROBOT_HOME=$WORK/.cache/huggingface/lerobot
# export LEROBOT_HOME=

export HF_DATASETS_OFFLINE=1 
export HF_HUB_OFFLINE=1

export WANDB_CACHE_DIR=/lustre/fsn1/projects/rech/dyf/ugz83ue/wandb
export WANDB_MODE=offline

export TOKENIZERS_PARALLELISM=false


cd ~/lerobot_pi

# ###### dgx
# source ~/miniconda3/bin/activate
# conda activate lerobot



# export WORK=/home/mustafa_shukor



# V3 So100
REPO_ID=danaaubakirova/svla_so100_task1_v3
DATASET_NAME=so100_v3_task_1


POLICY=smolvla
POLICY_NAME=smolvla


OFFLINE_STEPS=200000
BATCH_SIZE=64


# TASK_NAME=lerobot_${DATASET_NAME}_${POLICY_NAME}
# TRAIN_DIR=$WORK/logs/lerobot/$TASK_NAME
# echo $TRAIN_DIR
# rm -r $TRAIN_DIR
# python lerobot/scripts/train.py \
#      --policy.type=$POLICY  \
#      --dataset.repo_id=$REPO_ID \
#      --output_dir=$TRAIN_DIR \
#      --batch_size=$BATCH_SIZE \
#      --steps=$OFFLINE_STEPS



TASK_NAME=lerobot_${DATASET_NAME}_${POLICY_NAME}_pretrained
TRAIN_DIR=$WORK/logs/lerobot/$TASK_NAME
echo $TRAIN_DIR
rm -r $TRAIN_DIR
POLICY_PATH=/lustre/fswork/projects/rech/dyf/ugz83ue/logs/lerobot/lerobot_so100_community_v1_v2_v3clean2_smolpi0_lr1e-4bs64steps400000gpus4freeze32_imgtoktrue_cross_attn_gap1_vlml16_causalacttrue_sa2_smolvlm2500_nobs1_expw0.75_feat2_lrvlm1e-4_trans0true_decaylr2.5e-630000_camfalse_fps3030_idlefalse/checkpoints/280000/test_smolvla/
python lerobot/scripts/train.py \
     --policy.path=$POLICY_PATH  \
     --dataset.repo_id=$REPO_ID \
     --output_dir=$TRAIN_DIR \
     --batch_size=$BATCH_SIZE \
     --steps=$OFFLINE_STEPS

