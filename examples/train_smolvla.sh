#!/bin/bash


cd ~/lerobot
source ~/.bashrc
source activate lerobot_main

export TOKENIZERS_PARALLELISM=false

# V3 So100
REPO_ID=danaaubakirova/svla_so100_task1_v3
DATASET_NAME=so100_v3_task_1


POLICY=smolvla
POLICY_NAME=smolvla


OFFLINE_STEPS=200000
BATCH_SIZE=64


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
