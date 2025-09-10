#!/bin/bash

#SBATCH --job-name=lerobot_eval_smolpi0_libero_eval10ep_ca_sa2_16vlm_w075_smolvlm2b_lr7e5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lustre/fswork/projects/rech/dyf/ugz83ue/logs/slurm/lerobot_eval_smolpi0_libero_eval10ep_ca_sa2_16vlm_w075_smolvlm2b_lr7e5.out
###SBATCH --nodelist=jean-zay-a101
#SBATCH --cpus-per-task=45
###SBATCH --exclusive
#SBATCH --time=15:00:00
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
###SBATCH --qos=qos_gpu_h100-dev 
#SBATCH --qos=qos_gpu_h100-t3 

###SBATCH --begin=now+2hour

# cd ~/lerobot_pi
# source ~/.bashrc
# source activate lerobot
# export LC_ALL=C

# rm core-*
export CUDA_VISIBLE_DEVICES=3
# storage / caches
RAID=/raid/jade
export TRANSFORMERS_CACHE=$RAID/.cache/huggingface/transformers
export HF_HOME=$RAID/.cache/huggingface
export HF_DATASETS_CACHE=$RAID/.cache/huggingface/datasets
export HF_LEROBOT_HOME=$RAID/.cache/huggingface/lerobot
export WANDB_CACHE_DIR=$RAID/.cache/wandb
export TMPDIR=$RAID/.cache/tmp
mkdir -p $TMPDIR
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=3

PORT=29512

## then later
## wandb sync wandb/offline-run-*


ENV=libero

# TASK=libero_10
TASK=libero_spatial
# TASK=libero_spatial
# TASK=libero_10
# TASK=libero_spatial


POLICY_NAME=smolpi0

POLICY=smolpi0
ENV=libero






CKPT_KEYS_MAPPING=model._orig_mod.//model.
LOAD_VLM_WEIGHTS=true
PEFT_METHOD=freeze
SELF_ATTN_ONLY_ACTIONS=false
CAUSAL_ATTENTION_ON_HISTORY=false

PREDICT_RELATIVE_ACTIONS=false
RELATIVE_ACTIONS_MODE=first
SHUFFLE_CAMERA_POSITIONS=false

VLM_IMG_SIZE=-1
REGRESSION_LOSS=false


# ## Baseline for ablation study
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=max_length
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4_bs8_steps100000_gpus2_freeze32_onlyexpert_1act_promptfalse_imgtoktrue_nobs1_compiletrue_cross_attn_pref0_gap1_localimgfalse_reverseimgorderfalse_statetopreftrue/checkpoints/last/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=max_length
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4_bs8_steps100000_gpus2_freeze32_onlyexpert_1act_promptfalse_imgtoktrue_nobs1_compiletrue_cross_attn/checkpoints/last/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=false
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=max_length
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4_bs8_steps100000_gpus2_freeze32_onlyexpert_1act_promptfalse_imgtoktrue_nobs1_compiletrue_self_attn/checkpoints/last/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=self_attn
# STATE_TO_PREFIX=false
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_self_attn_gap1_localimgfalse_statetopreffalse_explay0_vlml0_causalacttrue_sa0/checkpoints/last/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=self_attn
# STATE_TO_PREFIX=false
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr5e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2250/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr8e-5bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm22b/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4_bs8_steps100000_gpus2_freeze32_onlyexpert_1act_promptfalse_imgtoktrue_nobs1_compiletrue_self_attn_pref0_gap1_localimgfalse_reverseimgorderfalse_statetopreftrue_toklongest_explay0_vlml0_causalacttrue/checkpoints/last/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=self_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa2/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4_bs8_steps100000_gpus2_freeze32_onlyexpert_1act_promptfalse_imgtoktrue_nobs1_compiletrue_cross_attn_pref0_gap1_localimgfalse_reverseimgorderfalse_statetopreffalse/checkpoints/last/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=false
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_self_attn_gap1_localimgfalse_statetopreffalse_explay0_vlml0_causalacttrue_sa0_smolvlm2500/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=self_attn
# STATE_TO_PREFIX=false
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_self_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=self_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=8
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml8_causalactfalse_sa0/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalactfalse_sa0/checkpoints/last/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=24
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml24_causalactfalse_sa0/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=100
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk100/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=30
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk30/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=10
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk10/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=1
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk1/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=16
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay16_vlml0_causalactfalse_sa0_smolvlm2500_chunk50_nobs1/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=2
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk50_nobs2/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=3
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4_bs8_steps100000_gpus2_freeze32_onlyexpert_1act_promptfalse_imgtoktrue_nobs3_compiletrue_cross_attn_pref0_gap1_localimgfalse_reverseimgorderfalse_statetopreftrue_toklongest/checkpoints/last/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="observation.state"
# N_OBS_STEPS=3
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk50_nobs3_paststates/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="observation.state,image"
# N_OBS_STEPS=3
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk50_nobs3_paststatesimgs/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct


# EXPERT_WIDTH_MULTIPLIER=1
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr9.5e-5bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk50_nobs1_expw1/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr9.5e-5bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk50_nobs1_expw0.75/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.25
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr2e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk50_nobs1_expw0.25/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="observation.state,image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalactfalse_sa0_smolvlm2500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="observation.state,image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs16steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalactfalse_sa0_smolvlm2500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct



# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="observation.state,image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalactfalse_sa0_smolvlm2500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM2-500M-Video-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr5e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm1250_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-256M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr8e-5bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm12b_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-Instruct



# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm1500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr8e-5bs8steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm1500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr8e-5bs8steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm1500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalactfalse_sa0_smolvlm2500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalactfalse_sa0_smolvlm2500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk50_nobs1_expw0.5_rep/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalactfalse_sa2_smolvlm2500_chunk50_nobs1_expw0.5_rep/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2full8_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-6/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2full8_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2full8_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# PEFT_METHOD=lora
# PEFT_TARGET_MODEL=text
# LORA_TARGET_MODULES=q_proj,v_proj,k_proj
# LORA_R=32
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2lora32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_loraqkv/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# PEFT_METHOD=lora
# PEFT_TARGET_MODEL=text
# LORA_TARGET_MODULES=q_proj,v_proj,k_proj
# LORA_R=32
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2lora32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-5_loraqkv/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# PEFT_METHOD=lora
# PEFT_TARGET_MODEL=text
# LORA_TARGET_MODULES=q_proj,v_proj,k_proj,up_proj,down_proj,gate_proj
# LORA_R=32
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2lora32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# PEFT_METHOD=lora
# PEFT_TARGET_MODEL=text
# LORA_TARGET_MODULES=q_proj,v_proj,k_proj,up_proj,down_proj,gate_proj
# LORA_R=32
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2lora32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# PEFT_METHOD=lora
# PEFT_TARGET_MODEL=text
# LORA_TARGET_MODULES=q_proj,v_proj,k_proj,up_proj,down_proj,gate_proj
# LORA_R=32
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2lora32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-6/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2lora32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-5/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_loraqkv/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs16steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs16steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs16steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtokfalse_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=false
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=true
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_saacttrue/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=max_length
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_saactfalse_droptrue_max_length/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=max_length
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_saactfalse_dropfalse_max_length/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=max_length
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr9.5e-5bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_saactfalse_dropfalse_max_length/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr9.5e-5bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_saactfalse_dropfalse_longest/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_saactfalse_dropfalse_longest/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_ptdroidfull/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_ptcomv3freeze/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_ptcomv1v2full/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_ptcomv1v2freeze/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr2e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans1true/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr2e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans3true/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_self_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans1false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=self_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_self_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalactfalse_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans1false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=self_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=false
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans1false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans1false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_ptcomv3freeze25_trans1false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_ptcomv3freeze50_trans1false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_ptcomv3freeze75_trans1false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_ptcomv3freeze100_trans1false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans6true/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans4true/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans7true/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans5true/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans2true/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans1true/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans8true/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans9true/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_ptcomv1v2full/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.25
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.25_lrvlm1e-4_longest_pt_trans0false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=1
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw1_lrvlm1e-4_longest_pt_trans0false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.25
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw0.25_lrvlm1e-4_longest_pt_trans0false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=1
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml0_causalacttrue_sa0_smolvlm2500_chunk50_nobs1_expw1_lrvlm1e-4_longest_pt_trans0false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="observation.state"
# N_OBS_STEPS=3
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs3statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image,observation.state"
# N_OBS_STEPS=3
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs3_expw0.75_lrvlm1e-4_longest_pt_trans0false/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image,observation.state"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr1e-5100000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image,observation.state"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr1e-530000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image,observation.state"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr5e-6100000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-6100000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0true_decaylr2.5e-630000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr1e-5200000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr5e-6200000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr1e-530000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-6200000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-6100000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr5e-6100000/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvla500base_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLA-500M-Base 



# PREDICT_RELATIVE_ACTIONS=true
# RELATIVE_ACTIONS_MODE=relative
# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relacttruerelative/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# PREDICT_RELATIVE_ACTIONS=true
# RELATIVE_ACTIONS_MODE=first
# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relacttruefirst/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvla500base_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLA-500M-Base 


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr8e-5bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-6100000_relactfalsefirst_camfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr6e-5bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-6100000_relactfalsefirst_camfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr4e-5bs8steps100000gpus2freeze32_imgtoktrue_cross_attn_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1statestrue_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-6100000_relactfalsefirst_camfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_ptcomv1v2freezebs64transv0_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans1true_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# REGRESSION_LOSS=true
# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs8steps100000gpus2freeze32_cross_attn_vlml0_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regtrue/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# REGRESSION_LOSS=true
# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regtrue/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-6100000_relactfalsefirst_camfalse_vim-1/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr2e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-6100000_relactfalsefirst_camfalse_vim-1/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr3e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-6100000_relactfalsefirst_camfalse_vim-1/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# REGRESSION_LOSS=true
# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr9e-5bs8steps100000gpus2freeze32_cross_attn_vlml0_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regtrue/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct



# REGRESSION_LOSS=true
# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.5
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=0
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr2e-4bs8steps100000gpus2freeze32_cross_attn_vlml0_sa0_smolvlm2500_chunk50_nobs1_expw0.5_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regtrue/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=0
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr9e-5bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2500_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-6100000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr5e-4bs8steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr8e-5bs8steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm22b_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr5e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm1250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr4e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm1250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr3e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm1250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr6e-5bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm12b_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm12b_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct



# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr9e-5bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm12b_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr7e-5bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm12b_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr8e-5bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm12b_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct


CAUSAL_ATTENTION_ON_HISTORY=true
SELF_ATTN_ONLY_ACTIONS=false
EXPERT_WIDTH_MULTIPLIER=0.75
PAST_OBS_KEYS="image"
N_OBS_STEPS=1
NUM_EXPERT_LAYERS=0
CHUNK_SIZE=50
NUM_VLM_LAYERS=16
PAD_LANG_TO=longest
EVAL_CKPT=/raid/jade/models/smolvlamust
ADD_IMAGE_TOKENS=true
ATTN_MODE=cross_attn
STATE_TO_PREFIX=true
CAUSAL_ACTION_ATTENTION_MASK=true
SELF_ATTN_EVERY_N_LAYERS=2
VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr6e-5bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm22b_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse_compilefalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr9e-5bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm22b_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse_compilefalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr8e-5bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm22b_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse_compilefalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm22b_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse_compilefalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct



# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr7e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm1250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr1e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse_compilefalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr3e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse_compilefalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr5e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse_compilefalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 


# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr7e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse_compilefalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr6e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse_compilefalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 

# CAUSAL_ATTENTION_ON_HISTORY=true
# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=$WORK/logs/lerobot/lerobot_physical_intelligence_libero_smolpi0_lr4e-4bs32steps100000gpus2freeze32_cross_attn_vlml16_sa2_smolvlm2250_chunk50_nobs1_expw0.75_lrvlm1e-4_longest_pt_trans0false_decaylr2.5e-630000_relactfalsefirst_camfalse_vim-1_regfalse_compilefalse/checkpoints/best/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct 


# # TASK=libero_spatial
# MULTITASK_EVAL=false
# N_EPISODES=50

# TASK=libero_spatial,libero_object,libero_goal,libero_10
MULTITASK_EVAL=true
# N_EPISODES=5
N_EPISODES=1

# MAX_PARRALLEL_TASKS=5
# MAX_PARRALLEL_TASKS=2
MAX_PARRALLEL_TASKS=1

# NUM_EVALS=2
# SEEDS=(1000 5000)
SEEDS=(5000)
# ACTION_STEPS_LIST=(1 10 30 50)
ACTION_STEPS_LIST=(1)
# ACTION_STEPS_LIST=(50)
TASK_LIST=(libero_spatial libero_object libero_goal libero_10)
TASK_LIST=(libero_spatial)
for SEED in "${SEEDS[@]}"; do
    for N_ACTION_STEPS in "${ACTION_STEPS_LIST[@]}"; do
        for TASK in "${TASK_LIST[@]}"; do
            echo "$TASK Evaluating: $EVAL_CKPT | N_ACTION_STEPS=$N_ACTION_STEPS | EVAL SEED=$SEED"
            python src/lerobot/scripts/eval.py \
                --output_dir=/raid/jade/logs/lerobot/tmp \
                --env.type=$ENV \
                --env.task=$TASK \
                --eval.batch_size=$N_EPISODES \
                --eval.n_episodes=$N_EPISODES \
                --seed=$SEED \
                --policy.use_amp=false \
                --policy.path=$EVAL_CKPT \
                --policy.n_action_steps=$N_ACTION_STEPS \
                --policy.checkpoint_path=$EVAL_CKPT \
                --env.multitask_eval=$MULTITASK_EVAL --env.max_parallel_tasks=$MAX_PARRALLEL_TASKS \
                --policy.add_image_special_tokens=$ADD_IMAGE_TOKENS \
                --policy.attention_mode=$ATTN_MODE \
                --policy.causal_action_attention_mask=$CAUSAL_ACTION_ATTENTION_MASK \
                --policy.state_to_prefix=$STATE_TO_PREFIX \
                --policy.self_attn_every_n_layers=$SELF_ATTN_EVERY_N_LAYERS \
                --policy.pad_language_to=$PAD_LANG_TO \
                --policy.load_vlm_weights=$LOAD_VLM_WEIGHTS \
                --policy.vlm_model_name=$VLM_NAME \
                --policy.num_vlm_layers=$NUM_VLM_LAYERS \
                --policy.chunk_size=$CHUNK_SIZE \
                --policy.n_obs_steps=$N_OBS_STEPS \
                --policy.past_obs_keys=$PAST_OBS_KEYS \
                --policy.num_expert_layers=$NUM_EXPERT_LAYERS \
                --policy.expert_width_multiplier=$EXPERT_WIDTH_MULTIPLIER \
                --policy.peft_method=$PEFT_METHOD \
                --policy.self_attn_only_actions=$SELF_ATTN_ONLY_ACTIONS \
                --policy.causal_attention_on_history=$CAUSAL_ATTENTION_ON_HISTORY \
                --policy.predict_relative_actions=$PREDICT_RELATIVE_ACTIONS --policy.relative_actions_mode=$RELATIVE_ACTIONS_MODE --policy.shuffle_camera_positions=$SHUFFLE_CAMERA_POSITIONS \
                --policy.vlm_img_size=$VLM_IMG_SIZE \
                --policy.regression_loss=$REGRESSION_LOSS 
                # --policy.peft_config.r=$LORA_R --policy.peft_config.target_modules=$LORA_TARGET_MODULES  --policy.peft_method=$PEFT_METHOD --policy.peft_target_model=$PEFT_TARGET_MODEL 

            echo "Done with: $EVAL_CKPT | Steps=$N_ACTION_STEPS | EVAL SEED=$SEED"
            echo "------------------------------------------------------"
        done
    done
done


# ############################################################################################################################################
# ############################################################################################################################################
# ############################################################################################################################################
# ########### Offline eval


# # ############################
# # # Community datasets V1
# # # REPO_ID=pranavsaroha/so100_legos4,pranavsaroha/so100_onelego2,jpata/so100_pick_place_tangerine,pranavsaroha/so100_onelego3,pranavsaroha/so100_carrot_2,pranavsaroha/so100_carrot_5,pandaRQ/pick_med_1,HITHY/so100_strawberry,vladfatu/so100_above,koenvanwijk/orange50-1,koenvanwijk/orange50-variation-2,FeiYjf/new_GtoR,CSCSXX/pick_place_cube_1.18,vladfatu/so100_office,dragon-95/so100_sorting,dragon-95/so100_sorting_1,nbaron99/so100_pick_and_place4,Beegbrain/pick_place_green_block,Ityl/so100_recording2,dragon-95/so100_sorting_2,dragon-95/so100_sorting_3,aractingi/push_cube_offline_data,HITHY/so100_peach3,HITHY/so100_peach4,shreyasgite/so100_legocube_50,shreyasgite/so100_base_env,triton7777/so100_dataset_mix,Deason11/Open_the_drawer_to_place_items,Deason11/PLACE_TAPE_PUSH_DRAWER,NONHUMAN-RESEARCH/SOARM100_TASK_VENDA,mikechambers/block_cup_14,samsam0510/tooth_extraction_3,samsam0510/tooth_extraction_4,samsam0510/cube_reorientation_2,samsam0510/cube_reorientation_4,samsam0510/glove_reorientation_1,DorayakiLin/so100_pick_charger_on_tissue,zijian2022/noticehuman3,liuhuanjim013/so100_th
# # # Inconsistent actions dim: Deason11/Open_the_drawer_to_place_items, Deason11/PLACE_TAPE_PUSH_DRAWER
# # # Filtered datasets
# # REPO_ID=pranavsaroha/so100_onelego2,pranavsaroha/so100_onelego3,pranavsaroha/so100_carrot_2,vladfatu/so100_above,koenvanwijk/orange50-1,CSCSXX/pick_place_cube_1.18,dragon-95/so100_sorting,dragon-95/so100_sorting_1,nbaron99/so100_pick_and_place4,Beegbrain/pick_place_green_block,dragon-95/so100_sorting_3,HITHY/so100_peach3,shreyasgite/so100_legocube_50,triton7777/so100_dataset_mix,NONHUMAN-RESEARCH/SOARM100_TASK_VENDA,mikechambers/block_cup_14,samsam0510/tooth_extraction_3,samsam0510/tooth_extraction_4,samsam0510/cube_reorientation_2,samsam0510/cube_reorientation_4,samsam0510/glove_reorientation_1,vladfatu/so100_office,pranavsaroha/so100_legos4,Ityl/so100_recording2,FeiYjf/new_GtoR,dragon-95/so100_sorting_2,HITHY/so100_peach4,jpata/so100_pick_place_tangerine,HITHY/so100_strawberry,shreyasgite/so100_base_env,koenvanwijk/orange50-variation-2,pranavsaroha/so100_carrot_5,pandaRQ/pick_med_1,aractingi/push_cube_offline_data,DorayakiLin/so100_pick_charger_on_tissue,zijian2022/noticehuman3,liuhuanjim013/so100_th
# # SAMPLING_WEIGHTS=
# # DATASET_NAME=so100_community_v1


# # # Community datasets V2
# # # Inconsistent actions: 1g0rrr/sam_openpi_solder1, 1g0rrr/sam_openpi03, 1g0rrr/sam_openpi_solder2
# # # Other issues: pierfabre/rabbit bensprenger/right_arm_p_brick_in_box_with_y_noise_v0 pierfabre/horse pierfabre/pig2 pierfabre/pig3 pierfabre/cow2,pierfabre/sheep
# # # REPO_ID=Chojins/chess_game_009_white,sihyun77/suho_3_17_1,sihyun77/sihyun_3_17_2,sihyun77/suho_3_17_3,sihyun77/sihyun_3_17_5,Odog16/so100_cube_drop_pick_v1,sihyun77/sihyun_main_2,sihyun77/suho_main_2,Bartm3/dice2,sihyun77/sihyun_main_3,Loki0929/so100_duck,pietroom/holdthis,pietroom/actualeasytask,Beegbrain/pick_lemon_and_drop_in_bowl,Beegbrain/sweep_tissue_cube,zijian2022/321,gxy1111/so100_pick_place,Odog16/so100_cube_stacking_v1,sihyun77/mond_1,andlyu/so100_indoor_1,andlyu/so100_indoor_3,frk2/so100large,lirislab/sweep_tissue_cube,lirislab/lemon_into_bowl,lirislab/red_cube_into_green_lego_block,lirislab/red_cube_into_blue_cube,00ri/so100_battery,frk2/so100largediffcam,FsqZ/so100_1,ZGGZZG/so100_drop0,Chojins/chess_game_000_white_red,smanni/train_so100_fluffy_box,ganker5/so100_push_20250328,ganker5/so100_dataline_0328,ganker5/so100_color_0328,CrazyYhang/A1234-B-C_mvA2B,RasmusP/so100_Orange2Green,sixpigs1/so100_pick_cube_in_box,ganker5/so100_push_20250331,ganker5/so100_dataline_20250331,lirislab/put_caps_into_teabox,lirislab/close_top_drawer_teabox,lirislab/open_top_drawer_teabox,lirislab/unfold_bottom_right,lirislab/push_cup_target,lirislab/put_banana_bowl,Chojins/chess_game_001_blue_stereo,Chojins/chess_game_001_red_stereo,ganker5/so100_toy_20250402,Gano007/so100_medic,00ri/so100_battery_bin_center,paszea/so100_whale_2,lirislab/fold_bottom_right,lirislab/put_coffee_cap_teabox,therarelab/so100_pick_place_2,paszea/so100_whale_3,paszea/so100_whale_4,paszea/so100_lego,LemonadeDai/so100_coca,zijian2022/backgrounda,zijian2022/backgroundb,356c/so100_nut_sort_1,Mwuqiu/so100_0408_muti,aimihat/so100_tape,lirislab/so100_demo,356c/so100_duck_reposition_1,zijian2022/sort1,weiye11/so100_410_zwy,VoicAndrei/so100_banana_to_plate_only,sixpigs1/so100_stack_cube_error,isadev/bougies3,zijian2022/close3,bensprenger/left_arm_yellow_brick_in_box_v0,lirislab/guess_who_so100,bensprenger/left_arm_yellow_brick_in_box_with_purple_noise_v0,roboticshack/team16-can-stacking,zijian2022/insert2,roboticshack/team-7-right-arm-grasp-tape,Jiangeng/so100_413,roboticshack/team9-pick_cube_place_static_plate,AndrejOrsula/lerobot_double_ball_stacking_random,roboticshack/left-arm-grasp-lego-brick,roboticshack/team-7-left-arm-grasp-motor,roboticshack/team9-pick_chicken_place_plate,roboticshack/team13-two-balls-stacking,tkc79/so100_lego_box_1,roboticshack/team13-three-balls-stacking,pierfabre/chicken,roboticshack/team16-water-pouring,ad330/cubePlace,Jiafei1224/so100_pa222per,paszea/so100_lego_2cam,bensprenger/chess_game_001_blue_stereo,Mohamedal/put_banana,tkc79/so100_lego_box_2,samanthalhy/so100_herding_1,jlesein/TestBoulon7
# # REPO_ID=pierfabre/rabbit,bensprenger/right_arm_p_brick_in_box_with_y_noise_v0,pierfabre/horse,pierfabre/pig2,pierfabre/pig3,pierfabre/cow2,pierfabre/sheep,Chojins/chess_game_009_white,sihyun77/suho_3_17_1,sihyun77/sihyun_3_17_2,sihyun77/suho_3_17_3,sihyun77/sihyun_3_17_5,Odog16/so100_cube_drop_pick_v1,sihyun77/sihyun_main_2,sihyun77/suho_main_2,Bartm3/dice2,sihyun77/sihyun_main_3,Loki0929/so100_duck,pietroom/holdthis,pietroom/actualeasytask,Beegbrain/pick_lemon_and_drop_in_bowl,Beegbrain/sweep_tissue_cube,zijian2022/321,gxy1111/so100_pick_place,Odog16/so100_cube_stacking_v1,sihyun77/mond_1,andlyu/so100_indoor_1,andlyu/so100_indoor_3,frk2/so100large,lirislab/sweep_tissue_cube,lirislab/lemon_into_bowl,lirislab/red_cube_into_green_lego_block,lirislab/red_cube_into_blue_cube,00ri/so100_battery,frk2/so100largediffcam,FsqZ/so100_1,ZGGZZG/so100_drop0,Chojins/chess_game_000_white_red,smanni/train_so100_fluffy_box,ganker5/so100_push_20250328,ganker5/so100_dataline_0328,ganker5/so100_color_0328,CrazyYhang/A1234-B-C_mvA2B,RasmusP/so100_Orange2Green,sixpigs1/so100_pick_cube_in_box,ganker5/so100_push_20250331,ganker5/so100_dataline_20250331,lirislab/put_caps_into_teabox,lirislab/close_top_drawer_teabox,lirislab/open_top_drawer_teabox,lirislab/unfold_bottom_right,lirislab/push_cup_target,lirislab/put_banana_bowl,Chojins/chess_game_001_blue_stereo,Chojins/chess_game_001_red_stereo,ganker5/so100_toy_20250402,Gano007/so100_medic,00ri/so100_battery_bin_center,paszea/so100_whale_2,lirislab/fold_bottom_right,lirislab/put_coffee_cap_teabox,therarelab/so100_pick_place_2,paszea/so100_whale_3,paszea/so100_whale_4,paszea/so100_lego,LemonadeDai/so100_coca,zijian2022/backgrounda,zijian2022/backgroundb,356c/so100_nut_sort_1,Mwuqiu/so100_0408_muti,aimihat/so100_tape,lirislab/so100_demo,356c/so100_duck_reposition_1,zijian2022/sort1,weiye11/so100_410_zwy,VoicAndrei/so100_banana_to_plate_only,sixpigs1/so100_stack_cube_error,isadev/bougies3,zijian2022/close3,bensprenger/left_arm_yellow_brick_in_box_v0,lirislab/guess_who_so100,bensprenger/left_arm_yellow_brick_in_box_with_purple_noise_v0,roboticshack/team16-can-stacking,zijian2022/insert2,roboticshack/team-7-right-arm-grasp-tape,Jiangeng/so100_413,roboticshack/team9-pick_cube_place_static_plate,AndrejOrsula/lerobot_double_ball_stacking_random,roboticshack/left-arm-grasp-lego-brick,roboticshack/team-7-left-arm-grasp-motor,roboticshack/team9-pick_chicken_place_plate,roboticshack/team13-two-balls-stacking,tkc79/so100_lego_box_1,roboticshack/team13-three-balls-stacking,pierfabre/chicken,roboticshack/team16-water-pouring,ad330/cubePlace,Jiafei1224/so100_pa222per,paszea/so100_lego_2cam,bensprenger/chess_game_001_blue_stereo,Mohamedal/put_banana,tkc79/so100_lego_box_2,samanthalhy/so100_herding_1,jlesein/TestBoulon7
# # SAMPLING_WEIGHTS=
# # DATASET_NAME=so100_community_v2

# # Community datasets V1+V2
# # REPO_ID=pierfabre/rabbit,bensprenger/right_arm_p_brick_in_box_with_y_noise_v0,pierfabre/horse,pierfabre/pig2,pierfabre/pig3,pierfabre/cow2,pierfabre/sheep,Chojins/chess_game_009_white,sihyun77/suho_3_17_1,sihyun77/sihyun_3_17_2,sihyun77/suho_3_17_3,sihyun77/sihyun_3_17_5,Odog16/so100_cube_drop_pick_v1,sihyun77/sihyun_main_2,sihyun77/suho_main_2,Bartm3/dice2,sihyun77/sihyun_main_3,Loki0929/so100_duck,pietroom/holdthis,pietroom/actualeasytask,Beegbrain/pick_lemon_and_drop_in_bowl,Beegbrain/sweep_tissue_cube,zijian2022/321,gxy1111/so100_pick_place,Odog16/so100_cube_stacking_v1,sihyun77/mond_1,andlyu/so100_indoor_1,andlyu/so100_indoor_3,frk2/so100large,lirislab/sweep_tissue_cube,lirislab/lemon_into_bowl,lirislab/red_cube_into_green_lego_block,lirislab/red_cube_into_blue_cube,00ri/so100_battery,frk2/so100largediffcam,FsqZ/so100_1,ZGGZZG/so100_drop0,Chojins/chess_game_000_white_red,smanni/train_so100_fluffy_box,ganker5/so100_push_20250328,ganker5/so100_dataline_0328,ganker5/so100_color_0328,CrazyYhang/A1234-B-C_mvA2B,RasmusP/so100_Orange2Green,sixpigs1/so100_pick_cube_in_box,ganker5/so100_push_20250331,ganker5/so100_dataline_20250331,lirislab/put_caps_into_teabox,lirislab/close_top_drawer_teabox,lirislab/open_top_drawer_teabox,lirislab/unfold_bottom_right,lirislab/push_cup_target,lirislab/put_banana_bowl,Chojins/chess_game_001_blue_stereo,Chojins/chess_game_001_red_stereo,ganker5/so100_toy_20250402,Gano007/so100_medic,00ri/so100_battery_bin_center,paszea/so100_whale_2,lirislab/fold_bottom_right,lirislab/put_coffee_cap_teabox,therarelab/so100_pick_place_2,paszea/so100_whale_3,paszea/so100_whale_4,paszea/so100_lego,LemonadeDai/so100_coca,zijian2022/backgrounda,zijian2022/backgroundb,356c/so100_nut_sort_1,Mwuqiu/so100_0408_muti,aimihat/so100_tape,lirislab/so100_demo,356c/so100_duck_reposition_1,zijian2022/sort1,weiye11/so100_410_zwy,VoicAndrei/so100_banana_to_plate_only,sixpigs1/so100_stack_cube_error,isadev/bougies3,zijian2022/close3,bensprenger/left_arm_yellow_brick_in_box_v0,lirislab/guess_who_so100,bensprenger/left_arm_yellow_brick_in_box_with_purple_noise_v0,roboticshack/team16-can-stacking,zijian2022/insert2,roboticshack/team-7-right-arm-grasp-tape,Jiangeng/so100_413,roboticshack/team9-pick_cube_place_static_plate,AndrejOrsula/lerobot_double_ball_stacking_random,roboticshack/left-arm-grasp-lego-brick,roboticshack/team-7-left-arm-grasp-motor,roboticshack/team9-pick_chicken_place_plate,roboticshack/team13-two-balls-stacking,tkc79/so100_lego_box_1,roboticshack/team13-three-balls-stacking,pierfabre/chicken,roboticshack/team16-water-pouring,ad330/cubePlace,Jiafei1224/so100_pa222per,paszea/so100_lego_2cam,bensprenger/chess_game_001_blue_stereo,Mohamedal/put_banana,tkc79/so100_lego_box_2,samanthalhy/so100_herding_1,jlesein/TestBoulon7,pranavsaroha/so100_onelego2,pranavsaroha/so100_onelego3,pranavsaroha/so100_carrot_2,vladfatu/so100_above,koenvanwijk/orange50-1,CSCSXX/pick_place_cube_1.18,dragon-95/so100_sorting,dragon-95/so100_sorting_1,nbaron99/so100_pick_and_place4,Beegbrain/pick_place_green_block,dragon-95/so100_sorting_3,HITHY/so100_peach3,shreyasgite/so100_legocube_50,triton7777/so100_dataset_mix,NONHUMAN-RESEARCH/SOARM100_TASK_VENDA,mikechambers/block_cup_14,samsam0510/tooth_extraction_3,samsam0510/tooth_extraction_4,samsam0510/cube_reorientation_2,samsam0510/cube_reorientation_4,samsam0510/glove_reorientation_1,vladfatu/so100_office,pranavsaroha/so100_legos4,Ityl/so100_recording2,FeiYjf/new_GtoR,dragon-95/so100_sorting_2,HITHY/so100_peach4,jpata/so100_pick_place_tangerine,HITHY/so100_strawberry,shreyasgite/so100_base_env,koenvanwijk/orange50-variation-2,pranavsaroha/so100_carrot_5,pandaRQ/pick_med_1,aractingi/push_cube_offline_data,DorayakiLin/so100_pick_charger_on_tissue,zijian2022/noticehuman3,liuhuanjim013/so100_th
# REPO_ID=pierfabre/rabbit,bensprenger/right_arm_p_brick_in_box_with_y_noise_v0,pierfabre/horse,pierfabre/pig2
# SAMPLING_WEIGHTS=

# # # Community V3
# # # issues, yskim2025/unitylerobot (version), cranberrysoft/so100 (don't exist),29  datasets different actions: nguyen-v/so100_rotate_red_button satvikahuja/mixer_on_off_new_1 ...
# # REPO_ID=satvikahuja/mixer_on_off_new_1,aergogo/so100_pick_place,andy309/so100_0314_fold_cloths,jchun/so100_pickplace_small_20250323_120056,astroyat/cube,Ofiroz91/so_100_cube2bowl,HappyPablo/dec3_data2,ZCM5115/so100_1210,francescocrivelli/orange_feeding,francescocrivelli/carrot_eating,0x00raghu/toffee_red,0x00raghu/toffee_red_2,0x00raghu/toffee_red_3__,0x00raghu/toffee_blue,0x00raghu/toffee_blue_2,0x00raghu/toffee_to_hand_1,0x00raghu/toffee_to_hand_2,liyitenga/so100_bi_hello,liyitenga/so100_bi_giveme5,ZCM5115/so100_2Arm3cameras_movebox,pranavsaroha/so100_carrot_1,pranavsaroha/so100_carrot_3,pranavsaroha/so100_carrot_4,maximilienroberti/so100_lego_red_box,pranavsaroha/so100_squishy,rabhishek100/so100_train_dataset,pranavsaroha/so100_squishy100,swarajgosavi/kikobot_pusht_real_v2,pandaRQ/pickmed,swarajgosavi/act_kikobot_pusht_real,pranavsaroha/so100_squishy2colors,pranavsaroha/so100_squishy2colors_1,Chojins/chess_game_001_white,jmrog/so100_sweet_pick,Chojins/chess_game_002_white,pranavsaroha/so100_squishy2colors_2_new,Chojins/chess_game_003_white,aractingi/pick_place_lego_cube,Chojins/chess_game_004_white,Chojins/chess_game_005_white,Chojins/chess_game_006_white,Chojins/chess_game_007_white,koenvanwijk/blue2,jlitch/so100multicam3,koenvanwijk/blue52,jlitch/so100multicam6,aractingi/pick_place_lego_cube_1,jlitch/so100multicam7,vladfatu/so100_ds,Chojins/chess_game_000_white,HITHY/so100-kiwi,HITHY/so100_peach1,HITHY/so100_redstrawberry,satvikahuja/orange_mixer_1,satvikahuja/mixer_on_off,satvikahuja/orange_pick_place_new1,satvikahuja/mixer_on_off_new,danmac1/real_real332,FeiYjf/Makalu_push,liyitenga/so100_pick_taffy1,chmadran/so100_dataset04,FeiYjf/Maklu_dataset,FeiYjf/new_Dataset,liyitenga/so100_pick_taffy2,satvikahuja/mixer_on_off_new_4,CSCSXX/pick_place_cube_1.17,liyitenga/so100_pick_taffy3,liyitenga/so100_pick_taffy4,yuz1wan/so100_pick_pink,yuz1wan/so100_pick_wahaha,yuz1wan/so100_pp_pink,yuz1wan/so100_pour_cup,liyitenga/so100_pick_taffy5,liyitenga/so100_pick_taffy6,yuz1wan/so100_button,yuz1wan/so100_pickplace,liyitenga/so100_pick_taffy7,FeiYjf/push_gg,FeiYjf/push_0094,swarajgosavi/act_kikobot_block_real,liyitenga/so100_pick_taffy8,phospho-ai/OrangeBrick3Cameras,vaishanthr/toy_pick_place,SeanLMH/so100_picknplace_v2,pepijn223/yellow_lego_in_box1,DimiSch/so100_50ep_2,DimiSch/so100_50ep_3,SeanLMH/so100_picknplace,nbaron99/so100_pick_and_place2,chmadran/so100_dataset08,vaishanthr/toy_pickplace_50ep,Beegbrain/pick_place_green_block_lr,Ityl/so100_recording1,vaishanthr/toy_pickplace,ad330/so100_box_pickPlace,Beegbrain/so100_put_cube_cup,aractingi/push_green_cube_hf,aractingi/push_green_cube_hf_cropped_resized,carpit680/giraffe_task,carpit680/giraffe_sock_demo_1,DimiSch/so100_terra_50_2,carpit680/giraffe_sock_demo_2,aractingi/push_cube_to_face_reward,aractingi/push_cube_to_face_reward_cropped_resized,aractingi/push_cube_reward_data,aractingi/push_cube_reward_data_cropped_resized,aractingi/push_cube_offline_data_cropped_resized,aractingi/push_cube_front_side_reward,aractingi/push_cube_front_side_reward_cropped_resized,aractingi/push_cube_front_side_reward_long,aractingi/push_cube_front_side_reward_long_cropped_resized,aractingi/push_cube_reward,aractingi/push_cube_reward_cropped_resized,aractingi/push_cube_square_reward_cropped_resized,aractingi/push_cube_square_reward_1,aractingi/push_cube_square_reward_1_cropped_resized,aractingi/push_cube_square_light_reward,aractingi/push_cube_square_light_offline_demo,aractingi/push_cube_square_light_offline_demo_cropped_resized,denghj/dataset_red_tape01,aractingi/push_cube_square_offline_demo,aractingi/push_cube_square_offline_demo_cropped_resized,Beegbrain/stack_two_cubes,FeiYjf/Test_NNNN,LegrandFrederic/Orange-brick-lower-resolution,aractingi/pick_place_lego_cube_cropped_resized,aractingi/push_cube_overfit,aractingi/push_cube_overfit_cropped_resized,HITHY/so100_peach,zaringleb/so100_cube_2,andreasBihlmaier/dual_arm_transfer_2025_02_16,zaringleb/so100_cube_4_binary,1g0rrr/reward_pickplace1,1g0rrr/reward_pickplace1_cropped_resized,FeiYjf/Hold_Pieces,FeiYjf/Grab_Pieces,hegdearyandev/so100_eraser_cup_v1,jbraumann/so100_1902,liyitenga/so100_pick_taffy10,mikechambers/block_cup_5,zaringleb/so100_cube_5_linear,yuz1wan/so100_pickplace_0223_2,yuz1wan/so100_pickplace_0223_3,samsam0510/mj_data_temp,samsam0510/tape_insert_1,samsam0510/tape_insert_2,pengjunkun/so100_push_to_hole,Deason11/Random_Kitchen,1g0rrr/reward_dataset_name2,1g0rrr/reward_dataset_name2_cropped_resized,1g0rrr/offline_dataset_name2,1g0rrr/offline_dataset_name2_cropped_resized,aractingi/push_cube_simp_cropped_resized,danielkr452/so100_work6,Loki0929/so100_100,yuz1wan/so100_fold_0227_1,yuz1wan/so100_fold_0227_2,speedyyoshi/so100_grasp_pink_block,lirislab/stack_two_red_cubes,lirislab/red_cube_into_mug,lirislab/green_lego_block_into_mug,lirislab/green_lego_block_into_mug_easy,kevin510/lerobot-cat-toy-placement,NONHUMAN-RESEARCH/SOARM100_TASK_VENDA_BOX,wangjl1512/pour_water,airthebear/so100_GL,zijian2022/noticehuman1,zijian2022/noticehuman2,kantine/so100_kapla_tower6,zijian2022/noticehuman5,zijian2022/llm40,Ashton3/lerobot-aloha,zijian2022/noticehuman50,AaronNewman/screwdriver_task_batch1,AaronNewman/screwdriver_task_batch2,AaronNewman/screwdriver_task_batch3,zijian2022/noticehuman60,zijian2022/noticehuman70,Bartm3/tape_to_bin,liuhuanjim013/so100_th_1,Pi-robot/barbecue_flip,Pi-robot/barbecue_put,wangjl1512/doll,sshh11/so100_orange_50ep_1,sshh11/so100_orange_50ep_2,DorayakiLin/so100_pick_cube_in_box,Bartm3/tape_to_bin2,luke250305/play_dice_250311.1,andy309/so100_0311_1152,sihyun77/suho_so100,sihyun77/si_so100,shreyasgite/so100_base_left,sihyun77/suho_red,liuhuanjim013/so100_block,andy309/so100_0313_no_wrist_camera,zijian2022/l9,zijian2022/n1_2,DorayakiLin/so100_stack_cube,andy309/so100_0313_no_wrist_camera_with_two_arms_cloths,joaoocruz00/so100_makeitD1,zijian2022/l10_1,zijian2022/l10_5,sihyun77/suho_red2,sihyun77/suho_angel,sihyun77/sihyun_king,acrampette/third_arm_01,Winster/so100_cube,1g0rrr/sam_openpi03,thedevansh/mar16_1336,hkphoooey/throw_stuffie,doujiangwang/task1_10epi_100000step,sihyun77/sihyun_3_17_1,acrampette/third_arm_02,imsyed00/so100_yellowbowl_pickplace_1,kumarhans/so100_tape_task,sihyun77/sihyun_main,doujiangwang/task2_10epi_100000step,kantine/industrial_robothon_buttons_expert,kantine/industrial_robothon_buttons_anomaly,kantine/industrial_robothon_hatchAndProbe_expert,kantine/industrial_robothon_hatchAndProbe_anomaly,Odog16/so100_tea_towel_folding_v1,zijian2022/so100_318,zijian2022/so100_318_1,Congying1112/so100_place_blue_bottle_with_two_cameras,Congying1112/so100_place_blue_bottle_with_two_cameras2,Congying1112/so100_place_blue_bottle_with_single_camera,pietroom/first_task_short,kantine/industrial_screws_sorting_expert,kantine/industrial_screws_sorting_anomaly,pietroom/second_task,zijian2022/c0,doujiangwang/task4_10epi_100000step,Congying1112/so100_switch_with_onhand_camera,HYAIYN/so100_get_orange_10epi,doujiangwang/task5_10epi_100000step,1g0rrr/sam_openpi_cube_low10,1g0rrr/sam_openpi_cube_top10,1g0rrr/sam_openpi_wire10,1g0rrr/sam_openpi_solder1,1g0rrr/sam_openpi_solder2,wcode/so100_put_pen_50,jchun/so100_pickplace_small_20250322_193929,bnarin/so100_tic_tac_toe_we_do_it_live,dc2ac/so100-t5,chmadran/so100_home_dataset,baladhurgesh97/so100_final_picking_3,bnarin/so100_tic_tac_toe_move_0_0,bnarin/so100_tic_tac_toe_move_1_0,bnarin/so100_tic_tac_toe_move_2_1,bnarin/so100_tic_tac_toe_move_4_0,zaringleb/so100_cube_6_2d,andlyu/so100_indoor_0,andlyu/so100_indoor_2,Winster/so100_sim,badwolf256/so100_twin_cam_duck,Congying1112/so100_simplepick_with_2_cameras_from_top,andlyu/so100_indoor_4,Zak-Y/so100_grap_dataset,kantine/domotic_pouringCoffee_expert,kantine/domotic_pouringCoffee_anomaly,lucasngoo/so100_strawberry_grape,kantine/domotic_makingCoffee_expert,kantine/domotic_makingCoffee_anomaly,ZGGZZG/so100_drop1,kantine/industrial_soldering_expert,kantine/industrial_soldering_anomaly,Yotofu/so100_sweeper_shoes,kantine/domotic_dishTidyUp_expert,kantine/domotic_dishTidyUp_anomaly,kantine/domotic_groceriesSorting_expert,kantine/domotic_groceriesSorting_anomaly,badwolf256/so100_twin_cam_duck_v2,kantine/domotic_vegetagblesAndFruitsSorting_expert,kantine/domotic_vegetagblesAndFruitsSorting_anomaly,kantine/domotic_setTheTable_expert,kantine/domotic_setTheTable_anomaly,therarelab/so100_pick_place,abhisb/so100_51_ep,andlyu/so100_indoor_val_0,allenchienxxx/so100Test,lizi178119985/so100_jia,badwolf256/so100_twin_cam_duck_v3,andrewcole712/so100_tape_bin_place,Gano007/so100_lolo,Zak-Y/so100_three_cameras_dataset,Gano007/so100_doliprane,XXRRSSRR/so100_v3_num_episodes_50,zijian2022/assemblyarm2,ganker5/so100_action_20250403,andlyu/so100_indoor_val2,Gano007/so100_gano,paszea/so100_whale_grab,paszea/so100_whale,Clementppr/lerobot_pick_and_place_dataset_world_model,andlyu/so100_indoor_10,RasmusP/so100_dataset50ep_a,RasmusP/so100_dataset50ep,Gano007/so100_second,zaringleb/so100_cude_linear_and_2d_comb,dsfsg/grasp_pens,zijian2022/digitalfix,zijian2022/digitalfix2,zijian2022/digitalfix3,T1g3rGE/so100_pickplace_small_20250407_171912,sihyun77/mond_13,abokinala/sputnik_100_11_pick_place_container,dsfsg/bring_bottle,duthvik/sputnik_100_13_pick_place_container,abokinala/sputnik_100_12_pick_place_container,Mwuqiu/so100_0408,AK51/4090_01,356c/so100_rope_reposition_1,paszea/so100_lego_mix,abokinala/sputnik_100_14_pick_place_container,abokinala/sputnik_100_23_pick_place_surface,jiajun001/eraser00_2,jlesein/TestBoulon2,duthvik/sputnik_100_31_pour_liquid,duthvik/sputnik_100_24_pick_place_surface,duthvik/sputnik_100_25_pick_place_surface,duthvik/sputnik_100_17_pick_place_container,duthvik/sputnik_100_26_pick_place_surface,VoicAndrei/so100_banana_to_plate_rebel_full,isadev/bougies1,danaaubakirova/so100_task_1,danaaubakirova/so100_task_2,danaaubakirova/so100_task_3,danaaubakirova/so100_task_4,sixpigs1/so100_pick_cube_in_box_error,sixpigs1/so100_push_cube_error,sixpigs1/so100_pull_cube_error,isadev/bougies2,therarelab/med_dis_rare_6,duthvik/sputnik_100_27_pick_place_surface,zijian2022/closer3,duthvik/sputnik_100_41_custom_tasks,duthvik/sputnik_100_42_custom_tasks,duthvik/sputnik_100_43_custom_tasks,duthvik/sputnik_100_44_custom_tasks,duthvik/sputnik_100_51_kitchen_tasks,duthvik/sputnik_100_52_kitchen_tasks,duthvik/sputnik_100_53_kitchen_tasks,duthvik/sputnik_100_45_custom_tasks,duthvik/sputnik_100_32_pour_liquid,duthvik/sputnik_100_29_pick_place_surface,duthvik/sputnik_100_18_pick_place_container,sixpigs1/so100_pull_cube_by_tool_error,sixpigs1/so100_insert_cylinder_error,abokinala/sputnik_100_54_kitchen_tasks,abokinala/sputnik_100_55_kitchen_tasks,m1b/so100_bluelego,abokinala/sputnik_100_46_custom_tasks,m1b/so100_bluelego_updt,kantine/flip_A0,kantine/flip_A1,kantine/flip_A2,kantine/flip_A3,lirislab/guess_who_no_cond,kantine/flip_A4,kantine/flip_A5,lirislab/guess_who_lighting,nguyen-v/so100_press_red_button,nguyen-v/so100_bimanual_grab_lemon_put_in_box2,pierfabre/cow,nguyen-v/press_red_button_new,nguyen-v/so100_rotate_red_button,raghav-katta-1/lerobot2,Cidoyi/so100_all_notes,roboticshack/team10-red-block,Cidoyi/so100_all_notes_1,roboticshack/team_5-QuiEstCe_everyBox,roboticshack/team11_pianobot,roboticshack/team2-guess_who_so100,roboticshack/team2-guess_who_so100_light,roboticshack/team2-guess_who_so100_edge_case,roboticshack/team2-guess_who_less_ligth,Cidoyi/so100_all_notes_3,dsfsg/grasp_pen_and_bottle,abokinala/sputnik_100_60_kitchen_tasks,abokinala/sputnik_100_58_kitchen_tasks,danaaubakirova/so100_v2_task_1,danaaubakirova/so100_v2_task_2,danaaubakirova/so100_v2_task_3,danaaubakirova/so100_v2_task_4,zijian2022/force1,zijian2022/force2,zijian2022/force3,jiajun001/eraser00_3,zijian2022/bi2,zijian2022/bi1,zijian2022/hand1,Setchii/so100_grab_ball,MossProphet/so100_square-1-2-3.2
# # SAMPLING_WEIGHTS=
# # DATASET_NAME=so100_community_v3

# ##########################

# ROBOT=so100
# export TOKENIZERS_PARALLELISM=false
# export MUJOCO_GL=egl



# SAMPLING_WEIGHTS=
# FEATURES_VERSION=2
# NUM_IMAGE_TRANSFORMS=10
# TRAIN_ON_ALL_FEATURES=true
# NORM_PER_ROBOT=true
# USE_IMAGENET_STATS=false

# MAX_STATE_DIM=6
# MAX_ACTION_DIM=6
# MAX_NUM_IMAGES=3
# MAX_IMAGE_DIM=256


# SEED=5000
# BATCH_SIZE=32
# # EVAL_STEPS=1000
# EVAL_STEPS=100





# SELF_ATTN_ONLY_ACTIONS=false
# EXPERT_WIDTH_MULTIPLIER=0.75
# PAST_OBS_KEYS="image"
# N_OBS_STEPS=1
# NUM_EXPERT_LAYERS=0
# CHUNK_SIZE=50
# NUM_VLM_LAYERS=16
# PAD_LANG_TO=longest
# EVAL_CKPT=/lustre/fswork/projects/rech/dyf/ugz83ue/logs/lerobot/lerobot_so100_community_v1_v2_smolpi0_lr1e-4bs64steps200000gpus4freeze32_imgtoktrue_cross_attn_gap1_localimgfalse_statetopreftrue_explay0_vlml16_causalacttrue_sa2_smolvlm2500_chunk50_nobs1_expw0.75_feat2_lrvlm1e-4_droptrue_max_length/checkpoints/080000/pretrained_model/
# ADD_IMAGE_TOKENS=true
# ATTN_MODE=cross_attn
# STATE_TO_PREFIX=true
# CAUSAL_ACTION_ATTENTION_MASK=true
# SELF_ATTN_EVERY_N_LAYERS=2
# VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct


# python lerobot/scripts/offline_inference.py \
#     --output_dir=$WORK/logs/lerobot/tmp \
#     --batch_size=$BATCH_SIZE \
#     --seed=$SEED \
#     --eval_steps=$EVAL_STEPS \
#     --use_amp=false \
#     --device=cuda \
#     --dataset.repo_id=$REPO_ID --dataset.local_files_only=true --dataset.sampling_weights=$SAMPLING_WEIGHTS --dataset.use_imagenet_stats=$USE_IMAGENET_STATS --policy.normalize_per_robot_type=$NORM_PER_ROBOT \
#     --dataset.image_transforms.max_num_transforms=$NUM_IMAGE_TRANSFORMS --dataset.image_transforms.enable=true --dataset.train_on_all_features=$TRAIN_ON_ALL_FEATURES \
#     --dataset.max_action_dim=$MAX_ACTION_DIM --dataset.max_state_dim=$MAX_STATE_DIM --dataset.max_num_images=$MAX_NUM_IMAGES --dataset.max_image_dim=$MAX_IMAGE_DIM --dataset.features_version=$FEATURES_VERSION \
#     --policy.type=$POLICY \
#     --policy.checkpoint_path=$EVAL_CKPT \
#     --policy.checkpoint_keys_mapping=$CKPT_KEYS_MAPPING \
#     --policy.add_image_special_tokens=$ADD_IMAGE_TOKENS \
#     --policy.attention_mode=$ATTN_MODE \
#     --policy.causal_action_attention_mask=$CAUSAL_ACTION_ATTENTION_MASK \
#     --policy.state_to_prefix=$STATE_TO_PREFIX \
#     --policy.self_attn_every_n_layers=$SELF_ATTN_EVERY_N_LAYERS \
#     --policy.vlm_model_name=$VLM_NAME \
#     --policy.pad_language_to=$PAD_LANG_TO \
#     --policy.load_vlm_weights=$LOAD_VLM_WEIGHTS \
#     --policy.num_vlm_layers=$NUM_VLM_LAYERS \
#     --policy.chunk_size=$CHUNK_SIZE \
#     --policy.n_obs_steps=$N_OBS_STEPS \
#     --policy.past_obs_keys=$PAST_OBS_KEYS \
#     --policy.num_expert_layers=$NUM_EXPERT_LAYERS \
#     --policy.expert_width_multiplier=$EXPERT_WIDTH_MULTIPLIER \
#     --policy.peft_method=$PEFT_METHOD \
#     --policy.self_attn_only_actions=$SELF_ATTN_ONLY_ACTIONS \
#     --policy.robot_type=$ROBOT 




# MULTITASK_EVAL=true
# N_EPISODES=5
# MAX_PARRALLEL_TASKS=1
# ACTION_STEPS_LIST=(1)
# TASK_LIST=(libero_10)
# for N_ACTION_STEPS in "${ACTION_STEPS_LIST[@]}"; do
#     for TASK in "${TASK_LIST[@]}"; do
#         echo "$TASK Evaluating: $EVAL_CKPT | N_ACTION_STEPS=$N_ACTION_STEPS"
#         python lerobot/scripts/eval.py \
#             --output_dir=$WORK/logs/lerobot/tmp \
#             --env.type=$ENV \
#             --env.task=$TASK \
#             --eval.batch_size=$N_EPISODES \
#             --eval.n_episodes=$N_EPISODES \
#             --use_amp=false \
#             --device=cuda \
#             --policy.n_action_steps=$N_ACTION_STEPS \
#             --policy.type=$POLICY \
#             --policy.checkpoint_path=$EVAL_CKPT \
#             --policy.checkpoint_keys_mapping=$CKPT_KEYS_MAPPING \
#             --env.multitask_eval=$MULTITASK_EVAL --env.max_parallel_tasks=$MAX_PARRALLEL_TASKS \
#             --policy.add_image_special_tokens=$ADD_IMAGE_TOKENS \
#             --policy.attention_mode=$ATTN_MODE \
#             --policy.causal_action_attention_mask=$CAUSAL_ACTION_ATTENTION_MASK \
#             --policy.state_to_prefix=$STATE_TO_PREFIX \
#             --policy.self_attn_every_n_layers=$SELF_ATTN_EVERY_N_LAYERS \
#             --policy.vlm_model_name=$VLM_NAME \
#             --policy.load_vlm_weights=$LOAD_VLM_WEIGHTS \
# --policy.num_vlm_layers=$NUM_VLM_LAYERS \
# --policy.chunk_size=$CHUNK_SIZE 

#         echo "Done with: $EVAL_CKPT | Steps=$N_ACTION_STEPS"
#         echo "------------------------------------------------------"
#     done
# done

