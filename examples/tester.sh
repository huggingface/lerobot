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
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

python examples/tester.py