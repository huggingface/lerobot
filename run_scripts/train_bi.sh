if [ -n "$1" ]; then
    HF_USER="$1"
    echo "Using HF_USER from command line argument: $HF_USER"
elif [ -z "$HF_USER" ]; then
    echo "Error: HF_USER environment variable is not set and no username provided."
    echo "Please either:"
    echo "  1. Set environment variable: export HF_USER=your_huggingface_username"
    echo "  2. Or run with username as argument: ./record_so101.sh your_huggingface_username"
    exit 1
else
    echo "Using HF_USER from environment: $HF_USER"
fi

# lerobot-train \
#   --dataset.repo_id=${HF_USER}/boxing-the-blocks  \
#   --policy.type=act \
#   --output_dir=outputs/train/act_so101_bi_boxing_block_2 \
#   --job_name=act_so101_bi_boxing_block_2 \
#   --policy.device=cuda \
#   --wandb.enable=false \
#   --policy.repo_id=${HF_USER}/act_so101_bi_boxing_block_2


lerobot-train \
  --dataset.repo_id=${HF_USER}/boxing-the-blocks  \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_so101_bi_boxing_block_2 \
  --job_name=diffusion_so101_bi_boxing_block_2 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=${HF_USER}/diffusion_so101_bi_boxing_block_2